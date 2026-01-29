#! /usr/bin/env python

import numpy as np
np.float = np.float64

from habitat.config.default_structured_configs import (
    HabitatSimSemanticSensorConfig,
    GPSSensorConfig,
    TopDownMapMeasurementConfig,
    HeadingSensorConfig,
)
from custom_sensors import register_sensors, AgentPositionSensorConfig
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

import rospy
import habitat
from std_msgs.msg import Int32
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from keyboard_agent import KeyboardAgent
from shortest_path_follower_agent import ShortestPathFollowerAgent
from greedy_path_follower_agent import GreedyPathFollowerAgent
from custom_sensors import AgentPositionSensor
from publishers import HabitatObservationPublisher
from habitat_baselines.config.default import get_config
from ddppo import DDPPOAgent

from skimage.io import imsave
from tqdm import tqdm
from habitat_map import env_orb
from cv_bridge import CvBridge
from PIL import Image
import cv2
import os
import roslaunch
import gc
import subprocess
import time
import tf
import math

DEFAULT_RATE = 30
DEFAULT_AGENT_TYPE = 'keyboard'
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1

class HabitatRunner():
    def __init__(self, name):
        # Initialize ROS node and take arguments
        self.name = name
        task_config = rospy.get_param('~task_config')
        rate_value = rospy.get_param('~rate', DEFAULT_RATE)
        agent_type = rospy.get_param('~agent_type', DEFAULT_AGENT_TYPE)
        if agent_type == 'shortest_path_follower':
            self.goal_positions_path = rospy.get_param('~goal_positions_path', None)
        self.goal_radius = rospy.get_param('~goal_radius', DEFAULT_GOAL_RADIUS)
        self.max_d_angle = rospy.get_param('~max_d_angle', DEFAULT_MAX_ANGLE)
        rgb_topic = rospy.get_param('~rgb_topic', None)
        depth_topic = rospy.get_param('~depth_topic', None)
        semantic_topic = rospy.get_param('~semantic_topic', None)
        camera_info_topic = rospy.get_param('~camera_info_topic', None)
        #semantic_mask_topic = rospy.get_param('~semantic_mask_topic', None)
        true_pose_topic = rospy.get_param('~true_pose_topic', None)
        camera_info_file = rospy.get_param('~camera_calib', None)
        scene_name = rospy.get_param('~scene_name', None)
        print('SCENE NAME:', scene_name)
        print('TASK CONFIG:', task_config)
        self.scene_name = scene_name

        self.rate = rospy.Rate(rate_value)
        self.publisher = HabitatObservationPublisher(rgb_topic, 
                                                    depth_topic, 
                                                    #semantic_topic,
                                                    camera_info_topic, 
                                                    true_pose_topic,
                                                    camera_info_file)
        # Now define the config for the sensor
        self.action_publisher = rospy.Publisher('habitat_action', Int32, latch=True, queue_size=100)
        self.map_publisher = rospy.Publisher('habitat/map', OccupancyGrid, latch=True, queue_size=100)
        self.reset_publisher = rospy.Publisher('/reset_exploration', String, latch=True, queue_size=100)
        self.robot_pose_publisher = rospy.Publisher('/robot_pose_in_habitat_coords', PoseStamped, latch=True, queue_size=100)

        self.pointgoal_sub = rospy.Subscriber(
            '/pointgoal',
            PoseStamped,
            self.pointgoal_callback,
            queue_size=1
        )

        self.goal_publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

        self.goal_received = False
        self.robot_pose_in_slam_coords = None
        self.robot_pose_in_habitat_coords = None
        self.goal_pose_in_slam_coords = None
        self.goal_pose_in_habitat_coords = None
        
        # Now define the config for the sensor
        habitat_path = '/habitat-lab/data'
        config = habitat.get_config(task_config)
        OmegaConf.set_readonly(config, False)
        OmegaConf.set_struct(config, False)

        config.habitat.environment.iterator_options.shuffle = False
        # config.habitat.environment.max_episode_steps = 5
        
        config.habitat.task.lab_sensors.agent_position_sensor = AgentPositionSensorConfig(
            answer_to_life=42
        )
        
        config.habitat.task.lab_sensors.heading_sensor = HeadingSensorConfig()
        
        config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor = HabitatSimSemanticSensorConfig()
        
        config.habitat.simulator.scene_dataset = os.path.join(
            habitat_path,
            "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )
        
        config.habitat.task.measurements.top_down_map = TopDownMapMeasurementConfig()
        config.habitat.task.lab_sensors["heading_sensor"] = HeadingSensorConfig()
        
        OmegaConf.set_struct(config, True)
        OmegaConf.set_readonly(config, True)
        self.config = config

        # Initialize the agent and environment
        self.env = habitat.Env(config=config)
        print('Environment created')

        if agent_type == 'keyboard':
           self.agent = KeyboardAgent()
        elif agent_type == 'shortest_path_follower':
            goal_positions = np.loadtxt(self.goal_positions_path)
            self.agent = ShortestPathFollowerAgent(self.env, self.goal_radius, goal_positions)
        elif agent_type == 'greedy_path_follower':
            self.agent = GreedyPathFollowerAgent(self.goal_radius, self.max_d_angle)
        elif agent_type == 'random_movement':
            self.agent = RandomMovementAgent()
        elif agent_type == 'ddppo':
            self.agent = DDPPOAgent()
        else:
            print('AGENT TYPE {} IS NOT DEFINED!!!'.format(agent_type))
            return
        
        self.dataset_save_path = '/data/datasets/opr_training_data/gibson'

    def publish_map(self):
        occupancy_map = self.mapper.mapper.map
        map_msg = OccupancyGrid()
        map_msg.header.stamp = rospy.Time.now()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = self.mapper.mapper.resolution / 100.
        map_msg.info.width = occupancy_map.shape[1]
        map_msg.info.height = occupancy_map.shape[0]
        map_msg.info.origin.position.x = -occupancy_map.shape[1] * self.mapper.mapper.resolution / 200.
        map_msg.info.origin.position.y = -occupancy_map.shape[0] * self.mapper.mapper.resolution / 200.
        map_data = np.ones((map_msg.info.height, map_msg.info.width), dtype=np.int8) * (-1)
        map_data[occupancy_map[:, :, 0] > 0] = 0
        map_data[occupancy_map[:, :, 1] > 0] = 100
        map_msg.data = list(map_data.ravel())
        self.map_publisher.publish(map_msg)

    def get_robot_pose(self, observations):
        current_x, current_y = observations['gps']
        current_y = -current_y
        robot_angle = observations['compass'][0]
        current_x_new = current_x * math.cos(-robot_angle) + current_y * math.sin(-robot_angle)
        current_y_new = -current_x * math.sin(-robot_angle) + current_y * math.cos(-robot_angle)
        return current_x, current_y, robot_angle

    def pointgoal_callback(self, msg):
        self.goal_received = True
        self.goal_received_once = True
        # Receive goal pose in SLAM coords
        # print('Received goal with coords: {}, {}'.format(msg.pose.position.x, msg.pose.position.y))
        goal_x, goal_y = msg.pose.position.x, msg.pose.position.y
        self.goal_pose_in_habitat_coords = np.array([goal_x, goal_y, msg.pose.position.z])

        # Find robot's position and orientation in SLAM and Habitat coords
        # habitat_position, habitat_orientation = self.robot_pose_in_habitat_coords
        # print('Robot pose in habitat coords:', habitat_position, habitat_orientation)
        # habitat_y, habitat_z, habitat_x = habitat_position
        # _, __, habitat_angle = tf.transformations.euler_from_quaternion([habitat_orientation.x, habitat_orientation.z, habitat_orientation.y, habitat_orientation.w])

        # # Calculate transform between SLAM and Habitat coordinate systems
        # d_angle = self.normalize(habitat_angle - self.slam_angle + np.pi)
        # #print('D_ANGLE:', d_angle)
        # dx = habitat_x - (self.slam_x * math.cos(d_angle) + self.slam_y * math.sin(d_angle))
        # dy = habitat_y - (-self.slam_x * math.sin(d_angle) + self.slam_y * math.cos(d_angle))

        # # Compute goal position in Habitat coords
        # goal_x_rotated = goal_x * math.cos(d_angle) + goal_y * math.sin(d_angle)
        # goal_y_rotated = -goal_x * math.sin(d_angle) + goal_y * math.cos(d_angle)
        #print('GOAL COORDS IN HABITAT SYSTEM:', self.goal_pose_in_habitat_coords)

    def normalize(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def publish_robot_pose(self, observations):
        self.slam_x, self.slam_y, self.slam_angle = self.get_robot_pose(observations)
        self.robot_pose_in_habitat_coords = observations['agent_position']
        robot_position, robot_rotation = self.robot_pose_in_habitat_coords
        robot_pose_msg = PoseStamped()
        robot_pose_msg.header.stamp = rospy.Time.now()
        robot_pose_msg.header.frame_id = 'habitat'
        robot_pose_msg.pose.position.x = robot_position[0]
        robot_pose_msg.pose.position.y = robot_position[1]
        robot_pose_msg.pose.position.z = robot_position[2]
        robot_pose_msg.pose.orientation.w = robot_rotation.w
        robot_pose_msg.pose.orientation.x = robot_rotation.x
        robot_pose_msg.pose.orientation.y = robot_rotation.y
        robot_pose_msg.pose.orientation.z = robot_rotation.z
        self.robot_pose_publisher.publish(robot_pose_msg)

    def publish_goal(self):
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = goal_map[0]
        goal_msg.pose.position.y = goal_map[1]
        goal_msg.pose.position.z = habitat_z

        quat = tf.transformations.quaternion_from_euler(0, 0, 1)
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]

        self.goal_publisher.publish(goal_msg)

    def run_episode(self, episode=0):
        observations = self.env.reset()
        self.agent.reset()
        self.goal_received_once = False

        reset_msg = String()
        reset_msg.data = 'reset'
        self.reset_publisher.publish(reset_msg)
        
        """
        points = self.env.get_navigable_points()
        orientations = [[0, 0, 0, 1], [0, 0.7071, 0, 0.7071], [0, 1, 0, 0], [0, -0.7071, 0, 0.7071]]
        poses = []
        i = 0
        if not os.path.exists(os.path.join(self.dataset_save_path, self.scene_name)):
            os.mkdir(os.path.join(self.dataset_save_path, self.scene_name))
        for pt in tqdm(points):
            for ori in orientations:
                i += 1
                observations = self.env.reset(start_position=pt, start_orientation=ori)
                #self.env.step(HabitatSimActions.TURN_LEFT)
                #print(observations.keys())
                rgb = observations['rgb']
                depth = observations['depth']
                depth = (depth * 255).astype(np.uint8)
                poses.append(pt + ori)
                imsave(os.path.join(self.dataset_save_path, self.scene_name, '{}_rgb.png'.format(i)), rgb)
                imsave(os.path.join(self.dataset_save_path, self.scene_name, '{}_depth.png'.format(i)), depth)
        np.savetxt(os.path.join(self.dataset_save_path, self.scene_name, 'poses.txt'), np.array(poses))
        return
        """
        self.env.step(HabitatSimActions.move_forward)
        step_start_time = rospy.Time.now()
        self.publisher.publish(observations, step_start_time)
        self.publish_robot_pose(observations)
        rospy.sleep(3)
        current_episode = self.env.current_episode
        goal_y, goal_z, goal_x = current_episode.goals[0].position

        self.slam_x, self.slam_y, self.slam_angle = self.get_robot_pose(observations)

        step = 0
        prev_dist = 0
        while not rospy.is_shutdown() and not self.env.episode_over:
            step_start_time = rospy.Time.now()
            t0 = rospy.Time.now().to_sec()
            self.publisher.publish(observations, step_start_time)
            self.publish_robot_pose(observations)
            t1 = rospy.Time.now().to_sec()
            #print('Publish time:', t1 - t0)
            metrics = self.env.get_metrics()
            distance_to_goal = metrics['distance_to_goal']

            self.publisher.publish(observations, rospy.Time.now())
            self.publish_robot_pose(observations)

            action = self.agent.act(observations)
            # action = HabitatSimActions.turn_left
            t2 = rospy.Time.now().to_sec()
            before_publish = rospy.Time.now().to_sec()
            #self.action_publisher.publish(action_msg)
            observations = self.env.step(action)
            robot_x, robot_y = observations['gps']
            robot_y = -robot_y
            robot_angle = observations['compass'][0]
            print('x y angle:', robot_x, robot_y, robot_angle)
            if isinstance(robot_angle, (list, np.ndarray)):
                robot_angle = float(robot_angle[0])
            after_publish = rospy.Time.now().to_sec()
            # trajectory.append((robot_x, robot_y, robot_angle, step_start_time.to_sec()))
            t3 = rospy.Time.now().to_sec()
            self.goal_received = False
            # self.rate.sleep()
            step += 1
            
        metrics = self.env.get_metrics()
        print('METRICS:', metrics)
        success = metrics['success']
        spl = metrics['spl']
        distance_to_goal = metrics['distance_to_goal']
        rospy.sleep(5)
        return success, spl, distance_to_goal, step


def main():
    register_sensors()
    rospy.init_node('habitat_ros_node', anonymous=True)
    scene_name = rospy.get_param('~scene_name', None)
    print(f'{scene_name=}')
    name_exp = f'pointnav_ddppo_mp3d_{scene_name}_20_topomap_no_resests'
    os.system('mkdir /data/gifs/' + name_exp)
    os.system('mkdir /data/bags/' + scene_name)
    runner = HabitatRunner(name_exp)
    runner.env.reset()
    successes = []
    spls = []
    distances_to_goal = []
    end_steps = []
    for ind, ep in enumerate(runner.env.episodes):
        print(f'Start episode {ind}')
        success, spl, distance_to_goal, step = runner.run_episode(ind)
        successes.append(success)
        spls.append(spl)
        end_steps.append(step)
        distances_to_goal.append(distance_to_goal)
        

if __name__ == '__main__':
    main()
