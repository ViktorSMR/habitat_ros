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
#from habitat_map.env_orb import Env
#from semantic_predictor import SemanticPredictor
#from semantic_predictor_segformer import SemanticPredictor
from std_msgs.msg import Int32
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from keyboard_agent import KeyboardAgent
from shortest_path_follower_agent import ShortestPathFollowerAgent
from greedy_path_follower_agent import GreedyPathFollowerAgent
# from random_movement_agent import RandomMovementAgent
from custom_sensors import AgentPositionSensor
from publishers import HabitatObservationPublisher
from habitat_map.mapper import Mapper
from habitat_baselines.config.default import get_config
from habitat_map.utils import draw_top_down_map
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
        if (self.scene_name == '2n8kARJN3HM'):
            self.T = np.array([
                [0, 0, 1, -14.29],
                [1, 0, 0, -3.79],
                [0, 1, 0, -2.26],
                [0, 0, 0, 1]
            ])
        elif (self.scene_name == 'D7N2EKCX4Sj'):
            self.T = np.array([
                [0, 0, 1, 5.15],
                [1, 0, 0, -1.5],
                [0, 1, 0, -0.89],
                [0, 0, 0, 1]
            ])
        elif (self.scene_name == 'E9uDoFAP3SH'):
            self.T = np.array([
                [0, 0, 1, 3.3228805],
                [1, 0, 0, -15.010473],
                [0, 1, 0, 5.0756187],
                [0, 0, 0, 1]
            ])
        elif (self.scene_name == 'JeFG25nYj2p'):
            self.T = np.array([
                [0, 0, 1, -3.5703154],
                [1, 0, 0, 3.0444067],
                [0, 1, 0, -0.973584],
                [0, 0, 0, 1]
            ])
        elif (self.scene_name == 'rPc6DW4iMge'):
            self.T = np.array([
                [0, 0, 1, 4.488],
                [1, 0, 0, 2.06],
                [0, 1, 0, -1.04],
                [0, 0, 0, 1]
            ])

        self.rate = rospy.Rate(rate_value)
        self.publisher = HabitatObservationPublisher(rgb_topic, 
                                                    depth_topic, 
                                                    #semantic_topic,
                                                    camera_info_topic, 
                                                    true_pose_topic,
                                                    camera_info_file,
                                                    self.T)
        # Now define the config for the sensor
        self.action_publisher = rospy.Publisher('habitat_action', Int32, latch=True, queue_size=100)
        self.map_publisher = rospy.Publisher('habitat/map', OccupancyGrid, latch=True, queue_size=100)
        #self.semantic_map_publisher = rospy.Publisher('habitat/semantic_map', OccupancyGrid, latch=True, queue_size=100)
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
        #self.env = env_orb.Env(config=config)
        print('Environment created')

        self.mapper = Mapper(self.env)
        #self.semantic_predictor = SemanticPredictor(threshold=0.35)

        # goal_positions = np.loadtxt('/catkin_ws/src/habitat_ros/goal_positions/mp3d/{}.txt'.format(scene_name))
        # goal_positions = np.loadtxt('/catkin_ws/src/habitat_ros/goal_positions/mipt.txt')
        #goal_positions = None
        if agent_type == 'keyboard':
           self.agent = KeyboardAgent()
        elif agent_type == 'shortest_path_follower':
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

    def transform_to_map_coords(self, position_habitat, T):
        position_habitat_4 = np.ones((4, 1))
        position_habitat_4[:3, 0] = position_habitat
        position_map = T @ position_habitat_4
        return position_map[:3, 0]

    def publish_robot_pose(self, observations, T):
        self.slam_x, self.slam_y, self.slam_angle = self.get_robot_pose(observations)
        self.robot_pose_in_habitat_coords = observations['agent_position']
        robot_position, robot_rotation = self.robot_pose_in_habitat_coords
        robot_position = self.transform_to_map_coords((robot_position[0], robot_position[1], robot_position[2]), T)
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

    def run_episode(self, episode=0):
        observations = self.env.reset()
        self.agent.reset()
        self.goal_received_once = False
        # self.env.step(HabitatSimActions.move_forward)

        self.mapper.reset()
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
        step_start_time = rospy.Time.now()
        self.publisher.publish(observations, step_start_time)
        self.publish_robot_pose(observations, self.T)
        rospy.sleep(3)
        current_episode = self.env.current_episode
        goal_y, goal_z, goal_x = current_episode.goals[0].position

        self.slam_x, self.slam_y, self.slam_angle = self.get_robot_pose(observations)
        dist, heading = observations['pointgoal_with_gps_compass']
        (habitat_y, habitat_z, habitat_x), habitat_orientation = observations['agent_position'] 

        _, __, habitat_angle = tf.transformations.euler_from_quaternion([habitat_orientation.x, habitat_orientation.z, habitat_orientation.y, habitat_orientation.w])

        # d_angle = self.normalize(-habitat_angle + self.slam_angle + np.pi)
        print('SLAM:', self.slam_x, self.slam_y, self.slam_angle)
        print('habitat in habitat:', habitat_y, habitat_z, habitat_x)
        print('habitat:', habitat_x, habitat_y, habitat_z, habitat_angle)
        # print('D_ANGLE:', d_angle)
        goal_map = np.array([-(self.slam_x + dist * math.cos(self.slam_angle + habitat_angle + heading)), 
                             -(self.slam_y + dist * math.sin(self.slam_angle + habitat_angle + heading)),
                             goal_z])
        print('Distance + heading:', dist, heading)
        print('Goal SLAM:', goal_map)
        print('Goal habitat:', goal_x, goal_y, goal_z)

        goal_map = self.transform_to_map_coords((goal_y, goal_z, goal_x), self.T)
        goal_map[2] = goal_z
        print('Goal SLAM new:', goal_map)
        # dx = habitat_x - (self.slam_x * math.cos(d_angle) + self.slam_y * math.sin(d_angle))
        # dy = habitat_y - (-self.slam_x * math.sin(d_angle) + self.slam_y * math.cos(d_angle))


        # goal_x_rotated = goal_x * math.cos(d_angle) + goal_y * math.sin(d_angle)
        # goal_y_rotated = -goal_x * math.sin(d_angle) + goal_y * math.cos(d_angle)
        # goal_map = np.array([goal_x_rotated - dx, goal_y_rotated - dy, goal_z])


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

        # trajectory = []
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # im_rgb = ax.imshow(observations['rgb'], animated=True)
        # ax.set_title(observations['pointgoal_with_gps_compass'])
        # data = []
        # times_inference = []
        step = 0
        prev_dist = 0
        while not rospy.is_shutdown() and not self.env.episode_over:
            step_start_time = rospy.Time.now()
            t0 = rospy.Time.now().to_sec()
            self.publisher.publish(observations, step_start_time)
            self.publish_robot_pose(observations, self.T)
            t1 = rospy.Time.now().to_sec()
            #print('Publish time:', t1 - t0)
            metrics = self.env.get_metrics()
            distance_to_goal = metrics['distance_to_goal']

            self.publisher.publish(observations, rospy.Time.now())
            self.publish_robot_pose(observations, self.T)

            # while not self.goal_received_once and not rospy.is_shutdown():
            if not self.goal_received_once:
                self.goal_publisher.publish(goal_msg)
                pointgoal_with_gps_compas = None
            else:
                while not self.goal_received and not rospy.is_shutdown():
                    self.rate.sleep()
                dx, dy, dz = self.goal_pose_in_habitat_coords[:3]
                # print(f"{dx=}, {dy=}")
                # rx, ry, rz = self.robot_pose_in_habitat_coords[0]
                q = self.robot_pose_in_habitat_coords[1]
                _, _, robot_yaw = tf.transformations.euler_from_quaternion([
                    q.x, q.y, q.z, q.w
                ])

                # dx = gx - rx
                # dy = gy - ry
                print('dx dy:', dx, dy)
                dist = float(math.hypot(dx, dy))
                goal_angle = float(math.atan2(dy, dx))
                print('Goal angle:', goal_angle)
                # if np.abs(dist - prev_dist) >= 2:
                #     self.agent.reset()
                #     print('Reset!')
                prev_dist = dist
                # heading = self.normalize(robot_yaw - goal_angle + np.pi)
                heading = goal_angle
                # print(observations['pointgoal_with_gps_compass'])
                pointgoal_with_gps_compas = np.array(
                    [dist, heading], dtype=np.float32
                )
                # data.append([observations['rgb'], f"Distance: {distance_to_goal:.4f} {pointgoal_with_gps_compas}"])
                # If it is final goal, and we are close to it, force to finish
                if dz > 0.5 and dist < 0.35:
                    pointgoal_with_gps_compas = np.array(
                        [0, 0], dtype=np.float32
                    )

            print('PointGoal from habitat:', observations['pointgoal_with_gps_compass'])
            print('PointGoal from TopoSLAM:', pointgoal_with_gps_compas)
            filtered_obs = {
                "rgb": observations["rgb_for_agent"],
                "pointgoal_with_gps_compass": pointgoal_with_gps_compas,
                #"pointgoal_with_gps_compass": observations['pointgoal_with_gps_compass']
            }

            action = self.agent.act(filtered_obs)
            t2 = rospy.Time.now().to_sec()
            #print('Action time:', t2 - t1)
            #if self.agent.goal_pose_in_habitat_coords is None:
            #    print('NO GOAL TO MOVE. FINISH')
            #    break
            #if step % 3 == 1:
            #    self.mapper.step(observations, observations['semantic'])
            #action_msg = Int32()
            #action_msg.data = action
            before_publish = rospy.Time.now().to_sec()
            #self.action_publisher.publish(action_msg)
            observations = self.env.step(action)
            robot_x, robot_y = observations['gps']
            robot_y = -robot_y
            robot_angle = observations['compass']
            if isinstance(robot_angle, (list, np.ndarray)):
                robot_angle = float(robot_angle[0])
            after_publish = rospy.Time.now().to_sec()
            # trajectory.append((robot_x, robot_y, robot_angle, step_start_time.to_sec()))
            t3 = rospy.Time.now().to_sec()
            #print('Step time:', t3 - t2)
            #if step % 10 == 1:
            #    self.publish_map()
            self.goal_received = False
            # self.rate.sleep()
            step += 1

        # def init():
        #     im_rgb.set_data(data[0][0])
        #     # im_goal.set_data(data[0][1])
        #     fig.suptitle(data[0][1])
        #     return im_rgb

        # def update(frame):
        #     if frame >= len(data):
        #         im_rgb.set_data(data[-1][0])
        #         # im_goal.set_data(data[-1][1])
        #         fig.suptitle(f"Success: {success:.0f} | Spl: {spl:.4f} | Distance: {distance_to_goal:.4f}")
        #         return im_rgb
        #     im_rgb.set_data(data[frame][0])
        #     # im_goal.set_data(data[frame][1])
        #     fig.suptitle(data[frame][1])
        #     return im_rgb
        
            
        metrics = self.env.get_metrics()
        print('METRICS:', metrics)
        success = metrics['success']
        spl = metrics['spl']
        distance_to_goal = metrics['distance_to_goal']
        # print('Length: ', len(data))
        # ani = animation.FuncAnimation(fig, update, frames=len(data) + 30, init_func=init)
        # # plt.show()
        # ani.save(filename="/data/gifs/" + self.name + "/episode" + str(episode) + ".mp4")

        # print(f"{trajectory=}")
        # np.savetxt('/catkin_ws/src/habitat_ros/trajectory.txt', trajectory)
        rospy.sleep(5)
        return success, spl, distance_to_goal, step


def main():
    register_sensors()
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()
    rospy.init_node('habitat_ros_node', anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    scene_name = rospy.get_param('~scene_name', None)
    print(f'{scene_name=}')
    prism_topomap_args = ['prism_topomap', 'habitat_mp3d_localization.launch', 
                          f'scene_name:={scene_name}',
                          f'path_to_load_json:=/data/maps/{scene_name}',
                          f'path_to_save_logs:=/logs']
    prism_roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(prism_topomap_args)[0]
    name_exp = f'pointnav_ddppo_mp3d_{scene_name}_20_topomap_no_resests'
    os.system('mkdir /data/gifs/' + name_exp)
    os.system('mkdir /data/bags/' + scene_name)
    runner = HabitatRunner(name_exp)
    # runner.run_episode()
    runner.env.reset()
    successes = []
    spls = []
    distances_to_goal = []
    end_steps = []
    for ind, ep in enumerate(runner.env.episodes):
        # import subprocess

        # rosbag_args = [
        #     'rosbag', 'record',
        #     '/topological_map', '/last_vertex', '/current_grid',
        #     '/local_grid', '/matched_points', '/unmatched_points',
        #     '/topological_path_marker', '/tf', '/pointgoal',
        #     '/true_pose',
        #     '-O', f'/data/bags/{scene_name}/path_for_{ind}.bag'
        # ]
        # proc = subprocess.Popen(rosbag_args)
        launch_prism = roslaunch.parent.ROSLaunchParent(uuid, [(prism_roslaunch_file, prism_topomap_args)])
        launch_prism.start()
        rospy.sleep(10)
        print(f'Start episode {ind}')
        success, spl, distance_to_goal, step = runner.run_episode(ind)
        launch_prism.shutdown()
        # proc.terminate()
        rospy.sleep(10)
        successes.append(success)
        spls.append(spl)
        end_steps.append(step)
        distances_to_goal.append(distance_to_goal)
    with open('/data/results.json', 'r') as f:
        data = json.load(f)
    data.update({name_exp : {'Average success': np.mean(successes), 'Average SPL': np.mean(spls), 'Lost': np.where(np.asarray(successes) == 0)[0].tolist(), 'End steps': end_steps}})
    with open('/data/results.json', 'w') as f:
        json.dump(data, f, indent=4)
        


if __name__ == '__main__':
    main()
