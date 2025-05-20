from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Any
from habitat.config.default_structured_configs import CompassSensorConfig, PointGoalSensorConfig, HeadingSensorConfig

import numpy as np
from gym import spaces
from omegaconf import MISSING

import habitat
from habitat.config.default_structured_configs import (
    LabSensorConfig,
    MeasurementConfig,
    HabitatSimRGBSensorConfig,
    HabitatSimDepthSensorConfig,
    HabitatSimSemanticSensorConfig
)

# Define the sensor and register it with habitat
# For the sensor, we will register it with a custom name
@habitat.registry.register_sensor(name="AgentPositionSensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        sensor_states = self._sim.get_agent_state().sensor_states
        return (sensor_states['rgb'].position, sensor_states['rgb'].rotation)


# define a configuration for this new sensor
@dataclass
class AgentPositionSensorConfig(LabSensorConfig):
    # Note that typing is required on all fields
    type: str = "AgentPositionSensor"
    # MISSING makes this field have no defaults
    answer_to_life: int = MISSING

@dataclass
class HabitatSimRGB4SensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "rbg"

@dataclass
class HabitatSimDepth4SensorConfig(HabitatSimDepthSensorConfig):
    uuid: str = "depth"

@dataclass
class HabitatSimSemantic4SensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "semantic"

def register_sensors():
    cs = ConfigStore.instance()
    cs.store(
        package="habitat.task.lab_sensors.compass_sensor",
        group="habitat/task/lab_sensors",
        name="compass_sensor",
        node=CompassSensorConfig,
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="rgb",
        node=HabitatSimRGBSensorConfig(),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="rgb_for_agent",
        node=HabitatSimRGB4SensorConfig(uuid="rgb_for_agent"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="rgb1",
        node=HabitatSimRGB4SensorConfig(uuid="rgb1"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="rgb2",
        node=HabitatSimRGB4SensorConfig(uuid="rgb2"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="rgb3",
        node=HabitatSimRGB4SensorConfig(uuid="rgb3"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="rgb4",
        node=HabitatSimRGB4SensorConfig(uuid="rgb4"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="depth1",
        node=HabitatSimDepth4SensorConfig(uuid="depth1"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="depth2",
        node=HabitatSimDepth4SensorConfig(uuid="depth2"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="depth3",
        node=HabitatSimDepth4SensorConfig(uuid="depth3"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="depth4",
        node=HabitatSimDepth4SensorConfig(uuid="depth4"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="semantic1",
        node=HabitatSimSemantic4SensorConfig(uuid="semantic1"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="semantic2",
        node=HabitatSimSemantic4SensorConfig(uuid="semantic2"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="semantic3",
        node=HabitatSimSemantic4SensorConfig(uuid="semantic3"),
    )
    cs.store(
        group="habitat/simulator/sim_sensors",
        name="semantic4",
        node=HabitatSimSemantic4SensorConfig(uuid="semantic4"),
    )
    cs.store(
        package="habitat.task.lab_sensors.pointgoal_sensor",
        group="habitat/task/lab_sensors",
        name="pointgoal_sensor",
        node=PointGoalSensorConfig,
    )
    cs.store(
        package="habitat.task.lab_sensors.agent_position_sensor",
        group="habitat/task/lab_sensors",
        name="agent_position_sensor",
        node=AgentPositionSensorConfig,
    )
    cs.store(
        package="habitat.task.lab_sensors.heading_sensor",
        group="habitat/task/lab_sensors",
        name="heading_sensor",
        node=HeadingSensorConfig,
    )
