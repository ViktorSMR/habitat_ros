
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from omegaconf import DictConfig, OmegaConf

import habitat
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs
from habitat.sims.habitat_simulator.actions import HabitatSimActions


@dataclass
class DDPPOAgentConfig:
    INPUT_TYPE: str = "rgb"
    MODEL_PATH: str = "/data/models/gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth"
    RESOLUTION: int = 256
    HIDDEN_SIZE: int = 512
    RANDOM_SEED: int = 7
    PTH_GPU_ID: int = 0
    GOAL_SENSOR_UUID: str = "pointgoal_with_gps_compass"


def get_default_config() -> DictConfig:
    return OmegaConf.create(DDPPOAgentConfig())  # type: ignore[call-overload]


class DDPPOAgent(Agent):
    def __init__(self) -> None:
        config = get_default_config()
        spaces = {
            get_default_config().GOAL_SENSOR_UUID: Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(config.RESOLUTION, config.RESOLUTION, 1),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(config.RESOLUTION, config.RESOLUTION, 3),
                dtype=np.uint8,
            )
        observation_spaces = SpaceDict(spaces)

        action_spaces = Discrete(4)

        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.HIDDEN_SIZE

        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="se_resneXt50",
            resnet_baseplanes=32,
            hidden_size=self.hidden_size,
            normalize_visual_inputs="rgb" in spaces,
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None
        self.pointgoal_history = []
        self.steps_on_place = 0
        self.stuck = False

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    def act(self, observations: Observations) -> Dict[str, int]:
        # If we didn't receive pointgoal message, perform random walking
        if observations['pointgoal_with_gps_compass'] is None:
            a = np.random.random()
            if a < 0.7:
                random_action = HabitatSimActions.move_forward
            else:
                random_action = HabitatSimActions.turn_left
            # else:
            #     random_action = HabitatSimActions.turn_right
            return {"action": random_action}
        self.pointgoal_history.append(observations['pointgoal_with_gps_compass'])
        # If we stuck on place for 6 or more steps, try to escape
        if len(self.pointgoal_history) > 1 and abs(self.pointgoal_history[-1][0] - self.pointgoal_history[-2][0]) < 0.05:
            self.steps_on_place += 1
        else:
            self.stuck = False
            self.steps_on_place = 0
        if self.steps_on_place == 15:
            self.stuck = True
            random_factor = np.random.random() - 0.4
            if observations['pointgoal_with_gps_compass'][1] * random_factor > 0:
                self.turn_action = HabitatSimActions.turn_left
            else:
                self.turn_action = HabitatSimActions.turn_right
        if self.stuck:
            print('Escape from stuck')
            if self.steps_on_place % 3 == 0:
                escape_action = self.turn_action
            else:
                escape_action = HabitatSimActions.move_forward
            return {"action": escape_action}
        batch = batch_obs([observations], device=self.device)
        # Otherwise, act by DDPPO policy
        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore

        return {"action": actions[0][0].item()}