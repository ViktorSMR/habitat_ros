import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import keyboard
import numpy as np

class KeyboardAgent(habitat.Agent):
    def __init__(self):
        pass

    def reset(self):
        pass

    def get_actions_from_keyboard(self):
        keyboard_commands = []
        if keyboard.is_pressed('left'):
            keyboard_commands += [HabitatSimActions.turn_left]
        if keyboard.is_pressed('right'):
            keyboard_commands += [HabitatSimActions.turn_right]
        if keyboard.is_pressed('up'):
            keyboard_commands += [HabitatSimActions.move_forward]
        return keyboard_commands

    def act(self, observations):
        # receive command from keyboard and move
        actions = self.get_actions_from_keyboard()
        if len(actions) > 0:
            return np.random.choice(actions)
        else:
            return HabitatSimActions.stop