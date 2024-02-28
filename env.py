from enum import IntEnum
from typing import Tuple, Optional, List, Dict
from gymnasium import Env, spaces, register
import gymnasium as gym
import numpy as np


def register_env(id: str, **kwargs):
    """Register custom gym environment so that we can use `gym.make()`
    Note: the max_episode_steps option controls the time limit of the environment.
    You can remove the argument to make FourRooms run without a timeout.
    """
    register(id=id, **kwargs)


class RandomWalkAction(IntEnum):
    LEFT = 0
    RIGHT = 1

def get_random_walk_env(num_states=7):
    """
    Get the RandomWalk environment
    Returns:
        env (RandomWalk): RandomWalk environment
    """
    try:
        spec = gym.spec('RandomWalk-v0')
    except:
        register_env("RandomWalk-v0", entry_point="env:RandomWalk", max_episode_steps=1000)
    finally:
        return gym.make('RandomWalk-v0', num_states=num_states)

# RandomWalk Env as described in Example 6.2 and 7.1 of Reinforcement Learning: An Introduction
class RandomWalk(Env):
    def __init__(self, num_states=7):
        self.num_states = num_states
        self.state = 0
        self.action_space = spaces.Discrete(len(RandomWalkAction))
        self.observation_space = spaces.Discrete(num_states)

    def step(self, action: RandomWalkAction):
        """
        Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).

        Args:
            action (Action): an action provided by the agent
        
        Returns:
            state (int): next state
            reward (float): reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            truncated (bool): whether the episode was truncated (not used in this environment)
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        assert self.action_space.contains(action)
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.num_states - 1, self.state + 1)

        if self.state == self.num_states - 1:
            reward = 1
            done = True
        elif self.state == 0:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        return self.state, reward, done, False, {}

    def reset(self, options: Dict = {}):
        """
        Reset the environment to the starting state

        Args:
            options (dict): options for the environment (not used in this environment)
        
        Returns:
            state (int): returns the initial state
        """
        # TODO: this might need to depend on the self.num_states
        self.state = 3
        return self.state, {}

# def get_four_rooms_env(goal_pos=(10, 10)):
#     """
#     Get the FourRooms environment
#     Args:
#         goal_pos (Tuple[int, int]): goal position
#     Returns:
#         env (FourRoomsEnv): FourRooms environment
#     """
#     try:
#         spec = gym.spec('FourRooms-v0')
#     except:
#         register_env("FourRooms-v0", entry_point="env:FourRoomsEnv", max_episode_steps=459)
#     finally:
#         return gym.make('FourRooms-v0', goal_pos=goal_pos)


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


class WindyGridWorldEnv(Env):
    def __init__(self):
        """Windy grid world gym environment
        This is the template for Q4a. You can use this class or modify it to create the variants for parts c and d.
        """

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        # TODO define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = None

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def reset(self, options: Dict = {}):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # TODO
        reward = None
        done = None

        return self.agent_pos, reward, done, {}
