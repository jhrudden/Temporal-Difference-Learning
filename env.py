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

def get_random_walk_env(**kwargs):
    """
    Get the RandomWalk environment

    Args:
        num_states (int): number of states in the environment
        rewards (List[int]): rewards for the leftmost and rightmost states

    Returns:
        env (RandomWalk): RandomWalk environment
    """
    try:
        spec = gym.spec('RandomWalk-v0')
    except:
        register_env("RandomWalk-v0", entry_point="env:RandomWalk", max_episode_steps=1000)
    finally:
        return gym.make('RandomWalk-v0', **kwargs)

# RandomWalk Env as described in Example 6.2 and 7.1 of Reinforcement Learning: An Introduction
class RandomWalk(Env):
    def __init__(self, num_states=7, rewards=[0,1]):
        self.num_states = num_states
        self.state = 0
        self.action_space = spaces.Discrete(len(RandomWalkAction))
        self.observation_space = spaces.Discrete(num_states)
        self.rewards = rewards
        self.start_pos = num_states // 2
    
    def calculate_optimal_value_function(self):
        """
        Calculate the optimal value function for the RandomWalk environment

        Returns:
            value_function (np.ndarray): optimal value function for the RandomWalk environment
        """
        value_function = self.rewards[1] * np.arange(1, self.num_states - 1) / (self.num_states - 1) 
        value_function += self.rewards[0] * value_function[::-1]
        return value_function

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
            reward = self.rewards[1]
            done = True
        elif self.state == 0:
            reward = self.rewards[0]
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
        self.state = self.start_pos
        return self.state, {}


def get_windy_gridworld_env(**kwargs):
    """
    Get the WindyGridWorld environment

    Args:
        kwargs (dict): keyword arguments for the WindyGridWorld environment

    Returns:
        env (WindyGridWorldEnv): WindyGridWorld environment
    """
    try:
        spec = gym.spec('WindyGridWorld-v0')
    except:
        register_env("WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv", max_episode_steps=8000)
    finally:
        return gym.make('WindyGridWorld-v0', **kwargs)

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
        self.cols = 10
        self.rows = 7

        # Wind
        # TODO define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = np.zeros(self.cols)
        # update wind strength for each column
        self.wind[3:6] = 1
        self.wind[6:8] = 2
        self.wind[8] = 1

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.cols), spaces.Discrete(self.rows))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def reset(self, options: Dict = {}):
        self.agent_pos = self.start_pos
        return self.agent_pos, {}

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
        assert self.action_space.contains(action)

        reward = -1
        done = False

        dx, dy = actions_to_dxdy(action)
        x, y = self.agent_pos

        new_x = max(0, min(x + dx, self.cols - 1))
        new_y = max(0, min(y + dy + self.wind[x], self.rows - 1))

        self.agent_pos = (new_x, new_y)

        if self.agent_pos == self.goal_pos:
            reward = 0
            done = True

        return self.agent_pos, reward, done, False, {}
