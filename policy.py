from gymnasium import Env
from typing import Tuple
import numpy as np

def get_equiprobable_policy(env: Env):
    def policy(state: Tuple):
        return np.random.choice(env.action_space.n)
    return policy