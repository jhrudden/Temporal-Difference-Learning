from gymnasium import Env
from typing import Tuple, Dict
import numpy as np

def get_equiprobable_policy(env: Env):
    def policy(state: Tuple):
        return np.random.choice(env.action_space.n), np.ones(env.action_space.n, dtype=float) / env.action_space.n
    return policy

def get_epsilon_greedy_policy(env: Env, epsilon: float, Q: Dict):
    def policy(state: Tuple):
        action_probs = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = np.argmax(Q[state])
        action_probs[best_action] += (1.0 - epsilon)
        chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return chosen_action, action_probs
    return policy