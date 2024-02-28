import gym
from typing import Optional
from collections import defaultdict
import numpy as np


def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    pass


def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    pass


def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    pass


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    pass


def td_prediction(env: gym.Env, gamma: float, episodes, alpha: float, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        alpha (float): Step size
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    V = defaultdict(float)

    for episode in episodes:
        targets = learning_targets(V, gamma, episode, n)
        for t, (s, a, r) in enumerate(episode):
            V[s] += (alpha * (targets[t] - V[s]))
    
    return dict(V)


def learning_targets(
    V: defaultdict, gamma: float, episode, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episode : the evaluation episode. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    targets = np.zeros(len(episode))

    if n is None:
        # Monte Carlo
        T = len(episode)
        G = 0
        for t in range(T - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            targets[t] = G
    else:
        # TD(n)
        T = len(episode)
        for t in range(len(episode)):
            G = 0
            final_t = min(t + n, T)
            for i in range(t, final_t):
                s, a, r = episode[i]
                G += (gamma ** (i - t)) * r
            
            # bootstrap
            if final_t < T:
                s, a, r = episode[final_t]
                G += (gamma ** (n+1)) * V[s]

            targets[t] = G
    
    return targets
        

