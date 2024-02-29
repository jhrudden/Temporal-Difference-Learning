import gymnasium as gym
from typing import Optional
from collections import defaultdict
import numpy as np
from tqdm import trange, tqdm

from policy import get_epsilon_greedy_policy

def mc_control_on_policy(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size=None, verbose: bool = False):
    """Monte Carlo control with epsilon-greedy policy. (Every-visit MC policy evaluation and improvement)

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
        verbose (bool): whether to show progress bar
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = get_epsilon_greedy_policy(env, epsilon, Q)
    episode_at_time = np.zeros(num_steps)

    if verbose:
        pbar = tqdm(num_steps, desc="MC Control")
    
    episode_num = 0
    curr_t = 0
    
    while True:
        s, _ = env.reset()
        done = False
        truncated = False
        episode = []
        while not done and not truncated and curr_t < num_steps:
            episode_at_time[curr_t] = episode_num
            a, _ = policy(s)
            s_prime, r, done, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s_prime
            curr_t+=1
            if verbose:
                pbar.update(1)
        
        episode_num += 1    
        
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            C[s][a] += 1
            if step_size is None:
                Q[s][a] += (1 / C[s][a]) * (G - Q[s][a])
            else:
                Q[s][a] += step_size * (G - Q[s][a])
        
        if curr_t >= num_steps:
            break
    
    return dict(Q), episode_at_time

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float, verbose: bool = False):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = get_epsilon_greedy_policy(env, epsilon, Q)
    episode_at_time = np.zeros(num_steps)
    
    if verbose:
        pbar = tqdm(num_steps, desc="SARSA")
    
    t = 0
    current_episode = 0

    while True:
        s, _ = env.reset()
        a, _ = policy(s)
        done = False
        truncated = False
        while not done and not truncated and t < num_steps:
            episode_at_time[t] = current_episode
            s_prime, r, done, truncated, _ = env.step(a)
            t += 1
            a_prime, _ = policy(s_prime)
            Q[s][a] += step_size * (r + gamma * Q[s_prime][a_prime] - Q[s][a])
            s = s_prime
            a = a_prime
            if verbose:
                pbar.update(1)
        
        current_episode += 1
        if current_episode >= num_steps:
            break

    return dict(Q), episode_at_time


def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    n: int,
    verbose = False
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = get_epsilon_greedy_policy(env, epsilon, Q)
    episode_at_time = np.zeros(num_steps)

    if verbose:
        _range = tqdm(num_steps, desc="N-step SARSA")
    
    curr_t = 0
    current_episode = 0
    while True:
        s, _ = env.reset()
        a, _ = policy(s)
        done = False
        truncated = False
        T = float('inf')
        S = [s]
        A = [a]
        R = [0]
        t = 0
        while True:
            if not done and not truncated:
                episode_at_time[curr_t] = current_episode
                s, r, done, truncated, _ = env.step(a)
                S.append(s)
                R.append(r)

                curr_t += 1
                if verbose:
                    _range.update(1)

                if curr_t >= num_steps:
                    truncated = True

                if done or truncated:
                    T = t + 1
                else:
                    a, _ = policy(s)
                    A.append(a)
            
            tau =  t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * R[i] for i in range(tau + 1, min(tau + n, T))])
                if tau + n < T:
                    G += gamma ** n * Q[S[tau + n]][A[tau + n]]
                Q[S[tau]][A[tau]] += step_size * (G - Q[S[tau]][A[tau]])
            
            if tau == T - 1:
                current_episode += 1
                break

            t += 1
            
        if curr_t >= num_steps:
            break
            
    
    return dict(Q), episode_at_time

def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    verbose = False
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = get_epsilon_greedy_policy(env, epsilon, Q)
    episode_at_time = np.zeros(num_steps)

    if verbose:
        _range = tqdm(num_steps, desc="Expected SARSA")
    
    t = 0
    current_episode = 0
    while True:
        s, _ = env.reset()
        a, _ = policy(s)
        done = False
        truncated = False
        while not done and not truncated and t < num_steps:
            episode_at_time[t] = current_episode
            s_prime, r, done, truncated, _ = env.step(a)
            a_prime, action_probs = policy(s_prime)
            expected_value = np.dot(Q[s_prime], action_probs)
            Q[s][a] += step_size * (r + gamma * expected_value - Q[s][a])
            s = s_prime
            a = a_prime
            t += 1

            if verbose:
                _range.update(1)

        current_episode += 1
        
        if t >= num_steps:
            break
    
    return dict(Q), episode_at_time


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    verbose = False
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
        verbose (bool): whether to show progress bar
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = get_epsilon_greedy_policy(env, epsilon, Q)
    episode_at_time = np.zeros(num_steps)

    if verbose:
        _range = tqdm(num_steps, desc="Q-learning")
    
    t = 0
    current_episode = 0
    while True:
        s, _ = env.reset()
        done = False
        truncated = False
        while not done and not truncated and t < num_steps:
            episode_at_time[t] = current_episode
            a, _ = policy(s)
            s_prime, r, done, truncated, _ = env.step(a)
            Q[s][a] += step_size * (r + gamma * np.max(Q[s_prime]) - Q[s][a])
            s = s_prime
            t += 1

            if verbose:
                _range.update(1)
            
        current_episode += 1

        if t >= num_steps:
            break
    
    return dict(Q), episode_at_time


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

    for episode in tqdm(episodes, desc="TD Prediction"):
        T = len(episode)
        for t in range(T):
            G = 0
            final_t = min(t + n, T)
            for i in range(t, final_t):
                s, a, r = episode[i]
                G += (gamma ** (i - t)) * r

            # bootstrap
            if final_t < T:
                s, a, r = episode[final_t]
                G += (gamma ** (n+1)) * V[s]
            
            # update
            s, a, r = episode[t]
            V[s] += alpha * (G - V[s])
    
    return dict(V)

