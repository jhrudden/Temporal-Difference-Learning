import gymnasium as gym
from typing import Optional
from collections import defaultdict
import numpy as np
from tqdm import trange

from policy import get_epsilon_greedy_policy

def mc_control_on_policy(env: gym.Env, num_episodes: int, gamma: float, epsilon: float, verbose: bool = False):
    """Monte Carlo control with epsilon-greedy policy. (Every-visit MC policy evaluation and improvement)

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = get_epsilon_greedy_policy(env, epsilon, Q)
    episode_at_time = []

    if verbose:
        _range = trange(num_episodes)
    else:
        _range = range(num_episodes)
    
    for i in _range:
        s = env.reset()
        done = False
        truncated = False
        episode = []
        while not done and not truncated:
            episode_at_time.append(i)
            a, _ = policy(s)
            s_prime, r, done, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s_prime
        
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            C[s][a] += 1
            Q[s][a] += (1 / C[s][a]) * (G - Q[s][a])
    
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
    episode_at_time = []
    
    if verbose:
        _range = trange(num_steps)
    else:
        _range = range(num_steps)

    for i in _range:
        s, _ = env.reset()
        a, _ = policy(s)
        done = False
        truncated = False
        while not done and not truncated:
            episode_at_time.append(i)
            s_prime, r, done, truncated, _ = env.step(a)
            a_prime, _ = policy(s_prime)
            Q[s][a] += step_size * (r + gamma * Q[s_prime][a_prime] - Q[s][a])
            s = s_prime
            a = a_prime
    
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
    episode_at_time = []

    if verbose:
        _range = trange(num_steps)
    else:
        _range = range(num_steps)
    
    for i in _range:
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
                episode_at_time.append(i)
                s, r, done, truncated, _ = env.step(a)
                S.append(s)
                R.append(r)
                if done:
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
                break
            t += 1
    
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
    episode_at_time = []

    if verbose:
        _range = trange(num_steps)
    else:
        _range = range(num_steps)
    
    for i in _range:
        s, _ = env.reset()
        a, _ = policy(s)
        done = False
        truncated = False
        while not done and not truncated:
            episode_at_time.append(i)
            s_prime, r, done, truncated, _ = env.step(a)
            a_prime, action_probs = policy(s_prime)
            expected_value = np.dot(Q[s_prime], action_probs)
            Q[s][a] += step_size * (r + gamma * expected_value - Q[s][a])
            s = s_prime
            a = a_prime
    
    return dict(Q), episode_at_time


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
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = get_epsilon_greedy_policy(env, epsilon, Q)
    episode_at_time = []
    
    for i in range(num_steps):
        s, _ = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            episode_at_time.append(i)
            a, _ = policy(s)
            s_prime, r, done, truncated, _ = env.step(a)
            Q[s][a] += step_size * (r + gamma * np.max(Q[s_prime]) - Q[s][a])
            s = s_prime
    
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

    for episode in episodes:
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
        

