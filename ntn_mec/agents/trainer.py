"""
agents/trainer.py
IL (independent learning) and CTDE episode runners.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

from env.ntn_env import NTNMECEnv
from agents.dqn import DQNAgent


# ─────────────────────────────────────────────────────────────────────────────
def run_episode_il(
    env: NTNMECEnv,
    agents: List[DQNAgent],
    steps_per_episode: int = 50,
    train: bool = True,
    greedy: bool = False,
) -> Dict[str, Any]:
    """
    Run one episode in Independent Learning (IL) mode.
    Each agent uses only its own local reward.
    """
    obs_list = env.reset()
    total_rewards = np.zeros(env.M)
    losses = []

    for _ in range(steps_per_episode):
        actions = [
            agent.select_action(obs_list[m], greedy=greedy)
            for m, agent in enumerate(agents)
        ]
        next_obs_list, hybrid_rewards, done, info = env.step(actions)
        local_rewards = info['rewards_local']

        if train:
            for m, agent in enumerate(agents):
                agent.store(obs_list[m], actions[m],
                            local_rewards[m],          # IL uses local reward
                            next_obs_list[m], done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

        obs_list = next_obs_list
        total_rewards += np.array(local_rewards)

        if done:
            break

    return {
        'total_reward': total_rewards.sum(),
        'per_agent_reward': total_rewards.tolist(),
        'mean_loss': float(np.mean(losses)) if losses else 0.0,
        'mode': 'IL',
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_episode_ctde(
    env: NTNMECEnv,
    agents: List[DQNAgent],
    steps_per_episode: int = 50,
    train: bool = True,
    greedy: bool = False,
) -> Dict[str, Any]:
    """
    Run one episode in CTDE mode.
    All agents receive the *hybrid* reward (local + global component),
    sharing information during training while keeping decentralised execution.
    """
    obs_list = env.reset()
    total_rewards = np.zeros(env.M)
    losses = []

    for _ in range(steps_per_episode):
        actions = [
            agent.select_action(obs_list[m], greedy=greedy)
            for m, agent in enumerate(agents)
        ]
        next_obs_list, hybrid_rewards, done, info = env.step(actions)

        if train:
            for m, agent in enumerate(agents):
                agent.store(obs_list[m], actions[m],
                            hybrid_rewards[m],         # CTDE uses hybrid reward
                            next_obs_list[m], done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

        obs_list = next_obs_list
        total_rewards += np.array(hybrid_rewards)

        if done:
            break

    return {
        'total_reward': total_rewards.sum(),
        'per_agent_reward': total_rewards.tolist(),
        'mean_loss': float(np.mean(losses)) if losses else 0.0,
        'mode': 'CTDE',
    }


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    env: NTNMECEnv,
    agents: List[DQNAgent],
    episodes: int = 10,
    steps_per_episode: int = 50,
) -> Dict[str, float]:
    """
    Greedy evaluation over multiple episodes.
    Returns aggregated metrics.
    """
    all_rewards: List[float] = []
    all_drops: List[int] = []
    all_completes: List[int] = []

    for _ in range(episodes):
        obs_list = env.reset()
        ep_reward = 0.0
        for _ in range(steps_per_episode):
            actions = [
                agent.select_action(obs_list[m], greedy=True)
                for m, agent in enumerate(agents)
            ]
            next_obs_list, rewards, done, _ = env.step(actions)
            ep_reward += sum(rewards)
            obs_list = next_obs_list
            if done:
                break
        all_rewards.append(ep_reward)

        drops = sum(len(ue.dropped_tasks) for ue in env.ues)
        completes = sum(len(ue.completed_tasks) for ue in env.ues)
        all_drops.append(drops)
        all_completes.append(completes)

    total_tasks = [d + c for d, c in zip(all_drops, all_completes)]
    drop_rates = [
        d / t if t > 0 else 0.0
        for d, t in zip(all_drops, total_tasks)
    ]
    return {
        'mean_reward':    float(np.mean(all_rewards)),
        'std_reward':     float(np.std(all_rewards)),
        'mean_drop_rate': float(np.mean(drop_rates)),
        'mean_throughput': float(np.mean([1 - dr for dr in drop_rates])),
    }
