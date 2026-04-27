"""
utils/replay_buffer.py
Fixed-size experience replay buffer.
"""
from __future__ import annotations
import numpy as np
from collections import deque
import random
from typing import Tuple


class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples.

    Parameters
    ----------
    capacity : int
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            np.array(obs,   dtype=np.float32),
            np.array(acts,  dtype=np.int64),
            np.array(rews,  dtype=np.float32),
            np.array(nobs,  dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
