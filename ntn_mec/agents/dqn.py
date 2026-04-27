"""
agents/dqn.py
Dueling DQN network + per-agent learner.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from utils.replay_buffer import ReplayBuffer


# ─────────────────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network.

    Architecture:
        input → feature MLP → split into value stream V(s) and
        advantage stream A(s,a) → Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, input_dim: int, n_actions: int,
                 hidden: int = 128, hidden2: int = 64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        val = self.value_stream(feat)             # (B, 1)
        adv = self.adv_stream(feat)               # (B, n_actions)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Single UE agent with Dueling DQN, ε-greedy exploration, target network.

    Parameters
    ----------
    agent_id   : int
    obs_dim    : int
    n_actions  : int
    lr         : float   Adam learning rate
    gamma      : float   discount factor
    eps_start  : float   initial exploration rate
    eps_end    : float   final exploration rate
    eps_decay  : int     number of steps to decay epsilon
    buffer_cap : int     replay buffer capacity
    batch_size : int
    target_update_freq : int  steps between target network syncs
    device     : str
    """

    def __init__(self, agent_id: int, obs_dim: int, n_actions: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 5000,
                 buffer_cap: int = 50_000, batch_size: int = 32,
                 target_update_freq: int = 100,
                 device: str = 'cpu'):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Networks
        self.online_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)

        # Epsilon schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.train_steps = 0

    # ------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        decay = np.exp(-self.steps_done / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * decay

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        self.steps_done += 1
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online_net(obs_t)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    def learn(self) -> Optional[float]:
        """One gradient step. Returns loss or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        obs, acts, rews, nobs, dones = self.buffer.sample(self.batch_size)
        obs_t  = torch.tensor(obs,   device=self.device)
        acts_t = torch.tensor(acts,  device=self.device)
        rews_t = torch.tensor(rews,  device=self.device)
        nobs_t = torch.tensor(nobs,  device=self.device)
        done_t = torch.tensor(dones, device=self.device)

        # Current Q values
        q_values = self.online_net(obs_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN style)
        with torch.no_grad():
            next_actions = self.online_net(nobs_t).argmax(dim=1)
            next_q = self.target_net(nobs_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rews_t + self.gamma * next_q * (1.0 - done_t)

        loss = nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            'online': self.online_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps_done = ckpt['steps_done']
