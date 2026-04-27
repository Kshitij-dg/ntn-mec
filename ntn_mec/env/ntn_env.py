"""
env/ntn_env.py
NTN-MEC multi-agent environment.

Action space per UE: 0 = LOCAL, 1..N = UAV_i, N+1 = LEO
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .ue import UE
from .edge_node import EdgeNode


# ─────────────────────────────────────────────────────────────────────────────
#  Default configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    # Network topology
    M=5,           # number of UEs
    N=3,           # number of UAVs
    # UE parameters
    f_loc=1e9,     # local CPU frequency (Hz)
    kappa=1e-28,   # energy coefficient
    cycles_per_bit=1000,   # CPU cycles per bit
    P_tx=0.5,      # transmit power (W)
    # Edge server parameters
    f_uav=10e9,    # UAV CPU frequency (Hz)
    f_leo=20e9,    # LEO CPU frequency (Hz)
    prop_delay_uav=0.001,   # 1 ms UAV propagation delay
    prop_delay_leo=0.010,   # 10 ms LEO propagation delay
    # Channel
    bandwidth=1e6,          # 1 MHz
    path_loss_uav=1e6,      # L_{m,n} linear scale (determines SNR)
    path_loss_leo=1e8,
    noise_power=1e-13,      # thermal noise (W)
    # Task generation
    p_arrive=0.6,           # Bernoulli arrival probability per UE per slot
    task_size_min=5000,     # bits
    task_size_max=50000,    # bits
    # Timing
    dt=0.01,                # slot duration (seconds)
    deadline_slots=5,       # max slots before drop
    # Reward
    w1=0.5,
    w2=0.5,
    drop_penalty=100.0,
    alpha=0.2,              # hybrid: (1-alpha)*local + alpha*global
)


# ─────────────────────────────────────────────────────────────────────────────
class NTNMECEnv:
    """
    Non-terrestrial network MEC multi-agent environment.

    Observation per UE:
        [local_queue_Mbits, tx_queue_count,
         *active_counts(N+1 nodes),
         *channel_rates_Mbps(N+1 nodes)]
    Length = 2 + (N+1) + (N+1)

    Action per UE:
        0 = LOCAL
        1..N = offload to UAV (1-indexed)
        N+1  = offload to LEO
    """

    ACTION_LOCAL = 0

    def __init__(self, config: Optional[Dict] = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        self.M: int = cfg['M']
        self.N: int = cfg['N']
        self.dt: float = cfg['dt']
        self.deadline: float = cfg['deadline_slots'] * cfg['dt']
        self.w1: float = cfg['w1']
        self.w2: float = cfg['w2']
        self.alpha: float = cfg['alpha']
        self.drop_penalty: float = cfg['drop_penalty']

        # arrival probabilities (per UE)
        p = cfg['p_arrive']
        self.p_arrive: np.ndarray = (
            np.full(self.M, p) if np.isscalar(p) else np.array(p)
        )

        # Build UEs
        self.ues: List[UE] = [
            UE(m, cfg['f_loc'], cfg['kappa'], cfg['cycles_per_bit'])
            for m in range(self.M)
        ]

        # Build edge nodes: UAVs + LEO
        self.uavs: List[EdgeNode] = [
            EdgeNode(f"UAV{n}", cfg['f_uav'], cfg['cycles_per_bit'],
                     cfg['prop_delay_uav'])
            for n in range(self.N)
        ]
        self.leo = EdgeNode("LEO", cfg['f_leo'], cfg['cycles_per_bit'],
                            cfg['prop_delay_leo'])

        # All edge nodes in order: [UAV0, UAV1, ..., UAV_{N-1}, LEO]
        self.edge_nodes: List[EdgeNode] = self.uavs + [self.leo]

        # Precompute channel rates (bits/s) for each UE → each edge node
        # R = B * log2(1 + P_tx / (L * noise))
        B = cfg['bandwidth']
        P = cfg['P_tx']
        N0 = cfg['noise_power']
        self._rates: Dict[Tuple[int, int], float] = {}  # (ue_id, node_idx) -> bps
        for m in range(self.M):
            for n_idx, node in enumerate(self.edge_nodes):
                if 'UAV' in node.node_id:
                    L = cfg['path_loss_uav']
                else:
                    L = cfg['path_loss_leo']
                snr = P / (L * N0)
                rate = B * np.log2(1.0 + snr)
                self._rates[(m, n_idx)] = rate

        # per-UE tx energy tracker: accumulated energy while transmitting current task
        self._tx_energy: Dict[int, float] = {m: 0.0 for m in range(self.M)}

        # per-UE in-progress offload info: ue_id -> {'node_idx', 'task', 'bits_left', 'energy'}
        self._offloading: Dict[int, Dict] = {}

        self.time: float = 0.0
        self._step_count: int = 0

        # observation / action dims
        n_nodes = self.N + 1
        self.obs_dim: int = 2 + n_nodes + n_nodes
        self.n_actions: int = 1 + self.N + 1   # LOCAL + N UAVs + LEO

    # ------------------------------------------------------------------
    def _channel_rates_for_ue(self, m: int) -> Dict[str, float]:
        return {
            node.node_id: self._rates[(m, n_idx)]
            for n_idx, node in enumerate(self.edge_nodes)
        }

    def _active_counts(self) -> Dict[str, int]:
        return {node.node_id: node.active_count for node in self.edge_nodes}

    # ------------------------------------------------------------------
    def reset(self) -> List[np.ndarray]:
        self.time = 0.0
        self._step_count = 0
        self._offloading.clear()
        self._tx_energy = {m: 0.0 for m in range(self.M)}
        for ue in self.ues:
            ue.reset()
        for node in self.edge_nodes:
            node.reset()
        return self._get_obs()

    def _get_obs(self) -> List[np.ndarray]:
        ac = self._active_counts()
        return [
            ue.get_observation(ac, self._channel_rates_for_ue(m))
            for m, ue in enumerate(self.ues)
        ]

    # ------------------------------------------------------------------
    def step(self, actions: List[int]) -> Tuple[
        List[np.ndarray], List[float], bool, Dict
    ]:
        """
        Advance environment by one time slot.

        Parameters
        ----------
        actions : list of int, one per UE (0=LOCAL, 1..N=UAV, N+1=LEO)

        Returns
        -------
        obs, rewards, done, info
        """
        assert len(actions) == self.M

        rewards_local = np.zeros(self.M)   # individual rewards
        rewards_global = 0.0

        # ── 1. Task Generation ───────────────────────────────────────
        for m, ue in enumerate(self.ues):
            if np.random.rand() < self.p_arrive[m]:
                size = int(np.random.uniform(
                    self.cfg['task_size_min'], self.cfg['task_size_max']
                ))
                task = {'size': size, 'bits_remaining': size,
                        'time': self.time, 'gen_time': self.time,
                        'task_id': ue.new_task_id(), 'ue_id': m}
                action = actions[m]
                if action == self.ACTION_LOCAL:
                    ue.local_queue.append(task)
                else:
                    # Offload
                    node_idx = action - 1   # action 1→UAV0, ..., N→UAV_{N-1}, N+1→LEO
                    if node_idx < 0 or node_idx >= len(self.edge_nodes):
                        # Invalid action → fallback to local
                        ue.local_queue.append(task)
                    else:
                        ue.tx_queue.append({**task, 'node_idx': node_idx})

        # ── 2 & 3. Transmission → Edge Reception ─────────────────────
        for m, ue in enumerate(self.ues):
            if ue.tx_queue:
                tx_task = ue.tx_queue[0]
                node_idx = tx_task['node_idx']
                rate = self._rates[(m, node_idx)]
                bits_this_slot = rate * self.dt
                sent = min(tx_task['bits_remaining'], bits_this_slot)
                tx_task['bits_remaining'] -= sent

                # energy for transmission
                tx_time = sent / rate
                energy = self.cfg['P_tx'] * tx_time
                self._tx_energy[m] += energy

                if tx_task['bits_remaining'] <= 1.0:
                    # Transmission done → enqueue at edge node
                    node = self.edge_nodes[node_idx]
                    arrival_task = {**tx_task,
                                    'bits_remaining': tx_task['size'],
                                    'tx_energy': self._tx_energy[m]}
                    node.enqueue(m, arrival_task)
                    self._tx_energy[m] = 0.0
                    ue.tx_queue.pop(0)

        # ── 4. Local CPU Processing ───────────────────────────────────
        for m, ue in enumerate(self.ues):
            r = ue.process_local(
                self.time, self.dt, self.w1, self.w2, self.deadline
            )
            rewards_local[m] += r

        # ── 5. Edge CPU Processing ────────────────────────────────────
        # Build tx_energy_map per node (last recorded tx energy for each UE)
        for node in self.edge_nodes:
            tx_energy_map: Dict[int, float] = {}
            for m in range(self.M):
                # find task at this node to get recorded tx_energy
                tasks = node.queues.get(m, [])
                if tasks:
                    tx_energy_map[m] = tasks[0].get('tx_energy', 0.0)
            edge_rewards = node.process(
                self.time, self.dt, self.w1, self.w2,
                self.deadline, tx_energy_map
            )
            for m, r in edge_rewards.items():
                rewards_local[m] += r

        # ── 6. Compute hybrid reward ──────────────────────────────────
        rewards_global = rewards_local.sum()
        hybrid_rewards = [
            (1.0 - self.alpha) * rewards_local[m] + self.alpha * rewards_global
            for m in range(self.M)
        ]

        # Advance time
        self.time += self.dt
        self._step_count += 1

        obs = self._get_obs()
        done = False
        info = {
            'rewards_local': rewards_local.tolist(),
            'rewards_global': float(rewards_global),
            'active_counts': self._active_counts(),
        }
        return obs, hybrid_rewards, done, info

    # ------------------------------------------------------------------
    def sample_actions(self) -> List[int]:
        """Random action for each UE (for testing)."""
        return [np.random.randint(0, self.n_actions) for _ in range(self.M)]
