"""
env/ue.py
User Equipment class: manages local CPU queue, transmit queue, energy tracking.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any


class UE:
    """
    Represents a ground User Equipment (UE) agent.

    Parameters
    ----------
    ue_id : int
    f_loc : float   local CPU frequency in Hz (e.g. 1e9 for 1 GHz)
    kappa : float   energy coefficient  E = kappa * f^2 * t_comp
    cycles_per_bit : float  CPU cycles needed per bit (s_m)
    """

    def __init__(self, ue_id: int, f_loc: float, kappa: float, cycles_per_bit: float):
        self.ue_id = ue_id
        self.f_loc = f_loc
        self.kappa = kappa
        self.cycles_per_bit = cycles_per_bit  # s_m

        # queues
        self.local_queue: List[Dict[str, Any]] = []   # [{'size', 'time', 'task_id'}]
        self.tx_queue: List[Dict[str, Any]] = []       # [{'size', 'target', 'time', 'task_id'}]

        # in-progress local task state
        self._current_task: Optional[Dict] = None
        self._local_time_remaining: float = 0.0
        self._local_energy_this_task: float = 0.0

        # per-step energy accumulator for offloaded transmissions
        self.tx_energy_pending: float = 0.0   # energy spent this slot on TX

        # running stats (reset each episode)
        self.completed_tasks: List[Dict] = []
        self.dropped_tasks: List[Dict] = []

        self._task_counter = 0

    # ------------------------------------------------------------------
    def reset(self):
        self.local_queue.clear()
        self.tx_queue.clear()
        self._current_task = None
        self._local_time_remaining = 0.0
        self._local_energy_this_task = 0.0
        self.tx_energy_pending = 0.0
        self.completed_tasks.clear()
        self.dropped_tasks.clear()

    # ------------------------------------------------------------------
    def new_task_id(self) -> str:
        self._task_counter += 1
        return f"ue{self.ue_id}_t{self._task_counter}"

    # ------------------------------------------------------------------
    def get_observation(self, edge_active_counts: Dict[str, int],
                        channel_rates: Dict[str, float]) -> np.ndarray:
        """
        Build observation vector for this UE.

        Returns a 1-D float32 numpy array:
          [local_queue_bits, tx_queue_size, *edge_active_counts, *channel_rates_Mbps]
        """
        loc_bits = sum(t['size'] for t in self.local_queue)
        if self._current_task is not None:
            loc_bits += self._local_time_remaining * self.f_loc / self.cycles_per_bit
        tx_items = len(self.tx_queue)
        edge_loads = np.array(list(edge_active_counts.values()), dtype=np.float32)
        rates = np.array(list(channel_rates.values()), dtype=np.float32) / 1e6  # Mbps
        obs = np.concatenate([
            [loc_bits / 1e6,   # normalise to Mbits
             float(tx_items)],
            edge_loads,
            rates,
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    def process_local(self, current_time: float, dt: float,
                      w1: float, w2: float,
                      deadline: float) -> float:
        """
        Advance local CPU by one time-slot (duration dt seconds).

        Returns the reward contribution from any tasks that finish
        (or are dropped) this slot.
        """
        reward = 0.0

        # If no task in progress, pull from queue
        if self._current_task is None and self.local_queue:
            self._current_task = self.local_queue.pop(0)
            bits = self._current_task['size']
            self._local_time_remaining = (bits * self.cycles_per_bit) / self.f_loc
            self._local_energy_this_task = 0.0

        if self._current_task is None:
            return reward

        # Check deadline before even processing
        age = current_time - self._current_task['time']
        if age > deadline:
            # Drop
            self.dropped_tasks.append({**self._current_task, 'drop_time': current_time})
            reward += -100.0   # large drop penalty
            self._current_task = None
            self._local_time_remaining = 0.0
            self._local_energy_this_task = 0.0
            return reward

        # Process up to dt seconds of CPU
        work_time = min(self._local_time_remaining, dt)
        self._local_time_remaining -= work_time
        energy_chunk = self.kappa * (self.f_loc ** 2) * work_time
        self._local_energy_this_task += energy_chunk

        if self._local_time_remaining <= 1e-12:
            # Task completed
            delay = (current_time + work_time) - self._current_task['time']
            energy = self._local_energy_this_task
            cost = w1 * delay + w2 * energy
            reward += -cost
            self.completed_tasks.append({
                **self._current_task,
                'delay': delay,
                'energy': energy,
                'finish_time': current_time + work_time,
                'mode': 'local',
            })
            self._current_task = None
            self._local_time_remaining = 0.0
            self._local_energy_this_task = 0.0

        return reward
