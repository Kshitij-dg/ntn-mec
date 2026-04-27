"""
env/edge_node.py
Edge server (UAV or LEO satellite) with equal-share CPU scheduling.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any


class EdgeNode:
    """
    Represents a UAV or LEO edge server.

    CPU is shared equally among all UEs that currently have tasks here.
    Energy cost to the UE comes only from transmission (tracked in NTNMECEnv).

    Parameters
    ----------
    node_id : str
    f_edge  : float   CPU frequency in Hz (e.g. 10e9 for 10 GHz)
    cycles_per_bit : float
    prop_delay : float  fixed propagation delay in seconds (0 for UAV, ~0.01 for LEO)
    """

    def __init__(self, node_id: str, f_edge: float,
                 cycles_per_bit: float, prop_delay: float = 0.0):
        self.node_id = node_id
        self.f_edge = f_edge
        self.cycles_per_bit = cycles_per_bit
        self.prop_delay = prop_delay

        # per-UE queues: ue_id -> list of task dicts (with 'size','gen_time','tx_done_time')
        self.queues: Dict[int, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    def reset(self):
        self.queues.clear()

    # ------------------------------------------------------------------
    @property
    def active_count(self) -> int:
        """Number of UEs that currently have pending bits here."""
        return sum(1 for tasks in self.queues.values() if tasks)

    # ------------------------------------------------------------------
    def enqueue(self, ue_id: int, task: Dict[str, Any]):
        """Add a received task to this node's queue for ue_id."""
        if ue_id not in self.queues:
            self.queues[ue_id] = []
        self.queues[ue_id].append(dict(task))  # shallow copy

    # ------------------------------------------------------------------
    def process(self, current_time: float, dt: float,
                w1: float, w2: float,
                deadline: float,
                tx_energy_map: Dict[int, float]) -> Dict[int, float]:
        """
        Advance edge CPU by one slot.

        tx_energy_map : {ue_id: tx_energy_spent_for_this_task}
            Used to compute the UE's total offload energy when task completes.

        Returns
        -------
        rewards : dict {ue_id -> reward_increment}
        """
        rewards: Dict[int, float] = {}
        B = self.active_count
        if B == 0:
            return rewards

        cpu_share = self.f_edge / B   # Hz per active UE

        for ue_id, task_list in list(self.queues.items()):
            if not task_list:
                continue
            task = task_list[0]   # FIFO: process head-of-line

            # Deadline check (from task generation time, including prop delay)
            age = current_time - task['gen_time']
            if age > deadline:
                task_list.pop(0)
                rewards[ue_id] = rewards.get(ue_id, 0.0) - 100.0
                if not task_list:
                    del self.queues[ue_id]
                continue

            # Process up to (cpu_share * dt) cycles worth of bits
            processable_bits = (cpu_share * dt) / self.cycles_per_bit
            work = min(task['bits_remaining'], processable_bits)
            task['bits_remaining'] -= work

            if task['bits_remaining'] <= 1e-6:
                # Task done
                finish_time = current_time + (work / (cpu_share / self.cycles_per_bit))
                delay = finish_time - task['gen_time']
                tx_energy = tx_energy_map.get(ue_id, 0.0)
                cost = w1 * delay + w2 * tx_energy
                rewards[ue_id] = rewards.get(ue_id, 0.0) - cost
                task_list.pop(0)
                if not task_list:
                    del self.queues[ue_id]

        return rewards
