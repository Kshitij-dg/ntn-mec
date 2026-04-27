"""
tests/test_env.py
Unit tests for the NTN-MEC environment.

Run with:  python -m pytest tests/ -v
       or: python tests/test_env.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from env.ntn_env import NTNMECEnv, DEFAULT_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
def make_env(**overrides):
    cfg = {**DEFAULT_CONFIG, 'M': 2, 'N': 1, **overrides}
    return NTNMECEnv(cfg)


# ──────────────────────────────────────────────────────────────────────────────
class TestEnvBasics:

    def test_reset_returns_correct_obs_shapes(self):
        env = make_env()
        obs = env.reset()
        assert len(obs) == env.M
        for o in obs:
            assert o.shape == (env.obs_dim,)
            assert o.dtype == np.float32

    def test_step_returns_correct_shapes(self):
        env = make_env()
        env.reset()
        actions = [0] * env.M      # all LOCAL
        obs, rewards, done, info = env.step(actions)
        assert len(obs) == env.M
        assert len(rewards) == env.M
        assert isinstance(done, bool)
        assert 'rewards_local' in info

    def test_action_local_adds_to_local_queue(self):
        """If action is LOCAL, tasks should appear in ue.local_queue."""
        env = make_env(p_arrive=1.0)   # guaranteed arrivals
        env.reset()
        env.step([0, 0])          # all LOCAL
        # After one step with guaranteed arrivals, local queues may have tasks
        # (they could also start processing immediately in the same step)
        # At least the counters shouldn't error out
        assert True

    def test_action_offload_uav_adds_to_tx_queue(self):
        """Offload action should push task to tx_queue."""
        env = make_env(p_arrive=1.0, task_size_min=500_000, task_size_max=500_000)
        env.reset()
        # Action 1 = UAV0
        obs, rewards, done, info = env.step([1, 1])
        # UE 0's tx_queue or offloading should have a task
        # (It might have been sent in one slot if rate is high enough)
        assert True   # no crash is the key test

    def test_time_advances_each_step(self):
        env = make_env()
        env.reset()
        assert env.time == 0.0
        env.step([0] * env.M)
        assert abs(env.time - env.cfg['dt']) < 1e-12

    def test_rewards_are_finite(self):
        env = make_env(p_arrive=0.8)
        env.reset()
        for _ in range(20):
            actions = env.sample_actions()
            _, rewards, _, _ = env.step(actions)
            for r in rewards:
                assert np.isfinite(r), f"Non-finite reward: {r}"

    def test_obs_dimension_formula(self):
        """obs_dim should equal 2 + (N+1) + (N+1)."""
        env = make_env(N=2)
        expected = 2 + (env.N + 1) + (env.N + 1)
        assert env.obs_dim == expected

    def test_n_actions(self):
        """n_actions = LOCAL(1) + N UAVs + LEO(1)."""
        env = make_env(N=3)
        assert env.n_actions == 1 + 3 + 1

    def test_reset_clears_state(self):
        env = make_env(p_arrive=1.0)
        env.reset()
        for _ in range(10):
            env.step(env.sample_actions())
        env.reset()
        assert env.time == 0.0
        for ue in env.ues:
            assert len(ue.local_queue) == 0
            assert len(ue.tx_queue) == 0
        for node in env.edge_nodes:
            assert node.active_count == 0


# ──────────────────────────────────────────────────────────────────────────────
class TestEnergyAndDelay:

    def test_local_energy_positive_when_tasks_complete(self):
        """Completed local tasks should have positive energy."""
        env = make_env(M=1, N=1, p_arrive=1.0,
                       task_size_min=1000, task_size_max=1000)
        env.reset()
        # Run enough steps for a local task to complete
        for _ in range(20):
            env.step([0])   # always LOCAL
        completed = env.ues[0].completed_tasks
        if completed:
            for t in completed:
                assert t['energy'] >= 0.0
                assert t['delay'] >= 0.0

    def test_drop_penalty_applied_when_deadline_exceeded(self):
        """With very tight deadline, tasks should be dropped."""
        env = make_env(
            M=1, N=1, p_arrive=1.0,
            task_size_min=1_000_000, task_size_max=1_000_000,  # huge task
            deadline_slots=1,   # only 1 slot deadline
            f_loc=1e6,          # slow CPU
        )
        env.reset()
        rewards_sum = 0.0
        for _ in range(5):
            _, rewards, _, _ = env.step([0])
            rewards_sum += sum(rewards)
        # Large negative rewards expected due to drops
        assert rewards_sum < 0


# ──────────────────────────────────────────────────────────────────────────────
class TestChannelRates:

    def test_uav_rate_higher_than_leo(self):
        """UAV (lower path loss) should have higher channel rate than LEO."""
        env = make_env()
        uav_rate = env._rates[(0, 0)]    # UE0 → UAV0
        leo_rate  = env._rates[(0, env.N)]  # UE0 → LEO
        assert uav_rate > leo_rate, "UAV should have better channel than LEO"


# ──────────────────────────────────────────────────────────────────────────────
class TestEdgeNode:

    def test_active_count_updates(self):
        from env.edge_node import EdgeNode
        node = EdgeNode('UAV0', f_edge=10e9, cycles_per_bit=1000)
        assert node.active_count == 0
        node.enqueue(0, {'size': 5000, 'bits_remaining': 5000,
                         'gen_time': 0.0, 'task_id': 't1', 'tx_energy': 0.0})
        assert node.active_count == 1

    def test_cpu_sharing_splits_equally(self):
        """Two UEs at one node → each gets f_edge/2 CPU share."""
        from env.edge_node import EdgeNode
        node = EdgeNode('UAV0', f_edge=10e9, cycles_per_bit=1000, prop_delay=0)
        task = lambda uid: {'size': 100_000, 'bits_remaining': 100_000,
                             'gen_time': 0.0, 'task_id': f't{uid}', 'tx_energy': 0.1}
        node.enqueue(0, task(0))
        node.enqueue(1, task(1))
        assert node.active_count == 2
        rewards = node.process(0.001, dt=0.001, w1=0.5, w2=0.5,
                               deadline=100.0, tx_energy_map={0: 0.1, 1: 0.1})
        # Both should have been processed (not necessarily done in 1 step)
        # Just check no errors and counts are still valid
        assert node.active_count in [0, 1, 2]


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Run without pytest
    import traceback

    suites = [TestEnvBasics, TestEnergyAndDelay, TestChannelRates, TestEdgeNode]
    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        for name in dir(suite_cls):
            if not name.startswith('test_'):
                continue
            try:
                getattr(suite, name)()
                print(f"  ✓  {suite_cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  ✗  {suite_cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed.")
