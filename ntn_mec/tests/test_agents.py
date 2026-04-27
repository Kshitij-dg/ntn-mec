"""
tests/test_agents.py
Unit tests for DQN agent and replay buffer.

Run with:  python -m pytest tests/ -v
       or: python tests/test_agents.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from agents.dqn import DuelingDQN, DQNAgent
from utils.replay_buffer import ReplayBuffer


# ──────────────────────────────────────────────────────────────────────────────
class TestDuelingDQN:

    def test_output_shape(self):
        net = DuelingDQN(input_dim=10, n_actions=5)
        x = torch.randn(4, 10)
        q = net(x)
        assert q.shape == (4, 5)

    def test_output_is_finite(self):
        net = DuelingDQN(input_dim=8, n_actions=3)
        x = torch.randn(16, 8)
        q = net(x)
        assert torch.all(torch.isfinite(q))

    def test_advantage_mean_zero(self):
        """By construction mean(A) is subtracted, so advantage branch mean ≈ 0."""
        net = DuelingDQN(input_dim=8, n_actions=4)
        x = torch.randn(32, 8)
        feat = net.feature(x)
        adv = net.adv_stream(feat)
        # The Q output subtracts adv.mean; the adv branch itself is not zero-mean,
        # but the Q - V decomposition should hold: Q.mean(dim=1) ≈ V(s)
        q = net(x)
        val = net.value_stream(feat)
        # Q = V + (A - mean(A)) → mean_action(Q) == V
        assert torch.allclose(q.mean(dim=1), val.squeeze(1), atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
class TestReplayBuffer:

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        obs     = np.zeros(8, dtype=np.float32)
        next_obs = np.ones(8, dtype=np.float32)
        for i in range(50):
            buf.push(obs, i % 3, float(i), next_obs, False)
        assert len(buf) == 50
        o, a, r, no, d = buf.sample(32)
        assert o.shape  == (32, 8)
        assert a.shape  == (32,)
        assert r.shape  == (32,)
        assert no.shape == (32, 8)
        assert d.shape  == (32,)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10)
        obs = np.zeros(4, dtype=np.float32)
        for i in range(20):
            buf.push(obs, 0, 0.0, obs, False)
        assert len(buf) == 10   # capped at capacity


# ──────────────────────────────────────────────────────────────────────────────
class TestDQNAgent:

    def _make_agent(self):
        return DQNAgent(agent_id=0, obs_dim=8, n_actions=3,
                        lr=1e-3, gamma=0.99,
                        eps_start=1.0, eps_end=0.05, eps_decay=100,
                        buffer_cap=1000, batch_size=16,
                        target_update_freq=50)

    def test_action_shape(self):
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)
        a = agent.select_action(obs)
        assert isinstance(a, int)
        assert 0 <= a < 3

    def test_greedy_action_consistent(self):
        """Greedy action for the same obs should be deterministic after training."""
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)
        actions = {agent.select_action(obs, greedy=True) for _ in range(5)}
        assert len(actions) == 1   # same action every time

    def test_learn_returns_loss_after_enough_data(self):
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)
        nobs = np.random.randn(8).astype(np.float32)
        for _ in range(20):
            agent.store(obs, 0, -1.0, nobs, False)
        loss = agent.learn()
        assert loss is not None
        assert np.isfinite(loss)

    def test_learn_returns_none_when_buffer_too_small(self):
        agent = self._make_agent()
        loss = agent.learn()
        assert loss is None

    def test_epsilon_decreases(self):
        agent = self._make_agent()
        obs = np.zeros(8, dtype=np.float32)
        eps_values = []
        for _ in range(200):
            agent.select_action(obs)
            eps_values.append(agent.epsilon)
        assert eps_values[-1] < eps_values[0]

    def test_save_load(self, tmp_path=None):
        import tempfile, os
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)
        nobs = np.random.randn(8).astype(np.float32)
        for _ in range(20):
            agent.store(obs, 1, -0.5, nobs, False)
        agent.learn()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        agent.save(path)

        agent2 = self._make_agent()
        agent2.load(path)
        os.unlink(path)

        # Check parameters match
        for p1, p2 in zip(agent.online_net.parameters(),
                          agent2.online_net.parameters()):
            assert torch.allclose(p1, p2)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import traceback

    suites = [TestDuelingDQN, TestReplayBuffer, TestDQNAgent]
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
