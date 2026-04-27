"""
train_il.py
Train all UE agents with Independent Learning (IL).

Usage:
    python train_il.py [--episodes 10000] [--steps 50] [--alpha 0.0] [--seed 42]
"""
import argparse, os, json, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from env.ntn_env import NTNMECEnv, DEFAULT_CONFIG
from agents.dqn import DQNAgent
from agents.trainer import run_episode_il, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10_000)
    p.add_argument('--steps',    type=int, default=50)
    p.add_argument('--alpha',    type=float, default=0.0)   # pure IL → 0
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--save_dir', type=str, default='checkpoints/il')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    cfg = {**DEFAULT_CONFIG, 'alpha': args.alpha}
    env = NTNMECEnv(cfg)
    os.makedirs(args.save_dir, exist_ok=True)

    agents = [
        DQNAgent(m, env.obs_dim, env.n_actions,
                 lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000)
        for m in range(env.M)
    ]

    log = []
    for ep in range(1, args.episodes + 1):
        result = run_episode_il(env, agents, steps_per_episode=args.steps, train=True)
        log.append(result)

        if ep % args.log_every == 0:
            recent = log[-args.log_every:]
            mean_r = np.mean([r['total_reward'] for r in recent])
            mean_l = np.mean([r['mean_loss'] for r in recent])
            eps0 = agents[0].epsilon
            print(f"[IL] ep={ep:5d}  reward={mean_r:8.2f}  "
                  f"loss={mean_l:.4f}  eps={eps0:.3f}")

        if ep % args.eval_every == 0:
            metrics = evaluate(env, agents, episodes=10, steps_per_episode=args.steps)
            print(f"  → EVAL: reward={metrics['mean_reward']:.2f}  "
                  f"drop_rate={metrics['mean_drop_rate']:.3f}  "
                  f"throughput={metrics['mean_throughput']:.3f}")
            for m, agent in enumerate(agents):
                agent.save(os.path.join(args.save_dir, f'agent_{m}_ep{ep}.pt'))

    # Final save
    for m, agent in enumerate(agents):
        agent.save(os.path.join(args.save_dir, f'agent_{m}_final.pt'))

    with open(os.path.join(args.save_dir, 'log_il.json'), 'w') as f:
        json.dump([{k: v for k, v in r.items()} for r in log], f, indent=2)

    print("IL training complete.")


if __name__ == '__main__':
    main()
