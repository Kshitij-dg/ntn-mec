"""
train_hybrid.py
Hybrid training: alternating CTDE and IL phases.

Default schedule: 20% CTDE → 5% IL → 10% CTDE → 5% IL → 10% CTDE → rest IL

Usage:
    python train_hybrid.py [--episodes 10000] [--steps 50] [--alpha 0.2] [--seed 42]
"""
import argparse, os, json, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from env.ntn_env import NTNMECEnv, DEFAULT_CONFIG
from agents.dqn import DQNAgent
from agents.trainer import run_episode_il, run_episode_ctde, evaluate


# Default phased schedule: list of (mode, fraction_of_total_episodes)
DEFAULT_SCHEDULE = [
    ('ctde', 0.20),
    ('il',   0.05),
    ('ctde', 0.10),
    ('il',   0.05),
    ('ctde', 0.10),
    ('il',   0.05),
    ('ctde', 0.10),
    ('il',   0.35),   # remainder
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes',  type=int,   default=10_000)
    p.add_argument('--steps',     type=int,   default=50)
    p.add_argument('--alpha',     type=float, default=0.2)
    p.add_argument('--seed',      type=int,   default=42)
    p.add_argument('--log_every', type=int,   default=200)
    p.add_argument('--eval_every',type=int,   default=1000)
    p.add_argument('--save_dir',  type=str,   default='checkpoints/hybrid')
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
                 eps_start=1.0, eps_end=0.05, eps_decay=6000)
        for m in range(env.M)
    ]

    # Expand schedule into episode counts
    schedule_eps = []
    for mode, frac in DEFAULT_SCHEDULE:
        n = max(1, int(frac * args.episodes))
        schedule_eps.append((mode, n))

    total_scheduled = sum(n for _, n in schedule_eps)
    remainder = args.episodes - total_scheduled
    if remainder > 0:
        schedule_eps[-1] = (schedule_eps[-1][0], schedule_eps[-1][1] + remainder)

    log = []
    ep = 0
    for phase_idx, (mode, n_eps) in enumerate(schedule_eps):
        print(f"\n── Phase {phase_idx+1}: {mode.upper()} for {n_eps} episodes ──")
        for _ in range(n_eps):
            ep += 1
            if mode == 'ctde':
                result = run_episode_ctde(env, agents,
                                          steps_per_episode=args.steps, train=True)
            else:
                result = run_episode_il(env, agents,
                                        steps_per_episode=args.steps, train=True)
            log.append({'ep': ep, **result})

            if ep % args.log_every == 0:
                recent = log[-args.log_every:]
                mean_r = np.mean([r['total_reward'] for r in recent])
                mean_l = np.mean([r['mean_loss'] for r in recent])
                print(f"  ep={ep:5d} [{mode}]  reward={mean_r:8.2f}  "
                      f"loss={mean_l:.4f}  eps={agents[0].epsilon:.3f}")

            if ep % args.eval_every == 0:
                metrics = evaluate(env, agents,
                                   episodes=10, steps_per_episode=args.steps)
                print(f"  → EVAL: reward={metrics['mean_reward']:.2f}  "
                      f"drop={metrics['mean_drop_rate']:.3f}  "
                      f"throughput={metrics['mean_throughput']:.3f}")
                for m, agent in enumerate(agents):
                    agent.save(os.path.join(args.save_dir, f'agent_{m}_ep{ep}.pt'))

    for m, agent in enumerate(agents):
        agent.save(os.path.join(args.save_dir, f'agent_{m}_final.pt'))

    with open(os.path.join(args.save_dir, 'log_hybrid.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print("\nHybrid training complete.")


if __name__ == '__main__':
    main()
