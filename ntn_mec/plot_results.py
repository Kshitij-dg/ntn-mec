"""
plot_results.py
Evaluate trained models and produce comparison plots.

Usage:
    python plot_results.py \
        --il_dir   checkpoints/il   \
        --ctde_dir checkpoints/ctde \
        --hybrid_dir checkpoints/hybrid \
        --episodes 50 --steps 50
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from env.ntn_env import NTNMECEnv, DEFAULT_CONFIG
from agents.dqn import DQNAgent
from agents.trainer import evaluate


def load_agents(env, save_dir):
    agents = [
        DQNAgent(m, env.obs_dim, env.n_actions)
        for m in range(env.M)
    ]
    for m, agent in enumerate(agents):
        path = os.path.join(save_dir, f'agent_{m}_final.pt')
        if os.path.exists(path):
            agent.load(path)
        else:
            print(f"  [warn] checkpoint not found: {path} — using random agent")
    return agents


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--il_dir',     type=str, default='checkpoints/il')
    p.add_argument('--ctde_dir',   type=str, default='checkpoints/ctde')
    p.add_argument('--hybrid_dir', type=str, default='checkpoints/hybrid')
    p.add_argument('--episodes',   type=int, default=50)
    p.add_argument('--steps',      type=int, default=50)
    p.add_argument('--out_dir',    type=str, default='plots')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = {**DEFAULT_CONFIG}
    env = NTNMECEnv(cfg)

    results = {}
    for label, save_dir in [
        ('IL',     args.il_dir),
        ('CTDE',   args.ctde_dir),
        ('Hybrid', args.hybrid_dir),
    ]:
        print(f"\nEvaluating {label} from {save_dir} …")
        agents = load_agents(env, save_dir)
        metrics = evaluate(env, agents,
                           episodes=args.episodes,
                           steps_per_episode=args.steps)
        results[label] = metrics
        print(f"  reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}  "
              f"drop={metrics['mean_drop_rate']:.3f}  "
              f"throughput={metrics['mean_throughput']:.3f}")

    # ── Bar chart: reward ───────────────────────────────────────────
    methods = list(results.keys())
    mean_r = [results[m]['mean_reward'] for m in methods]
    std_r  = [results[m]['std_reward']  for m in methods]
    drop_r = [results[m]['mean_drop_rate'] for m in methods]
    tput   = [results[m]['mean_throughput'] for m in methods]

    x = np.arange(len(methods))
    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Reward
    axes[0].bar(x, mean_r, yerr=std_r, color=colors,
                capsize=4, width=0.5, alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('Mean Cumulative Reward')
    axes[0].set_title('Reward Comparison')

    # Drop rate
    axes[1].bar(x, drop_r, color=colors, width=0.5, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods)
    axes[1].set_ylabel('Task Drop Rate')
    axes[1].set_title('Drop Rate')
    axes[1].set_ylim(0, 1)

    # Throughput
    axes[2].bar(x, tput, color=colors, width=0.5, alpha=0.85)
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods)
    axes[2].set_ylabel('Throughput (fraction)')
    axes[2].set_title('Throughput')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, 'comparison_bars.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot → {out_path}")

    # ── Alpha sweep (ablation) ──────────────────────────────────────
    # Quickly simulate alpha effect using random agents (illustrative)
    alphas = np.linspace(0, 1, 11)
    alpha_rewards = []
    print("\nAlpha sweep (random policy — illustrative) …")
    for a in alphas:
        cfg_a = {**DEFAULT_CONFIG, 'alpha': a}
        env_a = NTNMECEnv(cfg_a)
        random_agents = [
            DQNAgent(m, env_a.obs_dim, env_a.n_actions, eps_start=1.0, eps_end=1.0)
            for m in range(env_a.M)
        ]
        m = evaluate(env_a, random_agents, episodes=5, steps_per_episode=20)
        alpha_rewards.append(m['mean_reward'])

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(alphas, alpha_rewards, 'o-', color='#4C72B0')
    ax2.set_xlabel('α (local → global weight)')
    ax2.set_ylabel('Mean Reward (random policy)')
    ax2.set_title('Alpha Ablation (illustrative)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, 'alpha_ablation.png')
    plt.savefig(out2, dpi=150)
    print(f"Saved plot → {out2}")


if __name__ == '__main__':
    main()
