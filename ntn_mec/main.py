"""
main.py
Unified entry point.

Usage:
    python main.py --mode il     [--episodes 10000] [--steps 50] [--alpha 0.0]
    python main.py --mode ctde   [--episodes 10000] [--steps 50] [--alpha 0.5]
    python main.py --mode hybrid [--episodes 10000] [--steps 50] [--alpha 0.2]
    python main.py --mode eval   [--il_dir ...] [--ctde_dir ...] [--hybrid_dir ...]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description='NTN-MEC Multi-Agent RL')
    parser.add_argument('--mode', choices=['il', 'ctde', 'hybrid', 'eval'],
                        required=True)
    parser.add_argument('--episodes',  type=int,   default=10_000)
    parser.add_argument('--steps',     type=int,   default=50)
    parser.add_argument('--alpha',     type=float, default=0.2)
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--log_every', type=int,   default=200)
    parser.add_argument('--eval_every',type=int,   default=1000)
    parser.add_argument('--save_dir',  type=str,   default=None)
    # eval-only args
    parser.add_argument('--il_dir',     type=str, default='checkpoints/il')
    parser.add_argument('--ctde_dir',   type=str, default='checkpoints/ctde')
    parser.add_argument('--hybrid_dir', type=str, default='checkpoints/hybrid')
    args = parser.parse_args()

    if args.mode == 'il':
        from train_il import main as run
        sys.argv = [
            'train_il.py',
            f'--episodes={args.episodes}',
            f'--steps={args.steps}',
            f'--alpha={args.alpha}',
            f'--seed={args.seed}',
            f'--log_every={args.log_every}',
            f'--eval_every={args.eval_every}',
            f'--save_dir={args.save_dir or "checkpoints/il"}',
        ]
        run()

    elif args.mode == 'ctde':
        from train_ctde import main as run
        sys.argv = [
            'train_ctde.py',
            f'--episodes={args.episodes}',
            f'--steps={args.steps}',
            f'--alpha={args.alpha}',
            f'--seed={args.seed}',
            f'--log_every={args.log_every}',
            f'--eval_every={args.eval_every}',
            f'--save_dir={args.save_dir or "checkpoints/ctde"}',
        ]
        run()

    elif args.mode == 'hybrid':
        from train_hybrid import main as run
        sys.argv = [
            'train_hybrid.py',
            f'--episodes={args.episodes}',
            f'--steps={args.steps}',
            f'--alpha={args.alpha}',
            f'--seed={args.seed}',
            f'--log_every={args.log_every}',
            f'--eval_every={args.eval_every}',
            f'--save_dir={args.save_dir or "checkpoints/hybrid"}',
        ]
        run()

    elif args.mode == 'eval':
        from plot_results import main as run
        sys.argv = [
            'plot_results.py',
            f'--il_dir={args.il_dir}',
            f'--ctde_dir={args.ctde_dir}',
            f'--hybrid_dir={args.hybrid_dir}',
            f'--episodes=50',
            f'--steps={args.steps}',
        ]
        run()


if __name__ == '__main__':
    main()
