# NTN-MEC Multi-Agent Reinforcement Learning

> Multi-agent RL framework for task offloading in Non-Terrestrial Network (NTN) Mobile Edge Computing systems — featuring UAVs and LEO satellites.

---

## Overview

This project simulates a **space-air-ground edge computing network** where multiple User Equipment (UEs) must decide — every time slot — whether to:
- Process a task **locally** on their own CPU
- **Offload** to a UAV-mounted edge server
- **Offload** to a LEO satellite

Three learning strategies are implemented and compared:

| Mode | Description |
|------|-------------|
| **IL** (Independent Learning) | Each UE optimises its own cost independently using Dueling DQN |
| **CTDE** (Centralised Training, Decentralised Execution) | Agents share a global reward signal during training; act locally at runtime |
| **Hybrid** | Alternating schedule of IL and CTDE phases |

Two non-learning baselines are also included:

| Baseline | Description |
|----------|-------------|
| **Random** | Each UE picks an action uniformly at random every slot |
| **Local** | Each UE always processes tasks on its own CPU — never offloads |

---

## Project Structure

```
ntn_mec/
├── env/
│   ├── __init__.py
│   ├── ntn_env.py          # Main simulation environment
│   ├── ue.py               # User Equipment model (queues, CPU, energy)
│   └── edge_node.py        # Edge server model (UAV / LEO, CPU sharing)
│
├── agents/
│   ├── __init__.py
│   ├── dqn.py              # Dueling DQN network + DQN agent class
│   └── trainer.py          # IL and CTDE episode runners + evaluator
│
├── utils/
│   ├── __init__.py
│   └── replay_buffer.py    # Fixed-capacity experience replay buffer
│
├── tests/
│   ├── __init__.py
│   ├── test_env.py         # 14 unit tests for the environment
│   └── test_agents.py      # Unit tests for DQN agent and replay buffer
│
├── train_il.py             # Train with Independent Learning
├── train_ctde.py           # Train with CTDE
├── train_hybrid.py         # Train with hybrid IL/CTDE schedule
├── plot_training_curves.py # Plot episode-by-episode training metrics
├── plot_results.py         # Final evaluation bar charts and comparisons
├── ntn_mec_paper_sim.py    # Full paper-parameter simulation (40 UEs, 4 UAVs)
├── main.py                 # Unified entry point (all modes)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy matplotlib
```

> Python 3.8 or higher required. GPU is optional — training runs fine on CPU.

### 2. Run training

**Recommended — run all three modes:**
```bash
python train_il.py     --episodes 10000
python train_ctde.py   --episodes 10000 --alpha 0.5
python train_hybrid.py --episodes 10000 --alpha 0.2
```

**Or use the unified entry point:**
```bash
python main.py --mode il
python main.py --mode ctde
python main.py --mode hybrid
```

**Quick test (200 episodes, completes in minutes):**
```bash
python main.py --mode hybrid --episodes 200
```

### 3. Plot training curves

```bash
python plot_training_curves.py \
  --il     checkpoints/il/log_il.json \
  --ctde   checkpoints/ctde/log_ctde.json \
  --hybrid checkpoints/hybrid/log_hybrid.json
```

### 4. Evaluate and compare

```bash
python main.py --mode eval
```

---

## Parameters and Flags

### `train_il.py` / `train_ctde.py` / `train_hybrid.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | `10000` | Number of training episodes |
| `--steps` | `50` | Time slots per episode |
| `--alpha` | `0.2` | Global reward weight (0 = pure IL, 1 = pure CTDE) |
| `--seed` | `42` | Random seed for reproducibility |
| `--log_every` | `200` | Print training stats every N episodes |
| `--eval_every` | `1000` | Run greedy evaluation every N episodes |
| `--save_dir` | `checkpoints/<mode>` | Directory to save model checkpoints |

### `plot_training_curves.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--il` | `log_il.json` | Path to IL training log |
| `--ctde` | `log_ctde.json` | Path to CTDE training log |
| `--hybrid` | `log_hybrid.json` | Path to Hybrid training log |
| `--out` | `plots/` | Output directory for figures |
| `--window` | `200` | Smoothing window size for curves |

---

## System Model

### Environment defaults

| Parameter | Value | Description |
|-----------|-------|-------------|
| UEs (`M`) | 5 | Number of user devices |
| UAVs (`N`) | 3 | Number of UAV edge servers |
| Slot duration | 0.01 s | Length of one time slot |
| Task arrival prob. | 0.6 | Bernoulli arrival per UE per slot |
| Task size | 5k – 50k bits | Uniform random per task |
| Deadline | 5 slots | Tasks dropped if not completed in time |
| Drop penalty | −100 | Reward penalty for a missed deadline |
| UE CPU | 1 GHz | Local processing frequency |
| UAV CPU | 10 GHz | Edge server frequency (shared equally) |
| LEO CPU | 20 GHz | Satellite server frequency (shared equally) |
| Transmit power | 0.5 W | UE uplink power |
| Bandwidth | 1 MHz | Channel bandwidth |
| `w1`, `w2` | 0.5, 0.5 | Delay and energy cost weights |
| `α` (alpha) | 0.2 | Hybrid reward: local vs global balance |

### Reward function

Each agent's training reward is:

```
r_m = (1 - α) × (−Ψ_m)  +  α × (−Σ_k Ψ_k)
```

where `Ψ_m = w1 × delay + w2 × energy` is the per-UE cost.

- `α = 0` → pure independent learning (selfish)
- `α = 1` → pure global reward (fully cooperative)
- `α = 0.2` → default hybrid (mostly local, team-aware)

### Energy models

| Mode | Formula |
|------|---------|
| Local compute | `E = κ × f² × t_comp` |
| Offload (transmit) | `E = P_tx × t_tx` |

---

## Agent Architecture — Dueling DQN

```
Observation vector (10 values)
        │
  Feature MLP (128 → 128, ReLU)
        │
   ┌────┴────┐
   ▼         ▼
Value V(s)  Advantage A(s, a)
 (1 output)  (5 outputs)
   └────┬────┘
        ▼
  Q(s,a) = V(s) + A(s,a) − mean(A)
```

- **Optimiser:** Adam, lr = 1e-3
- **Target network:** synced every 100 training steps
- **Exploration:** ε-greedy, decaying 1.0 → 0.05
- **Replay buffer:** 50,000 transitions, batch size 32
- **Discount:** γ = 0.99

---

## Hybrid Training Schedule

```
Episodes:  [  0 – 2000  ] [2000–2500] [2500–3500] [3500–4000] ...
Mode:           CTDE           IL          CTDE          IL
```

The schedule is defined as fractions of total episodes:

```python
schedule = [
    ('ctde', 0.20),   # 20% CTDE first — learn to cooperate
    ('il',   0.05),   # 5%  IL  — refine locally
    ('ctde', 0.10),   # 10% CTDE
    ('il',   0.05),
    ('ctde', 0.10),
    ('il',   0.05),
    ('ctde', 0.10),
    ('il',   0.35),   # remainder IL
]
```

---

## Outputs

After training, the following files are created:

```
checkpoints/
├── il/
│   ├── agent_0_final.pt       # Saved weights for UE 0
│   ├── agent_1_final.pt
│   ├── ...
│   ├── agent_0_ep1000.pt      # Checkpoint at episode 1000
│   └── log_il.json            # Full episode-by-episode log
├── ctde/
│   └── ...
└── hybrid/
    └── ...

plots/
├── 1_total_reward_curves.png  # Reward vs episode (all methods)
├── 2_loss_curves.png          # Loss convergence
├── 3_per_agent_rewards.png    # Per-UE reward (fairness view)
├── 4_hybrid_phases.png        # Hybrid coloured by IL/CTDE phase
└── 5_rolling_stats.png        # Rolling mean ± std bands
```

---

## Running Unit Tests

```bash
# Run all environment tests (no PyTorch needed)
python tests/test_env.py

# Run agent tests (requires PyTorch)
python tests/test_agents.py

# Or with pytest
pytest tests/ -v
```

Expected output: **14 environment tests**, all passing.

---

## Paper Simulation (40 UEs, 4 UAVs)

To run the full paper-scale simulation matching the GLOBECOM 2024 parameters:

```bash
python ntn_mec_paper_sim.py --episodes 1000
```

Quick test:
```bash
python ntn_mec_paper_sim.py --episodes 100
```

This runs all 5 policies (Random, Local, IL, CTDE, Hybrid) and produces 6 comparison figures in `paper_plots/`.

| Figure | Content |
|--------|---------|
| `fig1_weighted_cost.png` | Ψ = w1·t + w2·e vs episodes |
| `fig2_avg_delay.png` | Average task completion delay |
| `fig3_avg_energy.png` | Average energy consumption |
| `fig4_drop_rate.png` | Task drop rate vs episodes |
| `fig5_summary_2x2.png` | All 4 metrics in one figure |
| `fig6_final_bar.png` | Final performance bar chart |

---

## Results Summary

| Method | Avg Reward | Drop Rate | Throughput | Fairness (std) |
|--------|-----------|-----------|------------|----------------|
| IL | −21.75 | 0.002 | 0.998 | 1.27 |
| CTDE | −68.86 | 0.000 | 1.000 | **0.50** |
| Hybrid | −44.16 | 0.001 | 0.999 | 1.19 |

- **IL** converges fastest, lowest per-agent cost, least fair
- **CTDE** slowest to converge, most equitable across UEs
- **Hybrid** best loss fit (0.070), balanced fairness and speed

---

## References

1. H. Li et al., "Multi-agent reinforcement learning based computation offloading for LEO satellite edge computing," *Computer Communications*, 2024.
2. J.-H. Lee et al., "Integrating LEO satellites and multi-UAV RL for hybrid NTN," *IEEE TNSE*, 2024.
3. T. Liu et al., "Energy-Efficient UAV-Assisted IoT and LEO Offloading in SAGIN," *Electronics*, 2025.
4. R. Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments," *NeurIPS*, 2017.

---

## License

This project was developed for academic research purposes.
