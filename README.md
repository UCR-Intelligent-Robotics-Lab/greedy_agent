# REFiNE: From Reward Fairness to Per-Agent Energy Regularization for Robust Coordination in RL-Driven Robots

Reference implementation and results for the REFiNE paper. This is the PC / training codebase; the on-robot energy and timing numbers in the paper are measured on TurtleBot3 robots with onboard NVIDIA Orin Nano compute.

The code is written against the TensorFlow 1 execution model and runs through `tensorflow.compat.v1` (`tf.disable_eager_execution()`), so a clean TensorFlow 2.x install works without a TF1 downgrade.

## Overview

The repo implements three incentive-based MARL methods on two social-dilemma benchmarks:

- **LIO** (Learning to Incentivize Others, Yang et al., NeurIPS 2020): the baseline, fully decentralized bidirectional incentive exchange.
- **EIA** (Exogenous Incentive Adjustment): a training-time incentive-channel stressor. It multiplicatively rescales the incentives one agent sends, action by action, to model channel faults such as quantization drift and lossy links. EIA is a diagnostic fault model: it changes nothing but the magnitude of transmitted incentives.
- **REFiNE** (Reward and Energy Fairness in Naturally Equilibrated Incentivization): augments LIO's policy objective with two regularizers, a direct per-agent energy penalty and a variance-based (`-Var`) reward-fairness term.

Benchmarks: **Escape Room** ER(N, M), a stateful threshold-coordination game, and the stateless **Iterated Prisoner's Dilemma** (IPD).

Central finding of the paper: for battery-limited robot teams, reward fairness is the wrong control variable for the per-agent energy that ends missions. An incentive redistributes reward without changing physical cost, so reward parity and energy parity are decoupled and the reward-fairness term is structurally inert here. The energy regularizer is the effective and sufficient lever. In the code, REFiNE = the FULL variant (both terms), but the energy term carries the result.

## Repository layout

```
greedy_agent/
├── lio/
│   ├── alg/                          # algorithms, configs, training, analysis
│   │   ├── config_room_lio.py            # LIO  on Escape Room
│   │   ├── config_room_REFiNE.py         # REFiNE on Escape Room
│   │   ├── config_ipd_lio.py             # LIO  on IPD
│   │   ├── config_ipd_REFiNE.py          # REFiNE on IPD
│   │   ├── lio_agent_ipd.py              # LIO agent (IPD)
│   │   ├── lio_eia_er.py, lio_eia_ipd.py # LIO + EIA agents (ER, IPD)
│   │   ├── REFiNE_er.py, REFiNE_ipd.py        # REFiNE agents (ER, IPD)
│   │   ├── REFiNE_eia_er.py, REFiNE_eia_ipd.py# REFiNE + EIA agents (ER, IPD)
│   │   ├── train_lio_{er,eia_er,ipd,eia_ipd}.py
│   │   ├── train_REFiNE_{er,eia_er,ipd,eia_ipd}.py
│   │   ├── evaluate.py                   # eval rollouts: test_room_symmetric / test_ipd
│   │   ├── verify_er.py, verify_ipd.py, analyze_quick.py   # checks / quick analysis
│   │   ├── plot_ER42_REFiNE.ipynb        # ER(4,2) figures (energy / reward / ablation / robustness)
│   │   └── vis_table3.ipynb              # results table
│   ├── env/                          # environments
│   │   ├── room_symmetric.py, room_symmetric_baseline.py, room_symmetric_centralized.py
│   │   ├── room_agent.py
│   │   ├── ipd_wrapper.py
│   │   └── maps.py
│   ├── utils/                        # configdict.py, util.py
│   └── results/                      # per-run logs (created at runtime)
└── sensys_runs/                      # experiment orchestration
    ├── run_single.py                 # one (method, size, weight, seed) run + provenance.json
    ├── run_batch.py                  # multi-seed matrix over methods / sizes / weights
    ├── run_full_sweep.sh             # canonical paper sweep
    ├── launch_tmux.sh                # smoke-gated tmux launcher
    └── ntfy_notify.py                # push notifications
```

## Setup

Reference environment: a conda env (`LIO_tecs` on our machines) with **Python 3.8+** and **TensorFlow 2.x**.

External dependencies:
- `numpy`, OpenAI `gym`
- [Sequential Social Dilemma games](https://github.com/eugenevinitsky/sequential_social_dilemma_games) (or the LIO author's fork) for the env infrastructure
- [LOLA](https://github.com/alshedivat/lola), needed only for the IPD environment

Install:
1. Create and activate the conda env, then run `pip install -e .` from the repo root.
2. Clone and `pip install` the Sequential Social Dilemma and LOLA repos above.

## Running experiments

All runs go through `sensys_runs/run_single.py`, which fixes seeds, writes a `provenance.json` (git commit, conda env, Python / TensorFlow / numpy versions, hyperparameters), and saves logs to `lio/results/<exp_name>/<dir_name>/log.csv`.

Methods (`--method`): `lio_er`, `lio_eia_er`, `lio_ipd`, `lio_eia_ipd`, `refine_er`, `refine_eia_er`, `refine_ipd`, `refine_eia_ipd`.

### Single run

```bash
cd ~/greedy_agent
# REFiNE on ER(4,2), default incentive distortion on agent A2 (w_lever=2.0, w_door=0.2)
python sensys_runs/run_single.py \
  --method refine_eia_er --exp_name er1 --dir_name er42_refine_seed0 --seed 12340 \
  --n_agents 4 --min_at_lever 2 --w_lever 2.0 --w_door 0.2 \
  --fairness_mult 0.2 --energy_weight 1.0 --fairness_clip \
  --n_episodes 25000 --period 500 --n_eval 10
```

### Full multi-seed sweep (paper matrix)

`run_batch.py` expands a method (or `all`) over sizes ER(3,1) / ER(4,2) / ER(6,4), attack weights {(2.0, 0.2) default, (1.1, 0.9) mild, (1.5, 1.5) symmetric control}, and 10 seeds (12340 to 12349).

```bash
# inspect the planned matrix without running anything
python sensys_runs/run_batch.py --method all --dry_run

# quick smoke test of every method (tiny episode count)
python sensys_runs/run_batch.py --method all --smoke

# run the REFiNE ER sweep, resumable (skips runs whose log.csv is already complete)
python sensys_runs/run_batch.py --method refine_er --n_seeds 10 --skip_existing
```

`run_full_sweep.sh` and `launch_tmux.sh` wrap this for the full paper sweep (ER 25k episodes / period 500; IPD 20k / 1000; ER(6,4) is run serially because of memory pressure).

### Direct training scripts

The `train_*.py` scripts also run standalone (`python train_REFiNE_eia_er.py er <i>`, saving to `results/er<i>`), but their N, M, and attack settings are edited inline in `__main__`. Prefer `run_single.py` / `run_batch.py`, which set everything via flags and record provenance.

## Key configuration

Defaults live in `config_room_{lio,REFiNE}.py` (ER) and `config_ipd_{lio,REFiNE}.py` (IPD); `run_single.py` flags override them per run.

| Knob | Default | Flag | Meaning |
|---|---|---|---|
| episodes | 25000 (ER), 20000 (IPD) | `--n_episodes` | training length |
| eval period | 500 (ER), 1000 (IPD) | `--period` | checkpoint / logging interval |
| energy weight (beta) | 1.0 | `--energy_weight` | per-agent energy regularizer strength (REFiNE) |
| reward-fairness weight (B) | 0.2 | `--fairness_mult` | `-Var` reward-fairness term strength (REFiNE) |
| fairness clip | on for the paper's FULL | `--fairness_clip` | clip per-agent fairness coefficient g_i to be non-negative |
| agents N / quorum M | 4 / 2 | `--n_agents` / `--min_at_lever` | ER(N, M) configuration |
| attack weights | (2.0, 0.2) | `--w_lever` / `--w_door` | EIA incentive distortion on agent A2 |

Other fixed defaults (in the config files): discount 0.99, epsilon-greedy 0.5 to 0.1, L1 regularization, two-layer policy network with 64 and 32 hidden units, incentive network shared with LIO.

Component ablation (use `refine_er`, no EIA): `--energy_weight 0` gives the reward-fairness-only variant (B), `--fairness_mult 0` gives the energy-only variant (beta), and both nonzero gives FULL (= REFiNE). The paper's deployed REFiNE is FULL with `--fairness_clip`.

## Results and figures

Each checkpoint appends to `log.csv`. Per-agent columns include `reward_total`, `reward_env`, `n_lever`, `n_door`, `received`, `given`, `win_rate`, `total_energy`, and `teamwork_fairness`, plus `steps_per_eps`. `reward_env` (environment-only return) is kept separate from `reward_total` (incentive-inclusive) on purpose: fairness and energy track `reward_env`, because it reflects the physical workload each robot bears.

Figures and tables:
- `plot_ER42_REFiNE.ipynb` reproduces the ER(4,2) energy, reward, component-attribution, and robustness plots.
- `vis_table3.ipynb` builds the results table.

Headline numbers (per-agent median over 10 seeds, no distortion): ER(4,2) settles at 92.3 vs LIO 101.6 modeled units (-9%), and ER(3,1) improves by -30%, with task return preserved (about 45 on ER(4,2)). Under the incentive stressor, REFiNE holds its median with standard deviation at or below 1 and converges in all 10 seeds at every attack weight, while LIO's energy spread inflates and develops a heavy high-energy tail beyond 200 units (early battery depletion). The energy regularizer carries the entire saving; the reward-fairness term does not reduce inter-agent reward dispersion and destabilizes a single seed (seed 5), consistent with the paper's finding that reward fairness is the wrong control variable for per-agent energy.

## Energy model

Per-action energy is derived from measured Orin Nano component power (active motor, communication, computation, sensing). The support / lever action draws the most power, and the per-action cost is context-dependent (for example, the first agent to engage the lever pays more). LIO and REFiNE share the same model (`calculate_energy_cost`), so energy comparisons are like-for-like. Per-agent cumulative energy is reported in modeled units; see the paper, Section 4.1, for the profiling.

## Reproducibility

Every run writes `provenance.json` next to its logs, recording the git commit, conda env, Python / TensorFlow / numpy versions, the seed, and the exact hyperparameters used. Seeds are fixed (`12340 + i`), so the same method, size, weight, and seed reproduce a run.

## Acknowledgement

Built on [Learning to Incentivize Other Learning Agents (LIO)](https://github.com/011235813/lio), Yang et al., NeurIPS 2020. We thank the authors for releasing their code.

## License

MIT. See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
