# SenSys-2027 LIO / LIO+EIA baseline orchestrator

Self-contained batch runner for frozen LIO and LIO+EIA baselines on Escape Room (ER) and IPD.
Training algorithms live in `lio/alg/`; this directory only launches them via subprocesses.

**Platform:** This repo’s NANO host uses conda env `LIO_tecs`. Real runs default to
`--platform nano`, so results land under `lio/results/nano_er{S}/` and `nano_ipd{S}/`.
Set `--platform ""` on `run_batch` to keep bare `er{S}` / `ipd{S}` names.

## Environment variables (`launch_tmux.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONDA_ENV` | `LIO_tecs` | Conda env with working TF + scipy |
| `NTFY_TOPIC` | (unset) | ntfy topic; unset = no-op |
| `NTFY_SERVER` | `https://ntfy.sh` | ntfy server |
| `PLATFORM` | `nano` | Prefix for real `exp_name` (`nano_er1`, …) |
| `MAX_PARALLEL` | `1` | Concurrent `run_single` jobs per tmux window |
| `THREADS` | `2` | Per-process OMP/BLAS/TF thread caps |
| `USE_GPU` | `0` | `1` → `--use_gpu` |
| `RUN_TIMEOUT` | (none) | Per-run subprocess timeout (seconds) |
| `RUN_PREFLIGHT` | `0` | `1` → run `--preflight` after smoke |
| `WINDOWS` | all 4 methods | Space-separated methods for tmux windows |
| `REFINE_NTFY_VERBOSE` | (unset) | `1` → per-run done ntfy |

**TensorFlow:** Active env must support TF1.x-style graphs (`tf.compat.v1` in train scripts).

## Commands

```bash
PY=/path/to/envs/LIO_tecs/bin/python

# Dry-run (260 runs; shows nano_-prefixed targets)
$PY sensys_runs/run_batch.py --method all --dry_run

# Smoke (fast wiring check)
$PY sensys_runs/run_batch.py --method all --smoke

# Preflight (real n_eval=10, ER(6,4) worst case — proves eval path)
$PY sensys_runs/run_batch.py --method all --preflight

# Real batch with resume
$PY sensys_runs/run_batch.py --method lio_er --platform nano --skip_existing --threads 2

# Full launch (smoke gate + tmux)
NTFY_TOPIC=secret CONDA_ENV=LIO_tecs PLATFORM=nano MAX_PARALLEL=1 THREADS=2 \
  WINDOWS="lio_er lio_eia_er" bash sensys_runs/launch_tmux.sh
```

## Provenance

Every run writes `lio/results/<exp_name>/<dir_name>/provenance.json` before training
(host, git branch/commit/dirty, conda env, platform, TF/numpy versions, resolved args).
`config.json` is still written by `train()` as before.

## Resume (`--skip_existing`, real mode only)

A run is **complete** when `log.csv` exists and data-row count
`(lines − header) >= n_episodes // period` (ER: 50, IPD: 20 with defaults).
Incomplete or missing runs are executed; complete runs are skipped.
Re-running an incomplete dir overwrites `log.csv` (`train()` opens it with `'w'`).

## Seed scheme

`S` in `1..20` → `seed = 12340 + S`.

## Real matrix (260 runs)

| Method | Count | `exp_name` (default platform) | `dir_name` |
|--------|------:|-------------------------------|------------|
| `lio_er` | 80 | `nano_er{S}` | `er_lio_{N}_{M}` |
| `lio_eia_er` | 140 | `nano_er{S}` | `er_eia_*` |
| `lio_ipd` | 20 | `nano_ipd{S}` | `ipd_lio` |
| `lio_eia_ipd` | 20 | `nano_ipd{S}` | `ipd_eia` |

Smoke / preflight use isolated `smoke_*` / `preflight_*` namespaces only.

## NANO load (Orin Nano-class: ~6 cores, 8GB shared RAM)

Concurrent processes ≈ `#WINDOWS × MAX_PARALLEL`; thread pressure ≈ that × `THREADS`.
Prefer two waves: ER methods first, IPD second (same launch command resumes via `--skip_existing`).
