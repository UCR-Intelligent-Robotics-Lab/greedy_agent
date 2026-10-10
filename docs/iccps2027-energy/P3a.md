# P3a: reproducible run_single launches

Base: `9d704b3e8a5399e2fad71262dedb0de6368755ae`. Existing-file changes are limited to
`sensys_runs/run_single.py` and `tests/test_energy_profiles.py`; all are insertions.
No new flag is added. The graph-mode re-seed is unconditional: no reproducible
legacy run_single output exists to preserve.

## Accepted findings (verbatim)

Nano, LIO_tecs, TF 2.21.0, 2026-10-10; scratch exports of 1d58b08 and 9d704b3; lio_er and refine_er at ER(4,2), preflight settings, seed 12341.

1. run_single.py set the TF seed while TF was still eager; the trainers' disable_eager_execution() left the default graph unseeded, so initializers and tf.multinomial sampling were unseeded in every run_single launch since a90467c (TF_DETERMINISTIC_OPS=1 raised "Random ops require a seed").
2. With a graph-mode re-seed and threads 1, lio_er is byte-identical across same-code runs (log.csv, config.json, checkpoint .index and .data).
3. With threads 2, lio_er checkpoints differed by float32 rounding until TF_ENABLE_ONEDNN_OPTS=0; with it they are byte-identical, with no measurable change in wall time.
4. Under both fixes, lio_er at 1d58b08 and at HEAD is byte-identical: P1-P3 leave the LIO legacy path unchanged.
5. refine_er reproduces only to float32 rounding on the same platform: same-code pairs give identical log.csv and config.json; checkpoints differ by at most 2.9e-6 absolute with identical Adam step counters; the first difference appears after the first joint reward/cost update of episode 1 (update index 11, agent_2 eta parameters). Not resolved by threads 1, PYTHONHASHSEED=0, oneDNN off, TF_DETERMINISTIC_OPS=1, or explicit inter/intra-op parallelism 1. The base-vs-HEAD refine_er difference has the same character and magnitude, and the graph node lists (15,744 nodes) are identical. This is a pre-existing property of the REFiNE training path and is outside this change set.

## Scope and validation

The launcher defaults `TF_ENABLE_ONEDNN_OPTS` to `0` before importing TensorFlow;
an explicit environment value wins. After importing the trainer, it sets the
graph-mode seed. Provenance records the actual default-graph `tf_graph_seed` and
effective environment string `onednn_opts`, not merely the requested CLI seed.

The private `room_symmetric.RandomState` remains unseeded and is used only when
`randomize=True`. Direct `python train_*.py` launches bypass these launcher fixes
and remain outside this reproducibility guarantee. Campaign runs use threads 1.
These same-platform observations are not a cross-platform determinism claim.

`tests/test_reproducibility.py` checks import/seed ordering without TensorFlow,
explicit environment precedence, and read-back provenance. TF tests import
run_batch's smoke overrides, use threads 1 and fresh temporary source/result
trees, and require exact LIO CSV/config/checkpoint bytes and REFiNE CSV/config
bytes. They also require different LIO seeds to change log.csv. Missing TF is
reported with `pytest.importorskip`; no checkpoint tolerance is introduced.
Raw diagnosis files remain outside this repository.

## Windows validation (2026-10-10)

Python 3.12.12, existing isolated pytest 9.1.1; TensorFlow is not installed.
Full local suite: **60 passed, 6 skipped, 0 failed** (2.66 seconds). All six
skips require TensorFlow: `test_tensorflow_propagation`, both parametrizations
of `test_tensorflow_final_evaluation`, `test_lio_smoke_byte_identity`,
`test_refine_smoke_log_and_config_identity`, and
`test_lio_different_seeds_change_log`. TF integration is not validated locally.

With explicit authorization, the existing P1 `test_launcher_dispatch_and_provenance`
test double now exposes `compat.v1.set_random_seed` and `get_default_graph`.
It preserves its top-level `set_random_seed` and `__version__`. Added assertions
verify an initially unseeded graph, exactly one graph-seed call with the CLI seed
after trainer import, actual graph-seed provenance, and the default oneDNN value.
All existing assertions are retained unchanged. No production change was made
while extending this test double.

Numstat against `9d704b3e8a5399e2fad71262dedb0de6368755ae`:
`sensys_runs/run_single.py`: 8 additions, 0 deletions;
`tests/test_energy_profiles.py`: 25 additions, 0 deletions.
`git diff --check` passes. No commit or tag was created.
