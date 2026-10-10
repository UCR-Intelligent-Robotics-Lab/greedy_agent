# P3b: static manifest launcher

Execute the reviewed P3 manifest with fixed, seed-major round-robin shards.
There is no scheduler, queue, resume, retry, or training change. Within each
seed, arm order comes from `energy_profile_campaign.ARMS`. Indices are
zero-based: shard i receives ordered runs i, i+K, i+2K, and so on.

The runner executes recorded argv unchanged, without a shell. Children inherit
the environment with only `PYTHONHASHSEED=0` added; consoles record the addition.
It does not set `TF_ENABLE_ONEDNN_OPTS`: P3a defaults it to 0 and respects explicit
overrides. All commands must use threads 1. Same-platform reruns reproduced
evaluation logs and configs exactly in every validated case; weights reproduce
to float32 rounding.

## Gates and audit outputs

Before execution (including dry-run), require the manifest commit to equal HEAD,
porcelain-empty git status including untracked files, a git-ignored results
root, NTFY_TOPIC, absent targets, and threads 1. The runner also validates
argv/target/arm consistency. Refusals return 2. Before each run, git-status and
target-absence checks repeat; a failure is recorded and the shard continues.
Any failed run makes the final shard exit code 1.

Logs and CSVs default to `~/iccps_runs/<campaign>_console/`, outside the repo
and results. The launcher requires this directory to be absent. Each shard
exclusively creates `shard-i-of-K.log` and `shard-i-of-K.csv`. `TIMING_FIELDS`
is the single registered CSV header. Wall time uses a monotonic clock and peak
RSS uses Linux `os.wait4` (KiB converted to MiB). Child stdout/stderr go into
the log; progress also appears in tmux. ntfy receives one message per run and
one summary, with high priority on failure. Direct-runner `--timeout SECONDS`
kills/reaps an overdue child, records rc=124, and continues; default: no timeout.

One detached tmux window runs each shard. A unique startup token provides a
one-time barrier: all shards validate all targets before any shard trains.
Ready files live in the external console directory. A shard that cannot see
every shard ready within 60 seconds refuses with exit 2 and a high-priority
notification. This is not a scheduler. Old tokens cannot release a new launch.
Do not manually use internal `--start-together` or edit a manifest during a run.

## Nano launch block (Yifan only)

These explicit tmux and NTFY_TOPIC requirements replace the generic nohup and
NTFY_TOPIC_URL example in skill_code.md Section 5.2. Close the Cursor remote
window during formal runs to release about 1 GB. K=1 is the conservative initial
choice; K x threads must not exceed nproc. CPU count is not a RAM-capacity check.
Run smoke first. Every failed command or gate means STOP.

```bash
cd ~/greedy_agent
git fetch --tags
tag=$(git describe --tags --exact-match) || { echo 'STOP: no exact tag'; exit 1; }
printf '%s\n' "$tag"
test "$tag" = iccps2027-energy-p3b || { echo 'STOP: wrong tag'; exit 1; }
status=$(git status --porcelain) || exit 1
printf '%s' "$status"
test -z "$status" || { echo 'STOP: dirty repository'; exit 1; }
export NTFY_TOPIC=refine-marl-ucr-99a1x NTFY_SERVER=https://ntfy.sh
mkdir -p ~/iccps_runs
campaign=iccps2027_energy_smokecheck
manifest="$HOME/iccps_runs/${campaign}_manifest.json"
~/miniconda3/envs/LIO_tecs/bin/python -B sensys_runs/energy_profile_campaign.py --campaign "$campaign" --platform nano --threads 1 --smoke --out "$manifest"
bash sensys_runs/energy_profile_launch.sh "$manifest" 1 iccps_energy_smoke --dry-run
bash sensys_runs/energy_profile_launch.sh "$manifest" 1 iccps_energy_smoke
tmux attach -t iccps_energy_smoke
```

After reviewing smoke results, Yifan may use the formal block. No assistant
tool launches the formal campaign.

```bash
cd ~/greedy_agent
git fetch --tags
tag=$(git describe --tags --exact-match) || { echo 'STOP: no exact tag'; exit 1; }
printf '%s\n' "$tag"
test "$tag" = iccps2027-energy-p3b || { echo 'STOP: wrong tag'; exit 1; }
status=$(git status --porcelain) || exit 1
printf '%s' "$status"
test -z "$status" || { echo 'STOP: dirty repository'; exit 1; }
export NTFY_TOPIC=refine-marl-ucr-99a1x NTFY_SERVER=https://ntfy.sh
mkdir -p ~/iccps_runs
campaign=iccps2027_energy_v1
manifest="$HOME/iccps_runs/${campaign}_manifest.json"
~/miniconda3/envs/LIO_tecs/bin/python -B sensys_runs/energy_profile_campaign.py --campaign "$campaign" --platform nano --threads 1 --out "$manifest"
bash sensys_runs/energy_profile_launch.sh "$manifest" 1 iccps_energy --dry-run
bash sensys_runs/energy_profile_launch.sh "$manifest" 1 iccps_energy
tmux attach -t iccps_energy
```

The launcher prints attach and tail commands and returns immediately. Dry-run
prints only tmux commands and creates nothing; a real launch prints `free -h`.
For an individual portable plan, use the confirmed interpreter with
`-B sensys_runs/energy_profile_runner.py MANIFEST --shard 0 --of K --dry-run`.
Actual execution requires Linux wait4. Existing outputs are never reused.

## Windows validation

Tests use real fake-training subprocesses and injected git responses without
TensorFlow or network notifications. Windows simulates the unavailable wait4
API; native Linux RSS measurement and real tmux launches remain target-platform
checks. Existing production files are unchanged.

Local full suite: `81 passed, 6 skipped in 4.52s` (60 existing plus 21 new
tests). The six skips require unavailable TensorFlow: two evaluator tests,
one launcher/provenance test, and the three reproducibility tests. Command:
`python -B -m pytest tests -q -rs -p no:cacheprovider`, with the existing local
test dependencies on PYTHONPATH and PYTHONDONTWRITEBYTECODE=1. Git Bash syntax
validation (`bash -n sensys_runs/energy_profile_launch.sh`) passed. No Nano
command or campaign was run; no existing tracked file was edited.
