#!/usr/bin/env bash
# run_noeia_ablation.sh
# ---------------------------------------------------------------------------
# Launch the MISSING *no-EIA* REFiNE ablations for ER(4,2):
#     energy_only    : --energy_weight <beta_full> --fairness_mult 0
#     fairness_only  : --energy_weight 0           --fairness_mult <B_full>
# matched to the FULL config that `sweep_er_refine_4_2_s*` was trained with
# (auto-read from that run's provenance.json so beta/B/fairness_clip cannot drift).
#
# FULL (beta_full, B_full) already exists as sweep_er_refine_4_2_s0-9, so it is
# NOT re-run. energy_only/fairness_only s0,s1 already exist as probe_* and are
# skipped if complete; only s2-s9 are launched (16 new runs by default).
#
# Safe on the ~8GB Jetson box: a single controller maintains a bounded pool of
# background run_single.py jobs with a free-RAM guard, instead of many tmux
# windows that can OOM. Run it INSIDE one tmux window for persistence:
#     cd ~/greedy_agent && tmux new -s abl
#     MAX_PARALLEL=2 bash sensys_runs/run_noeia_ablation.sh
#     # detach: Ctrl-b d   reattach: tmux attach -t abl
#
# Preview without launching:   DRY_RUN=1 bash sensys_runs/run_noeia_ablation.sh
# ---------------------------------------------------------------------------
set -uo pipefail

# -------------------------- config (override via env) ----------------------
CONDA_ENV="${CONDA_ENV:-LIO_tecs}"
PY="${PY:-$HOME/miniconda3/envs/${CONDA_ENV}/bin/python}"
REPO_ROOT="${REPO_ROOT:-$HOME/greedy_agent}"
EXP_NAME="${EXP_NAME:-er}"                 # parent dir under lio/results/
N_AGENTS="${N_AGENTS:-4}"
MIN_AT_LEVER="${MIN_AT_LEVER:-2}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"      # full matched set; existing s0,s1 are skipped if complete
MAX_PARALLEL="${MAX_PARALLEL:-2}"          # concurrent runs (2 is safe on 8GB w/ Cursor closed; try 3 cautiously)
THREADS="${THREADS:-2}"                    # OMP/BLAS threads per run (keep MAX_PARALLEL*THREADS <= cores)
MIN_FREE_MB="${MIN_FREE_MB:-1500}"         # do not start a new run if MemAvailable below this
MIN_COMPLETE_ROWS="${MIN_COMPLETE_ROWS:-50}"   # log.csv line count to treat a run as already complete
STAGGER_S="${STAGGER_S:-8}"                # delay between launches so TF graph init doesn't spike RAM together
REF_FULL_DIR="${REF_FULL_DIR:-sweep_er_refine_4_2_s0}"  # an existing FULL run to read config from
NTFY_TOPIC="${NTFY_TOPIC:-refine-marl-ucr-99a1x}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
LOGDIR="${LOGDIR:-$REPO_ROOT/sensys_runs/abl_logs}"
DRY_RUN="${DRY_RUN:-0}"
# Optional hard overrides (skip provenance) -- leave empty to auto-read:
BETA_OVERRIDE="${BETA_FULL:-}"
B_OVERRIDE="${B_FULL:-}"
CLIP_OVERRIDE="${FAIRNESS_CLIP:-}"
# ---------------------------------------------------------------------------

RESULTS="$REPO_ROOT/lio/results"
mkdir -p "$LOGDIR"

notify() {  # $1=body $2=title $3=tags
  [ -n "$NTFY_TOPIC" ] || return 0
  curl -s -H "Title: ${2:-REFiNE ablation}" -H "Tags: ${3:-information_source}" \
       -d "$1" "$NTFY_SERVER/$NTFY_TOPIC" >/dev/null 2>&1 || true
}
free_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
n_live()  { jobs -rp | wc -l | tr -d ' '; }
is_complete() {  # $1 = full run dir
  local f="$1/log.csv"
  [ -f "$f" ] && [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -ge "$MIN_COMPLETE_ROWS" ]
}

# ---- sanity checks ----
[ -x "$PY" ] || { echo "ERROR: python not found/executable: $PY" >&2; exit 1; }
[ -f "$REPO_ROOT/sensys_runs/run_single.py" ] || { echo "ERROR: $REPO_ROOT/sensys_runs/run_single.py missing" >&2; exit 1; }

# ---- resolve FULL config (provenance, then overrides, then config defaults) ----
PROV="$RESULTS/$EXP_NAME/$REF_FULL_DIR/provenance.json"
PBETA=""; PB=""; PCLIP=""
if [ -f "$PROV" ]; then
  read -r PBETA PB PCLIP < <("$PY" - "$PROV" <<'PYEOF'
import json, sys
try:
    p = json.load(open(sys.argv[1]))
    bw = p.get("energy_weight"); bm = p.get("fairness_mult"); cl = p.get("fairness_clip", False)
    print(bw if bw is not None else "", bm if bm is not None else "", "1" if cl else "0")
except Exception:
    print("", "", "")
PYEOF
)
  echo "Read FULL config from provenance: $PROV"
else
  echo "WARN: $PROV not found; will fall back to config defaults (1.0 / 0.2)."
fi

BETA_FULL="${BETA_OVERRIDE:-${PBETA:-1.0}}"
B_FULL="${B_OVERRIDE:-${PB:-0.2}}"
CLIP="${CLIP_OVERRIDE:-${PCLIP:-0}}"
CLIP_FLAG=""; [ "$CLIP" = "1" ] && CLIP_FLAG="--fairness_clip"

# dir-name templates (match existing probe_* naming so s0,s1 are reused)
EN_DIR() { echo "probe_er_${N_AGENTS}_${MIN_AT_LEVER}_B0_beta${BETA_FULL}_s$1"; }
FA_DIR() { echo "probe_er_${N_AGENTS}_${MIN_AT_LEVER}_B${B_FULL}_beta0_s$1"; }

echo "=================================================================="
echo " no-EIA ablation plan  (ER(${N_AGENTS},${MIN_AT_LEVER}))"
echo "   FULL ref dir : $REF_FULL_DIR  ->  beta_full=$BETA_FULL  B_full=$B_FULL  fairness_clip=$CLIP"
echo "   energy_only  : --energy_weight $BETA_FULL --fairness_mult 0   dir=$(EN_DIR '<s>')"
echo "   fairness_only: --energy_weight 0 --fairness_mult $B_FULL      dir=$(FA_DIR '<s>')"
echo "   method=refine_er (NO EIA)   seeds=[$SEEDS]"
echo "   MAX_PARALLEL=$MAX_PARALLEL  THREADS=$THREADS  MIN_FREE_MB=$MIN_FREE_MB  DRY_RUN=$DRY_RUN"
echo "   per-run logs -> $LOGDIR/<dir>.out ;  results -> $RESULTS/$EXP_NAME/"
echo "=================================================================="

# ---- build queue: (kind, seed) interleaved ----
QUEUE=()
for s in $SEEDS; do QUEUE+=("EN:$s" "FA:$s"); done

launch_one() {  # $1=dir ; rest = run_single args
  local dir="$1"; shift
  local out="$LOGDIR/${dir}.out"
  echo "[$(date +%H:%M:%S)] LAUNCH $dir   (free=$(free_mb)MB live=$(n_live))"
  if [ "$DRY_RUN" = "1" ]; then
    echo "    DRY: $PY sensys_runs/run_single.py --method refine_er --exp_name $EXP_NAME --dir_name $dir --n_agents $N_AGENTS --min_at_lever $MIN_AT_LEVER --threads $THREADS $CLIP_FLAG $*"
    return 0
  fi
  (
    cd "$REPO_ROOT" && \
    NTFY_TOPIC="$NTFY_TOPIC" NTFY_SERVER="$NTFY_SERVER" \
    "$PY" sensys_runs/run_single.py \
      --method refine_er --exp_name "$EXP_NAME" --dir_name "$dir" \
      --n_agents "$N_AGENTS" --min_at_lever "$MIN_AT_LEVER" \
      --threads "$THREADS" $CLIP_FLAG "$@" > "$out" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE  $dir" \
    || { echo "[$(date +%H:%M:%S)] FAIL  $dir (see $out)"; notify "FAIL $dir" "REFiNE ablation" "rotating_light"; }
  ) &
}

TOTAL=${#QUEUE[@]}; idx=0; started=0; skipped=0
notify "no-EIA ablation start: up to $TOTAL jobs (MAX_PARALLEL=$MAX_PARALLEL)" "REFiNE ablation" "rocket"

for item in "${QUEUE[@]}"; do
  kind="${item%%:*}"; s="${item##*:}"; idx=$((idx+1))
  if [ "$kind" = "EN" ]; then
    dir="$(EN_DIR "$s")"; eargs=(--seed "$s" --energy_weight "$BETA_FULL" --fairness_mult 0)
  else
    dir="$(FA_DIR "$s")"; eargs=(--seed "$s" --energy_weight 0 --fairness_mult "$B_FULL")
  fi

  if is_complete "$RESULTS/$EXP_NAME/$dir"; then
    echo "[skip $idx/$TOTAL] complete: $dir"; skipped=$((skipped+1)); continue
  fi

  if [ "$DRY_RUN" != "1" ]; then
    # wait for a free slot, then for enough RAM
    while [ "$(n_live)" -ge "$MAX_PARALLEL" ]; do sleep 15; done
    while [ "$(free_mb)" -lt "$MIN_FREE_MB" ]; do
      echo "[$(date +%H:%M:%S)] RAM low ($(free_mb)MB < ${MIN_FREE_MB}MB) — waiting for a run to finish..."
      sleep 30
    done
  fi

  launch_one "$dir" "${eargs[@]}"
  started=$((started+1))
  [ "$DRY_RUN" = "1" ] || sleep "$STAGGER_S"
done

wait
notify "no-EIA ablation done: launched=$started skipped=$skipped total=$TOTAL" "REFiNE ablation" "white_check_mark"
echo "=================================================================="
echo "ALL DONE  launched=$started  skipped(complete)=$skipped  total=$TOTAL"
echo "Per-run logs: $LOGDIR/   |   Results: $RESULTS/$EXP_NAME/"
echo "Next: re-run comprehensive_ablation_stats.py — energy_only/fairness_only now have s0-9 (no-EIA),"
echo "      and FULL = refine_default (sweep_er_refine_4_2). Add contrast (refine_default, fairness_only(...))."
echo "=================================================================="