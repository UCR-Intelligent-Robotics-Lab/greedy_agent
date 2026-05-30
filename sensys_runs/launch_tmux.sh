#!/usr/bin/env bash
# Human-facing launcher: smoke gate (+ optional preflight), then tmux per method.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-LIO_tecs}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
PLATFORM="${PLATFORM:-nano}"
N_SEEDS="${N_SEEDS:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
THREADS="${THREADS:-2}"
USE_GPU="${USE_GPU:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
WINDOWS="${WINDOWS:-lio_er lio_eia_er lio_ipd lio_eia_ipd}"
SESSION="${SESSION:-sensys}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY="${PY:-$HOME/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: interpreter not found/executable at ${PY} (set CONDA_ENV or PY)" >&2
  exit 1
fi

export NTFY_TOPIC NTFY_SERVER

echo "REPO_ROOT=${REPO_ROOT}"
echo "PY=${PY}"
echo "SESSION=${SESSION}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "NTFY_TOPIC=${NTFY_TOPIC:-<not set>}"
echo "NTFY_SERVER=${NTFY_SERVER}"
echo "PLATFORM=${PLATFORM}"
echo "N_SEEDS=${N_SEEDS}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"
echo "THREADS=${THREADS}"
echo "USE_GPU=${USE_GPU}"
echo "RUN_TIMEOUT=${RUN_TIMEOUT:-<none>}"
echo "RUN_PREFLIGHT=${RUN_PREFLIGHT}"
echo "WINDOWS=${WINDOWS}"

_gpu_flag() {
  if [[ "${USE_GPU}" == "1" ]]; then
    echo "--use_gpu"
  fi
}

_timeout_flag() {
  if [[ -n "${RUN_TIMEOUT}" ]]; then
    echo "--run_timeout ${RUN_TIMEOUT}"
  fi
}

_window_exists() {
  local name="$1"
  tmux list-windows -t "${SESSION}" -F '#{window_name}' 2>/dev/null | grep -qx "${name}"
}

cd "${REPO_ROOT}"

echo ""
echo "=== Smoke gate (all methods) ==="
if ! NTFY_TOPIC="${NTFY_TOPIC}" NTFY_SERVER="${NTFY_SERVER}" \
     "${PY}" sensys_runs/run_batch.py --method all --smoke --threads "${THREADS}"; then
  NTFY_TOPIC="${NTFY_TOPIC}" NTFY_SERVER="${NTFY_SERVER}" \
    "${PY}" -c "import sys; sys.path.insert(0,'sensys_runs'); from ntfy_notify import notify; notify('SMOKE FAILED', title='REFiNE ERROR', priority='high', tags='rotating_light')" 2>/dev/null || true
  echo "SMOKE FAILED — not launching real batch"
  exit 1
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  echo ""
  echo "=== Preflight (real n_eval, ER 6,4 worst case) ==="
  if ! NTFY_TOPIC="${NTFY_TOPIC}" NTFY_SERVER="${NTFY_SERVER}" \
       "${PY}" sensys_runs/run_batch.py --method all --preflight --threads "${THREADS}"; then
    NTFY_TOPIC="${NTFY_TOPIC}" NTFY_SERVER="${NTFY_SERVER}" \
      "${PY}" -c "import sys; sys.path.insert(0,'sensys_runs'); from ntfy_notify import notify; notify('PREFLIGHT FAILED', title='REFiNE ERROR', priority='high', tags='rotating_light')" 2>/dev/null || true
    echo "PREFLIGHT FAILED — not launching real batch"
    exit 1
  fi
fi

GPU_FLAG="$(_gpu_flag)"
TIMEOUT_FLAG="$(_timeout_flag)"

read -r -a WINDOW_ARR <<< "${WINDOWS}"
N_WINDOWS=${#WINDOW_ARR[@]}
TOTAL_PROCS=$(( N_WINDOWS * MAX_PARALLEL ))
TOTAL_THREADS=$(( TOTAL_PROCS * THREADS ))

echo ""
echo "=== NANO LOAD WARNING (Jetson Orin Nano-class: ~6 cores, 8GB CPU+GPU RAM) ==="
echo "  Concurrent training processes = #windows (${N_WINDOWS}) × MAX_PARALLEL (${MAX_PARALLEL}) = ${TOTAL_PROCS}"
echo "  Approx thread pressure       ≈ ${TOTAL_PROCS} × THREADS (${THREADS}) = ${TOTAL_THREADS}"
echo "  4 windows × MAX_PARALLEL>1 can OOM-kill on 8GB shared memory. RECOMMENDED two-wave launch:"
echo "    WINDOWS=\"lio_er lio_eia_er\" bash sensys_runs/launch_tmux.sh"
echo "    WINDOWS=\"lio_ipd lio_eia_ipd\" bash sensys_runs/launch_tmux.sh   # after ER or anytime (resume)"
echo "  Keep THREADS × #windows ≤ core count; USE_GPU=0 for these tiny MLPs unless overridden."

_run_cmd() {
  local method="$1"
  cat <<EOF
cd "${REPO_ROOT}" && NTFY_TOPIC="${NTFY_TOPIC}" NTFY_SERVER="${NTFY_SERVER}" "${PY}" sensys_runs/run_batch.py --method ${method} --platform "${PLATFORM}" --n_seeds ${N_SEEDS} --max_parallel ${MAX_PARALLEL} --threads ${THREADS} --skip_existing ${GPU_FLAG} ${TIMEOUT_FLAG}; exec bash
EOF
}

ADDED=0
SKIPPED=0

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo ""
  echo "=== Session ${SESSION} exists — adding missing windows only ==="
  for method in "${WINDOW_ARR[@]}"; do
    if _window_exists "${method}"; then
      echo "  skip (exists): ${method}"
      SKIPPED=$((SKIPPED + 1))
    else
      tmux new-window -t "${SESSION}" -n "${method}" \
        bash -c "$(_run_cmd "${method}")"
      echo "  added: ${method}"
      ADDED=$((ADDED + 1))
    fi
  done
else
  echo ""
  echo "=== Creating session ${SESSION} ==="
  FIRST=1
  for method in "${WINDOW_ARR[@]}"; do
    if [[ "${FIRST}" -eq 1 ]]; then
      tmux new-session -d -s "${SESSION}" -n "${method}" \
        bash -c "$(_run_cmd "${method}")"
      echo "  created: ${method}"
      ADDED=$((ADDED + 1))
      FIRST=0
    else
      tmux new-window -t "${SESSION}" -n "${method}" \
        bash -c "$(_run_cmd "${method}")"
      echo "  added: ${method}"
      ADDED=$((ADDED + 1))
    fi
  done
fi

echo ""
echo "=== tmux session ${SESSION} ==="
echo "  windows added: ${ADDED}, skipped (already present): ${SKIPPED}"
echo "  attach:  tmux attach -t ${SESSION}"
echo "  logs:    ${REPO_ROOT}/sensys_runs/run_logs/"
echo "  results: ${REPO_ROOT}/lio/results/${PLATFORM}_er*/ and ${PLATFORM}_ipd*/"
echo ""
echo "Re-run the same launch command after interruption; --skip_existing resumes incomplete runs."
echo "(train() truncates log.csv on re-run, so incomplete dirs are overwritten cleanly.)"
