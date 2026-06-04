#!/usr/bin/env bash
# REFiNE SenSys full sweep -- light runs 3-way parallel, (6,4) serial (OOM), per-run logs, ntfy start/error/done.
# FULL method = clipped amplifier B=0.2 + energy beta=1.0. ER 25k ep / period 500 / 10 seeds; IPD 20k / 1000.
set -u
cd ~/greedy_agent
PY="$HOME/miniconda3/envs/LIO_tecs/bin/python"
export NTFY_TOPIC="${NTFY_TOPIC:-refine-marl-ucr-99a1x}"
export NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
mkdir -p sensys_runs/sweep_logs
EXP=er
notify(){ curl -s -H "Title: $1" -H "Priority: ${2:-default}" -d "$3" "$NTFY_SERVER/$NTFY_TOPIC" >/dev/null 2>&1 || true; }

ER="--n_episodes 25000 --period 500 --n_eval 10"
IPD_A="--n_episodes 20000 --period 1000 --n_eval 10"
FULL="--fairness_mult 0.2 --energy_weight 1.0 --fairness_clip"
S31="--n_agents 3 --min_at_lever 1 --w_lever 2.0 --w_door 0.2"
S42="--n_agents 4 --min_at_lever 2 --w_lever 2.0 --w_door 0.2"
S64="--n_agents 6 --min_at_lever 4 --w_lever 2.0 --w_door 0.2"

run_one(){  # <dir> <method> <args...>  (skips if already complete -> resumable)
  local dir="$1" method="$2"; shift 2
  local log="sensys_runs/sweep_logs/${dir}.log"
  local exp=50; [[ "$dir" == *ipd* ]] && exp=20
  local have; have=$(ls lio/results/*/"$dir"/log.csv 2>/dev/null | head -1)
  if [ -n "$have" ] && [ "$(( $(wc -l < "$have") - 1 ))" -ge "$exp" ]; then echo "[skip] $dir (complete)"; return 0; fi
  $PY sensys_runs/run_single.py --method "$method" --exp_name "$EXP" --platform nano --threads 2 \
      "$@" --dir_name "$dir" > "$log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && notify "REFiNE sweep ERROR" high "FAILED(rc=$rc): $dir | $(tail -n2 "$log" | tr '\n' ' ')"
  return 0
}
throttle(){ while (( $(jobs -rp | wc -l) >= $1 )); do wait -n; done; }

# --- SMOKE GATE: abort before committing ~1.5 days if anything is broken ---
notify "REFiNE sweep: smoke check" min "verifying run_single"
$PY sensys_runs/run_single.py --method refine_eia_er --exp_name "$EXP" --platform nano --threads 2 \
   --n_agents 4 --min_at_lever 2 --w_lever 2.0 --w_door 0.2 --fairness_mult 0.2 --energy_weight 1.0 --fairness_clip \
   --n_episodes 200 --period 100 --n_eval 10 --seed 0 --dir_name smoke_sweep_check \
   > sensys_runs/sweep_logs/smoke_sweep_check.log 2>&1
if [ $? -ne 0 ] || ! ls lio/results/*/smoke_sweep_check/log.csv >/dev/null 2>&1; then
  notify "REFiNE sweep ABORTED" urgent "SMOKE FAILED -- not launched. See sweep_logs/smoke_sweep_check.log"
  echo "SMOKE FAILED -- aborting."; exit 1
fi
echo "smoke OK"; T0=$SECONDS
notify "REFiNE FULL SWEEP started" default "host=$(hostname). core ER -> EIA-weights -> IPD (parallel x3) -> (6,4) SERIAL."

# --- PHASE 1: CORE ER (light, 3-way parallel): full +EIA & no-EIA at (3,1),(4,2) + {beta=0}/{B=0} ablations at (4,2) ---
for S in $(seq 0 9); do
  throttle 3; run_one sweep_er_refine_eia_3_1_s$S       refine_eia_er $ER $FULL $S31 --seed $S &
  throttle 3; run_one sweep_er_refine_eia_4_2_s$S       refine_eia_er $ER $FULL $S42 --seed $S &
  throttle 3; run_one sweep_er_refine_3_1_s$S           refine_er     $ER $FULL $S31 --seed $S &
  throttle 3; run_one sweep_er_refine_4_2_s$S           refine_er     $ER $FULL $S42 --seed $S &
  throttle 3; run_one sweep_er_refine_eia_4_2_beta0_s$S refine_eia_er $ER --fairness_mult 0.2 --energy_weight 0.0 --fairness_clip $S42 --seed $S &
  throttle 3; run_one sweep_er_refine_eia_4_2_B0_s$S    refine_eia_er $ER --fairness_mult 0.0 --energy_weight 1.0 --fairness_clip $S42 --seed $S &
done
wait
notify "REFiNE sweep: PHASE 1 core done" default "elapsed $(( (SECONDS-T0)/60 ))min"

# --- PHASE 2: EIA-weight sensitivity @(4,2) via --w_lever/--w_door (baseline used er_eia_4_2_w{wh}-{wl}); (2.0,0.2) already done ---
for S in $(seq 0 9); do
  throttle 3; run_one sweep_er_refine_eia_4_2_w1.1-0.9_s$S refine_eia_er $ER $FULL --n_agents 4 --min_at_lever 2 --w_lever 1.1 --w_door 0.9 --seed $S &
  throttle 3; run_one sweep_er_refine_eia_4_2_w1.5-1.5_s$S refine_eia_er $ER $FULL --n_agents 4 --min_at_lever 2 --w_lever 1.5 --w_door 1.5 --seed $S &
done
wait
notify "REFiNE sweep: PHASE 2 EIA-weights done" default "elapsed $(( (SECONDS-T0)/60 ))min"

# --- PHASE 3: IPD limitation (light, parallel) ---
EXP=ipd
for S in $(seq 0 9); do
  throttle 3; run_one sweep_ipd_refine_eia_B0.1_s$S refine_eia_ipd $IPD_A --fairness_mult 0.1 --fairness_clip --seed $S &
done
wait
EXP=er
notify "REFiNE sweep: PHASE 3 IPD done. Starting (6,4) SERIAL (slow)" default "elapsed $(( (SECONDS-T0)/60 ))min"

# --- PHASE 4: (6,4) scalability (SERIAL, OOM-safe) ---
for S in $(seq 0 9); do
  run_one sweep_er_refine_eia_6_4_s$S refine_eia_er $ER $FULL $S64 --seed $S
  run_one sweep_er_refine_6_4_s$S     refine_er     $ER $FULL $S64 --seed $S
  notify "REFiNE sweep: (6,4) seed $S/9 done" min "elapsed $(( (SECONDS-T0)/60 ))min"
done

notify "REFiNE FULL SWEEP DONE" high "total $(( (SECONDS-T0)/60 ))min on $(hostname)."
echo "ALL DONE in $(( (SECONDS-T0)/60 )) min"
