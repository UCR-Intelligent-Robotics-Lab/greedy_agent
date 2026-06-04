#!/bin/bash
cd ~/greedy_agent
PY=$HOME/miniconda3/envs/LIO_tecs/bin/python
TOPIC="https://ntfy.sh/refine-marl-ucr-99a1x"
C="--exp_name er --platform nano --threads 1 --n_episodes 25000 --n_eval 10 --period 500 --w_lever 2.0 --w_door 0.2 --fairness_clip"
curl -s -d "(6,4) b0.1 n=10 run STARTED (seeds 5-9) $(date +%H:%M)" "$TOPIC" >/dev/null
for S in 5 6 7 8 9; do
  echo "=== (6,4) b0.1 seed $S  $(date +%H:%M:%S) ==="
  $PY sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_6_4_b0.1_s$S --seed $S --n_agents 6 --min_at_lever 4 --fairness_mult 0.2 --energy_weight 0.1 $C
  curl -s -d "(6,4) b0.1 seed $S done $(date +%H:%M)" "$TOPIC" >/dev/null
done
RESULT=$($PY sensys_runs/analyze_quick.py 2>&1)
curl -s -H "Title: REFiNE (6,4) b0.1 n=10 DONE" -d "$RESULT" "$TOPIC" >/dev/null
echo "=== ALL DONE $(date +%H:%M:%S) ==="
