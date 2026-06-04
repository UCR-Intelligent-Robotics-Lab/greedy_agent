#!/bin/bash
cd ~/greedy_agent
PY=$HOME/miniconda3/envs/LIO_tecs/bin/python
C="--exp_name er --platform nano --threads 1 --n_episodes 25000 --n_eval 10 --period 500 --w_lever 2.0 --w_door 0.2 --fairness_clip"
for S in 1 2 3 4; do
  echo "=== (6,4) b0.1 seed $S  $(date +%H:%M:%S) ==="
  $PY sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_6_4_b0.1_s$S --seed $S --n_agents 6 --min_at_lever 4 --fairness_mult 0.2 --energy_weight 0.1 $C
done
echo "=== DONE  $(date +%H:%M:%S) ==="
