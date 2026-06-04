#!/bin/bash
cd ~/greedy_agent
PY=$HOME/miniconda3/envs/LIO_tecs/bin/python
C="--exp_name er --platform nano --threads 1 --n_episodes 25000 --n_eval 10 --period 500 --w_lever 2.0 --w_door 0.2 --fairness_clip"

echo "=== [1/4] (4,2) b1.0 sanity   $(date +%H:%M:%S) ==="
$PY sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_4_2_b1.0_s0 --seed 0 --n_agents 4 --min_at_lever 2 --fairness_mult 0.2 --energy_weight 1.0 $C

echo "=== [2/4] (6,4) b0.1 probe    $(date +%H:%M:%S) ==="
$PY sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_6_4_b0.1_s0 --seed 0 --n_agents 6 --min_at_lever 4 --fairness_mult 0.2 --energy_weight 0.1 $C

echo "=== [3/4] (6,4) b0.3 probe    $(date +%H:%M:%S) ==="
$PY sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_6_4_b0.3_s0 --seed 0 --n_agents 6 --min_at_lever 4 --fairness_mult 0.2 --energy_weight 0.3 $C

echo "=== [4/4] (6,4) b1.0 control  $(date +%H:%M:%S) ==="
$PY sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_6_4_b1.0_s0 --seed 0 --n_agents 6 --min_at_lever 4 --fairness_mult 0.2 --energy_weight 1.0 $C

echo "=== ALL DONE  $(date +%H:%M:%S) ==="
