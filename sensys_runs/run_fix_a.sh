#!/bin/bash
cd ~/greedy_agent
/home/yifan/miniconda3/envs/LIO_tecs/bin/python sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_4_2_b1.0_s0 --seed 0 --n_agents 4 --min_at_lever 2 --fairness_mult 0.2 --energy_weight 1.0 --exp_name er --platform nano --threads 1 --n_episodes 25000 --n_eval 10 --period 500 --w_lever 2.0 --w_door 0.2 --fairness_clip
echo "=== fixA done: (4,2) b1.0 sanity ==="
