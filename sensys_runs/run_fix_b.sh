#!/bin/bash
cd ~/greedy_agent
/home/yifan/miniconda3/envs/LIO_tecs/bin/python sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_6_4_b0.1_s0 --seed 0 --n_agents 6 --min_at_lever 4 --fairness_mult 0.2 --energy_weight 0.1 --exp_name er --platform nano --threads 1 --n_episodes 25000 --n_eval 10 --period 500 --w_lever 2.0 --w_door 0.2 --fairness_clip
/home/yifan/miniconda3/envs/LIO_tecs/bin/python sensys_runs/run_single.py --method refine_eia_er --dir_name fix_er_refine_eia_6_4_b1.0_s0 --seed 0 --n_agents 6 --min_at_lever 4 --fairness_mult 0.2 --energy_weight 1.0 --exp_name er --platform nano --threads 1 --n_episodes 25000 --n_eval 10 --period 500 --w_lever 2.0 --w_door 0.2 --fairness_clip
echo "=== fixB done: (6,4) b0.1, b1.0 ==="
