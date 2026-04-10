#!/bin/bash
cd /Users/zexinli/Downloads/greedy_agent

for agents in 2 3 4
do
  for i in {1..10}
  do
    echo "Running baseline Staghunt($agents) trial $i..."
    TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python lio/alg/train_lio_staghunt.py staghunt "$i" --n_agents $agents
    
    echo "Running ADMO Staghunt($agents) trial $i..."
    TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python lio/alg/train_lio_staghunt_admo.py "$i" --n_agents $agents
  done
done

echo "All experiments completed. Generating figure..."
/Users/zexinli/miniforge3/envs/torch-gpu/bin/python plot_staghunt_fig6.py
echo "Done!"
