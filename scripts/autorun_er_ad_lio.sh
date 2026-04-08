#!/bin/bash

NUM_EXP=$1

cd ../lio/alg

# Loop to run the experiment 10 times
for i in $(seq 1 $NUM_EXP)
do
  echo "Running experiment $i..."
  TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python train_lio_er_admo.py "$i" --min_at_lever 2 --n_agents 4
  TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python train_lio_er_admo.py "$i" --min_at_lever 3 --n_agents 4
  TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python train_lio_er_admo.py "$i" --min_at_lever 1 --n_agents 2
  echo "Experiment $i completed."
done

echo "All experiments completed."