#!/bin/bash

NUM_EXP=$1

cd ../lio/alg

# Loop to run the experiment 10 times
for i in $(seq 1 $NUM_EXP)
do
  echo "Running experiment $i..."
  TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python train_lio_er.py er "$i"
  TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python train_lio_explotitive_attack_er.py er "$i"
  TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python train_lio_defense_er.py er "$i"
  echo "Experiment $i completed."
done

echo "All experiments completed."