#!/bin/bash

NUM_EXP=$1

cd ../lio/alg

# Loop to run the experiment 10 times
for i in $(seq 1 $NUM_EXP)
do
  echo "Running experiment $i..."
  LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python train_lio_ipd.py ipd "$i"
  LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python train_lio_explotitive_attack_ipd.py ipd "$i"
  LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python train_lio_defense_ipd.py ipd "$i"
  echo "Experiment $i completed."
done

echo "All experiments completed."