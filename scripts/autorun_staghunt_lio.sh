#!/bin/bash

NUM_EXP=$1

cd ../lio/alg

for i in $(seq 1 $NUM_EXP)
do
  echo "Running experiment $i..."
  TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python train_lio_staghunt.py staghunt "$i"
  echo "Experiment $i completed."
done

echo "All experiments completed."
