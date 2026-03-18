#!/bin/bash

NUM_EXP=$1

cd ../reciprocators

for i in $(seq 1 $NUM_EXP)
do
  echo "Running experiment $i..."
  LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python run_staghunt.py
  echo "Experiment $i completed."
done

echo "All experiments completed."
