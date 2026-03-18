#!/bin/bash

NUM_EXP=$1

cd ../lio/alg

for i in $(seq 1 $NUM_EXP)
do
  echo "Running experiment $i..."
  LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so python lio/train_lio_staghunt.py staghunt "$i"
  echo "Experiment $i completed."
done

echo "All experiments completed."
