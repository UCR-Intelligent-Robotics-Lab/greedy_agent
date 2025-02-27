#!/bin/bash

NUM_EXP=$1

cd ../lio/alg

# Loop to run the experiment 10 times
for i in $(seq 1 $NUM_EXP)
do
  echo "Running experiment $i..."
  python train_lio.py er "$i"
  echo "Experiment $i completed."
done

echo "All experiments completed."