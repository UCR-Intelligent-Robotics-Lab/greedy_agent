#!/bin/bash
cd /Users/zexinli/Downloads/greedy_agent/reciprocators
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so

PYTHON_BIN="/Users/zexinli/miniforge3/envs/torch-gpu/bin/python"

for i in {1..10}
do
  # ER
  echo "Running Reciprocators Baseline ER trial $i..."
  $PYTHON_BIN run_er.py --num "$i" --n_agents 2
  echo "Running Reciprocators ADMO ER trial $i..."
  $PYTHON_BIN run_er_admo.py --num "$i" --n_agents 2
  
  # IPD
  echo "Running Reciprocators Baseline IPD trial $i..."
  $PYTHON_BIN run_ipd.py --num "$i"
  echo "Running Reciprocators ADMO IPD trial $i..."
  $PYTHON_BIN run_ipd_admo.py --num "$i"

  # Stag Hunt (2, 3, 4)
  for agents in 2 3 4
  do
    echo "Running Reciprocators Baseline Staghunt($agents) trial $i..."
    $PYTHON_BIN run_staghunt.py --num "$i" --n_agents $agents
    echo "Running Reciprocators ADMO Staghunt($agents) trial $i..."
    $PYTHON_BIN run_staghunt_admo.py --num "$i" --n_agents $agents
  done
done

echo "All Reciprocators experiments completed. Generating figures..."
cd ..
$PYTHON_BIN plot_reciprocators_all.py
echo "Done!"
