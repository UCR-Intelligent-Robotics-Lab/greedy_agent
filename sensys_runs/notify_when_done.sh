#!/bin/bash
TOPIC="https://ntfy.sh/refine-marl-ucr-99a1x"
# 等 'seeds' session 结束(run_64seeds.sh 跑完后该 session 会自动关闭)
while tmux has-session -t seeds 2>/dev/null; do sleep 30; done
sleep 5
cd ~/greedy_agent
RESULT=$($HOME/miniconda3/envs/LIO_tecs/bin/python sensys_runs/analyze_quick.py 2>&1)
curl -s -H "Title: REFiNE (6,4) b0.1 seeds DONE" -d "$RESULT" "$TOPIC"
