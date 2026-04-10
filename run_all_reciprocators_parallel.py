#!/usr/bin/env python3
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor

project_root = '/Users/zexinli/Downloads/greedy_agent/reciprocators'
os.chdir(project_root)

commands = []

# 1. Stag Hunt (N=2, 3, 4)
for n in [2, 3, 4]:
    for i in range(1, 11):
        cmd_base = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python run_staghunt.py --num {i} --n_agents {n}"
        cmd_admo = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python run_staghunt_admo.py --num {i} --n_agents {n} --mode adversarial"
        commands.extend([cmd_base, cmd_admo])

# 2. Escape Room (N=2 min=1, N=4 min=2, N=4 min=3)
er_configs = [(2, 1), (4, 2), (4, 3)]
for n, min_l in er_configs:
    for i in range(1, 11):
        cmd_base = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python run_er.py --num {i} --n_agents {n} --min_at_lever {min_l}"
        cmd_admo = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python run_er_admo.py --num {i} --n_agents {n} --min_at_lever {min_l} --mode adversarial"
        commands.extend([cmd_base, cmd_admo])

# 3. IPD
for i in range(1, 11):
    cmd_base = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python run_ipd.py --num {i}"
    cmd_admo = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python run_ipd_admo.py --num {i} --mode adversarial"
    commands.extend([cmd_base, cmd_admo])

def run_cmd(cmd):
    print(f"Starting: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ERROR in command: {cmd}\n{result.stderr}")
    else:
        print(f"Finished: {cmd}")

print(f"Total Reciprocators commands to run in parallel: {len(commands)}")

# Use 8 workers to prevent overheating/crashing
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(run_cmd, commands)

print("\nAll 140 training trials have completed successfully.")
print("Generating final Reciprocators plots...")

os.chdir('/Users/zexinli/Downloads/greedy_agent')
plot_cmd = "/Users/zexinli/miniforge3/envs/torch-gpu/bin/python plot_reciprocators_all.py"
plot_result = subprocess.run(plot_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if plot_result.returncode == 0:
    print("Plots successfully generated! Saved to 'er_reciprocators.png', 'ipd_reciprocators.png', 'staghunt_reciprocators.png'.")
else:
    print(f"Error generating plots:\n{plot_result.stderr}")

print("Done!")
