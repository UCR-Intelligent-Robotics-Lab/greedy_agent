#!/usr/bin/env python3
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Force working directory to the project root
project_root = '/Users/zexinli/Downloads/greedy_agent'
os.chdir(project_root)

# Prepare all 60 commands (3 agent counts * 10 trials * 2 modes)
commands = []
for n in [2, 3, 4]:
    for i in range(1, 11):
        # Baseline LIO command
        cmd1 = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python lio/alg/train_lio_staghunt.py staghunt {i} --n_agents {n}"
        # ADMO Attack command
        cmd2 = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python lio/alg/train_lio_staghunt_admo.py {i} --n_agents {n}"
        commands.extend([cmd1, cmd2])

def run_cmd(cmd):
    print(f"Starting: {cmd}")
    # We suppress standard output to avoid mangling the console, but capture standard error if something goes wrong
    result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ERROR in command: {cmd}\n{result.stderr}")
    else:
        print(f"Finished: {cmd}")

print(f"Total commands to run in parallel: {len(commands)}")

# Use 8 workers for parallel execution. This is a safe sweet spot for TF/CPU overhead on the Mac.
# Adjust max_workers down if you experience memory crashes.
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(run_cmd, commands)

print("\nAll 60 training trials have completed successfully.")
print("Generating the final 2x3 Stag Hunt plot...")

plot_cmd = "/Users/zexinli/miniforge3/envs/torch-gpu/bin/python plot_staghunt_fig6.py"
plot_result = subprocess.run(plot_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if plot_result.returncode == 0:
    print("Plot successfully generated! Saved to 'fig6_staghunt_replication.png'.")
else:
    print(f"Error generating plot:\n{plot_result.stderr}")

print("Done!")
