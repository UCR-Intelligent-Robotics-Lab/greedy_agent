import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

os.chdir('/Users/zexinli/Downloads/greedy_agent')

commands = []
for n in [2, 3, 4]:
    for i in range(1, 11):
        cmd1 = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python lio/alg/train_lio_staghunt.py staghunt {i} --n_agents {n}"
        cmd2 = f"TF_USE_LEGACY_KERAS=1 LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so /Users/zexinli/miniforge3/envs/torch-gpu/bin/python lio/alg/train_lio_staghunt_admo.py {i} --n_agents {n}"
        commands.extend([cmd1, cmd2])

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"Total commands to run: {len(commands)}")
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(run_cmd, commands)
print("All done!")