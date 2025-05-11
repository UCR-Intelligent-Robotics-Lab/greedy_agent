#!/usr/bin/env python3
import subprocess, os

algs = [
    "train_lio_er.py",
    "train_lio_eia_er.py",
    "train_REFiNE_eia_er.py"
]

os.makedirs("logs", exist_ok=True)

for i in range(1, 21):
    for script in algs:
        name = script.rsplit(".",1)[0]
        print(f"Running {name} ER {i} …")
        logfile = f"logs/{name}_er{i}.log"
        with open(logfile, "w") as f:
            subprocess.run(
                ["python", script, "er", str(i)],
                stdout=f, stderr=subprocess.STDOUT,
                check=True
            )
        print(f"  → done, log → {logfile}")
print("All experiments complete!")
