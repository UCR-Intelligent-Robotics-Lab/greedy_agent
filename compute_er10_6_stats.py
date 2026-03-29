import pandas as pd
import numpy as np
import os

BASE = os.path.join(os.path.dirname(__file__), "lio", "results")
N_AGENTS = 10
N_RUNS = 11  # er1 .. er11
TARGET_EP = 25000

methods = {
    "LIO":    "er_lio_10_6",
    "EIA":    "er_attack_10_6",
    "REFiNE": "er_REFiNE_attack_10_6",
}

agent_cols_total = [f"A{i}_reward_total" for i in range(1, N_AGENTS + 1)]
agent_cols_env = [f"A{i}_reward_env" for i in range(1, N_AGENTS + 1)]

for method_name, folder_name in methods.items():
    # Shape: (n_runs, n_agents)
    all_runs_total = []
    all_runs_env = []
    missing = []
    for run_id in range(1, N_RUNS + 1):
        log_path = os.path.join(BASE, f"er{run_id}", folder_name, "log.csv")
        if not os.path.exists(log_path):
            missing.append(run_id)
            continue
        df = pd.read_csv(log_path)
        # Some logs have spaced headers like " A1_reward_total" or "A1_reward_total "
        df.columns = df.columns.str.strip()
        # Get the last row (should be episode 25000)
        last = df.iloc[-1]
        rewards_total = [last[c] for c in agent_cols_total]
        rewards_env = [last[c] for c in agent_cols_env]
        all_runs_total.append(rewards_total)
        all_runs_env.append(rewards_env)

    if missing:
        print(f"[{method_name}] WARNING: missing runs {missing}")

    if not all_runs_total:
        print(f"[{method_name}] runs=0  Total=nan  Stdev=nan")
        print("  Per-agent means: nan")
        print()
        continue

    arr_total = np.array(all_runs_total)  # (n_valid_runs, 10)
    arr_env = np.array(all_runs_env)  # (n_valid_runs, 10)
    # Average each agent across runs
    agent_means_total = arr_total.mean(axis=0)  # (10,) total rewards
    agent_means_env = arr_env.mean(axis=0)  # (10,) environment rewards

    reward_stdev_total = np.std(agent_means_total, ddof=0)
    reward_stdev_env = np.std(agent_means_env, ddof=0)
    total_reward_total = np.sum(agent_means_total)
    total_reward_env = np.sum(agent_means_env)

    print(f"[{method_name}] runs={arr_total.shape[0]}  "
          f"Total={total_reward_total:.2f}  Stdev={reward_stdev_total:.2f}")
    print(f"  Per-agent means: {np.round(agent_means_total, 2)}")
    print(f"[{method_name}] runs={arr_env.shape[0]}  "
          f"Env={total_reward_env:.2f}  Stdev={reward_stdev_env:.2f}")
    print(f"  Per-agent means: {np.round(agent_means_env, 2)}")
    print()