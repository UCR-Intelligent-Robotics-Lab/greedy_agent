import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'baseline': 'blue', 'admo': 'red'}
LABELS = {'baseline': 'Baseline (Clean)', 'admo': 'ADMO (Attacked)'}

def get_latest_log_for_env(env_id, is_admo, is_reciprocator=False):
    """
    Search lio/results for the corresponding log.csv files
    """
    prefix = f"{env_id}_admo_" if is_admo else f"{env_id}_"
    suffix = f"{env_id}_reciprocators" if is_reciprocator else f"{env_id}_lio"
    if is_admo and not is_reciprocator:
        suffix += "_admo"
        
    search_pattern = f"lio/results/{prefix}*/{suffix}/log.csv"
    # LIO logs are generally structured differently than the quick script test
    # If no number matched, try a generic search
    if not glob.glob(search_pattern):
        search_pattern = f"lio/results/{env_id}*/{suffix}/log.csv"
    
    files = glob.glob(search_pattern)
    if not files: return None
    # Sort by modification time to grab the most recent run
    return sorted(files, key=os.path.getmtime)[-1]

def smooth_curve(x, y, weight=0.8):
    if len(y) == 0: return y
    last = y[0]
    smoothed = []
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_csv_results(is_reciprocator=True):
    """
    Plot metrics extracted from log.csv files for LIO/Reciprocator evaluations.
    Referencing Figure 6 logic from the paper, we want to look at "Success Rate" (win_rate)
    or Reward per Energy. We'll plot Environment Reward (A1_reward_env) and Reward Per Energy.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    envs = [('er', 'Escape Room'), ('ipd', 'IPD'), ('staghunt', 'Stag Hunt')]
    agent_type = "Reciprocator" if is_reciprocator else "LIO"
    
    for i, (env_id, env_name) in enumerate(envs):
        clean_log = get_latest_log_for_env(env_id, is_admo=False, is_reciprocator=is_reciprocator)
        admo_log = get_latest_log_for_env(env_id, is_admo=True, is_reciprocator=is_reciprocator)
        
        # Plot 1: Environmental Reward (Row 0)
        ax_rew = axes[0, i]
        # Plot 2: Total Reward (Row 1)
        ax_tot = axes[1, i]
        
        if clean_log and os.path.exists(clean_log):
            df_c = pd.read_csv(clean_log)
            if 'episode' in df_c.columns and 'A1_reward_env' in df_c.columns:
                ax_rew.plot(df_c['episode'], smooth_curve(df_c['episode'], df_c['A1_reward_env']), 
                           label=LABELS['baseline'], color=COLORS['baseline'], alpha=0.8)
                ax_tot.plot(df_c['episode'], smooth_curve(df_c['episode'], df_c['A1_reward_total']), 
                           label=LABELS['baseline'], color=COLORS['baseline'], alpha=0.8)

        if admo_log and os.path.exists(admo_log):
            df_a = pd.read_csv(admo_log)
            if 'episode' in df_a.columns and 'A1_reward_env' in df_a.columns:
                ax_rew.plot(df_a['episode'], smooth_curve(df_a['episode'], df_a['A1_reward_env']), 
                           label=LABELS['admo'], color=COLORS['admo'], alpha=0.8)
                ax_tot.plot(df_a['episode'], smooth_curve(df_a['episode'], df_a['A1_reward_total']), 
                           label=LABELS['admo'], color=COLORS['admo'], alpha=0.8)

        ax_rew.set_title(f"{env_name} (Env Reward)")
        ax_tot.set_title(f"{env_name} (Total Reward)")
        
        ax_rew.set_xlabel("Episodes")
        ax_tot.set_xlabel("Episodes")
        
        if i == 0:
            ax_rew.set_ylabel(f"A1 Env Reward ({agent_type})")
            ax_tot.set_ylabel(f"A1 Total Reward ({agent_type})")
            
        ax_rew.legend()
        ax_tot.legend()
        
    plt.tight_layout()
    out_file = f'results_{agent_type.lower()}_comparison.png'
    plt.savefig(out_file, dpi=300)
    print(f"Saved results comparison to '{out_file}'")

if __name__ == "__main__":
    print("Generating Reciprocator plots...")
    plot_csv_results(is_reciprocator=True)
    print("Generating LIO plots...")
    plot_csv_results(is_reciprocator=False)
