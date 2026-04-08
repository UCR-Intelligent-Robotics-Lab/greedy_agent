import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Define styles similar to Figure 6
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'baseline': 'blue', 'admo': 'red'}
LABELS = {'baseline': 'Baseline (Clean)', 'admo': 'ADMO (Attacked)'}

def get_metric(files, metric='A1_reward_env'):
    if not files: return None
    latest_file = sorted(files, key=os.path.getmtime)[-1]
    df = pd.read_csv(latest_file)
    if metric in df.columns:
        return df[metric].values
    return None

def smooth(y, weight=0.8):
    if y is None or len(y) == 0: return []
    last = y[0]
    smoothed = []
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_csv_results():
    """
    Plot metrics extracted from log.csv files for LIO evaluations.
    Referencing Figure 6 logic from the paper, plotting A1_reward_total.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    envs = [('er', 'Escape Room'), ('ipd', 'IPD'), ('staghunt', 'Stag Hunt')]
    
    for i, (env_id, env_name) in enumerate(envs):
        ax = axes[i]
        
        # Match glob paths specifically to what is in lio/results
        clean_glob = f"lio/results/{env_id}*/{env_id}_lio_2/log.csv"
        admo_glob = f"lio/results/{env_id}_admo*/staghunt_lio_admo/log.csv" if env_id == 'staghunt' else f"lio/results/{env_id}_admo*/{env_id}_lio_admo/log.csv"
        
        # Add fallback for LIO IPD
        if env_id == 'ipd':
            clean_glob = f"lio/results/ipd_*/ipd_lio_2/log.csv"
            admo_glob = f"lio/results/ipd_admo_*/ipd_lio_admo/log.csv"

        clean_files = glob.glob(clean_glob)
        admo_files = glob.glob(admo_glob)

        clean_vals = get_metric(clean_files, 'A1_reward_total')
        admo_vals = get_metric(admo_files, 'A1_reward_total')

        has_data = False
        if clean_vals is not None and len(clean_vals) > 0:
            ax.plot(smooth(clean_vals), label=LABELS['baseline'], color=COLORS['baseline'], alpha=0.8, linewidth=2)
            has_data = True
        if admo_vals is not None and len(admo_vals) > 0:
            ax.plot(smooth(admo_vals), label=LABELS['admo'], color=COLORS['admo'], alpha=0.8, linewidth=2)
            has_data = True
            
        ax.set_title(f"{env_name} (A1 Total Reward)")
        ax.set_xlabel("Evaluation Periods")
        if i == 0:
            ax.set_ylabel(f"Total Reward (LIO)")
            
        if has_data:
            ax.legend()
        
    plt.tight_layout()
    out_file = f'results_lio_comparison.png'
    plt.savefig(out_file, dpi=300)
    print(f"Saved LIO results comparison to '{out_file}'")


def plot_reciprocator_results():
    """
    Plot metrics extracted from log.csv files for Reciprocator evaluations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    envs = [('er', 'Escape Room'), ('ipd', 'IPD'), ('staghunt', 'Stag Hunt')]
    
    for i, (env_id, env_name) in enumerate(envs):
        ax = axes[i]
        
        # Match glob paths specifically to Reciprocators
        clean_glob = f"lio/results/{env_id}_*/{env_id}_reciprocators/log.csv"
        admo_glob = f"lio/results/{env_id}_admo_*/{env_id}_reciprocators/log.csv"

        # Fix staghunt name mapping in glob 
        if env_id == 'staghunt':
             clean_glob = f"lio/results/{env_id}*/{env_id}_reciprocators/log.csv"
             admo_glob = f"lio/results/{env_id}_admo*/{env_id}_reciprocators/log.csv"

        clean_files = glob.glob(clean_glob)
        admo_files = glob.glob(admo_glob)

        clean_vals = get_metric(clean_files, 'A1_reward_total')
        admo_vals = get_metric(admo_files, 'A1_reward_total')

        has_data = False
        if clean_vals is not None and len(clean_vals) > 0:
            ax.plot(smooth(clean_vals), label=LABELS['baseline'], color=COLORS['baseline'], alpha=0.8, linewidth=2)
            has_data = True
        if admo_vals is not None and len(admo_vals) > 0:
            ax.plot(smooth(admo_vals), label=LABELS['admo'], color=COLORS['admo'], alpha=0.8, linewidth=2)
            has_data = True
            
        ax.set_title(f"{env_name} (A1 Total Reward)")
        ax.set_xlabel("Evaluation Periods")
        if i == 0:
            ax.set_ylabel(f"Total Reward (Reciprocator)")
            
        if has_data:
            ax.legend()
        
    plt.tight_layout()
    out_file = f'results_reciprocator_comparison.png'
    plt.savefig(out_file, dpi=300)
    print(f"Saved Reciprocator results comparison to '{out_file}'")


if __name__ == "__main__":
    print("Generating LIO plots...")
    plot_csv_results()
    print("Generating Reciprocator plots...")
    plot_reciprocator_results()
