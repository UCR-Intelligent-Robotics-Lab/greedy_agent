import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from matplotlib.ticker import PercentFormatter

plt.style.use('seaborn-v0_8-whitegrid')

# Fig 6 style parameters
COLOR_BASELINE = 'tab:orange'
COLOR_ADMO = 'tab:blue'
ALPHA_FILL = 0.2

def get_metric_arrays(files, metric='A1_reward_env', expected_episodes=5000):
    if not files: return None, None
    
    all_vals = []
    episodes = None
    for f in files:
        df = pd.read_csv(f)
        # Find the correct metric column
        col_name = None
        if metric in df.columns:
            col_name = metric
        elif f'A1_{metric}' in df.columns:
            col_name = f'A1_{metric}'
            
        if col_name:
            vals = df[col_name].values
            all_vals.append(vals)
            if episodes is None and 'episode' in df.columns:
                episodes = df['episode'].values
                
    if not all_vals: return None, None
    
    # ensure same length for averaging
    min_len = min(len(v) for v in all_vals)
    trimmed_vals = [v[:min_len] for v in all_vals]
    episodes = episodes[:min_len] if episodes is not None else np.arange(min_len)
    
    return np.array(trimmed_vals), episodes

def generate_staghunt_plot():
    """
    Generate a plot mimicking Figure 6 for the Stag Hunt environment specifically,
    across multiple agent counts.
    """
    agent_counts = [2, 3, 4]
    fig, axes = plt.subplots(2, len(agent_counts), figsize=(12, 8))
        
    for idx, n_agents in enumerate(agent_counts):
        ax_base = axes[0, idx]
        ax_admo = axes[1, idx]
        env_id = 'staghunt'
        
        # General glob patterns
        clean_glob = f"lio/results/{env_id}*/{env_id}_lio_{n_agents}/log.csv"
        admo_glob = f"lio/results/{env_id}_admo_{n_agents}*/{env_id}_lio_admo_{n_agents}/log.csv"
        
        # For N=2 we know the existing path structure from prior runs
        if n_agents == 2:
            clean_glob = f"lio/results/{env_id}*/{env_id}_lio_2/log.csv"
            admo_glob = f"lio/results/{env_id}_admo*/staghunt_lio_admo/log.csv"

        clean_files = glob.glob(clean_glob)
        admo_files = glob.glob(admo_glob)

        metric = 'A1_win_rate'
        
        clean_arrs, episodes_clean = get_metric_arrays(clean_files, metric)
        admo_arrs, episodes_admo = get_metric_arrays(admo_files, metric)
        
        if clean_arrs is None:
             clean_arrs, episodes_clean = get_metric_arrays(clean_files, 'A1_reward_env')
             # Normalizing approximation
             if clean_arrs is not None: clean_arrs = clean_arrs / 5.0
        
        if admo_arrs is None:
             admo_arrs, episodes_admo = get_metric_arrays(admo_files, 'A1_reward_env')
             # Normalizing approximation
             if admo_arrs is not None: admo_arrs = admo_arrs / 5.0

        # Plot Clean data
        if clean_arrs is not None:
            means_clean = np.mean(clean_arrs, axis=0)
            mins_clean = np.min(clean_arrs, axis=0)
            maxs_clean = np.max(clean_arrs, axis=0)
            
            t_clean = episodes_clean / 1000.0

            ax_base.fill_between(t_clean, mins_clean, maxs_clean, alpha=ALPHA_FILL, color=COLOR_ADMO)
            ax_base.plot(t_clean, means_clean, color=COLOR_ADMO, label='LIO Baseline', linewidth=1.5, marker='.', markersize=3)
            
            ax_base.grid(color='silver', linestyle='--', linewidth=0.5)
            ax_base.set_xlim([0, 25])
            if idx == 0:
                ax_base.set_ylabel('Success Rate', fontsize=12)
            ax_base.set_title(f'Stag Hunt({n_agents})', fontsize=14)
            ax_base.yaxis.set_major_formatter(PercentFormatter(1.0))
            if idx == 0:
                ax_base.legend(loc='best', fontsize=10)
        else:
            ax_base.set_title(f'Stag Hunt({n_agents})\n(No Data)', fontsize=14)
            ax_base.axis('off')

        # Plot ADMO data
        if admo_arrs is not None:
            means_admo = np.mean(admo_arrs, axis=0)
            mins_admo = np.min(admo_arrs, axis=0)
            maxs_admo = np.max(admo_arrs, axis=0)
            
            t_admo = episodes_admo / 1000.0

            ax_admo.fill_between(t_admo, mins_admo, maxs_admo, alpha=ALPHA_FILL, color=COLOR_BASELINE)
            ax_admo.plot(t_admo, means_admo, color=COLOR_BASELINE, label='ADMO Attack', linewidth=1.5, marker='.', markersize=3)
            
            ax_admo.grid(color='silver', linestyle='--', linewidth=0.5)
            ax_admo.set_xlim([0, 25])
            ax_admo.set_xlabel('Episodes (×1000)', fontsize=12)
            if idx == 0:
                ax_admo.set_ylabel('Success Rate', fontsize=12)
            ax_admo.yaxis.set_major_formatter(PercentFormatter(1.0))
            if idx == 0:
                ax_admo.legend(loc='best', fontsize=10)
        else:
            ax_admo.set_title(f'Stag Hunt({n_agents})\n(No Data)', fontsize=14)
            ax_admo.axis('off')
            
    plt.tight_layout()
    plt.savefig('fig6_staghunt_replication.png', dpi=300)
    print("Saved Figure 6 replication to 'fig6_staghunt_replication.png'")

if __name__ == "__main__":
    generate_staghunt_plot()
