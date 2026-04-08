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
    fig, axes = plt.subplots(1, len(agent_counts), figsize=(12, 4))
    if len(agent_counts) == 1:
        axes = [axes]
        
    for idx, n_agents in enumerate(agent_counts):
        ax = axes[idx]
        env_id = 'staghunt'
        
        # Staghunt(2) corresponds to our previous runs, others might need a different naming pattern
        # If no other files exist yet, the plot will just show what's available
        clean_glob = f"lio/results/{env_id}*/{env_id}_lio_{n_agents}/log.csv"
        admo_glob = f"lio/results/{env_id}_admo_{n_agents}*/{env_id}_lio_admo/log.csv"
        
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
             admo_arrs, episodes_admo = get_metric_arrays(admo_files, 'A1_reward_env')
             # Normalizing approximation
             if clean_arrs is not None: clean_arrs = clean_arrs / 5.0
             if admo_arrs is not None: admo_arrs = admo_arrs / 5.0

        if clean_arrs is not None and admo_arrs is not None:
            means_clean = np.mean(clean_arrs, axis=0)
            
            means_admo = np.mean(admo_arrs, axis=0)
            mins_admo = np.min(admo_arrs, axis=0)
            maxs_admo = np.max(admo_arrs, axis=0)
            
            t_admo = episodes_admo / 1000.0
            t_clean = episodes_clean / 1000.0

            ax.fill_between(t_admo, mins_admo, maxs_admo, alpha=ALPHA_FILL, color=COLOR_ADMO)
            ax.plot(t_admo, means_admo, color=COLOR_ADMO, label='ADMO', linewidth=1.5)

            ax.plot(t_clean, means_clean, color=COLOR_BASELINE, label='Baseline', linewidth=1.5, marker='.', markersize=3)
            
            ax.grid(color='silver', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Episodes (×1000)', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Success Rate', fontsize=12)
            ax.set_title(f'Stag Hunt({n_agents})', fontsize=14)
            
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            if idx == 0:
                ax.legend(loc='best', fontsize=10)
        else:
            ax.set_title(f'Stag Hunt({n_agents})\n(No Data)', fontsize=14)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig('fig6_staghunt_replication.png', dpi=300)
    print("Saved Figure 6 replication to 'fig6_staghunt_replication.png'")

if __name__ == "__main__":
    generate_staghunt_plot()
