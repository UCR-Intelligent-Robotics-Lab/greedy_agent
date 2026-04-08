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
    Generate a plot mimicking Figure 6 for the Stag Hunt environment specifically.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    env_id = 'staghunt'
    
    clean_glob = f"lio/results/{env_id}*/{env_id}_lio_2/log.csv"
    admo_glob = f"lio/results/{env_id}_admo*/staghunt_lio_admo/log.csv"

    clean_files = glob.glob(clean_glob)
    admo_files = glob.glob(admo_glob)

    # Use win_rate if available, else fallback to A1_reward_env as a proxy for success
    metric = 'A1_win_rate'
    
    clean_arrs, episodes_clean = get_metric_arrays(clean_files, metric)
    admo_arrs, episodes_admo = get_metric_arrays(admo_files, metric)
    
    # If win_rate isn't explicitly there, fallback to a proxy (e.g., A1_reward_env scaled to 0-1)
    if clean_arrs is None:
         clean_arrs, episodes_clean = get_metric_arrays(clean_files, 'A1_reward_env')
         admo_arrs, episodes_admo = get_metric_arrays(admo_files, 'A1_reward_env')
         # Proxy normalization for stag hunt (max reward ~ 5)
         if clean_arrs is not None: clean_arrs = clean_arrs / 5.0
         if admo_arrs is not None: admo_arrs = admo_arrs / 5.0

    if clean_arrs is not None and admo_arrs is not None:
        # Baseline (Clean)
        means_clean = np.mean(clean_arrs, axis=0)
        
        # ADMO (Attacked)
        means_admo = np.mean(admo_arrs, axis=0)
        mins_admo = np.min(admo_arrs, axis=0)
        maxs_admo = np.max(admo_arrs, axis=0)
        
        # Convert x-axis to thousands
        t_admo = episodes_admo / 1000.0
        t_clean = episodes_clean / 1000.0

        # Plot ADMO (Area + Line) - Mapped to tab:blue (like Partial Communication in Fig 6)
        ax.fill_between(t_admo, mins_admo, maxs_admo, alpha=ALPHA_FILL, color=COLOR_ADMO)
        ax.plot(t_admo, means_admo, color=COLOR_ADMO, label='ADMO', linewidth=1.5)

        # Plot Baseline (Line + Dots) - Mapped to tab:orange (like LIO in Fig 6)
        ax.plot(t_clean, means_clean, color=COLOR_BASELINE, label='Baseline', linewidth=1.5, marker='.', markersize=3)
        
        # Formatting
        ax.grid(color='silver', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Episodes (×1000)', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Stag Hunt', fontsize=14)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('fig6_staghunt_replication.png', dpi=300)
        print("Saved Figure 6 replication to 'fig6_staghunt_replication.png'")
    else:
        print("Could not find sufficient data to generate plot.")

if __name__ == "__main__":
    generate_staghunt_plot()
