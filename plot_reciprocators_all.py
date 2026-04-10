import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.ticker import PercentFormatter

plt.style.use('seaborn-v0_8-whitegrid')

COLOR_LINE = 'tab:blue'
COLOR_FILL = 'gray'
ALPHA_FILL = 0.3

def get_metric_arrays(files, metric='A1_reward_env'):
    if not files: return None, None
    all_vals = []
    episodes = None
    for f in files:
        df = pd.read_csv(f)
        col_name = metric if metric in df.columns else f'A1_{metric}' if f'A1_{metric}' in df.columns else None
        if col_name:
            all_vals.append(df[col_name].values)
            if episodes is None and 'episode' in df.columns:
                episodes = df['episode'].values
                
    if not all_vals: return None, None
    min_len = min(len(v) for v in all_vals)
    trimmed_vals = [v[:min_len] for v in all_vals]
    episodes = episodes[:min_len] if episodes is not None else np.arange(min_len)
    return np.array(trimmed_vals), episodes

def plot_section(ax, data_arrs, episodes, title, ylabel=None, is_percent=False):
    if data_arrs is not None:
        means = np.mean(data_arrs, axis=0)
        mins = np.min(data_arrs, axis=0)
        maxs = np.max(data_arrs, axis=0)
        t = episodes / 1000.0
        
        ax.fill_between(t, mins, maxs, alpha=ALPHA_FILL, color=COLOR_FILL, linewidth=0)
        ax.plot(t, means, color=COLOR_LINE, label='Mean', linewidth=1.5, marker='.', markersize=3)
        ax.grid(color='silver', linestyle='--', linewidth=0.5)
        ax.set_title(title, fontsize=14)
        if ylabel: ax.set_ylabel(ylabel, fontsize=12)
        if is_percent: ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel('Episodes (×1000)', fontsize=12)
    else:
        ax.set_title(f"{title}\n(No Data)", fontsize=14)
        ax.axis('off')

def plot_er():
    configs = [(2, 1), (4, 2), (4, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, (n, min_l) in enumerate(configs):
        base_files = glob.glob(f"lio/results/er_{n}_{min_l}_trail_*/er_reciprocators_{n}_{min_l}/log.csv")
        admo_files = glob.glob(f"lio/results/er_admo_{n}_{min_l}_trail_*/er_reciprocators_admo_{n}_{min_l}/log.csv")
        
        base_arrs, ep_base = get_metric_arrays(base_files, 'win_rate')
        admo_arrs, ep_admo = get_metric_arrays(admo_files, 'win_rate')
        
        plot_section(axes[0, idx], base_arrs, ep_base, f"ER({n},{min_l}) Baseline", "Success Rate" if idx == 0 else None, True)
        plot_section(axes[1, idx], admo_arrs, ep_admo, f"ER({n},{min_l}) ADMO Attack", "Success Rate" if idx == 0 else None, True)
    
    plt.tight_layout()
    plt.savefig('er_reciprocators.png', dpi=300, bbox_inches='tight')

def plot_ipd():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    base_files = glob.glob(f"lio/results/ipd_2_trail_*/ipd_reciprocators_2/log.csv")
    admo_files = glob.glob(f"lio/results/ipd_admo_2_trail_*/ipd_reciprocators_2/log.csv")
    
    base_arrs, ep_base = get_metric_arrays(base_files, 'reward_env')
    admo_arrs, ep_admo = get_metric_arrays(admo_files, 'reward_env')
    
    plot_section(axes[0], base_arrs, ep_base, "IPD(2,1) Baseline", "Reward")
    plot_section(axes[1], admo_arrs, ep_admo, "IPD(2,1) ADMO Attack")
    
    plt.tight_layout()
    plt.savefig('ipd_reciprocators.png', dpi=300, bbox_inches='tight')

def plot_staghunt():
    configs = [2, 3, 4]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, n in enumerate(configs):
        base_files = glob.glob(f"lio/results/staghunt_{n}_trail_*/staghunt_reciprocators_{n}/log.csv")
        admo_files = glob.glob(f"lio/results/staghunt_admo_{n}_trail_*/staghunt_reciprocators_{n}/log.csv")
        
        base_arrs, ep_base = get_metric_arrays(base_files, 'reward_env')
        admo_arrs, ep_admo = get_metric_arrays(admo_files, 'reward_env')
        
        if base_arrs is not None: base_arrs = base_arrs / 5.0
        if admo_arrs is not None: admo_arrs = admo_arrs / 5.0
        
        plot_section(axes[0, idx], base_arrs, ep_base, f"Stag Hunt({n}) Baseline", "Success Rate" if idx == 0 else None, True)
        plot_section(axes[1, idx], admo_arrs, ep_admo, f"Stag Hunt({n}) ADMO Attack", "Success Rate" if idx == 0 else None, True)
    
    plt.tight_layout()
    plt.savefig('staghunt_reciprocators.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    plot_er()
    plot_ipd()
    plot_staghunt()
