import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'baseline': '#1f77b4', 'admo': '#d62728'}
LABELS = {'baseline': 'Baseline', 'admo': 'ADMO'}

def get_metric(files, metric='A1_reward_env'):
    if not files: return None
    # For multiple files (e.g. 10 trials), we can average them out to be more robust
    all_vals = []
    for f in files:
        df = pd.read_csv(f)
        if metric in df.columns:
            all_vals.append(df[metric].values)
            
    if not all_vals: return None
    
    # ensure same length for mean
    min_len = min(len(v) for v in all_vals)
    trimmed_vals = [v[:min_len] for v in all_vals]
    return np.mean(trimmed_vals, axis=0)

def smooth(y, weight=0.98):
    if y is None or len(y) == 0: return []
    # Use pandas rolling mean for better visual smoothing on high-frequency data
    window = int(len(y) * 0.05) # 5% of data window
    if window < 5: window = 5
    return pd.Series(y).rolling(window, min_periods=1).mean().values

def extract_from_logs(log_file, regex_pattern):
    if not os.path.exists(log_file): return []
    vals = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(regex_pattern, line)
            if match:
                vals.append(float(match.group(1)))
    return vals

def plot_stdout_logs():
    """
    Plot metrics extracted directly from the console .log files in scripts/.
    Focuses on 'Mean reciprocal reward' (proxy for success rate / impact) or similar metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    envs = [
        ('er', 'Escape Room', 'scripts/autorun_er_reciprocators.log', 'scripts/autorun_er_ad_reciprocators.log'),
        ('ipd', 'IPD', 'scripts/autorun_ipd_reciprocators.log', 'scripts/autorun_ipd_ad_reciprocators.log'),
        ('staghunt', 'Stag Hunt', 'scripts/autorun_staghunt_reciprocators.log', 'scripts/autorun_staghunt_ad_reciprocators.log')
    ]
    
    # We will track max future reward influence as a proxy for impact since the scripts log it reliably
    metric_regex = r'Max future reward influence:\s*([-\d\.]+)'
    
    for i, (env_id, env_name, clean_log, admo_log) in enumerate(envs):
        ax = axes[i]
        
        clean_vals = extract_from_logs(clean_log, metric_regex)
        admo_vals = extract_from_logs(admo_log, metric_regex)
        
        has_data = False
        if clean_vals:
            ax.plot(smooth(clean_vals, 0.95), label=LABELS['baseline'], color=COLORS['baseline'], alpha=0.9, linewidth=2)
            has_data = True
        if admo_vals:
            ax.plot(smooth(admo_vals, 0.95), label=LABELS['admo'], color=COLORS['admo'], alpha=0.9, linewidth=2)
            has_data = True
            
        ax.set_title(f"{env_name} (Max Future Reward Infl.)")
        ax.set_xlabel("Updates")
        if i == 0:
            ax.set_ylabel("Influence Magnitude")
        if has_data:
            ax.legend()
        
    plt.tight_layout()
    plt.savefig('scripts_logs_comparison.png', dpi=300)
    print("Saved scripts stdout comparison to 'scripts_logs_comparison.png'")


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
        elif env_id == 'er':
            clean_glob = f"lio/results/er_*/er_lio_2/log.csv"
            admo_glob = f"lio/results/er_admo_*/er_lio_admo/log.csv"

        clean_files = glob.glob(clean_glob)
        admo_files = glob.glob(admo_glob)

        clean_vals = get_metric(clean_files, 'A1_reward_total')
        admo_vals = get_metric(admo_files, 'A1_reward_total')

        has_data = False
        if clean_vals is not None and len(clean_vals) > 0:
            ax.plot(smooth(clean_vals, 0.95), label=LABELS['baseline'], color=COLORS['baseline'], alpha=0.9, linewidth=2)
            has_data = True
        if admo_vals is not None and len(admo_vals) > 0:
            ax.plot(smooth(admo_vals, 0.95), label=LABELS['admo'], color=COLORS['admo'], alpha=0.9, linewidth=2)
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


if __name__ == "__main__":
    print("Generating LIO plots...")
    plot_csv_results()
    print("Generating Script Log plots...")
    plot_stdout_logs()
