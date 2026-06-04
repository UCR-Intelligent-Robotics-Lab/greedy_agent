import glob, os
import numpy as np, pandas as pd
TAIL, RESULTS = 5, "lio/results"
def per_run(path):
    df = pd.read_csv(path); cols = df.columns.tolist()
    n = sum(1 for c in cols if c.endswith("_reward_env"))
    env = [df[f"A{i}_reward_env"].tail(TAIL).mean() for i in range(1, n+1)]
    has_e = any(c.endswith("_total_energy") for c in cols)
    en  = [df[f"A{i}_total_energy"].tail(TAIL).mean() for i in range(1, n+1)] if has_e else None
    return float(np.mean(env)), (float(np.mean(en)) if en else float('nan')), float(np.var(env, ddof=0))
def cell(label, pattern):
    paths = sorted(glob.glob(os.path.join(RESULTS, "*", pattern, "log.csv")))
    if not paths: print(f"{label:26s}  (none: {pattern})"); return
    EM, EN, EV = [], [], []
    for p in paths:
        try: em,en,ev = per_run(p); EM.append(em); EN.append(en); EV.append(ev)
        except Exception as e: print(f"  ! {p}: {e}")
    ms = lambda x:(lambda a:f"{a.mean():7.2f} +/-{a.std(ddof=1) if len(a)>1 else 0.0:5.2f}")(np.array(x))
    print(f"{label:26s} envR {ms(EM)}  energy {ms(EN)}  envVar {ms(EV)}  (n={len(EM)})")
print("=== corrected J_fair quick check (env-only, tail-5) ===")
cell("(4,2) b1.0  [sanity]", "fix_er_refine_eia_4_2_b1.0_s*")
cell("(6,4) b0.1  [probe] ", "fix_er_refine_eia_6_4_b0.1_s*")
cell("(6,4) b0.3  [probe] ", "fix_er_refine_eia_6_4_b0.3_s*")
cell("(6,4) b1.0  [control]", "fix_er_refine_eia_6_4_b1.0_s*")
print("-- baselines (your sweep) --")
cell("(4,2) LIO", "er_lio_4_2"); cell("(6,4) LIO", "er_lio_6_4")
