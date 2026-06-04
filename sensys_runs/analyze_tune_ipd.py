#!/usr/bin/env python3
"""IPD B-tuning (clipped) vs EIA/LIO. env_std ~ half the payoff gap (lower=fairer).
CORRECTED verdict: 'cooperation' needs HIGH mean (near cooperative payoff), NOT just low std.
The all-(-100) mutual-defection floor has std=0 but is the WORST outcome; flagged as such."""
import glob, os, re, numpy as np, pandas as pd
TAIL = 5
G_TUNE = 'lio/results/*/tune_ipd_refine_eia_clip_B*'
G_EIA  = 'lio/results/*/ipd_eia'; G_LIO = 'lio/results/*/ipd_lio'

def acols(df, suf):
    cs = [c for c in df.columns if c.startswith('A') and c.endswith(suf)]
    return sorted(cs, key=lambda c: int(c[1:c.index('_')]))
def metr(lp):
    df = pd.read_csv(lp); n = len(df); df = df.tail(TAIL)
    R = df[acols(df, '_reward_env')].mean(0).values
    return dict(rows=n, env_mean=float(R.mean()), env_std=float(R.std(ddof=0)),
               env_min=float(R.min()), env_max=float(R.max()))
def parse(nm):
    mB = re.search(r'_B([0-9.]+)_s', nm); mS = re.search(r'_s(\d+)$', nm)
    return (float(mB.group(1)) if mB else np.nan, int(mS.group(1)) if mS else -1)
def coll(p):
    o = []
    for d in sorted(glob.glob(p)):
        lp = os.path.join(d, 'log.csv')
        if os.path.exists(lp):
            try: o.append((os.path.basename(d), metr(lp)))
            except Exception as e: print('skip', d, e)
    return o
rows = []
for nm, m in coll(G_TUNE):
    B, S = parse(nm); rows.append(dict(B=B, seed=S, **m))
tune = pd.DataFrame(rows)
def bl(p): return pd.DataFrame([m for _, m in coll(p)])
eia, lio = bl(G_EIA), bl(G_LIO); ff = lambda x: f"{x:8.2f}"
print("\n" + "#"*96)
print("# IPD B-tuning (clipped) | env_std ~ half payoff gap (lower=fairer) | GOOD = HIGH mean AND low std")
print("#"*96)
if tune.empty:
    print("\n[!] no tune_ipd_refine_eia_clip_B* dirs found.")
else:
    print("\n=== PER-SEED ===")
    print(tune.sort_values(['B','seed'])[['B','seed','env_mean','env_std','env_min','env_max','rows']].to_string(index=False, float_format=ff))
    print("\n=== SEED-AVG by B ===")
    g = tune.groupby('B').agg(env_mean=('env_mean','mean'), env_std=('env_std','mean'),
        env_min=('env_min','mean'), env_max=('env_max','mean'),
        seeds=('seed','count'), min_rows=('rows','min')).reset_index()
    print(g.to_string(index=False, float_format=ff))
def show(n, df):
    if df.empty: print(f"{n:10s}: (none)"); return None
    r = dict(env_mean=df.env_mean.mean(), env_std=df.env_std.mean(), env_min=df.env_min.mean(), env_max=df.env_max.mean())
    print(f"{n:10s}: mean={r['env_mean']:7.2f}  std={r['env_std']:7.2f}  min={r['env_min']:7.2f}  max={r['env_max']:7.2f}  (n={len(df)})")
    return r
print("\n=== BASELINES ===")
beia = show('IPD_EIA', eia); blio = show('IPD_LIO', lio)
if not tune.empty and beia and blio:
    print("\n=== DECISION (CORRECTED: high mean = cooperation; mean~-100 = mutual defection) ===")
    hdr = f"{'B':>7s}{'mean':>9s}{'std':>8s}{'vs_LIO_mean':>12s}  verdict"
    print(hdr); print('-'*len(hdr))
    for _, r in g.sort_values('B').iterrows():
        dlt = r.env_mean - blio['env_mean']
        if r.env_mean <= -99:
            v = "MUTUAL DEFECTION (worst; std=0 is the equal-bad floor, NOT cooperation)"
        elif r.env_mean > blio['env_mean'] and r.env_std < blio['env_std']:
            v = "<<< beats LIO: higher payoff AND fairer (genuine cooperation gain)"
        elif r.env_mean > beia['env_mean'] and r.env_std < beia['env_std']:
            v = "~ recovers toward LIO (better than attacked EIA; not beating clean LIO)"
        else:
            v = "no clear gain"
        print(f"{r.B:7.3f}{r.env_mean:9.2f}{r.env_std:8.2f}{dlt:+12.2f}  {v}")
    print("\n  Want: a B with mean clearly above LIO's {:.1f} AND std below LIO's {:.1f}.".format(blio['env_mean'], blio['env_std']))
    print("  If every B is either ~EIA (no gain) or ~-100 (collapse) -> IPD reward-fairness does not hold;")
    print("  energy-equity headline carries the paper, IPD becomes an honest characterization/limitation.")
