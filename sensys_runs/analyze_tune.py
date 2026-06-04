#!/usr/bin/env python3
"""Analyze REFiNE ER(4,2) B-tuning runs (clipped + unclipped) vs EIA/LIO baselines.
env_* = per-agent ENV reward over the last TAIL eval rows.
fairness = env_std (lower=better); leveling-UP if env_min rises vs baseline (not just env_std down)."""
import glob, os, re, numpy as np, pandas as pd

TAIL = 5                          # final eval snapshots to average
TUNE_GLOB = 'lio/results/*/tune_er_refine_eia_4_2_*'
EIA_GLOB  = 'lio/results/*/er_eia_4_2_w2.0-0.2'
LIO_GLOB  = 'lio/results/*/er_lio_4_2'
EXPECT_ROWS = 20                  # 10k ep / period 500

def acols(df, suf):
    cs = [c for c in df.columns if c.startswith('A') and c.endswith(suf)]
    return sorted(cs, key=lambda c: int(c[1:c.index('_')]))

def metrics(lp, tail=TAIL):
    df = pd.read_csv(lp); nrow = len(df); df = df.tail(tail)
    renv = acols(df, '_reward_env'); en = acols(df, '_total_energy')
    R = df[renv].mean(axis=0).values
    d = dict(rows=nrow, env_mean=float(R.mean()), env_std=float(R.std(ddof=0)),
             env_min=float(R.min()), env_max=float(R.max()),
             nan=int(df[renv].isna().any().any()))
    if en:
        E = df[en].mean(axis=0).values
        d['en_mean']=float(E.mean()); d['en_var']=float(E.var(ddof=0))
    else:
        d['en_mean']=np.nan; d['en_var']=np.nan
    return d

def parse(name):
    clip = 'clip' in name
    mB = re.search(r'_B([0-9.]+)_s', name); mS = re.search(r'_s(\d+)$', name)
    return ('clip' if clip else 'raw',
            float(mB.group(1)) if mB else np.nan,
            int(mS.group(1)) if mS else -1)

def collect(pat):
    out=[]
    for d in sorted(glob.glob(pat)):
        lp=os.path.join(d,'log.csv')
        if os.path.exists(lp):
            try: out.append((os.path.basename(d), metrics(lp)))
            except Exception as e: print(f"  [skip] {d}: {e}")
    return out

rows=[]
for name,m in collect(TUNE_GLOB):
    v,B,S = parse(name); rows.append(dict(variant=v,B=B,seed=S,**m))
tune = pd.DataFrame(rows)
def bl(pat): return pd.DataFrame([m for _,m in collect(pat)])
eia, lio = bl(EIA_GLOB), bl(LIO_GLOB)

pd.set_option('display.width',240); pd.set_option('display.max_columns',40)
ff = lambda x: f"{x:8.3f}"
print("\n"+"#"*108)
print(f"# REFiNE ER(4,2) B-tuning  | env_* over last {TAIL} eval rows | fairness=env_std (lower better)")
print("# leveling-UP if env_min rises vs baseline (good); leveling-DOWN if env_mean<0 or env_std->0 (bad)")
print("#"*108)

if tune.empty:
    print("\n[!] no tune_er_refine_eia_4_2_* dirs found yet (run the tuning sweep first).")
else:
    inc = tune[tune.rows < EXPECT_ROWS]
    if len(inc): print(f"\n[!] {len(inc)} run(s) have < {EXPECT_ROWS} rows (still running / incomplete) — see 'rows'.")
    print("\n=== PER-SEED ===")
    cols=['variant','B','seed','env_mean','env_std','env_min','env_max','en_var','rows','nan']
    print(tune.sort_values(['variant','B','seed'])[cols].to_string(index=False,float_format=ff))
    print("\n=== SEED-AVERAGED by (variant, B) ===")
    g = (tune.groupby(['variant','B'])
            .agg(env_mean=('env_mean','mean'),env_std=('env_std','mean'),
                 env_min=('env_min','mean'),env_max=('env_max','mean'),
                 en_var=('en_var','mean'),seeds=('seed','count'),
                 min_rows=('rows','min')).reset_index())
    print(g.to_string(index=False,float_format=ff))

def show(name,df):
    if df.empty: print(f"{name:16s}: (no dirs found)"); return None
    r=dict(env_mean=df.env_mean.mean(),env_std=df.env_std.mean(),
           env_min=df.env_min.mean(),env_max=df.env_max.mean(),en_var=df.en_var.mean())
    print(f"{name:16s}: env_mean={r['env_mean']:7.2f}  env_std={r['env_std']:7.2f}  "
          f"env_min={r['env_min']:7.2f}  env_max={r['env_max']:7.2f}  en_var={r['en_var']:9.2f}  (n={len(df)})")
    return r
print("\n=== BASELINES (4,2) ===")
beia = show('EIA_baseline', eia); blio = show('LIO_baseline', lio)

if (not tune.empty) and beia:
    print("\n=== DECISION VIEW (seed-avg, relative to EIA baseline) ===")
    print(f"  EIA ref: env={beia['env_mean']:.2f}  std={beia['env_std']:.2f}  "
          f"env_min={beia['env_min']:.2f}  en_var={beia['en_var']:.2f}")
    hdr=f"{'var':4s}{'B':>5s}{'env_ret%':>9s}{'std_red%':>9s}{'enVarRed%':>10s}{'env_min':>9s}{'min_vs_base':>12s}  verdict"
    print(hdr); print('-'*len(hdr))
    for _,r in g.sort_values(['variant','B']).iterrows():
        env_ret = 100*r.env_mean/beia['env_mean'] if beia['env_mean'] else np.nan
        std_red = 100*(1-r.env_std/beia['env_std']) if beia['env_std'] else np.nan
        ev_red  = 100*(1-r.en_var/beia['en_var']) if beia['en_var'] else np.nan
        min_up  = r.env_min - beia['env_min']
        if   r.env_mean < 0 or r.env_std < 1.0:        v="COLLAPSE / leveling-down"
        elif env_ret>=85 and std_red>10 and min_up>0:  v="<<< STRONG: perf kept, fairer, min UP"
        elif env_ret>=85 and std_red>10:               v="<< perf kept + fairer"
        elif std_red>30 and env_ret<70:                v="fairness via perf loss"
        else:                                           v=""
        print(f"{r.variant:4s}{r.B:5.2f}{env_ret:9.1f}{std_red:9.1f}{ev_red:10.1f}{r.env_min:9.3f}{min_up:12.3f}  {v}")
    print("\nPick: variant+B with the largest std_red% while env_ret% >= ~85 and ideally min_vs_base > 0,")
    print("and B within the Prop-1 bounded regime (B <~ 0.5 for ER(4,2)).")
