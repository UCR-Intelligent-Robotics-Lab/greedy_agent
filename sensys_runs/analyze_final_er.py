#!/usr/bin/env python3
"""ER(4,2) FINAL probe analysis -- noise-aware (2 seeds/point: read DIRECTION+consistency, not precision).
A) beta-sweep (B=0, energy alone): does a larger beta cut total energy? Term=-(beta*E)/steps -> en_mean.
B) 2x2 (LIO / amplifier-only / energy-only@0.1 / FULL): what moved the FULL en_var (2827)?
Noise floor = LIO baseline's run-to-run std over its 10 seeds; every 2-seed number is judged against it.
2 seeds => NO significance test; read per-seed agreement + whether the effect clears LIO's band.
metrics() identical to analyze_tune.py (TAIL=5, ddof=0)."""
import glob, os, re, numpy as np, pandas as pd
TAIL = 5
def acols(df, suf):
    cs=[c for c in df.columns if c.startswith('A') and c.endswith(suf)]
    return sorted(cs, key=lambda c: int(c[1:c.index('_')]))
def metrics(lp):
    df=pd.read_csv(lp); nrow=len(df); df=df.tail(TAIL)
    R=df[acols(df,'_reward_env')].mean(0).values
    E=df[acols(df,'_total_energy')].mean(0).values
    return dict(rows=nrow, env_mean=float(R.mean()), en_mean=float(E.mean()), en_var=float(E.var(ddof=0)))
def frame(glb):
    rows=[]
    for d in sorted(glob.glob(glb)):
        lp=os.path.join(d,'log.csv')
        if os.path.exists(lp):
            try: rows.append(dict(dir=os.path.basename(d), **metrics(lp)))
            except Exception as e: print(f"  [skip] {d}: {e}")
    return pd.DataFrame(rows)
ff=lambda x:f"{x:9.2f}"
lio=frame('lio/results/*/er_lio_4_2'); eia=frame('lio/results/*/er_eia_4_2_w2.0-0.2')
print("\n"+"#"*98)
print("# ER(4,2) FINAL PROBE ANALYSIS  (2 seeds per REFiNE point -- read DIRECTION, not precision)")
print("#"*98)
def stat(df,c):
    if len(df)>1: return (df[c].mean(), df[c].std(ddof=1))
    return (df[c].mean() if len(df) else float('nan'), float('nan'))
print("\n=== NOISE FLOOR (baseline run-to-run std across seeds) ===")
for nm,df in [('LIO',lio),('EIA',eia)]:
    if df.empty: print(f"  {nm}: (no dirs)"); continue
    em,es=stat(df,'en_mean'); vm,vs=stat(df,'en_var'); rm,rs=stat(df,'env_mean')
    print(f"  {nm:4s} (n={len(df)}): en_mean={em:7.2f} +/-{es:6.2f}   en_var={vm:8.1f} +/-{vs:7.1f}   env_mean={rm:6.2f} +/-{rs:5.2f}")
lio_em,lio_es=stat(lio,'en_mean'); lio_vm,lio_vs=stat(lio,'en_var')
sw=frame('lio/results/*/probe_er_4_2_B0_beta*')
beta_of=lambda nm:(float(re.search(r'_beta([0-9.]+)_s',nm).group(1)) if re.search(r'_beta([0-9.]+)_s',nm) else np.nan)
seed_of=lambda nm:(int(re.search(r'_s(\d+)$',nm).group(1)) if re.search(r'_s(\d+)$',nm) else -1)
print("\n"+"="*98)
print("A) BETA-SWEEP (B=0, energy term alone) | does a larger beta cut total energy (en_mean)?")
print("="*98)
if sw.empty:
    print("  [!] no probe_er_4_2_B0_beta* dirs.")
else:
    sw['beta']=sw['dir'].map(beta_of); sw['seed']=sw['dir'].map(seed_of)
    print("\n--- PER-SEED (do the 2 seeds agree?) ---")
    print(sw.sort_values(['beta','seed'])[['beta','seed','env_mean','en_mean','en_var','rows']].to_string(index=False,float_format=ff))
    g=sw.groupby('beta').agg(env_mean=('env_mean','mean'),
        en_mean=('en_mean','mean'), en_mean_gap=('en_mean',lambda x:x.max()-x.min()),
        en_var=('en_var','mean'), seeds=('seed','count'), min_rows=('rows','min')).reset_index()
    print("\n--- SEED-AVG by beta (en_mean_gap = gap between the 2 seeds = the noise) ---")
    print(g.to_string(index=False,float_format=ff))
    print(f"\n--- READ vs LIO noise band (en_mean {lio_em:.1f} +/-{lio_es:.1f}) ---")
    for _,r in g.sort_values('beta').iterrows():
        d=r.en_mean-lio_em
        if r.env_mean<15: read="TASK COLLAPSED (energy 'saved' by failing the task -- unusable)"
        elif d<0 and abs(d)>lio_es and r.en_mean_gap<abs(d): read="en_mean DOWN, clears noise, seeds agree -> plausibly REAL"
        elif d<0 and abs(d)>lio_es: read="down & clears noise BUT seeds disagree -> shaky"
        elif abs(d)<=lio_es: read="within LIO noise -> NOT distinguishable from baseline"
        else: read="no help / energy up"
        print(f"  beta={r.beta:6.2f}: env_mean={r.env_mean:7.2f}  en_mean={r.en_mean:7.2f} (dLIO {d:+7.2f}, seedgap {r.en_mean_gap:6.2f})  -> {read}")
    print("  (en_var on 2 seeds is a variance-of-variance: very noisy -- suggestive only, never decisive.)")
print("\n"+"="*98)
print("B) 2x2 DECOMPOSITION -- did the AMPLIFIER or the ENERGY term move the FULL en_var (2827)?")
print("="*98)
cells=[]
def cell(label,glb):
    df=frame(glb)
    if df.empty: print(f"  [!] no dirs: {label} ({glb})"); return
    cells.append(dict(label=label,n=len(df),min_rows=int(df.rows.min()),
        env_mean=df.env_mean.mean(),en_mean=df.en_mean.mean(),
        en_var=df.en_var.mean(),en_var_gap=(df.en_var.max()-df.en_var.min()) if len(df)>1 else 0.0))
cell('(b=0,  B=0)   LIO',            'lio/results/*/er_lio_4_2')
cell('(b=0,  B=0.2) amplifier-only', 'lio/results/*/probe_er_4_2_B0.2_beta0_s*')
cell('(b=0.1,B=0)   energy-only',    'lio/results/*/probe_er_4_2_B0_beta0.1_s*')
cell('(b=0.1,B=0.2) FULL',           'lio/results/*/tune_er_refine_eia_4_2_clip_B0.2_s*')
if cells:
    cdf=pd.DataFrame(cells)
    print("\n"+cdf[['label','n','min_rows','env_mean','en_mean','en_var','en_var_gap']].to_string(index=False,float_format=ff))
    print(f"\n  LIO en_var noise band = {lio_vm:.0f} +/-{lio_vs:.0f} (1 std over 10 seeds).")
    print(f"  Any cell within +/-{lio_vs:.0f} of {lio_vm:.0f} is NOT distinguishable from LIO on 2 seeds.")
    for c in cells:
        if c['label'].startswith('(b=0,  B=0)'): continue
        d=c['en_var']-lio_vm; tag="WITHIN noise (likely not real)" if abs(d)<=lio_vs else "outside band (worth a 10-seed check)"
        print(f"    {c['label']:28s} en_var={c['en_var']:8.1f} (dLIO {d:+8.1f}, seedgap {c['en_var_gap']:7.1f}) -> {tag}")
print("\n"+"#"*98)
print("# 2 seeds => NO significance test. Read direction + seed agreement + clearing LIO's band.")
print("# Then COMMIT to the 10-seed full sweep at the best config -- do NOT run more 2-seed probes.")
print("#"*98)
