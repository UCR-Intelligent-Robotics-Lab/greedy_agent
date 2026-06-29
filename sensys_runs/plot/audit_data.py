#!/usr/bin/env python3
"""Ground-truth audit of on-disk REFiNE data: inventory + per-seed + outliers.
Run: conda run -n LIO_tecs python3 sensys_runs/plot/audit_data.py"""
import os, glob, re, numpy as np, pandas as pd
ROOT, TAIL = 'lio/results', 5

def n_agents_of(group):
    if 'ipd' in group: return 2
    m = re.search(r'_(\d+)_(\d+)', group);  return int(m.group(1)) if m else 4

def metrics(csv, na):
    df = pd.read_csv(csv); df.columns=[c.strip() for c in df.columns]; t=df.tail(TAIL)
    has_en = 'A1_total_energy' in df.columns
    en  = np.mean([t[f'A{i}_total_energy'].mean() for i in range(1,na+1)]) if has_en else np.nan
    env = np.mean([t[f'A{i}_reward_env'].mean()   for i in range(1,na+1)])
    return en, env

def flags(vals):                       # MAD-based outlier indices
    v=np.asarray(vals,float)
    if np.all(np.isnan(v)): return np.nan, []
    med=np.nanmedian(v); mad=np.nanmedian(np.abs(v-med))*1.4826
    thr=max(3*mad, 0.25*abs(med), 1e-9)
    return med, [i for i,x in enumerate(v) if not np.isnan(x) and abs(x-med)>thr]

# ---- collect every run ----
recs=[]
for csv in glob.glob(f'{ROOT}/*/*/log.csv'):
    rundir=os.path.basename(os.path.dirname(csv))
    parent=os.path.basename(os.path.dirname(os.path.dirname(csv)))
    m=re.search(r'_s(\d+)$', rundir); seed=int(m.group(1)) if m else -1
    recs.append(dict(parent=parent, group=re.sub(r'_s\d+$','',rundir), seed=seed, csv=csv))
inv=pd.DataFrame(recs)
if inv.empty: print('NO log.csv under', ROOT); raise SystemExit

# ---- 1) inventory ----
print('='*78); print('1) DIRECTORY INVENTORY  (group = rundir minus _s<seed>)'); print('='*78)
for (p,g),sub in inv.groupby(['parent','group']):
    sd=sorted(s for s in sub.seed if s>=0)
    print(f'{p:>14} / {g:<40} n={sub.seed.nunique():>2}  seeds={sd}')

# ---- 2) per-group robust summary + outlier flags ----
print('\n'+'='*78); print('2) PER-GROUP SUMMARY  (median; mean±std; converged count; diverged seeds)'); print('='*78)
detail=[]
for (p,g),sub in sorted(inv.groupby(['parent','group'])):
    na=n_agents_of(g); rows=[]
    for _,r in sub.sort_values('seed').iterrows():
        try: en,env=metrics(r.csv,na); rows.append((r.seed,en,env))
        except Exception as e: rows.append((r.seed,np.nan,np.nan))
    seeds=[s for s,_,_ in rows]; ens=[e for _,e,_ in rows]; envs=[v for _,_,v in rows]
    med_en,fe=flags(ens); med_env,fv=flags(envs)
    bad=sorted({seeds[i] for i in set(fe)|set(fv)})
    nconv=len(seeds)-len(bad)
    mu=np.nanmean(ens); sd=np.nanstd(ens,ddof=1) if np.sum(~np.isnan(ens))>1 else 0.0
    print(f'{g:<42} n={len(seeds):>2}  med(en={med_en:7.1f}, env={med_env:6.1f})  '
          f'mean_en={mu:7.1f}±{sd:5.1f}  conv={nconv}/{len(seeds)}  diverged={bad}')
    if bad: detail.append((g,rows,bad))

# ---- 3) per-seed dump for groups with any divergence ----
if detail:
    print('\n'+'='*78); print('3) PER-SEED DUMP for groups with outliers'); print('='*78)
    for g,rows,bad in detail:
        print(f'-- {g} (diverged: {bad}) --')
        for s,en,env in rows:
            tag=' <-- OUTLIER' if s in bad else ''
            print(f'   seed{s}: en={en:7.1f}  env={env:6.1f}{tag}')

# ---- 4) focused ER(4,2): headline candidates + EIA-weight robustness ----
print('\n'+'='*78); print('4) FOCUS ER(4,2): headline candidates & EIA-weight spread'); print('='*78)
def group_stats(pattern, na=4):
    g=inv[inv.group.str.match(pattern)]
    if g.empty: return None
    out={}
    for grp,sub in g.groupby('group'):
        ens=[];envs=[]
        for _,r in sub.iterrows():
            try: e,v=metrics(r.csv,na); ens.append(e); envs.append(v)
            except: pass
        med_en,fe=flags(ens); med_env,_=flags(envs)
        out[grp]=dict(n=len(ens), med_en=med_en, med_env=med_env,
                      mean_en=np.nanmean(ens), std_en=np.nanstd(ens,ddof=1) if len(ens)>1 else 0.0,
                      nconv=len(ens)-len(fe))
    return out
for label,pat in [('LIO',          r'.*er_lio_4_2$'),
                  ('LIO+EIA (all weights)', r'.*er_eia_4_2_w.*'),
                  ('REFiNE no-EIA', r'.*er_refine_4_2$'),
                  ('REFiNE+EIA (all weights)', r'.*er_refine_eia_4_2.*')]:
    st=group_stats(pat)
    print(f'\n[{label}]  pattern={pat}')
    if not st: print('   (no dirs)'); continue
    for grp,s in sorted(st.items()):
        print(f'   {grp:<42} n={s["n"]:>2} conv={s["nconv"]}/{s["n"]}  '
              f'med_en={s["med_en"]:7.1f}  med_env={s["med_env"]:6.1f}  '
              f'mean_en={s["mean_en"]:7.1f}±{s["std_en"]:5.1f}')
print('\nDONE. Compare med_en across LIO vs REFiNE for the true %Δ; compare std_en/spread '
      'across EIA weights for the destabilization story.')
