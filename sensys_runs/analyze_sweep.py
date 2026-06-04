#!/usr/bin/env python3
"""REFiNE SenSys full-sweep analysis (110 runs, 10 seeds/condition).

Per-run metric: average each agent over the LAST TAIL=5 eval rows, then across agents.
  env_mean / env_std : reward_env (env-only). env_std = inter-agent dispersion (ddof=0).
  en_mean  / en_var  : total_energy. en_mean = sustainability; en_var = inter-agent variance (ddof=0).
Across the 10 seeds: report  mean +/- std (ddof=1).
  -> en_mean SEED-STD is the run-to-run energy (IN)STABILITY (the EIA destabilization signal).

Energy story = en_mean (NOT en_var). Reward = reward_env (NOT reward_total).
NEVER trust teamwork_fairness or win_rate columns.
"""
import glob, os, numpy as np, pandas as pd
pd.set_option('display.width', 200)

TAIL = 5
ROOT = 'lio/results'
ER_ROWS, IPD_ROWS = 50, 20

def acols(df, suf):
    cs = [c for c in df.columns if c.startswith('A') and c.endswith(suf)]
    return sorted(cs, key=lambda c: int(c[1:c.index('_')]))

def run_metrics(lp):
    df = pd.read_csv(lp); nrow = len(df); tail = df.tail(TAIL)
    renv = acols(tail, '_reward_env'); en = acols(tail, '_total_energy')
    R = tail[renv].mean(axis=0).values
    d = dict(rows=nrow,
             env_mean=float(R.mean()), env_std=float(R.std(ddof=0)),
             env_min=float(R.min()), env_max=float(R.max()),
             nan=int(tail[renv].isna().any().any()))
    if en:
        E = tail[en].mean(axis=0).values
        d['en_mean'] = float(E.mean()); d['en_var'] = float(E.var(ddof=0))
    else:
        d['en_mean'] = np.nan; d['en_var'] = np.nan
    return d

def collect(dirpat):
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, '*', dirpat))):
        lp = os.path.join(d, 'log.csv')
        if os.path.exists(lp):
            try:
                m = run_metrics(lp); m['dir'] = os.path.basename(d); rows.append(m)
            except Exception as e:
                print(f"  [skip] {d}: {e}")
    return pd.DataFrame(rows)

def agg(dirpat):
    df = collect(dirpat)
    if df.empty:
        return None
    a = {}
    for k in ['env_mean', 'env_std', 'en_mean', 'en_var']:
        col = df[k]
        if col.notna().any():
            a[k + '_m'] = float(col.mean())
            a[k + '_s'] = float(col.std(ddof=1)) if len(col) > 1 else 0.0
        else:
            a[k + '_m'] = np.nan; a[k + '_s'] = np.nan
    a['n'] = len(df); a['min_rows'] = int(df['rows'].min())
    return a

def red(cond, base, key='en_mean_m'):
    if not cond or not base or not base.get(key) or base[key] != base[key]:
        return float('nan')
    return 100.0 * (1.0 - cond[key] / base[key])

def ret(cond, base, key='env_mean_m'):
    if not cond or not base or not base.get(key) or base[key] != base[key]:
        return float('nan')
    return 100.0 * cond[key] / base[key]

def prow(label, a):
    if a is None:
        print(f"  {label:16s} (not found)"); return
    print(f"  {label:16s} envR {a['env_mean_m']:7.2f} +/-{a['env_mean_s']:6.2f} | "
          f"energy {a['en_mean_m']:7.2f} +/-{a['en_mean_s']:6.2f} | "
          f"enVar {a['en_var_m']:8.1f} +/-{a['en_var_s']:7.1f}   "
          f"(n={a['n']}, rows>={a['min_rows']})")

ER = {
    '(3,1)': dict(lio='er_lio_3_1', eia='er_eia_3_1_w2.0-0.2',
                  refine_eia='sweep_er_refine_eia_3_1_s*', refine='sweep_er_refine_3_1_s*'),
    '(4,2)': dict(lio='er_lio_4_2', eia='er_eia_4_2_w2.0-0.2',
                  refine_eia='sweep_er_refine_eia_4_2_s*', refine='sweep_er_refine_4_2_s*'),
    '(6,4)': dict(lio='er_lio_6_4', eia='er_eia_6_4_w2.0-0.2',
                  refine_eia='sweep_er_refine_eia_6_4_s*', refine='sweep_er_refine_6_4_s*'),
}

print("=" * 100)
print("# 0. INVENTORY  (count x basename, min_rows over matches)")
print("=" * 100)
inv = {}
for lp in glob.glob(os.path.join(ROOT, '*', '*', 'log.csv')):
    name = os.path.basename(os.path.dirname(lp))
    try:
        with open(lp) as f:
            n = max(0, sum(1 for _ in f) - 1)
    except OSError:
        n = -1
    inv.setdefault(name, []).append(n)
for name in sorted(inv):
    rr = inv[name]
    print(f"  {len(rr):3d} x  {name:44s} min_rows={min(rr)}")
print(f"\n  TOTAL sweep_* run-dirs: {sum(len(v) for k, v in inv.items() if k.startswith('sweep_'))}\n")

print("=" * 100)
print("# A. ER SUSTAINABILITY + EIA DESTABILIZATION  (energy = sustainability metric)")
print("#    want: REFiNE+EIA energy ~10% < LIO; REFiNE+EIA energy +/- SMALL (re-stabilized);")
print("#          EIA energy +/- LARGE (destabilized); envR preserved (task kept).")
print("=" * 100)
A = {}
for size, c in ER.items():
    print(f"\n-- {size} --")
    A[size] = dict(lio=agg(c['lio']), eia=agg(c['eia']),
                   refine_eia=agg(c['refine_eia']), refine=agg(c['refine']))
    prow('LIO (no atk)', A[size]['lio'])
    prow('EIA (atk)', A[size]['eia'])
    prow('REFiNE+EIA', A[size]['refine_eia'])
    prow('REFiNE (no atk)', A[size]['refine'])

print("\n" + "=" * 100)
print("# B. ENERGY ATTRIBUTION @ (4,2)  ({B=0}=energy-only isolates beta; {beta=0}=amplifier-only)")
print("=" * 100)
aL = A['(4,2)']['lio']; aFull = A['(4,2)']['refine_eia']
aBeta0 = agg('sweep_er_refine_eia_4_2_beta0_s*')
aB0 = agg('sweep_er_refine_eia_4_2_B0_s*')
print()
prow('LIO', aL); prow('FULL b=1,B=.2', aFull)
prow('{B=0} energyOnly', aB0); prow('{b=0} amplOnly', aBeta0)
print("\n  en_mean reduction vs LIO:  "
      f"FULL={red(aFull, aL):.1f}%   energy-only={red(aB0, aL):.1f}%   ampl-only={red(aBeta0, aL):.1f}%")
print("  -> energy term drives sustainability IF energy-only ~= FULL and ampl-only ~= 0.")

print("\n" + "=" * 100)
print("# C. SCALABILITY  (REFiNE+EIA energy reduction vs LIO + task retention, per size)")
print("=" * 100)
print(f"\n  {'size':8s}{'LIO en':>10s}{'REF+EIA en':>14s}{'en_red%':>10s}{'envR_ret%':>12s}")
for size, d in A.items():
    aL2, aR2 = d['lio'], d['refine_eia']
    le = f"{aL2['en_mean_m']:.2f}" if aL2 else 'NA'
    rv = f"{aR2['en_mean_m']:.2f}" if aR2 else 'NA'
    print(f"  {size:8s}{le:>10s}{rv:>14s}{red(aR2, aL2):>10.1f}{ret(aR2, aL2):>12.1f}")

print("\n" + "=" * 100)
print("# D. ATTACK-WEIGHT ROBUSTNESS @ (4,2)  (REFiNE+EIA energy/task vs attack weights)")
print("=" * 100)
print()
for lbl, aR, aE in [('w2.0-0.2 (default)', A['(4,2)']['refine_eia'], A['(4,2)']['eia']),
                    ('w1.1-0.9', agg('sweep_er_refine_eia_4_2_w1.1-0.9_s*'), agg('er_eia_4_2_w1.1-0.9')),
                    ('w1.5-1.5', agg('sweep_er_refine_eia_4_2_w1.5-1.5_s*'), agg('er_eia_4_2_w1.5-1.5'))]:
    print(f"-- EIA weights {lbl} --")
    prow('  REFiNE+EIA', aR); prow('  EIA base', aE)
print("\n  -> robust if REFiNE+EIA energy stays low/stable and envR preserved across weights.")

print("\n" + "=" * 100)
print("# E. IPD LIMITATION  (want LOW env_std + HIGH env_mean = cooperation restored)")
print("#    no restoration if REFiNE+EIA env_mean ~= EIA (~ -75) -> characterized limitation.")
print("=" * 100)
def iprow(label, a):
    if a is None:
        print(f"  {label:16s} (not found)"); return
    print(f"  {label:16s} env_mean {a['env_mean_m']:8.2f} +/-{a['env_mean_s']:6.2f} | "
          f"env_std {a['env_std_m']:7.2f} +/-{a['env_std_s']:6.2f}   (n={a['n']}, rows>={a['min_rows']})")
print()
iprow('IPD_LIO', agg('ipd_lio'))
iprow('IPD_EIA', agg('ipd_eia'))
iprow('REFiNE+EIA B=.1', agg('sweep_ipd_refine_eia_B0.1_s*'))

print("\n" + "=" * 100)
print("# F. PER-SEED CHECK  REFiNE+EIA (collapse if env_mean<15; incomplete if rows<50)")
print("=" * 100)
for size, c in ER.items():
    df = collect(c['refine_eia'])
    print(f"\n-- {size}  REFiNE+EIA --")
    if df.empty:
        print("  (none found)"); continue
    for _, r in df.sort_values('dir').iterrows():
        flag = (' INCOMPLETE' if r['rows'] < ER_ROWS else '') + (' COLLAPSE' if r['env_mean'] < 15 else '') + (' NAN' if r['nan'] else '')
        print(f"  {r['dir']:34s} env_mean={r['env_mean']:7.2f}  en_mean={r['en_mean']:7.2f}  rows={int(r['rows']):3d}{flag}")
for pat, lbl in [('sweep_er_refine_eia_4_2_beta0_s*', '{b=0} amplOnly'),
                 ('sweep_er_refine_eia_4_2_B0_s*', '{B=0} energyOnly')]:
    df = collect(pat)
    print(f"\n-- (4,2) {lbl} --")
    if df.empty:
        print("  (none found)"); continue
    for _, r in df.sort_values('dir').iterrows():
        flag = (' INCOMPLETE' if r['rows'] < ER_ROWS else '') + (' COLLAPSE' if r['env_mean'] < 15 else '')
        print(f"  {r['dir']:34s} env_mean={r['env_mean']:7.2f}  en_mean={r['en_mean']:7.2f}  rows={int(r['rows']):3d}{flag}")
print("\n[done]")
