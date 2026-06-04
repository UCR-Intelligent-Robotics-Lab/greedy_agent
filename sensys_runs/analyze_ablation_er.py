#!/usr/bin/env python3
"""ER(4,2) energy-decomposition: isolate the energy term (beta) vs the fairness amplifier (B).
Metric of record = en_var (inter-agent variance of A{i}_total_energy, last TAIL eval rows); LOWER = more equitable.
metrics() identical to analyze_tune.py (TAIL=5, ddof=0)."""
import glob, os, numpy as np, pandas as pd

TAIL = 5
EXPECT_ROWS = 20  # 10k ep / period 500
SPECS = [
    ('lio/results/*/er_lio_4_2',                         'LIO   (b=0,   B=0)    no-attack baseline', 0.0, 0.0),
    ('lio/results/*/probe_er_4_2_B0.2_beta0_s*',         '{b=0} (b=0,   B=0.2)  amplifier only',     0.0, 0.2),
    ('lio/results/*/probe_er_4_2_B0_beta0.1_s*',         '{B=0} (b=0.1, B=0)    energy only',        0.1, 0.0),
    ('lio/results/*/tune_er_refine_eia_4_2_clip_B0.2_s*','FULL  (b=0.1, B=0.2)  both (from tuning)', 0.1, 0.2),
    ('lio/results/*/er_eia_4_2_w2.0-0.2',                'EIA   (under attack)  baseline',           np.nan, np.nan),
]

def acols(df, suf):
    cs = [c for c in df.columns if c.startswith('A') and c.endswith(suf)]
    return sorted(cs, key=lambda c: int(c[1:c.index('_')]))

def metrics(lp):
    df = pd.read_csv(lp); nrow = len(df); df = df.tail(TAIL)
    renv = acols(df, '_reward_env'); en = acols(df, '_total_energy')
    R = df[renv].mean(axis=0).values
    d = dict(rows=nrow, env_mean=float(R.mean()), env_std=float(R.std(ddof=0)), env_min=float(R.min()))
    if en:
        E = df[en].mean(axis=0).values
        d['en_mean'] = float(E.mean()); d['en_var'] = float(E.var(ddof=0))
    else:
        d['en_mean'] = np.nan; d['en_var'] = np.nan
    return d

def agg(glb, label):
    rows = []
    for d in sorted(glob.glob(glb)):
        lp = os.path.join(d, 'log.csv')
        if os.path.exists(lp):
            try: rows.append(metrics(lp))
            except Exception as e: print(f"  [skip] {d}: {e}")
    if not rows:
        print(f"  [!] no dirs matched for {label}: {glb}"); return None
    df = pd.DataFrame(rows)
    return dict(label=label, n=len(df), min_rows=int(df.rows.min()),
                env_mean=df.env_mean.mean(), env_std=df.env_std.mean(),
                env_min=df.env_min.mean(), en_mean=df.en_mean.mean(), en_var=df.en_var.mean())

cells = [c for c in (agg(g, lab) for g, lab, *_ in SPECS) if c]
ff = lambda x: f"{x:9.2f}"
print("\n" + "#"*104)
print("# ER(4,2) ENERGY DECOMPOSITION  |  en_var = inter-agent variance of total_energy (LOWER = more equitable)")
print("# Q: is it the ENERGY TERM (beta) -- not just the amplifier (B) -- that reduces energy inequity?")
print("#"*104)
print(pd.DataFrame(cells)[['label','n','min_rows','env_mean','env_std','en_mean','en_var']].to_string(index=False, float_format=ff))

def get(key):
    for c in cells:
        if c['label'].startswith(key): return c
    return None
lio, be0, b0, full, eia = get('LIO'), get('{b=0}'), get('{B=0}'), get('FULL'), get('EIA')

for c in (be0, b0, full):
    if c and c['min_rows'] < EXPECT_ROWS:
        print(f"  [!] {c['label']} incomplete: {c['min_rows']} rows (< {EXPECT_ROWS}).")
    if c and c['env_mean'] < 20:
        print(f"  [!] {c['label']} env_mean={c['env_mean']:.1f} << baseline ~42 -> may not have converged; en_var unreliable.")

if lio and b0 and be0:
    base = lio['en_var']; red = lambda c: 100*(1 - c['en_var']/base) if base else float('nan')
    print(f"\n=== ATTRIBUTION (en_var reduction vs LIO = {base:.0f}) ===")
    print(f"  amplifier only {{b=0}} : en_var={be0['en_var']:8.1f}   ({red(be0):+5.1f} %)")
    print(f"  energy    only {{B=0}} : en_var={b0['en_var']:8.1f}   ({red(b0):+5.1f} %)   <- isolates the ENERGY TERM")
    if full: print(f"  both (FULL)          : en_var={full['en_var']:8.1f}   ({red(full):+5.1f} %)")
    if eia:  print(f"  EIA (no defense)     : en_var={eia['en_var']:8.1f}   ({red(eia):+5.1f} %)")
    rb0, rbe0 = red(b0), red(be0)
    print("\n=== SUGGESTED VERDICT (confirm with 10 seeds before claiming) ===")
    if rb0 >= 10 and (rb0 - rbe0) >= 8:
        print("  >>> ENERGY STORY HOLDS: energy term drives it (energy-only clearly < LIO AND < amplifier-only).")
        print("      -> proceed to 10-seed confirmation.")
    elif rb0 >= 10 and rbe0 >= 10:
        print("  ~ JOINT EFFECT: both terms lower en_var; can't credit beta alone. Frame jointly; still run 10 seeds.")
    elif max(rb0, rbe0) < 10:
        print("  !!! ENERGY TERM WEAK: neither isolated term is clearly < LIO. Full gain may be 2-seed noise.")
        print("      Do NOT launch the full sweep on the energy headline -> lean on attack-char + Prop 2 + hardware; re-discuss.")
    else:
        print("  ? mixed -- decide from the table.")
    print("\n  NOTE: 2 seeds = PRELIMINARY. This picks the direction, not the final claim.")
