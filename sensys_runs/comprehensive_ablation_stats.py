#!/usr/bin/env python3
# comprehensive_ablation_stats.py  (definitive: provenance-audited, no-EIA, equivalence verdict)
# ---------------------------------------------------------------------------
# Rigorous multi-statistic audit of whether the reward-fairness target F_T is
# inert on ER(N,M), now that the matched no-EIA ablation exists (10 seeds each):
#
#   LIO            nano_er{1..10}/er_lio_{N}_{M}                (baseline)
#   FULL           er/sweep_er_refine_{N}_{M}_s*   (beta=beta_full, B=B_full)
#   energy_only    er/probe_er_{N}_{M}_B0_beta{beta_full}_s*    (beta=beta_full, B=0)
#   fairness_only  er/probe_er_{N}_{M}_B{B_full}_beta0_s*       (beta=0,        B=B_full)
#
# Config (beta,B,n_episodes,fairness_clip,git_commit) is read from each run's
# provenance.json (authoritative), FALLING BACK to the directory name. The FULL
# operating point (beta_full,B_full) is auto-detected from the FULL runs.
#
# Four per-run scalars from the last TAIL eval rows (1-indexed agent cols A1.., A2..):
#   en_mean        mean over agents of A{i}_total_energy   (sustainability)
#   env_std_agents std  over agents of A{i}_reward_env     (= sigma_agent; F_T = -Var targets this)
#   en_std_agents  std  over agents of A{i}_total_energy   (energy spread)
#   env_mean       mean over agents of A{i}_reward_env     (task performance)
#
# Run: $HOME/miniconda3/envs/LIO_tecs/bin/python comprehensive_ablation_stats.py
# ---------------------------------------------------------------------------
import os, re, glob, json
import numpy as np
import pandas as pd
from collections import Counter

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ============================== CONFIG ==============================
RESULTS_ROOT = os.path.expanduser("~/greedy_agent/lio/results")
TAIL        = 5
SIZES       = [(4, 2)]        # add (3, 1) if that ablation also exists
EXCLUDE_EIA = True
BETA_FULL   = None            # None -> auto-detect from FULL runs (else force, e.g. 1.0)
B_FULL      = None            # None -> auto-detect            (else force, e.g. 0.2)
BOOT        = 10000
SEED        = 0
EQUIV_DELTA = 0.2             # equivalence margin = EQUIV_DELTA * pooled_SD (small Cohen's d)
# ===================================================================

NUM = r"[-+]?\d*\.?\d+"
rng = np.random.default_rng(SEED)
CANON = ["LIO", "FULL", "energy_only", "fairness_only"]

# F_T-isolating contrasts get an EQUIV verdict; energy ones report the effect.
CONTRASTS = [
    ("energy_only",   "FULL",          "F_T",      "add F_T on top of energy term  (expect NO change -> inert)"),
    ("LIO",           "fairness_only", "F_T",      "F_T alone vs baseline          (expect NO change -> inert)"),
    ("fairness_only", "FULL",          "energy",   "add energy term on top of F_T  (expect change -> energy works)"),
    ("LIO",           "energy_only",   "energy",   "energy term vs baseline        (expect change)"),
    ("LIO",           "FULL",          "combined", "full REFiNE vs baseline"),
]
METRICS = [
    ("en_mean",        "per-agent mean energy"),
    ("env_std_agents", "inter-agent reward_env std = sigma_agent  [F_T's OWN target]"),
    ("en_std_agents",  "inter-agent energy std"),
    ("env_mean",       "per-agent mean reward_env (task perf.)"),
]


def find_logs(root):
    return sorted(glob.glob(os.path.join(root, "**", "log.csv"), recursive=True))


def in_size(rel, N, M):
    return re.search(rf"[_/]{N}[_-]{M}(?=[_/.]|$)", rel) is not None


def name_meta(path):
    rel = os.path.relpath(path, RESULTS_ROOT); low = rel.lower()
    m = {"rel": rel, "method": None, "beta": None, "B": None, "eia": False, "seed": None}
    m["eia"] = ("eia" in low) or (re.search(rf"_w{NUM}-{NUM}", low) is not None)
    mB = re.search(rf"[_/]B({NUM})(?=[_/.]|$)", rel)            # uppercase fairness weight
    if mB: m["B"] = float(mB.group(1))
    mbeta = re.search(rf"[_/]beta({NUM})(?=[_/.]|$)", rel)
    if mbeta:
        m["beta"] = float(mbeta.group(1))
    else:
        mb = re.search(rf"[_/]b({NUM})(?=[_/.]|$)", rel)        # short lowercase form
        if mb: m["beta"] = float(mb.group(1))
    if ("refine" in low) or (m["beta"] is not None) or (m["B"] is not None):
        m["method"] = "refine"
    elif "lio" in low:
        m["method"] = "lio"
    ms = re.search(rf"[_/]s(\d+)(?=[_/.]|$)", low)
    if ms:
        m["seed"] = int(ms.group(1))
    elif re.search(r"nano_er(\d+)", low):
        m["seed"] = int(re.search(r"nano_er(\d+)", low).group(1))
    return m


def read_provenance(run_dir):
    p = os.path.join(run_dir, "provenance.json")
    out = {"beta": None, "B": None, "nep": None, "clip": None, "commit": None, "has": False}
    if os.path.isfile(p):
        try:
            d = json.load(open(p))
            out.update(beta=d.get("energy_weight"), B=d.get("fairness_mult"),
                       nep=d.get("n_episodes"), clip=d.get("fairness_clip"),
                       commit=((d.get("git_commit") or "")[:8] or None), has=True)
        except Exception:
            pass
    return out


def per_run_metrics(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"read-fail:{type(e).__name__}"
    if len(df) == 0:
        return None, "empty"
    env_cols = sorted([c for c in df.columns if re.fullmatch(r"A\d+_reward_env", c)],
                      key=lambda c: int(re.findall(r"\d+", c)[0]))
    en_cols = sorted([c for c in df.columns if re.fullmatch(r"A\d+_total_energy", c)],
                     key=lambda c: int(re.findall(r"\d+", c)[0]))
    if not env_cols or not en_cols:
        return None, "missing-cols"
    t = df.tail(TAIL)
    env = t[env_cols].mean(axis=0).to_numpy(float)
    en = t[en_cols].mean(axis=0).to_numpy(float)
    if np.isnan(env).any() or np.isnan(en).any():
        return None, "nan"
    return {"en_mean": float(en.mean()), "en_std_agents": float(en.std(ddof=0)),
            "env_mean": float(env.mean()), "env_std_agents": float(env.std(ddof=0)),
            "n_rows": int(len(df)), "n_agents": len(env_cols)}, "ok"


def near(x, y, tol=1e-6):
    if x is None or y is None:
        return False
    return abs(x - y) <= tol + 1e-6 * abs(y)


def canon(method, beta, B, bf, Bf):
    if method == "lio":
        return "LIO"
    if method == "refine":
        if beta is None or B is None:
            return "REFiNE(unknown)"
        if near(beta, bf) and near(B, Bf): return "FULL"
        if near(beta, bf) and near(B, 0):  return "energy_only"
        if near(beta, 0) and near(B, Bf):  return "fairness_only"
        if near(beta, 0) and near(B, 0):   return "LIO_equiv"
        return f"other(b={beta:g},B={B:g})"
    return "OTHER"


def dist(vals):
    x = np.asarray([v for v in vals if v is not None and not np.isnan(v)], float)
    if x.size == 0:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    return dict(n=x.size, mean=x.mean(), median=float(np.median(x)),
                std=(float(x.std(ddof=1)) if x.size > 1 else 0.0),
                min=x.min(), q1=q1, q3=q3, max=x.max(), iqr=q3 - q1)


def boot_ci(a, b, paired):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan)
    if paired:
        d = b - a
        bs = np.array([rng.choice(d, d.size, True).mean() for _ in range(BOOT)])
    else:
        bs = np.array([rng.choice(b, b.size, True).mean() - rng.choice(a, a.size, True).mean()
                       for _ in range(BOOT)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def hr(c="="):
    print(c * 100)


def main():
    print(f"RESULTS_ROOT = {RESULTS_ROOT}")
    print(f"TAIL={TAIL}  EXCLUDE_EIA={EXCLUDE_EIA}  EQUIV_DELTA={EQUIV_DELTA}*SD  BOOT={BOOT}  scipy={HAVE_SCIPY}")
    logs = find_logs(RESULTS_ROOT)
    print(f"found {len(logs)} log.csv\n")

    for (N, M) in SIZES:
        hr(); print(f"ER({N},{M})"); hr()

        runs = []
        for p in logs:
            nm = name_meta(p)
            if not in_size(nm["rel"], N, M):
                continue
            if EXCLUDE_EIA and nm["eia"]:
                continue
            prov = read_provenance(os.path.dirname(p))
            beta = prov["beta"] if prov["beta"] is not None else nm["beta"]
            B = prov["B"] if prov["B"] is not None else nm["B"]
            mets, status = per_run_metrics(p)
            runs.append({"rel": nm["rel"], "method": nm["method"], "seed": nm["seed"],
                         "beta": beta, "B": B, "nep": prov["nep"], "clip": prov["clip"],
                         "commit": prov["commit"], "prov": prov["has"],
                         "mets": mets, "status": status})

        # ---- auto-detect FULL operating point ----
        bf, Bf = BETA_FULL, B_FULL
        if bf is None or Bf is None:
            cand = [(r["beta"], r["B"]) for r in runs
                    if r["method"] == "refine" and r["beta"] and r["B"]
                    and r["beta"] > 0 and r["B"] > 0]
            if cand:
                (abf, aBf), _ = Counter(cand).most_common(1)[0]
                bf = bf if bf is not None else abf
                Bf = Bf if Bf is not None else aBf
            else:
                bf = bf or 1.0; Bf = Bf or 0.2
        autotag = "[auto-detected]" if (BETA_FULL is None or B_FULL is None) else "[forced]"
        print(f"FULL operating point (beta_full, B_full) = ({bf:g}, {Bf:g})  {autotag}\n")

        for r in runs:
            r["group"] = canon(r["method"], r["beta"], r["B"], bf, Bf)

        # ---------------- INVENTORY ----------------
        print("INVENTORY")
        print(f"{'group':<16}{'beta':>6}{'B':>6}{'seed':>5}{'ep':>7}{'clip':>5}{'commit':>10}{'rows':>6}{'prov':>5}  status      rel")
        for r in sorted(runs, key=lambda r: (r["group"], (r["seed"] if r["seed"] is not None else -1))):
            b = "" if r["beta"] is None else f"{r['beta']:g}"
            B = "" if r["B"] is None else f"{r['B']:g}"
            sd = "" if r["seed"] is None else str(r["seed"])
            ep = "" if r["nep"] is None else str(r["nep"])
            cl = "" if r["clip"] is None else ("1" if r["clip"] else "0")
            cm = r["commit"] or ""
            rw = "" if r["mets"] is None else str(r["mets"]["n_rows"])
            print(f"{r['group']:<16}{b:>6}{B:>6}{sd:>5}{ep:>7}{cl:>5}{cm:>10}{rw:>6}{('Y' if r['prov'] else 'n'):>5}  "
                  f"{r['status']:<11} {r['rel']}")
        print()

        # ---------------- CONFIG AUDIT ----------------
        hr("-"); print("CONFIG AUDIT  (the comparison is only valid if it is apples-to-apples)"); hr("-")

        def cfgset(group, keys):
            return {tuple(r[k] for k in keys)
                    for r in runs if r["group"] == group and r["mets"] is not None}

        for g in CANON:
            members = [r for r in runs if r["group"] == g and r["mets"] is not None]
            if not members:
                print(f"  {g:<14} : MISSING (no usable runs)")
                continue
            seeds = sorted(r["seed"] for r in members)
            cfgs = cfgset(g, ["beta", "B", "nep", "clip", "commit"])
            flag = "" if len(cfgs) == 1 else "   [!] MIXED CONFIGS within group"
            c = sorted(cfgs)[0]
            print(f"  {g:<14} n={len(members):2d} seeds={seeds}")
            print(f"  {'':<14}   beta={c[0]} B={c[1]} ep={c[2]} clip={c[3]} commit={c[4]}{flag}")

        def cross_ok(gA, gB, keys):
            A = cfgset(gA, keys); B = cfgset(gB, keys)
            if not A or not B:
                return None
            return A == B
        print("  cross-group consistency (an ablation is valid only if the two groups differ solely in the ablated knob):")
        for gA, gB, keys, note in [
            ("energy_only",   "FULL", ["nep", "clip", "commit"], "match ep/clip/commit; differ only in B"),
            ("fairness_only", "FULL", ["nep", "clip", "commit"], "match ep/clip/commit; differ only in beta"),
            ("LIO",           "FULL", ["nep"],                   "match episodes (LIO has no clip/beta/B)"),
        ]:
            ok = cross_ok(gA, gB, keys)
            tag = "OK" if ok else ("[!] MISMATCH" if ok is False else "n/a (missing)")
            print(f"    {gA:<14} vs {gB:<14} : {tag:<14} ({note})")
        print("  (a MISMATCH caused only by missing provenance is benign if the INVENTORY 'rows' column")
        print("   matches across groups: equal row counts => same episode count.)")
        print()

        # ---- gather metric vectors keyed by seed ----
        groups = {g: {} for g in CANON}
        for r in runs:
            if r["group"] in groups and r["mets"] is not None:
                groups[r["group"]][r["seed"]] = r["mets"]

        # ---------------- PER-CONDITION DISTRIBUTIONS ----------------
        for mkey, mdesc in METRICS:
            hr("-"); print(f"[{mkey}]  {mdesc}"); hr("-")
            print(f"{'cond':<15}{'n':>3}{'mean':>10}{'median':>10}{'std':>9}{'min':>9}{'Q1':>9}{'Q3':>9}{'max':>9}{'IQR':>9}")
            for g in CANON:
                d = dist([m[mkey] for m in groups[g].values()])
                if d is None:
                    print(f"{g:<15}{'--':>3}"); continue
                print(f"{g:<15}{d['n']:>3}{d['mean']:>10.3f}{d['median']:>10.3f}{d['std']:>9.3f}"
                      f"{d['min']:>9.3f}{d['q1']:>9.3f}{d['q3']:>9.3f}{d['max']:>9.3f}{d['iqr']:>9.3f}")
            print()

        # ---------------- PER-SEED RAW DUMP ----------------
        hr("-"); print("PER-SEED RAW VALUES (outliers cannot hide here)"); hr("-")
        seeds = sorted({s for g in CANON for s in groups[g]})
        for mkey, _ in METRICS:
            print(f"\n  metric = {mkey}")
            print(f"  {'seed':<6}" + "".join(f"{g:>16}" for g in CANON))
            for s in seeds:
                print(f"  {str(s):<6}" + "".join(
                    (f"{groups[g][s][mkey]:>16.3f}" if s in groups[g] else f"{'--':>16}") for g in CANON))
        print()

        # ---------------- CONTRASTS ----------------
        for gA, gB, kind, desc in CONTRASTS:
            hr(); print(f"CONTRAST [{kind}]: {gA} -> {gB}   {desc}"); hr()
            A, B = groups.get(gA, {}), groups.get(gB, {})
            if not A or not B:
                print(f"  skip: missing {[g for g in (gA, gB) if not groups.get(g)]}\n"); continue
            common = sorted(set(A) & set(B))
            print(f"  n({gA})={len(A)} seeds={sorted(A)} | n({gB})={len(B)} seeds={sorted(B)} | paired n={len(common)}")
            for mkey, _ in METRICS:
                a = [A[s][mkey] for s in A]; b = [B[s][mkey] for s in B]
                lo, hi = boot_ci(a, b, paired=False)
                line = (f"  [{mkey:<15}] unpaired med_gap={np.median(b) - np.median(a):+.3f} "
                        f"mean_gap={np.mean(b) - np.mean(a):+.3f} CI=[{lo:+.3f},{hi:+.3f}]")
                if HAVE_SCIPY:
                    try:
                        _, p = sps.mannwhitneyu(a, b, alternative="two-sided"); line += f" MWU_p={p:.2g}"
                    except Exception:
                        line += " MWU_p=NA"
                print(line)
                if len(common) >= 2:
                    da = np.array([A[s][mkey] for s in common]); db = np.array([B[s][mkey] for s in common])
                    d = db - da
                    plo, phi = boot_ci(da, db, paired=True)
                    sd_pool = float(np.std(np.concatenate([da, db]), ddof=1))
                    eps = max(EQUIV_DELTA * sd_pool, 1e-6)          # absolute floor absorbs float dust
                    sd_d = float(d.std(ddof=1))
                    if sd_d <= 1e-12:
                        dz = 0.0 if abs(d.mean()) <= 1e-12 else float("inf")
                    else:
                        dz = d.mean() / sd_d
                    dz_str = "huge" if (np.isinf(dz) or abs(dz) >= 100) else f"{dz:+.2f}"
                    pl = (f"       paired(n={len(common)}) diff={d.mean():+.3f} med={np.median(d):+.3f} "
                          f"rng=[{d.min():+.3f},{d.max():+.3f}] CI=[{plo:+.3f},{phi:+.3f}] dz={dz_str}")
                    if HAVE_SCIPY and np.any(np.abs(d) > 1e-12):
                        try:
                            _, pw = sps.wilcoxon(da, db); pl += f" Wilx_p={pw:.2g}"
                        except Exception:
                            pl += " Wilx_p=NA"
                    if kind == "F_T":
                        equiv = (not np.isnan(plo)) and (plo >= -eps) and (phi <= eps)
                        pl += f"   | EQUIV(+/-{eps:.4f}) = {'YES' if equiv else 'NO'}"
                    print(pl)
            if common:
                print(f"\n  per-seed paired diff ({gB} - {gA}):")
                print(f"  {'seed':<6}" + "".join(f"{mk:>18}" for mk, _ in METRICS))
                for s in common:
                    print(f"  {str(s):<6}" + "".join(f"{(B[s][mk] - A[s][mk]):>18.3f}" for mk, _ in METRICS))
            print()

        # ---------------- OTHER REFiNE CONFIGS ----------------
        others = sorted({r["group"] for r in runs
                         if r["group"].startswith("other") or r["group"] in ("LIO_equiv", "REFiNE(unknown)")})
        if others:
            hr("-"); print("OTHER REFiNE CONFIGS (outside the FULL-matched ablation; reference only)"); hr("-")
            for g in others:
                mem = {r["seed"]: r["mets"] for r in runs if r["group"] == g and r["mets"] is not None}
                de = dist([m["en_mean"] for m in mem.values()])
                dv = dist([m["env_std_agents"] for m in mem.values()])
                if de:
                    print(f"  {g:<24} n={de['n']} | en_mean med={de['median']:.2f} | sigma_agent med={dv['median']:.2f}")
            print()

        hr(); print("HOW TO READ / VERDICT"); hr()
        print(reading_guide(bf, Bf))


def reading_guide(bf, Bf):
    return (
f"  Claim under audit: the reward-fairness target F_T is INERT (the characterized-negative).\n"
f"  Judge on ALL metrics, foremost sigma_agent (env_std_agents) -- the quantity F_T = -Var\n"
f"  literally minimizes. The two F_T-isolating contrasts are decisive:\n"
f"     energy_only(b={bf:g},B=0) -> FULL(b={bf:g},B={Bf:g})   adds F_T on top of the energy term\n"
f"     LIO -> fairness_only(b=0,B={Bf:g})                       adds F_T alone\n"
f"  * Trust the PAIRED rows (same seed cancels seed-to-seed noise). 'INERT' is SUPPORTED when,\n"
f"    for these two contrasts and every metric, EQUIV=YES (paired 95% CI within a small-effect\n"
f"    margin) AND |dz| is tiny AND no single seed shows a large diff in the per-seed dump.\n"
f"    In particular, EQUIV=YES on sigma_agent means F_T does not move even its own objective --\n"
f"    the strongest possible evidence of inertness.\n"
f"  * If any F_T contrast/metric is EQUIV=NO, or a seed shows a large diff, SCOPE the claim\n"
f"    ('inert in median/energy but shifts sigma_agent by X on k/N seeds') -- do not assert it\n"
f"    does nothing.\n"
f"  * The 'energy' contrasts (fairness_only->FULL, LIO->energy_only) SHOULD show a real effect\n"
f"    (energy reduction): that is the positive result the energy regularizer earns.\n"
f"  * CONFIG AUDIT must be clean: an F_T contrast is only valid if its two groups match on\n"
f"    episodes/clip/commit and differ solely in the ablated knob. Any [!] there invalidates the\n"
f"    comparison -- fix it before trusting any number below.\n")


if __name__ == "__main__":
    main()