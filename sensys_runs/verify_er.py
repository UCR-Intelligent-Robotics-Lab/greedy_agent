#!/usr/bin/env python3
"""Health + sanity report for completed ER baseline runs (read-only)."""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_GLOB = os.path.join(REPO_ROOT, "lio", "results", "nano_er*", "*", "log.csv")
EXPECTED_ROWS = 50
SEEDS = list(range(1, 11))

LIO_DIRS = ["er_lio_3_1", "er_lio_4_2", "er_lio_6_4"]
EIA_DIRS = [
    "er_eia_3_1_w2.0-0.2",
    "er_eia_4_2_w2.0-0.2",
    "er_eia_6_4_w2.0-0.2",
    "er_eia_4_2_w1.1-0.9",
    "er_eia_4_2_w1.5-1.5",
]
ALL_DIRS = LIO_DIRS + EIA_DIRS

CONVERGENCE_PCT = 0.25  # final window within 25% of previous window
COLLAPSE_THRESH = 1.0  # mean |env reward| below this => collapsed


@dataclass
class RunRecord:
    seed_dir: str
    run_dir: str
    method: str
    n_agents: int
    min_at_lever: int
    weights: str | None
    path: str
    n_rows: int
    expected: int
    status: str
    nan_inf: bool
    n_agents_found: int
    final_mean_env: float | None
    final_fairness_var: float | None
    final_fairness_sigma: float | None
    converged: bool
    suspicious_reason: str = ""


def parse_dir(run_dir: str) -> tuple[str, int, int, str | None]:
    if run_dir.startswith("er_lio_"):
        parts = run_dir.replace("er_lio_", "").split("_")
        return "lio", int(parts[0]), int(parts[1]), None
    m = re.match(r"er_eia_(\d+)_(\d+)_w([\d.]+)-([\d.]+)", run_dir)
    if m:
        w = f"{m.group(3)}-{m.group(4)}"
        return "eia", int(m.group(1)), int(m.group(2)), w
    raise ValueError(f"unrecognized dir: {run_dir}")


def discover_agent_env_cols(header: list[str]) -> list[str]:
    cols = [c for c in header if re.match(r"A\d+_reward_env$", c)]
    cols.sort(key=lambda c: int(re.search(r"A(\d+)", c).group(1)))
    return cols


def discover_agent_energy_cols(header: list[str]) -> list[str]:
    return sorted(
        [c for c in header if re.match(r"A\d+_total_energy$", c)],
        key=lambda c: int(re.search(r"A(\d+)", c).group(1)),
    )


def load_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def rows_to_float_matrix(rows: list[dict], cols: list[str]) -> np.ndarray:
    data = []
    for r in rows:
        data.append([float(r[c]) for c in cols])
    return np.array(data, dtype=float)


def print_schema(lio_path: str, eia_path: str) -> tuple[list[str], list[str]]:
    print("=" * 70)
    print("STEP 0 — SCHEMA DISCOVERY")
    print("=" * 70)
    for label, path in [("LIO", lio_path), ("EIA", eia_path)]:
        header, rows = load_csv(path)
        env_cols = discover_agent_env_cols(header)
        energy_cols = discover_agent_energy_cols(header)
        idx_cols = [c for c in ("episode", "step_train", "step") if c in header]
        print(f"\n--- {label}: {path}")
        print(f"Columns ({len(header)}): {', '.join(header)}")
        print(f"Episode index: {idx_cols}")
        print(f"Per-agent env reward: {env_cols}")
        print(f"Per-agent energy: {energy_cols}")
        print(f"NOTE: A*_teamwork_fairness is UNRELIABLE — not used.")
        print("First 2 data rows:")
        for r in rows[:2]:
            print("  " + ",".join(str(r[c]) for c in header[:6]) + ", ...")
        print("Last 2 data rows:")
        for r in rows[-2:]:
            print("  " + ",".join(str(r[c]) for c in header[:6]) + ", ...")
    return discover_agent_env_cols(load_csv(lio_path)[0]), discover_agent_energy_cols(
        load_csv(lio_path)[0]
    )


def analyze_run(seed_dir: str, run_dir: str, path: str, env_cols: list[str]) -> RunRecord:
    method, n_agents, min_at_lever, weights = parse_dir(run_dir)
    rec = RunRecord(
        seed_dir=seed_dir,
        run_dir=run_dir,
        method=method,
        n_agents=n_agents,
        min_at_lever=min_at_lever,
        weights=weights,
        path=path,
        n_rows=0,
        expected=EXPECTED_ROWS,
        status="MISSING",
        nan_inf=False,
        n_agents_found=0,
        final_mean_env=None,
        final_fairness_var=None,
        final_fairness_sigma=None,
        converged=False,
    )
    if not os.path.isfile(path):
        rec.suspicious_reason = "missing file"
        return rec

    header, rows = load_csv(path)
    env_cols_run = discover_agent_env_cols(header)
    rec.n_agents_found = len(env_cols_run)
    rec.n_rows = len(rows)

    if rec.n_rows == EXPECTED_ROWS:
        rec.status = "OK"
    elif rec.n_rows == 0:
        rec.status = "MISSING"
    else:
        rec.status = "SHORT"

    if not rows or not env_cols_run:
        rec.suspicious_reason = "no data or no env reward cols"
        return rec

    mat = rows_to_float_matrix(rows, env_cols_run)
    rec.nan_inf = bool(np.any(~np.isfinite(mat)))

    final_w = mat[-5:] if len(mat) >= 5 else mat
    prev_w = mat[-10:-5] if len(mat) >= 10 else mat[: max(1, len(mat) // 2)]

    agent_means_final = np.mean(final_w, axis=0)
    rec.final_mean_env = float(np.mean(agent_means_final))
    rec.final_fairness_var = float(np.var(agent_means_final))
    rec.final_fairness_sigma = float(np.std(agent_means_final))

    prev_agent_means = np.mean(prev_w, axis=0)
    prev_mean = float(np.mean(prev_agent_means))
    final_mean = rec.final_mean_env

    reasons = []
    if rec.nan_inf:
        reasons.append("NaN/inf")
    if rec.status != "OK":
        reasons.append(rec.status)
    if abs(final_mean) < COLLAPSE_THRESH and abs(prev_mean) < COLLAPSE_THRESH:
        reasons.append("collapsed~0")
    if len(mat) >= 10 and prev_mean != 0:
        rel_change = abs(final_mean - prev_mean) / max(abs(prev_mean), 1e-8)
        if rel_change > CONVERGENCE_PCT:
            reasons.append(f"unstable({rel_change:.0%})")
    elif len(mat) >= 10 and prev_mean == 0 and abs(final_mean) > COLLAPSE_THRESH:
        reasons.append("jump from ~0")

    rec.converged = len(reasons) == 0 or (len(reasons) == 1 and reasons[0] == rec.status and rec.status == "OK")
    rec.converged = rec.status == "OK" and not rec.nan_inf and not any(
        r.startswith(("collapsed", "unstable", "jump")) for r in reasons
    )
    if not rec.converged:
        rec.suspicious_reason = "; ".join(reasons) if reasons else "unknown"
    return rec


def group_key(rec: RunRecord) -> tuple:
    return (rec.method, rec.n_agents, rec.min_at_lever, rec.weights or "")


def main() -> int:
    out_runs = os.path.join(os.path.dirname(__file__), "verify_er_runs.csv")
    out_summary = os.path.join(os.path.dirname(__file__), "verify_er_summary.csv")

    lio_sample = os.path.join(REPO_ROOT, "lio/results/nano_er1/er_lio_3_1/log.csv")
    eia_sample = os.path.join(REPO_ROOT, "lio/results/nano_er1/er_eia_4_2_w2.0-0.2/log.csv")
    env_cols, _ = print_schema(lio_sample, eia_sample)

    print("\n" + "=" * 70)
    print("STEP 1 — COMPLETENESS & INTEGRITY")
    print("=" * 70)

    records: list[RunRecord] = []
    found_paths: set[str] = set()

    for seed in SEEDS:
        seed_dir = f"nano_er{seed}"
        for run_dir in ALL_DIRS:
            path = os.path.join(REPO_ROOT, "lio", "results", seed_dir, run_dir, "log.csv")
            rec = analyze_run(seed_dir, run_dir, path, env_cols)
            records.append(rec)
            if os.path.isfile(path):
                found_paths.add(path)

    # Also scan any extra ER logs not in expected list
    for path in glob.glob(RESULTS_GLOB):
        parts = path.split(os.sep)
        if len(parts) >= 2:
            run_dir = parts[-2]
            seed_dir = parts[-3]
            if run_dir in ALL_DIRS and path not in found_paths:
                records.append(analyze_run(seed_dir, run_dir, path, env_cols))

    ok = sum(1 for r in records if r.status == "OK")
    short = sum(1 for r in records if r.status == "SHORT")
    missing = sum(1 for r in records if r.status == "MISSING")
    nan_runs = [r for r in records if r.nan_inf]

    print(f"Expected matrix: {len(ALL_DIRS)} dirs × {len(SEEDS)} seeds = {len(ALL_DIRS)*len(SEEDS)}")
    print(f"OK={ok}  SHORT={short}  MISSING={missing}  NaN/inf={len(nan_runs)}")

    group_counts: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in records:
        group_counts[group_key(r)].append(r)

    print("\nSeeds per group:")
    for g in sorted(group_counts.keys()):
        cnt = sum(1 for r in group_counts[g] if r.status == "OK")
        flag = "" if cnt == 10 else f"  *** GROUP <10 ({cnt}/10) ***"
        print(f"  {g}: {cnt}/10 OK{flag}")

    print("\n" + "=" * 70)
    print("STEP 2 — CONVERGENCE")
    print("=" * 70)
    suspicious = [r for r in records if not r.converged and r.status != "MISSING"]
    converged = [r for r in records if r.converged]
    print(f"Converged: {len(converged)}/{ok} OK runs")
    print(f"Suspicious: {len(suspicious)}")
    if suspicious:
        print("Suspicious runs:")
        for r in suspicious[:20]:
            print(f"  {r.seed_dir}/{r.run_dir}: {r.suspicious_reason}")
        if len(suspicious) > 20:
            print(f"  ... and {len(suspicious)-20} more")

    print("\n" + "=" * 70)
    print("STEP 3 — LIO vs EIA SANITY")
    print("=" * 70)

    summary_rows: list[dict[str, Any]] = []

    def agg_group(method: str, n: int, m: int, weights: str | None) -> dict | None:
        key = (method, n, m, weights or "")
        rs = [r for r in group_counts.get(key, []) if r.status == "OK" and r.final_mean_env is not None]
        if not rs:
            return None
        means = [r.final_mean_env for r in rs]
        vars_ = [r.final_fairness_var for r in rs]
        sigmas = [r.final_fairness_sigma for r in rs]
        return {
            "method": method,
            "n_agents": n,
            "min_at_lever": m,
            "weights": weights or "",
            "n_seeds": len(rs),
            "mean_env": float(np.mean(means)),
            "std_env": float(np.std(means)),
            "mean_fairness_var": float(np.mean(vars_)),
            "std_fairness_var": float(np.std(vars_)),
            "mean_fairness_sigma": float(np.mean(sigmas)),
            "std_fairness_sigma": float(np.std(sigmas)),
        }

    sizes = [(3, 1), (4, 2), (6, 4)]
    eia_fairness_worse_at: list[str] = []
    lio_eia_pairs: list[tuple] = []

    print("\nPer-size LIO vs EIA (default w=2.0-0.2):")
    for n, m in sizes:
        lio = agg_group("lio", n, m, None)
        eia = agg_group("eia", n, m, "2.0-0.2")
        if lio and eia:
            worse = eia["mean_fairness_var"] > lio["mean_fairness_var"]
            lio_eia_pairs.append((n, m, worse, lio, eia))
            tag = "EIA worse fairness" if worse else "LIO worse or equal"
            if worse:
                eia_fairness_worse_at.append(f"({n},{m})")
            print(
                f"  ER({n},{m}): LIO env={lio['mean_env']:.1f}±{lio['std_env']:.1f} "
                f"Var={lio['mean_fairness_var']:.1f}±{lio['std_fairness_var']:.1f} | "
                f"EIA env={eia['mean_env']:.1f}±{eia['std_env']:.1f} "
                f"Var={eia['mean_fairness_var']:.1f}±{eia['std_fairness_var']:.1f}  [{tag}]"
            )
            summary_rows.append({**lio, "group": f"lio_{n}_{m}"})
            summary_rows.append({**eia, "group": f"eia_{n}_{m}_w2.0-0.2"})

    print("\nER(4,2) weight sweep (fairness Var, lower=milder):")
    sweep_weights = ["2.0-0.2", "1.1-0.9", "1.5-1.5"]
    sweep_stats = []
    for w in sweep_weights:
        s = agg_group("eia", 4, 2, w)
        if s:
            sweep_stats.append((w, s))
            print(
                f"  w={w}: env={s['mean_env']:.1f}±{s['std_env']:.1f} "
                f"Var={s['mean_fairness_var']:.1f}±{s['std_fairness_var']:.1f} "
                f"sigma={s['mean_fairness_sigma']:.1f}"
            )
            summary_rows.append({**s, "group": f"eia_4_2_w{w}"})

    # Weight ordering verdict
    weight_verdict = "unexpected"
    if len(sweep_stats) == 3:
        wmap = {w: s["mean_fairness_var"] for w, s in sweep_stats}
        v_strong = wmap["2.0-0.2"]
        v_mild = wmap["1.1-0.9"]
        v_sym = wmap["1.5-1.5"]
        if v_strong > v_mild and v_strong >= v_sym:
            weight_verdict = "as-expected"
        elif v_strong > v_mild or v_strong > v_sym:
            weight_verdict = "weak"
        print(
            f"\nWeight ordering: strong(2.0-0.2)={v_strong:.1f}, "
            f"mild(1.1-0.9)={v_mild:.1f}, sym(1.5-1.5)={v_sym:.1f}  => [{weight_verdict}]"
        )

    lio_eia_verdict = "unexpected"
    if lio_eia_pairs:
        n_worse = sum(1 for _, _, w, _, _ in lio_eia_pairs if w)
        if n_worse == len(lio_eia_pairs):
            lio_eia_verdict = "as-expected"
        elif n_worse >= 1:
            lio_eia_verdict = "weak"
        print(f"\nEIA worse fairness than LIO at {eia_fairness_worse_at or 'none'}  => [{lio_eia_verdict}]")

    # Write CSVs
    run_fields = [
        "seed_dir", "run_dir", "method", "n_agents", "min_at_lever", "weights",
        "n_rows", "expected", "status", "nan_inf", "n_agents_found",
        "final_mean_env", "final_fairness_var", "final_fairness_sigma",
        "converged", "suspicious_reason", "path",
    ]
    with open(out_runs, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x.run_dir, x.seed_dir)):
            w.writerow({k: getattr(r, k) for k in run_fields})

    sum_fields = [
        "group", "method", "n_agents", "min_at_lever", "weights", "n_seeds",
        "mean_env", "std_env", "mean_fairness_var", "std_fairness_var",
        "mean_fairness_sigma", "std_fairness_sigma",
    ]
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in sum_fields})

    print("\n" + "=" * 70)
    print("TOP-LINE SUMMARY")
    print("=" * 70)
    print(f"Complete: {ok}/80 runs OK ({short} SHORT, {missing} MISSING)")
    print(f"Suspicious (convergence): {len(suspicious)}")
    print(f"EIA > LIO fairness (higher Var) at sizes: {eia_fairness_worse_at or 'none'} [{lio_eia_verdict}]")
    print(f"Weight sweep ER(4,2) ordering: [{weight_verdict}]")
    print(f"\nWrote: {out_runs}")
    print(f"Wrote: {out_summary}")

    return 0 if ok == 80 and missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
