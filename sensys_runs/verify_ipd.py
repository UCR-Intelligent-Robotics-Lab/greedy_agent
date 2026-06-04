#!/usr/bin/env python3
"""Health + sanity report for completed IPD baseline + REFiNE runs (read-only)."""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from verify_er import discover_agent_env_cols, load_csv, rows_to_float_matrix

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_GLOB = os.path.join(REPO_ROOT, "lio", "results", "nano_ipd*", "*", "log.csv")
EXPECTED_ROWS = 20
SEEDS = list(range(1, 11))
BASELINE_DIRS = ["ipd_lio", "ipd_eia"]
REFINE_DIRS = ["ipd_refine", "ipd_refine_eia", "ipd_refine_eia_B0"]
ALL_DIRS = BASELINE_DIRS + REFINE_DIRS

CONVERGENCE_PCT = 0.25
COLLAPSE_THRESH = 1.0
FINAL_WINDOW = 3
PREV_WINDOW = 3


@dataclass
class RunRecord:
    seed_dir: str
    run_dir: str
    method: str
    ablation: str
    path: str
    n_rows: int
    expected: int
    status: str
    nan_inf: bool
    final_mean_env: float | None
    final_reward_env_stdev: float | None
    final_coop_rate: float | None
    final_payoff_gap: float | None
    final_fairness_var: float | None
    converged: bool
    suspicious_reason: str = ""


def parse_dir(run_dir: str) -> tuple[str, str]:
    if run_dir == "ipd_lio":
        return "lio", ""
    if run_dir == "ipd_eia":
        return "eia", ""
    if run_dir == "ipd_refine":
        return "refine", ""
    if run_dir == "ipd_refine_eia":
        return "refine_eia", ""
    if run_dir == "ipd_refine_eia_B0":
        return "refine_eia", "B0"
    raise ValueError(f"unrecognized dir: {run_dir}")


def discover_coop_cols(header: list[str]) -> tuple[list[str], list[str]]:
    nc = sorted(
        [c for c in header if re.match(r"A\d+_n_c$", c)],
        key=lambda c: int(re.search(r"A(\d+)", c).group(1)),
    )
    nd = sorted(
        [c for c in header if re.match(r"A\d+_n_d$", c)],
        key=lambda c: int(re.search(r"A(\d+)", c).group(1)),
    )
    return nc, nd


def coop_rate_from_row(row: dict, nc_cols: list[str], nd_cols: list[str]) -> float | None:
    rates = []
    for nc, nd in zip(nc_cols, nd_cols):
        c, d = float(row[nc]), float(row[nd])
        total = c + d
        if total > 0:
            rates.append(c / total)
    return float(np.mean(rates)) if rates else None


def _sample_path(*candidates: str) -> str:
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def print_schema(lio_path: str, eia_path: str, refine_path: str) -> None:
    print("=" * 70)
    print("STEP 0 — SCHEMA DISCOVERY")
    print("=" * 70)
    for label, path in [
        ("LIO ipd_lio", lio_path),
        ("EIA ipd_eia", eia_path),
        ("REFiNE ipd_refine", refine_path),
    ]:
        if not os.path.isfile(path):
            print(f"\n--- {label}: {path} (missing, skipped)")
            continue
        header, rows = load_csv(path)
        env_cols = discover_agent_env_cols(header)
        nc, nd = discover_coop_cols(header)
        idx_cols = [c for c in ("episode", "step_train", "step") if c in header]
        print(f"\n--- {label}: {path}")
        print(f"Columns ({len(header)}): {', '.join(header)}")
        print(f"Episode index: {idx_cols}")
        print(f"Per-agent env reward: {env_cols}")
        print(f"Cooperation action counts: {nc} / defect: {nd}")
        print("Per-agent energy: (none in IPD logs)")
        print("NOTE: A*_teamwork_fairness is UNRELIABLE — not used.")
        if rows:
            print("First 2 data rows:")
            for r in rows[:2]:
                print("  " + ",".join(str(r[c]) for c in header[:6]) + ", ...")
            print("Last 2 data rows:")
            for r in rows[-2:]:
                print("  " + ",".join(str(r[c]) for c in header[:6]) + ", ...")


def analyze_run(
    seed_dir: str,
    run_dir: str,
    path: str,
    env_cols: list[str],
    nc_cols: list[str],
    nd_cols: list[str],
) -> RunRecord:
    method, ablation = parse_dir(run_dir)
    rec = RunRecord(
        seed_dir=seed_dir,
        run_dir=run_dir,
        method=method,
        ablation=ablation,
        path=path,
        n_rows=0,
        expected=EXPECTED_ROWS,
        status="MISSING",
        nan_inf=False,
        final_mean_env=None,
        final_reward_env_stdev=None,
        final_coop_rate=None,
        final_payoff_gap=None,
        final_fairness_var=None,
        converged=False,
    )
    if not os.path.isfile(path):
        rec.suspicious_reason = "missing file"
        return rec

    header, rows = load_csv(path)
    env_cols_run = discover_agent_env_cols(header)
    nc_run, nd_run = discover_coop_cols(header)
    if not nc_cols:
        nc_cols = nc_run
        nd_cols = nd_run

    rec.n_rows = len(rows)
    rec.status = (
        "OK"
        if rec.n_rows >= EXPECTED_ROWS
        else ("MISSING" if rec.n_rows == 0 else "SHORT")
    )

    if not rows or len(env_cols_run) < 2:
        rec.suspicious_reason = "no data or <2 agents"
        return rec

    env_mat = rows_to_float_matrix(rows, env_cols_run)
    rec.nan_inf = bool(np.any(~np.isfinite(env_mat)))

    n = len(rows)
    fw = env_mat[-FINAL_WINDOW:] if n >= FINAL_WINDOW else env_mat
    pw = (
        env_mat[-(FINAL_WINDOW + PREV_WINDOW) : -FINAL_WINDOW]
        if n >= FINAL_WINDOW + PREV_WINDOW
        else env_mat[: max(1, n // 2)]
    )

    agent_means_final = np.mean(fw, axis=0)
    rec.final_mean_env = float(np.mean(agent_means_final))
    rec.final_reward_env_stdev = float(np.std(agent_means_final))
    rec.final_payoff_gap = float(abs(agent_means_final[0] - agent_means_final[1]))
    rec.final_fairness_var = float(np.var(agent_means_final))

    coop_rates = []
    for r in rows[-FINAL_WINDOW:]:
        cr = coop_rate_from_row(r, nc_run, nd_run)
        if cr is not None:
            coop_rates.append(cr)
    rec.final_coop_rate = float(np.mean(coop_rates)) if coop_rates else None

    prev_agent_means = np.mean(pw, axis=0)
    prev_mean = float(np.mean(prev_agent_means))
    final_mean = rec.final_mean_env

    reasons = []
    if rec.nan_inf:
        reasons.append("NaN/inf")
    if rec.status != "OK":
        reasons.append(rec.status)
    if abs(final_mean) < COLLAPSE_THRESH and abs(prev_mean) < COLLAPSE_THRESH:
        reasons.append("collapsed~0")
    if n >= FINAL_WINDOW + PREV_WINDOW and prev_mean != 0:
        rel_change = abs(final_mean - prev_mean) / max(abs(prev_mean), 1e-8)
        if rel_change > CONVERGENCE_PCT:
            reasons.append(f"unstable({rel_change:.0%})")
    elif n >= FINAL_WINDOW + PREV_WINDOW and prev_mean == 0 and abs(final_mean) > COLLAPSE_THRESH:
        reasons.append("jump from ~0")

    rec.converged = rec.status == "OK" and not rec.nan_inf and not any(
        r.startswith(("collapsed", "unstable", "jump")) for r in reasons
    )
    if not rec.converged:
        rec.suspicious_reason = "; ".join(reasons) if reasons else "unknown"
    return rec


def group_key(rec: RunRecord) -> tuple[str, str]:
    return (rec.method, rec.ablation)


def agg_group(
    group_counts: dict[tuple[str, str], list[RunRecord]],
    method: str,
    ablation: str = "",
) -> dict[str, Any] | None:
    rs = [
        r
        for r in group_counts.get((method, ablation), [])
        if r.status == "OK" and r.final_reward_env_stdev is not None
    ]
    if not rs:
        return None
    stdevs = [r.final_reward_env_stdev for r in rs]
    return {
        "method": method,
        "ablation": ablation,
        "n_seeds": len(rs),
        "mean_env": float(np.mean([r.final_mean_env for r in rs])),
        "std_env": float(np.std([r.final_mean_env for r in rs])),
        "mean_reward_env_stdev": float(np.mean(stdevs)),
        "std_reward_env_stdev": float(np.std(stdevs)),
        "mean_coop_rate": float(np.mean([r.final_coop_rate for r in rs if r.final_coop_rate is not None]))
        if any(r.final_coop_rate is not None for r in rs)
        else None,
        "std_coop_rate": float(np.std([r.final_coop_rate for r in rs if r.final_coop_rate is not None]))
        if any(r.final_coop_rate is not None for r in rs)
        else None,
        "mean_payoff_gap": float(np.mean([r.final_payoff_gap for r in rs])),
        "std_payoff_gap": float(np.std([r.final_payoff_gap for r in rs])),
        "mean_fairness_var": float(np.mean([r.final_fairness_var for r in rs])),
        "std_fairness_var": float(np.std([r.final_fairness_var for r in rs])),
    }


def verdict_coop(lio: dict, eia: dict) -> str:
    signals = 0
    total = 0
    if lio["mean_env"] is not None and eia["mean_env"] is not None:
        total += 1
        if eia["mean_env"] < lio["mean_env"]:
            signals += 1
    if lio.get("mean_coop_rate") is not None and eia.get("mean_coop_rate") is not None:
        total += 1
        if eia["mean_coop_rate"] < lio["mean_coop_rate"]:
            signals += 1
    if total == 0:
        return "unexpected"
    if signals == total:
        return "as-expected"
    if signals >= 1:
        return "weak"
    return "unexpected"


def verdict_gap(lio: dict, eia: dict) -> str:
    if eia["mean_payoff_gap"] > lio["mean_payoff_gap"]:
        return "as-expected"
    if eia["mean_payoff_gap"] >= lio["mean_payoff_gap"] * 0.9:
        return "weak"
    return "unexpected"


def main() -> int:
    out_runs = os.path.join(os.path.dirname(__file__), "verify_ipd_runs.csv")
    out_summary = os.path.join(os.path.dirname(__file__), "verify_ipd_summary.csv")

    lio_sample = _sample_path(
        os.path.join(REPO_ROOT, "lio/results/nano_ipd1/ipd_lio/log.csv")
    )
    eia_sample = _sample_path(
        os.path.join(REPO_ROOT, "lio/results/nano_ipd1/ipd_eia/log.csv")
    )
    refine_sample = _sample_path(
        os.path.join(REPO_ROOT, "lio/results/nano_ipd1/ipd_refine/log.csv"),
        os.path.join(REPO_ROOT, "lio/results/nano_ipd1/ipd_refine_eia/log.csv"),
        lio_sample,
    )
    print_schema(lio_sample, eia_sample, refine_sample)

    header, _ = load_csv(lio_sample)
    env_cols = discover_agent_env_cols(header)
    nc_cols, nd_cols = discover_coop_cols(header)

    print("\n" + "=" * 70)
    print("STEP 1 — COMPLETENESS & INTEGRITY")
    print("=" * 70)

    records: list[RunRecord] = []
    found_paths: set[str] = set()
    for seed in SEEDS:
        seed_dir = f"nano_ipd{seed}"
        for run_dir in ALL_DIRS:
            path = os.path.join(REPO_ROOT, "lio", "results", seed_dir, run_dir, "log.csv")
            records.append(analyze_run(seed_dir, run_dir, path, env_cols, nc_cols, nd_cols))
            if os.path.isfile(path):
                found_paths.add(path)

    for path in glob.glob(RESULTS_GLOB):
        parts = path.split(os.sep)
        if len(parts) >= 2 and path not in found_paths:
            run_dir = parts[-2]
            seed_dir = parts[-3]
            try:
                parse_dir(run_dir)
            except ValueError:
                continue
            records.append(
                analyze_run(seed_dir, run_dir, path, env_cols, nc_cols, nd_cols)
            )

    baseline_records = [r for r in records if r.run_dir in BASELINE_DIRS]
    ok = sum(1 for r in baseline_records if r.status == "OK")
    short = sum(1 for r in baseline_records if r.status == "SHORT")
    missing = sum(1 for r in baseline_records if r.status == "MISSING")
    nan_runs = [r for r in records if r.nan_inf]

    print(f"Baseline: {len(BASELINE_DIRS)} dirs × {len(SEEDS)} seeds = {len(BASELINE_DIRS) * len(SEEDS)}")
    print(f"Full matrix (incl. REFiNE): {len(ALL_DIRS)} dirs × {len(SEEDS)} seeds")
    print(f"Baseline OK={ok}  SHORT={short}  MISSING={missing}  NaN/inf={len(nan_runs)}")

    group_counts: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        group_counts[group_key(r)].append(r)

    for method in ("lio", "eia", "refine", "refine_eia"):
        for ablation in ("", "B0"):
            cnt = sum(
                1
                for r in records
                if r.method == method and r.ablation == ablation and r.status == "OK"
            )
            if cnt == 0 and method == "refine_eia" and ablation == "B0":
                continue
            if cnt == 0 and method in ("refine",) and ablation:
                continue
            if cnt == 0 and method in ("lio", "eia") and ablation:
                continue
            flag = "" if cnt == 10 else f"  *** {method}/{ablation or 'full'} <10 ({cnt}/10) ***"
            print(f"  {method}/{ablation or 'full'}: {cnt}/10 OK{flag}")

    print("\n" + "=" * 70)
    print(f"STEP 2 — CONVERGENCE (final {FINAL_WINDOW} vs prior {PREV_WINDOW} rows)")
    print("=" * 70)
    suspicious = [r for r in records if not r.converged and r.status != "MISSING"]
    converged = [r for r in records if r.converged]
    print(f"Converged: {len(converged)} OK runs")
    print(f"Suspicious: {len(suspicious)}")
    for r in suspicious[:30]:
        print(f"  {r.seed_dir}/{r.run_dir}: {r.suspicious_reason}")
    if len(suspicious) > 30:
        print(f"  ... and {len(suspicious) - 30} more")

    print("\n" + "=" * 70)
    print("STEP 3 — LIO vs EIA SANITY (IPD)")
    print("=" * 70)

    lio = agg_group(group_counts, "lio")
    eia = agg_group(group_counts, "eia")
    summary_rows: list[dict[str, Any]] = []

    if lio and eia:
        print(
            f"\nLIO:  mean_env={lio['mean_env']:.2f}±{lio['std_env']:.2f}  "
            f"stdev={lio['mean_reward_env_stdev']:.2f}±{lio['std_reward_env_stdev']:.2f}  "
            f"coop_rate={lio['mean_coop_rate']:.3f}±{lio['std_coop_rate']:.3f}  "
            f"payoff_gap={lio['mean_payoff_gap']:.2f}±{lio['std_payoff_gap']:.2f}"
        )
        print(
            f"EIA:  mean_env={eia['mean_env']:.2f}±{eia['std_env']:.2f}  "
            f"stdev={eia['mean_reward_env_stdev']:.2f}±{eia['std_reward_env_stdev']:.2f}  "
            f"coop_rate={eia['mean_coop_rate']:.3f}±{eia['std_coop_rate']:.3f}  "
            f"payoff_gap={eia['mean_payoff_gap']:.2f}±{eia['std_payoff_gap']:.2f}"
        )
        coop_v = verdict_coop(lio, eia)
        gap_v = verdict_gap(lio, eia)
        print(f"\nCooperation (EIA lower env/coop vs LIO): [{coop_v}]")
        print(f"Payoff gap (EIA higher |A1-A2| vs LIO):   [{gap_v}]")
        summary_rows.append({**lio, "group": "ipd_lio"})
        summary_rows.append({**eia, "group": "ipd_eia"})
    else:
        coop_v = gap_v = "unexpected"

    print("\n" + "=" * 70)
    print("STEP 4 — REFiNE SANITY (IPD)")
    print("=" * 70)

    refine = agg_group(group_counts, "refine")
    refine_eia = agg_group(group_counts, "refine_eia")
    refine_eia_b0 = agg_group(group_counts, "refine_eia", "B0")

    if refine:
        print(
            f"\nREFiNE plain: env={refine['mean_env']:.2f}±{refine['std_env']:.2f} "
            f"stdev={refine['mean_reward_env_stdev']:.2f}±{refine['std_reward_env_stdev']:.2f} "
            f"payoff_gap={refine['mean_payoff_gap']:.2f}"
        )
        summary_rows.append({**refine, "group": "ipd_refine"})

    for label, stats, group in [
        ("full", refine_eia, "ipd_refine_eia"),
        ("B0", refine_eia_b0, "ipd_refine_eia_B0"),
    ]:
        if stats:
            print(
                f"REFiNE+EIA [{label}]: env={stats['mean_env']:.2f}±{stats['std_env']:.2f} "
                f"stdev={stats['mean_reward_env_stdev']:.2f}±{stats['std_reward_env_stdev']:.2f} "
                f"payoff_gap={stats['mean_payoff_gap']:.2f}"
            )
            summary_rows.append({**stats, "group": group})

    refine_vs_eia_flags: list[str] = []
    refine_vs_eia_verdict = "missing-data"
    if refine_eia and eia:
        stdev_ok = refine_eia["mean_reward_env_stdev"] < eia["mean_reward_env_stdev"]
        gap_ok = refine_eia["mean_payoff_gap"] < eia["mean_payoff_gap"]
        if not stdev_ok:
            refine_vs_eia_flags.append("reward_env_stdev NOT lower than EIA")
        if not gap_ok:
            refine_vs_eia_flags.append("payoff_gap NOT lower than EIA")
        if stdev_ok and gap_ok:
            refine_vs_eia_verdict = "as-expected"
        elif stdev_ok or gap_ok:
            refine_vs_eia_verdict = "weak"
        else:
            refine_vs_eia_verdict = "unexpected"
        print(
            f"\nREFiNE+EIA vs EIA: stdev {refine_eia['mean_reward_env_stdev']:.2f} vs "
            f"{eia['mean_reward_env_stdev']:.2f} | gap {refine_eia['mean_payoff_gap']:.2f} vs "
            f"{eia['mean_payoff_gap']:.2f}"
        )
        if refine_vs_eia_flags:
            print("  *** FLAG: " + "; ".join(refine_vs_eia_flags) + " ***")
        print(f"  => [{refine_vs_eia_verdict}]")

    run_fields = [
        "seed_dir",
        "run_dir",
        "method",
        "ablation",
        "n_rows",
        "expected",
        "status",
        "nan_inf",
        "final_mean_env",
        "final_reward_env_stdev",
        "final_coop_rate",
        "final_payoff_gap",
        "final_fairness_var",
        "converged",
        "suspicious_reason",
        "path",
    ]
    with open(out_runs, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for r in sorted(records, key=lambda x: (x.run_dir, x.seed_dir)):
            w.writerow({k: getattr(r, k) for k in run_fields})

    sum_fields = [
        "group",
        "method",
        "ablation",
        "n_seeds",
        "mean_env",
        "std_env",
        "mean_reward_env_stdev",
        "std_reward_env_stdev",
        "mean_coop_rate",
        "std_coop_rate",
        "mean_payoff_gap",
        "std_payoff_gap",
        "mean_fairness_var",
        "std_fairness_var",
    ]
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in sum_fields})

    print("\n" + "=" * 70)
    print("TOP-LINE SUMMARY")
    print("=" * 70)
    print(f"Baseline complete: {ok}/20 OK ({short} SHORT, {missing} MISSING)")
    print(f"Suspicious (convergence): {len(suspicious)}")
    print(f"Cooperation EIA vs LIO: [{coop_v}]")
    print(f"Payoff gap EIA vs LIO:    [{gap_v}]")
    print(f"REFiNE+EIA vs EIA:        [{refine_vs_eia_verdict}]")
    if refine_vs_eia_flags:
        print(f"  Flags: {'; '.join(refine_vs_eia_flags)}")
    print(f"\nWrote: {out_runs}")
    print(f"Wrote: {out_summary}")

    exit_fail = ok != 20 or missing > 0
    if refine_vs_eia_verdict == "unexpected":
        exit_fail = True
    return 1 if exit_fail else 0


if __name__ == "__main__":
    sys.exit(main())
