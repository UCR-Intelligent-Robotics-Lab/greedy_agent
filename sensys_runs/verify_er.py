#!/usr/bin/env python3
"""Health + sanity report for completed ER baseline + REFiNE runs (read-only)."""
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_GLOB = os.path.join(REPO_ROOT, "lio", "results", "nano_er*", "*", "log.csv")
EXPECTED_ROWS = 50
SEEDS = list(range(1, 11))
SIZES = [(3, 1), (4, 2), (6, 4)]
DEFAULT_W = (2.0, 0.2)
SWEEP_W = [(1.1, 0.9), (1.5, 1.5)]
ABLATIONS = ("", "_B0", "_beta0")

LIO_DIRS = [f"er_lio_{n}_{m}" for n, m in SIZES]
EIA_DIRS = [f"er_eia_{n}_{m}_w{DEFAULT_W[0]}-{DEFAULT_W[1]}" for n, m in SIZES] + [
    f"er_eia_4_2_w{wl}-{wd}" for wl, wd in SWEEP_W
]
REFINE_DIRS = [f"er_refine_{n}_{m}" for n, m in SIZES]


def _refine_eia_dirs() -> list[str]:
    dirs: list[str] = []
    for n, m in SIZES:
        wl, wd = DEFAULT_W
        for ab in ABLATIONS:
            dirs.append(f"er_refine_eia_{n}_{m}_w{wl}-{wd}{ab}")
    for wl, wd in SWEEP_W:
        for ab in ABLATIONS:
            dirs.append(f"er_refine_eia_4_2_w{wl}-{wd}{ab}")
    return dirs


REFINE_EIA_DIRS = _refine_eia_dirs()
BASELINE_DIRS = LIO_DIRS + EIA_DIRS
ALL_DIRS = BASELINE_DIRS + REFINE_DIRS + REFINE_EIA_DIRS

CONVERGENCE_PCT = 0.25
COLLAPSE_THRESH = 1.0
WINRATE_COLLAPSE_THRESH = 0.01
FINAL_WINDOW = 5


@dataclass
class RunRecord:
    seed_dir: str
    run_dir: str
    method: str
    n_agents: int
    min_at_lever: int
    weights: str | None
    ablation: str
    path: str
    n_rows: int
    expected: int
    status: str
    nan_inf: bool
    n_agents_found: int
    final_mean_env: float | None
    final_reward_env_stdev: float | None
    final_fairness_var: float | None
    final_fairness_sigma: float | None
    final_energy_var: float | None
    final_mean_winrate: float | None
    converged: bool
    suspicious_reason: str = ""


def parse_dir(run_dir: str) -> tuple[str, int, int, str | None, str]:
    m = re.match(r"er_refine_eia_(\d+)_(\d+)_w([\d.]+)-([\d.]+)(_B0|_beta0)?$", run_dir)
    if m:
        ablation = ""
        if m.group(5) == "_B0":
            ablation = "B0"
        elif m.group(5) == "_beta0":
            ablation = "beta0"
        return (
            "refine_eia",
            int(m.group(1)),
            int(m.group(2)),
            f"{m.group(3)}-{m.group(4)}",
            ablation,
        )

    m = re.match(r"er_refine_(\d+)_(\d+)$", run_dir)
    if m:
        return "refine", int(m.group(1)), int(m.group(2)), None, ""

    if run_dir.startswith("er_lio_"):
        parts = run_dir.replace("er_lio_", "").split("_")
        return "lio", int(parts[0]), int(parts[1]), None, ""

    m = re.match(r"er_eia_(\d+)_(\d+)_w([\d.]+)-([\d.]+)$", run_dir)
    if m:
        return "eia", int(m.group(1)), int(m.group(2)), f"{m.group(3)}-{m.group(4)}", ""

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


def discover_agent_winrate_cols(header: list[str]) -> list[str]:
    return sorted(
        [c for c in header if re.match(r"A\d+_win_rate$", c)],
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


def _sample_path(*candidates: str) -> str:
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def print_schema(lio_path: str, eia_path: str, refine_path: str) -> tuple[list[str], list[str]]:
    print("=" * 70)
    print("STEP 0 — SCHEMA DISCOVERY")
    print("=" * 70)
    for label, path in [
        ("LIO", lio_path),
        ("EIA", eia_path),
        ("REFiNE", refine_path),
    ]:
        if not os.path.isfile(path):
            print(f"\n--- {label}: {path} (missing, skipped)")
            continue
        header, rows = load_csv(path)
        env_cols = discover_agent_env_cols(header)
        energy_cols = discover_agent_energy_cols(header)
        win_cols = discover_agent_winrate_cols(header)
        idx_cols = [c for c in ("episode", "step_train", "step") if c in header]
        print(f"\n--- {label}: {path}")
        print(f"Columns ({len(header)}): {', '.join(header)}")
        print(f"Episode index: {idx_cols}")
        print(f"Per-agent env reward: {env_cols}")
        print(f"Per-agent energy: {energy_cols}")
        print(f"Per-agent win_rate: {win_cols}")
        print("NOTE: A*_teamwork_fairness is UNRELIABLE — not used.")
        if rows:
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
    method, n_agents, min_at_lever, weights, ablation = parse_dir(run_dir)
    rec = RunRecord(
        seed_dir=seed_dir,
        run_dir=run_dir,
        method=method,
        n_agents=n_agents,
        min_at_lever=min_at_lever,
        weights=weights,
        ablation=ablation,
        path=path,
        n_rows=0,
        expected=EXPECTED_ROWS,
        status="MISSING",
        nan_inf=False,
        n_agents_found=0,
        final_mean_env=None,
        final_reward_env_stdev=None,
        final_fairness_var=None,
        final_fairness_sigma=None,
        final_energy_var=None,
        final_mean_winrate=None,
        converged=False,
    )
    if not os.path.isfile(path):
        rec.suspicious_reason = "missing file"
        return rec

    header, rows = load_csv(path)
    env_cols_run = discover_agent_env_cols(header)
    energy_cols_run = discover_agent_energy_cols(header)
    win_cols_run = discover_agent_winrate_cols(header)
    rec.n_agents_found = len(env_cols_run)
    rec.n_rows = len(rows)

    if rec.n_rows >= EXPECTED_ROWS:
        rec.status = "OK"
    elif rec.n_rows == 0:
        rec.status = "MISSING"
    else:
        rec.status = "SHORT"

    if not rows or not env_cols_run:
        rec.suspicious_reason = "no data or no env reward cols"
        return rec

    env_mat = rows_to_float_matrix(rows, env_cols_run)
    rec.nan_inf = bool(np.any(~np.isfinite(env_mat)))

    final_w = env_mat[-FINAL_WINDOW:] if len(env_mat) >= FINAL_WINDOW else env_mat
    prev_w = (
        env_mat[-(2 * FINAL_WINDOW) : -FINAL_WINDOW]
        if len(env_mat) >= 2 * FINAL_WINDOW
        else env_mat[: max(1, len(env_mat) // 2)]
    )

    agent_means_final = np.mean(final_w, axis=0)
    rec.final_mean_env = float(np.mean(agent_means_final))
    rec.final_fairness_var = float(np.var(agent_means_final))
    rec.final_reward_env_stdev = float(np.std(agent_means_final))
    rec.final_fairness_sigma = rec.final_reward_env_stdev

    if energy_cols_run:
        energy_mat = rows_to_float_matrix(rows, energy_cols_run)
        energy_final = energy_mat[-FINAL_WINDOW:] if len(energy_mat) >= FINAL_WINDOW else energy_mat
        agent_energy_means = np.mean(energy_final, axis=0)
        rec.final_energy_var = float(np.var(agent_energy_means))

    if win_cols_run:
        win_mat = rows_to_float_matrix(rows, win_cols_run)
        win_final = win_mat[-FINAL_WINDOW:] if len(win_mat) >= FINAL_WINDOW else win_mat
        rec.final_mean_winrate = float(np.mean(win_final))

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
    if len(env_mat) >= 2 * FINAL_WINDOW and prev_mean != 0:
        rel_change = abs(final_mean - prev_mean) / max(abs(prev_mean), 1e-8)
        if rel_change > CONVERGENCE_PCT:
            reasons.append(f"unstable({rel_change:.0%})")
    elif len(env_mat) >= 2 * FINAL_WINDOW and prev_mean == 0 and abs(final_mean) > COLLAPSE_THRESH:
        reasons.append("jump from ~0")

    if (
        n_agents == 6
        and min_at_lever == 4
        and method in ("refine", "refine_eia")
        and rec.status == "OK"
    ):
        env_collapsed = abs(rec.final_mean_env or 0.0) < COLLAPSE_THRESH
        win_collapsed = (
            rec.final_mean_winrate is not None
            and rec.final_mean_winrate < WINRATE_COLLAPSE_THRESH
        )
        if env_collapsed or win_collapsed:
            reasons.append("refine_er64_collapse")

    rec.converged = rec.status == "OK" and not rec.nan_inf and not any(
        r.startswith(("collapsed", "unstable", "jump", "refine_er64_collapse")) for r in reasons
    )
    if not rec.converged:
        rec.suspicious_reason = "; ".join(reasons) if reasons else "unknown"
    return rec


def group_key(rec: RunRecord) -> tuple:
    return (rec.method, rec.n_agents, rec.min_at_lever, rec.weights or "", rec.ablation)


def agg_group(
    group_counts: dict[tuple, list[RunRecord]],
    method: str,
    n: int,
    m: int,
    weights: str | None = None,
    ablation: str = "",
) -> dict[str, Any] | None:
    key = (method, n, m, weights or "", ablation)
    rs = [
        r
        for r in group_counts.get(key, [])
        if r.status == "OK" and r.final_reward_env_stdev is not None
    ]
    if not rs:
        return None
    stdevs = [r.final_reward_env_stdev for r in rs]
    env_means = [r.final_mean_env for r in rs if r.final_mean_env is not None]
    vars_ = [r.final_fairness_var for r in rs if r.final_fairness_var is not None]
    energy_vars = [r.final_energy_var for r in rs if r.final_energy_var is not None]
    return {
        "method": method,
        "n_agents": n,
        "min_at_lever": m,
        "weights": weights or "",
        "ablation": ablation,
        "n_seeds": len(rs),
        "mean_env": float(np.mean(env_means)) if env_means else None,
        "std_env": float(np.std(env_means)) if env_means else None,
        "mean_reward_env_stdev": float(np.mean(stdevs)),
        "std_reward_env_stdev": float(np.std(stdevs)),
        "mean_fairness_var": float(np.mean(vars_)) if vars_ else None,
        "std_fairness_var": float(np.std(vars_)) if vars_ else None,
        "mean_fairness_sigma": float(np.mean(stdevs)),
        "std_fairness_sigma": float(np.std(stdevs)),
        "mean_energy_var": float(np.mean(energy_vars)) if energy_vars else None,
        "std_energy_var": float(np.std(energy_vars)) if energy_vars else None,
    }


def main() -> int:
    out_runs = os.path.join(os.path.dirname(__file__), "verify_er_runs.csv")
    out_summary = os.path.join(os.path.dirname(__file__), "verify_er_summary.csv")

    lio_sample = _sample_path(
        os.path.join(REPO_ROOT, "lio/results/nano_er1/er_lio_3_1/log.csv")
    )
    eia_sample = _sample_path(
        os.path.join(REPO_ROOT, "lio/results/nano_er1/er_eia_4_2_w2.0-0.2/log.csv")
    )
    refine_sample = _sample_path(
        os.path.join(REPO_ROOT, "lio/results/nano_er1/er_refine_4_2/log.csv"),
        os.path.join(REPO_ROOT, "lio/results/nano_er1/er_refine_eia_4_2_w2.0-0.2/log.csv"),
        lio_sample,
    )
    print_schema(lio_sample, eia_sample, refine_sample)

    print("\n" + "=" * 70)
    print("STEP 1 — COMPLETENESS & INTEGRITY")
    print("=" * 70)

    records: list[RunRecord] = []
    found_paths: set[str] = set()

    for seed in SEEDS:
        seed_dir = f"nano_er{seed}"
        for run_dir in ALL_DIRS:
            path = os.path.join(REPO_ROOT, "lio", "results", seed_dir, run_dir, "log.csv")
            rec = analyze_run(seed_dir, run_dir, path, [])
            records.append(rec)
            if os.path.isfile(path):
                found_paths.add(path)

    for path in glob.glob(RESULTS_GLOB):
        parts = path.split(os.sep)
        if len(parts) >= 2:
            run_dir = parts[-2]
            seed_dir = parts[-3]
            if path not in found_paths:
                try:
                    parse_dir(run_dir)
                except ValueError:
                    continue
                records.append(analyze_run(seed_dir, run_dir, path, []))

    baseline_records = [r for r in records if r.run_dir in BASELINE_DIRS]
    ok = sum(1 for r in baseline_records if r.status == "OK")
    short = sum(1 for r in baseline_records if r.status == "SHORT")
    missing = sum(1 for r in baseline_records if r.status == "MISSING")
    nan_runs = [r for r in records if r.nan_inf]

    print(f"Baseline matrix: {len(BASELINE_DIRS)} dirs × {len(SEEDS)} seeds = {len(BASELINE_DIRS) * len(SEEDS)}")
    print(f"Full matrix (incl. REFiNE): {len(ALL_DIRS)} dirs × {len(SEEDS)} seeds")
    print(f"Baseline OK={ok}  SHORT={short}  MISSING={missing}  NaN/inf={len(nan_runs)}")

    group_counts: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in records:
        group_counts[group_key(r)].append(r)

    print("\nSeeds per group (baseline + REFiNE):")
    for g in sorted(group_counts.keys()):
        cnt = sum(1 for r in group_counts[g] if r.status == "OK")
        flag = "" if cnt == 10 else f"  *** GROUP <10 ({cnt}/10) ***"
        print(f"  {g}: {cnt}/10 OK{flag}")

    print("\n" + "=" * 70)
    print("STEP 2 — CONVERGENCE")
    print("=" * 70)
    suspicious = [r for r in records if not r.converged and r.status != "MISSING"]
    converged = [r for r in records if r.converged]
    print(f"Converged: {len(converged)} OK runs")
    print(f"Suspicious: {len(suspicious)}")
    if suspicious:
        print("Suspicious runs:")
        for r in suspicious[:30]:
            print(f"  {r.seed_dir}/{r.run_dir}: {r.suspicious_reason}")
        if len(suspicious) > 30:
            print(f"  ... and {len(suspicious) - 30} more")

    print("\n" + "=" * 70)
    print("STEP 3 — LIO vs EIA SANITY")
    print("=" * 70)

    summary_rows: list[dict[str, Any]] = []
    sizes = SIZES
    eia_fairness_worse_at: list[str] = []
    lio_eia_pairs: list[tuple] = []

    print("\nPer-size LIO vs EIA (default w=2.0-0.2):")
    for n, m in sizes:
        lio = agg_group(group_counts, "lio", n, m)
        eia = agg_group(group_counts, "eia", n, m, "2.0-0.2")
        if lio and eia:
            worse = eia["mean_fairness_var"] > lio["mean_fairness_var"]
            lio_eia_pairs.append((n, m, worse, lio, eia))
            tag = "EIA worse fairness" if worse else "LIO worse or equal"
            if worse:
                eia_fairness_worse_at.append(f"({n},{m})")
            print(
                f"  ER({n},{m}): LIO env={lio['mean_env']:.1f}±{lio['std_env']:.1f} "
                f"stdev={lio['mean_reward_env_stdev']:.2f}±{lio['std_reward_env_stdev']:.2f} | "
                f"EIA env={eia['mean_env']:.1f}±{eia['std_env']:.1f} "
                f"stdev={eia['mean_reward_env_stdev']:.2f}±{eia['std_reward_env_stdev']:.2f}  [{tag}]"
            )
            summary_rows.append({**lio, "group": f"lio_{n}_{m}"})
            summary_rows.append({**eia, "group": f"eia_{n}_{m}_w2.0-0.2"})

    print("\nER(4,2) weight sweep (EIA fairness Var, lower=milder):")
    sweep_weights = ["2.0-0.2", "1.1-0.9", "1.5-1.5"]
    sweep_stats = []
    for w in sweep_weights:
        s = agg_group(group_counts, "eia", 4, 2, w)
        if s:
            sweep_stats.append((w, s))
            print(
                f"  w={w}: env={s['mean_env']:.1f}±{s['std_env']:.1f} "
                f"stdev={s['mean_reward_env_stdev']:.2f}±{s['std_reward_env_stdev']:.2f} "
                f"energy_var={s['mean_energy_var']:.1f}"
            )
            summary_rows.append({**s, "group": f"eia_4_2_w{w}"})

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

    print("\n" + "=" * 70)
    print("STEP 4 — REFiNE SANITY")
    print("=" * 70)

    print("\nPer-size REFiNE (plain) reward_env stdev over seeds:")
    for n, m in sizes:
        refine = agg_group(group_counts, "refine", n, m)
        if refine:
            print(
                f"  ER({n},{m}): refine env={refine['mean_env']:.1f}±{refine['std_env']:.1f} "
                f"stdev={refine['mean_reward_env_stdev']:.2f}±{refine['std_reward_env_stdev']:.2f}"
            )
            summary_rows.append({**refine, "group": f"refine_{n}_{m}"})

    print("\nPer-size REFiNE+EIA default w=2.0-0.2 (full, no ablation):")
    for n, m in sizes:
        for ablation in ("", "B0", "beta0"):
            rs = agg_group(group_counts, "refine_eia", n, m, "2.0-0.2", ablation)
            if rs:
                ab_tag = ablation or "full"
                print(
                    f"  ER({n},{m}) [{ab_tag}]: env={rs['mean_env']:.1f}±{rs['std_env']:.1f} "
                    f"stdev={rs['mean_reward_env_stdev']:.2f}±{rs['std_reward_env_stdev']:.2f} "
                    f"energy_var={rs['mean_energy_var']:.1f}±{rs['std_energy_var']:.1f}"
                )
                summary_rows.append(
                    {**rs, "group": f"refine_eia_{n}_{m}_w2.0-0.2_{ab_tag}"}
                )

    print("\nER(4,2) REFiNE+EIA weight sweep (full):")
    for w in sweep_weights:
        rs = agg_group(group_counts, "refine_eia", 4, 2, w)
        if rs:
            print(
                f"  w={w}: stdev={rs['mean_reward_env_stdev']:.2f}±{rs['std_reward_env_stdev']:.2f} "
                f"energy_var={rs['mean_energy_var']:.1f}"
            )
            summary_rows.append({**rs, "group": f"refine_eia_4_2_w{w}"})

    refine_vs_eia_flags: list[str] = []
    refine_eia_42 = agg_group(group_counts, "refine_eia", 4, 2, "2.0-0.2")
    eia_42 = agg_group(group_counts, "eia", 4, 2, "2.0-0.2")
    refine_vs_eia_verdict = "missing-data"
    if refine_eia_42 and eia_42:
        stdev_ok = refine_eia_42["mean_reward_env_stdev"] < eia_42["mean_reward_env_stdev"]
        energy_ok = True
        if (
            refine_eia_42["mean_energy_var"] is not None
            and eia_42["mean_energy_var"] is not None
        ):
            energy_ok = refine_eia_42["mean_energy_var"] < eia_42["mean_energy_var"]
        if not stdev_ok:
            refine_vs_eia_flags.append("reward_env_stdev NOT lower than EIA")
        if not energy_ok:
            refine_vs_eia_flags.append("total_energy variance NOT lower than EIA")
        if stdev_ok and energy_ok:
            refine_vs_eia_verdict = "as-expected"
        elif stdev_ok or energy_ok:
            refine_vs_eia_verdict = "weak"
        else:
            refine_vs_eia_verdict = "unexpected"
        print(
            f"\nER(4,2) default w: REFiNE+EIA vs EIA "
            f"stdev {refine_eia_42['mean_reward_env_stdev']:.2f} vs {eia_42['mean_reward_env_stdev']:.2f} | "
            f"energy_var {refine_eia_42['mean_energy_var']:.1f} vs {eia_42['mean_energy_var']:.1f}"
        )
        if refine_vs_eia_flags:
            print("  *** FLAG: " + "; ".join(refine_vs_eia_flags) + " ***")
        print(f"  => [{refine_vs_eia_verdict}]")

    er64_collapse = [
        r
        for r in records
        if r.n_agents == 6
        and r.min_at_lever == 4
        and r.method in ("refine", "refine_eia")
        and "refine_er64_collapse" in r.suspicious_reason
    ]
    print(f"\nER(6,4) REFiNE collapse flags: {len(er64_collapse)}")
    for r in er64_collapse:
        print(
            f"  *** {r.seed_dir}/{r.run_dir}: env={r.final_mean_env:.3f} "
            f"win_rate={r.final_mean_winrate} ***"
        )

    run_fields = [
        "seed_dir",
        "run_dir",
        "method",
        "n_agents",
        "min_at_lever",
        "weights",
        "ablation",
        "n_rows",
        "expected",
        "status",
        "nan_inf",
        "n_agents_found",
        "final_mean_env",
        "final_reward_env_stdev",
        "final_fairness_var",
        "final_fairness_sigma",
        "final_energy_var",
        "final_mean_winrate",
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
        "n_agents",
        "min_at_lever",
        "weights",
        "ablation",
        "n_seeds",
        "mean_env",
        "std_env",
        "mean_reward_env_stdev",
        "std_reward_env_stdev",
        "mean_fairness_var",
        "std_fairness_var",
        "mean_fairness_sigma",
        "std_fairness_sigma",
        "mean_energy_var",
        "std_energy_var",
    ]
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in sum_fields})

    print("\n" + "=" * 70)
    print("TOP-LINE SUMMARY")
    print("=" * 70)
    print(f"Baseline complete: {ok}/80 OK ({short} SHORT, {missing} MISSING)")
    print(f"Suspicious (convergence): {len(suspicious)}")
    print(f"EIA > LIO fairness (higher Var) at sizes: {eia_fairness_worse_at or 'none'} [{lio_eia_verdict}]")
    print(f"Weight sweep ER(4,2) ordering: [{weight_verdict}]")
    print(f"REFiNE+EIA vs EIA ER(4,2): [{refine_vs_eia_verdict}]")
    if refine_vs_eia_flags:
        print(f"  Flags: {'; '.join(refine_vs_eia_flags)}")
    print(f"ER(6,4) REFiNE collapse flags: {len(er64_collapse)}")
    print(f"\nWrote: {out_runs}")
    print(f"Wrote: {out_summary}")

    exit_fail = ok != 80 or missing > 0
    if refine_vs_eia_verdict == "unexpected":
        exit_fail = True
    if er64_collapse:
        exit_fail = True
    return 1 if exit_fail else 0


if __name__ == "__main__":
    sys.exit(main())
