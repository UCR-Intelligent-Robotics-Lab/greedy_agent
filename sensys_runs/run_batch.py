#!/usr/bin/env python3
"""Batch orchestrator for SenSys LIO / LIO+EIA baseline experiments."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_SINGLE = os.path.join(os.path.dirname(__file__), "run_single.py")
RUN_LOGS_DIR = os.path.join(os.path.dirname(__file__), "run_logs")
RESULTS_ROOT = os.path.join(REPO_ROOT, "lio", "results")

SEED_BASE = 12340
SIZES = [(3, 1), (4, 2), (6, 4)]
SWEEP_W = [(1.1, 0.9), (1.5, 1.5)]
DEFAULT_W = (2.0, 0.2)

METHODS = ("lio_er", "lio_eia_er", "lio_ipd", "lio_eia_ipd")

ER_DEFAULT_EPISODES = 25000
ER_DEFAULT_PERIOD = 500
IPD_DEFAULT_EPISODES = 20000
IPD_DEFAULT_PERIOD = 1000

ER_SUFFIXES = [
    "reward_total",
    "reward_env",
    "n_lever",
    "n_door",
    "received",
    "given",
    "r-lever",
    "r-start",
    "r-door",
    "win_rate",
    "total_energy",
    "teamwork_fairness",
]
IPD_SUFFIXES = [
    "given",
    "received",
    "reward_env",
    "reward_total",
    "n_c",
    "n_d",
    "teamwork_fairness",
]

Mode = Literal["real", "smoke", "preflight"]


def _seed(s_idx: int) -> int:
    return SEED_BASE + s_idx


def _er_exp_name(s: int, platform: str) -> str:
    base = f"er{s}"
    return f"{platform}_{base}" if platform else base


def _ipd_exp_name(s: int, platform: str) -> str:
    base = f"ipd{s}"
    return f"{platform}_{base}" if platform else base


def _er_header(n_agents: int) -> str:
    cols = ["episode", "step_train", "step"]
    for aid in range(1, n_agents + 1):
        for suffix in ER_SUFFIXES:
            cols.append(f"A{aid}_{suffix}")
    cols.append("steps_per_eps")
    return ",".join(cols) + "\n"


def _ipd_header(n_agents: int) -> str:
    cols = ["episode", "step_train", "step"]
    for aid in range(1, n_agents + 1):
        for suffix in IPD_SUFFIXES:
            cols.append(f"A{aid}_{suffix}")
    return ",".join(cols) + "\n"


def _count_agent_groups(header_line: str) -> int:
    import re

    return len(set(re.findall(r"A(\d+)_", header_line)))


def _default_episodes_period(method: str) -> tuple[int, int]:
    if method in ("lio_er", "lio_eia_er"):
        return ER_DEFAULT_EPISODES, ER_DEFAULT_PERIOD
    return IPD_DEFAULT_EPISODES, IPD_DEFAULT_PERIOD


def expected_data_rows(spec: dict[str, Any]) -> int:
    default_ep, default_period = _default_episodes_period(spec["method"])
    n_ep = int(spec.get("n_episodes", default_ep))
    period = int(spec.get("period", default_period))
    return n_ep // period


def result_log_csv(spec: dict[str, Any]) -> str:
    return os.path.join(RESULTS_ROOT, spec["exp_name"], spec["dir_name"], "log.csv")


def count_data_rows(log_csv: str) -> int:
    if not os.path.isfile(log_csv):
        return 0
    with open(log_csv, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    return max(0, len(lines) - 1)


def is_run_complete(spec: dict[str, Any]) -> bool:
    log_csv = result_log_csv(spec)
    need = expected_data_rows(spec)
    return count_data_rows(log_csv) >= need


def build_matrix(
    method: str, mode: Mode, platform: str = "nano", n_seeds: int = 10
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    plat = platform if mode == "real" else ""

    def add(**kw: Any) -> None:
        if plat:
            kw.setdefault("platform", plat)
        specs.append(kw)

    if mode == "smoke":
        smoke_overrides = dict(n_episodes=10, period=2, seed=_seed(1))
        families = METHODS if method == "all" else (method,)
        for fam in families:
            base = dict(method=fam, exp_name=f"smoke_{fam}", **smoke_overrides)
            if fam == "lio_er":
                add(
                    **base,
                    dir_name="smoke_lio_3_1",
                    n_agents=3,
                    min_at_lever=1,
                    n_eval=3,
                )
            elif fam == "lio_eia_er":
                add(
                    **base,
                    dir_name="smoke_eia_4_2_w2.0-0.2",
                    n_agents=4,
                    min_at_lever=2,
                    w_lever=2.0,
                    w_door=0.2,
                    n_eval=4,
                )
            elif fam == "lio_ipd":
                add(**base, dir_name="smoke_lio", n_eval=2)
            elif fam == "lio_eia_ipd":
                add(**base, dir_name="smoke_eia", n_eval=2)
        return specs

    if mode == "preflight":
        families = METHODS if method == "all" else (method,)
        for fam in families:
            if fam == "lio_er":
                add(
                    method=fam,
                    exp_name="preflight_lio_er",
                    dir_name="preflight_lio_6_4",
                    seed=_seed(1),
                    n_agents=6,
                    min_at_lever=4,
                    n_episodes=600,
                    period=500,
                    n_eval=10,
                )
            elif fam == "lio_eia_er":
                add(
                    method=fam,
                    exp_name="preflight_lio_eia_er",
                    dir_name="preflight_eia_6_4_w2.0-0.2",
                    seed=_seed(1),
                    n_agents=6,
                    min_at_lever=4,
                    w_lever=2.0,
                    w_door=0.2,
                    n_episodes=600,
                    period=500,
                    n_eval=10,
                )
            elif fam == "lio_ipd":
                add(
                    method=fam,
                    exp_name="preflight_lio_ipd",
                    dir_name="preflight_lio",
                    seed=_seed(1),
                    n_episodes=1000,
                    period=1000,
                    n_eval=10,
                )
            elif fam == "lio_eia_ipd":
                add(
                    method=fam,
                    exp_name="preflight_lio_eia_ipd",
                    dir_name="preflight_eia",
                    seed=_seed(1),
                    n_episodes=1000,
                    period=1000,
                    n_eval=10,
                )
        return specs

    methods = METHODS if method == "all" else (method,)

    for m in methods:
        if m == "lio_er":
            for n, min_lev in SIZES:
                for s in range(1, n_seeds + 1):
                    add(
                        method=m,
                        exp_name=_er_exp_name(s, plat),
                        dir_name=f"er_lio_{n}_{min_lev}",
                        n_agents=n,
                        min_at_lever=min_lev,
                        seed=_seed(s),
                    )
        elif m == "lio_eia_er":
            wl, wd = DEFAULT_W
            for n, min_lev in SIZES:
                for s in range(1, n_seeds + 1):
                    add(
                        method=m,
                        exp_name=_er_exp_name(s, plat),
                        dir_name=f"er_eia_{n}_{min_lev}_w{wl}-{wd}",
                        n_agents=n,
                        min_at_lever=min_lev,
                        w_lever=wl,
                        w_door=wd,
                        seed=_seed(s),
                    )
            for wl, wd in SWEEP_W:
                for s in range(1, n_seeds + 1):
                    add(
                        method=m,
                        exp_name=_er_exp_name(s, plat),
                        dir_name=f"er_eia_4_2_w{wl}-{wd}",
                        n_agents=4,
                        min_at_lever=2,
                        w_lever=wl,
                        w_door=wd,
                        seed=_seed(s),
                    )
        elif m == "lio_ipd":
            for s in range(1, n_seeds + 1):
                add(
                    method=m,
                    exp_name=_ipd_exp_name(s, plat),
                    dir_name="ipd_lio",
                    seed=_seed(s),
                )
        elif m == "lio_eia_ipd":
            for s in range(1, n_seeds + 1):
                add(
                    method=m,
                    exp_name=_ipd_exp_name(s, plat),
                    dir_name="ipd_eia",
                    seed=_seed(s),
                )

    return specs


def spec_to_argv(
    spec: dict[str, Any], use_gpu: bool, threads: int, platform: str
) -> list[str]:
    cmd = [
        sys.executable,
        RUN_SINGLE,
        "--method",
        spec["method"],
        "--exp_name",
        spec["exp_name"],
        "--dir_name",
        spec["dir_name"],
        "--seed",
        str(spec["seed"]),
        "--platform",
        spec.get("platform", platform),
        "--threads",
        str(spec.get("threads", threads)),
    ]
    for key in ("n_episodes", "n_eval", "period", "n_agents", "min_at_lever"):
        if key in spec:
            cmd.extend([f"--{key}", str(spec[key])])
    for key in ("w_lever", "w_door"):
        if key in spec:
            cmd.extend([f"--{key}", str(spec[key])])
    if use_gpu:
        cmd.append("--use_gpu")
    return cmd


def _log_path(spec: dict[str, Any]) -> str:
    return os.path.join(RUN_LOGS_DIR, f"{spec['exp_name']}__{spec['dir_name']}.log")


def _tail_lines(path: str, n: int = 25) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError:
        return ""


def validate_run(spec: dict[str, Any], exit_code: int) -> list[str]:
    errors: list[str] = []
    label = f"{spec['method']} {spec['exp_name']}/{spec['dir_name']}"

    if exit_code != 0:
        errors.append(f"{label}: exit code {exit_code} != 0")
        return errors

    log_dir = os.path.join(RESULTS_ROOT, spec["exp_name"], spec["dir_name"])
    if not os.path.isdir(log_dir):
        errors.append(f"{label}: log dir missing: {log_dir}")
        return errors

    log_csv = os.path.join(log_dir, "log.csv")
    if not os.path.isfile(log_csv):
        errors.append(f"{label}: log.csv missing")
        return errors

    with open(log_csv, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if len(lines) < 2:
        errors.append(f"{label}: log.csv has no data rows")
        return errors

    header = lines[0] + "\n"
    method = spec["method"]
    n_agents = spec.get("n_agents", 2)

    if method in ("lio_er", "lio_eia_er"):
        expected = _er_header(n_agents)
    else:
        expected = _ipd_header(n_agents)

    if header != expected:
        errors.append(f"{label}: log.csv header mismatch")
        errors.append(f"  expected: {expected.strip()}")
        errors.append(f"  got:      {lines[0]}")

    if _count_agent_groups(lines[0]) != n_agents:
        errors.append(
            f"{label}: header agent groups {_count_agent_groups(lines[0])} "
            f"!= n_agents {n_agents}"
        )

    prov_path = os.path.join(log_dir, "provenance.json")
    if not os.path.isfile(prov_path):
        errors.append(f"{label}: provenance.json missing")
    else:
        with open(prov_path, encoding="utf-8") as f:
            prov = json.load(f)
        if prov.get("exp_name") != spec["exp_name"]:
            errors.append(f"{label}: provenance exp_name mismatch")
        if prov.get("dir_name") != spec["dir_name"]:
            errors.append(f"{label}: provenance dir_name mismatch")

    cfg_path = os.path.join(log_dir, "config.json")
    if not os.path.isfile(cfg_path):
        errors.append(f"{label}: config.json missing")
        return errors

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if method in ("lio_er", "lio_eia_er"):
        n = spec["n_agents"]
        m = spec["min_at_lever"]
        if cfg.get("env", {}).get("n_agents") != n:
            errors.append(f"{label}: config env.n_agents != {n}")
        if cfg.get("env", {}).get("min_at_lever") != m:
            errors.append(f"{label}: config env.min_at_lever != {m}")
        if cfg.get("lio", {}).get("min_at_lever") != m:
            errors.append(f"{label}: config lio.min_at_lever != {m}")
        if method == "lio_eia_er":
            if cfg.get("lio", {}).get("eia_w_lever") != spec.get("w_lever"):
                errors.append(f"{label}: config lio.eia_w_lever mismatch")
            if cfg.get("lio", {}).get("eia_w_door") != spec.get("w_door"):
                errors.append(f"{label}: config lio.eia_w_door mismatch")

    return errors


def run_one(
    spec: dict[str, Any],
    use_gpu: bool,
    threads: int,
    platform: str,
    validate: bool,
    run_timeout: int | None,
) -> tuple[dict[str, Any], str, int, list[str]]:
    """Returns (spec, status, exit_code, validation_errors). status in ok|fail|timeout."""
    os.makedirs(RUN_LOGS_DIR, exist_ok=True)
    cmd = spec_to_argv(spec, use_gpu, threads, platform)
    log_path = _log_path(spec)
    exit_code = 0
    status = "ok"

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n\n")
        logf.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=run_timeout,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
            exit_code = -1
            logf.write(f"\n# TIMEOUT after {run_timeout}s\n")

    validation_errors: list[str] = []
    if validate and status == "ok":
        validation_errors = validate_run(spec, exit_code)
        if validation_errors:
            status = "fail"
    elif exit_code != 0:
        status = "fail"

    return spec, status, exit_code, validation_errors


def _print_spec_table(specs: list[dict[str, Any]], platform: str) -> None:
    from collections import Counter

    by_method = Counter(s["method"] for s in specs)
    print(f"Planned runs: {len(specs)} total")
    for m in METHODS:
        if m in by_method:
            print(f"  {m}: {by_method[m]}")
    if platform:
        sample_er = next((s for s in specs if s["method"] == "lio_er"), None)
        sample_ipd = next((s for s in specs if s["method"] == "lio_ipd"), None)
        if sample_er:
            print(
                f"  ER target example: {os.path.join(RESULTS_ROOT, sample_er['exp_name'], sample_er['dir_name'])}/"
            )
        if sample_ipd:
            print(
                f"  IPD target example: {os.path.join(RESULTS_ROOT, sample_ipd['exp_name'], sample_ipd['dir_name'])}/"
            )
    print()
    for i, spec in enumerate(specs, 1):
        parts = [
            f"{i:4d}",
            spec["method"],
            spec["exp_name"],
            spec["dir_name"],
            f"seed={spec['seed']}",
        ]
        for k in ("n_agents", "min_at_lever", "w_lever", "w_door", "n_episodes", "period", "n_eval"):
            if k in spec:
                parts.append(f"{k}={spec[k]}")
        print("  ".join(str(p) for p in parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["lio_er", "lio_eia_er", "lio_ipd", "lio_eia_ipd", "all"],
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--platform", type=str, default="nano")
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Real mode only: skip runs whose log.csv has enough data rows",
    )
    parser.add_argument(
        "--run_timeout",
        type=int,
        default=None,
        help="Per-run subprocess timeout in seconds (default: none)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of seeds S in 1..n_seeds (seed = 12340 + S); real matrix only",
    )
    args = parser.parse_args()

    if args.smoke:
        run_mode: Mode = "smoke"
    elif args.preflight:
        run_mode = "preflight"
    else:
        run_mode = "real"

    if args.skip_existing and run_mode != "real":
        parser.error("--skip_existing applies to real mode only")

    specs = build_matrix(args.method, run_mode, args.platform, args.n_seeds)

    if args.dry_run:
        _print_spec_table(specs, args.platform if run_mode == "real" else "")
        sys.exit(0)

    sys.path.insert(0, os.path.dirname(__file__))
    from ntfy_notify import notify

    host = socket.gethostname()
    t0 = time.time()

    to_run: list[dict[str, Any]] = []
    skipped = 0
    if args.skip_existing:
        for spec in specs:
            if is_run_complete(spec):
                skipped += 1
            else:
                to_run.append(spec)
    else:
        to_run = specs

    notify(
        f"method={args.method} mode={run_mode} planned={len(specs)} "
        f"run={len(to_run)} skipped={skipped} host={host}",
        title="REFiNE start",
    )

    ok = 0
    fail = 0
    timed_out = 0
    failed_specs: list[dict[str, Any]] = []
    validate = run_mode in ("smoke", "preflight")

    verbose_ntfy = os.environ.get("REFINE_NTFY_VERBOSE", "").strip() == "1"

    if not to_run:
        print(f"No runs to execute ({skipped} skipped as complete).")
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
            futures = {
                pool.submit(
                    run_one,
                    spec,
                    args.use_gpu,
                    args.threads,
                    args.platform,
                    validate,
                    args.run_timeout,
                ): spec
                for spec in to_run
            }
            for fut in as_completed(futures):
                spec, status, exit_code, validation_errors = fut.result()
                label = f"{spec['method']} {spec['dir_name']} seed={spec['seed']}"
                log_path = _log_path(spec)
                rows = count_data_rows(result_log_csv(spec))

                if status == "timeout":
                    timed_out += 1
                    fail += 1
                    failed_specs.append(spec)
                    tail = _tail_lines(log_path)
                    notify(
                        f"TIMEOUT {label}\n{tail}",
                        title="REFiNE ERROR",
                        priority="high",
                        tags="rotating_light",
                    )
                    print(f"TIMEOUT: {label} (see {log_path})", file=sys.stderr)
                elif exit_code != 0 or validation_errors:
                    fail += 1
                    failed_specs.append(spec)
                    tail = _tail_lines(log_path)
                    notify(
                        f"{label} FAILED (exit={exit_code})\n{tail}",
                        title="REFiNE ERROR",
                        priority="high",
                        tags="rotating_light",
                    )
                    if validation_errors:
                        for err in validation_errors:
                            print(f"VALIDATION FAIL: {err}", file=sys.stderr)
                    else:
                        print(f"FAIL: {label} (see {log_path})", file=sys.stderr)
                else:
                    ok += 1
                    print(f"OK: {label} log_rows={rows}")
                    if verbose_ntfy:
                        notify(f"{label} done rows={rows}", title="REFiNE", priority="low")

    elapsed = time.time() - t0
    notify(
        f"method={args.method} mode={run_mode} ok={ok} fail={fail} "
        f"skipped={skipped} timeout={timed_out} elapsed={elapsed:.0f}s host={host}",
        title="REFiNE done",
    )

    print()
    print("=" * 60)
    print(
        f"Batch {args.method} ({run_mode}): ok={ok} fail={fail} "
        f"skipped={skipped} timeout={timed_out} elapsed={elapsed:.1f}s"
    )
    print(f"Per-run logs: {RUN_LOGS_DIR}")
    print(f"Results:      {RESULTS_ROOT}")
    if failed_specs:
        print("Failed runs:")
        for spec in failed_specs:
            print(f"  {spec['exp_name']}/{spec['dir_name']} ({spec['method']})")

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
