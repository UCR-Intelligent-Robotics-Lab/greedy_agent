#!/usr/bin/env python3
"""Non-intrusive ntfy watchdog for sensys baseline batches (reads env at runtime)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_STATE = os.path.join(os.path.dirname(__file__), ".watch_state.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor ER/IPD batch progress via ntfy.")
    p.add_argument("--interval", type=int, default=300, help="Loop interval (seconds)")
    p.add_argument("--stall", type=int, default=1800, help="Stall threshold (seconds)")
    p.add_argument("--heartbeat", type=int, default=7200, help="Heartbeat interval (seconds)")
    p.add_argument(
        "--results-glob",
        default="lio/results/nano_*/*/log.csv",
        help="Glob for result log.csv files (relative to repo root)",
    )
    p.add_argument(
        "--runlog-glob",
        default="sensys_runs/run_logs/nano_*.log",
        help="Glob for per-run subprocess logs",
    )
    p.add_argument("--state", default=DEFAULT_STATE, help="Persistent state JSON path")
    return p.parse_args()


def _expected_rows(log_csv: str) -> int:
    return 20 if "ipd" in log_csv.lower() else 50


def _count_data_rows(log_csv: str) -> int:
    try:
        with open(log_csv, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        return max(0, len(lines) - 1)
    except OSError:
        return 0


def _tail_lines(path: str, n: int = 15) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def _load_state(path: str) -> dict:
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "errored": [],
        "stall_pinged": False,
        "last_heartbeat": 0.0,
        "last_max_mtime": 0.0,
    }


def _save_state(path: str, state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as exc:
        print(f"[watch] WARNING: could not save state: {exc}")


def _expected_total() -> int | None:
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from run_batch import build_matrix

        return len(build_matrix("all", "real", "nano", 10))
    except Exception as exc:
        print(f"[watch] could not compute expected total: {exc}")
        return None


def _scan_results(results_glob: str) -> tuple[list[str], int, int, float]:
    """Return (paths, complete_count, incomplete_count, max_mtime)."""
    pattern = os.path.join(REPO_ROOT, results_glob)
    paths = sorted(glob.glob(pattern))
    complete = 0
    incomplete = 0
    max_mtime = 0.0
    for p in paths:
        try:
            mt = os.path.getmtime(p)
            max_mtime = max(max_mtime, mt)
        except OSError:
            continue
        need = _expected_rows(p)
        rows = _count_data_rows(p)
        if rows >= need:
            complete += 1
        else:
            incomplete += 1
    return paths, complete, incomplete, max_mtime


def _run_name_from_log(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".log"):
        return base[:-4].replace("__", "/")
    return base


def _check_errors(runlog_glob: str, state: dict, notify) -> None:
    pattern = os.path.join(REPO_ROOT, runlog_glob)
    errored: set[str] = set(state.get("errored", []))
    for path in glob.glob(pattern):
        if path in errored:
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError:
            continue
        if "Traceback (most recent call last)" not in body:
            continue
        name = _run_name_from_log(path)
        tail = _tail_lines(path, 15)
        msg = f"ERROR in {name}\n{tail}"
        print(f"[watch] ERROR detected: {name}")
        notify(msg, title="REFiNE ERROR", priority="high", tags="rotating_light")
        errored.add(path)
    state["errored"] = sorted(errored)


def _fmt_age(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m:02d}:{s:02d}"


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)

    sys.path.insert(0, os.path.dirname(__file__))
    from ntfy_notify import notify

    state_path = args.state
    if not os.path.isabs(state_path):
        state_path = os.path.join(REPO_ROOT, state_path)

    state = _load_state(state_path)
    total = _expected_total()
    host = socket.gethostname()

    paths, complete, incomplete, max_mtime = _scan_results(args.results_glob)
    n_found = len(paths)
    total_s = str(total) if total is not None else "?"

    startup_msg = (
        f"watchdog up: {n_found} log.csv found, {complete} complete "
        f"(>=expected rows), host={host}, matrix total={total_s}"
    )
    print(f"[watch] {startup_msg}")
    notify(startup_msg, title="REFiNE watch", priority="default")

    state["last_heartbeat"] = time.time()
    state["last_max_mtime"] = max_mtime
    _save_state(state_path, state)

    while True:
        try:
            time.sleep(args.interval)

            paths, complete, incomplete, max_mtime = _scan_results(args.results_glob)
            now = time.time()

            _check_errors(args.runlog_glob, state, notify)

            prev_max = float(state.get("last_max_mtime", 0.0))
            if max_mtime > prev_max + 1e-6:
                if state.get("stall_pinged"):
                    print("[watch] log.csv updated; clearing stall flag")
                state["stall_pinged"] = False
            state["last_max_mtime"] = max_mtime

            if (
                paths
                and incomplete > 0
                and max_mtime > 0
                and (now - max_mtime) > args.stall
                and not state.get("stall_pinged", False)
            ):
                mins = int((now - max_mtime) // 60)
                msg = (
                    f"possible stall: no log.csv update in {mins} min "
                    f"while {incomplete} run(s) incomplete ({complete} complete)"
                )
                print(f"[watch] STALL: {msg}")
                notify(msg, title="REFiNE STALL", priority="high", tags="rotating_light")
                state["stall_pinged"] = True

            last_hb = float(state.get("last_heartbeat", 0.0))
            if now - last_hb >= args.heartbeat:
                age = _fmt_age(now - max_mtime) if max_mtime else "n/a"
                total_s = str(total) if total is not None else "?"
                msg = (
                    f"progress: {complete}/{total_s} runs complete, "
                    f"latest update {age} ago, {len(paths)} log.csv tracked"
                )
                print(f"[watch] HEARTBEAT: {msg}")
                notify(msg, title="REFiNE watch", priority="low")
                state["last_heartbeat"] = now

            _save_state(state_path, state)

            if total is not None and complete >= total and len(paths) >= total:
                done_msg = f"ALL runs complete ({complete}/{total}), host={host}"
                print(f"[watch] {done_msg}")
                notify(done_msg, title="REFiNE done", priority="default")
                sys.exit(0)

        except KeyboardInterrupt:
            print("[watch] interrupted; exiting")
            sys.exit(0)
        except Exception as exc:
            print(f"[watch] loop error (continuing): {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
