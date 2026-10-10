"""Execute reviewed energy manifests in fixed shards, with external audit logs.

No command regeneration, resume, retry or work queue. Linux wait4 supplies RSS.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sensys_runs.energy_profile_campaign import ARMS
from sensys_runs.ntfy_notify import notify

CHILD_ENV_ADDITIONS = {"PYTHONHASHSEED": "0"}
TIMING_FIELDS = ["time", "exp_name", "method", "profile", "seed", "rc",
                 "wall_seconds", "peak_rss_mb", "error"]
STARTUP_TIMEOUT_SECONDS = 60
WAIT4_SUPPORTED = hasattr(os, "wait4") and sys.platform.startswith("linux")


class RunnerError(ValueError):
    """Invalid launch configuration, reported with exit code 2."""


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True,
                            capture_output=True)
    if result.returncode:
        raise RunnerError(result.stderr.strip() or "git " + " ".join(args) + " failed")
    return result.stdout.strip()


def option(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise RunnerError("manifest argv requires exactly one " + name)
    index = argv.index(name) + 1
    if index == len(argv):
        raise RunnerError("missing value for " + name)
    return argv[index]


def default_console_dir(manifest: dict) -> Path:
    """The results-root name separates smoke from formal without touching results."""
    roots = {Path(row["output_directory"]).parent for row in manifest["runs"]}
    if len(roots) != 1:
        raise RunnerError("Manifest requires one results root")
    return Path.home() / "iccps_runs" / (next(iter(roots)).name + "_console")


def check_fresh(row: dict) -> None:
    if git("status", "--porcelain"):
        raise RunnerError("repository is not porcelain-empty")
    target = Path(row["output_directory"])
    if target.exists() or target.is_symlink():
        raise RunnerError("target already exists: " + str(target))


def shard_plan(manifest: dict, shard: int, count: int) -> list[dict]:
    if count < 1 or not 0 <= shard < count:
        raise RunnerError("require --of >= 1 and 0 <= --shard < --of")
    if manifest.get("wave") == 3:
        from sensys_runs.energy_reward_campaign import WAVE3_HARDWARE
        ordered = sorted(manifest["runs"], key=lambda r: (
            r["seed"], WAVE3_HARDWARE.index(r["hardware_profile"])))
        return ordered[shard::count]
    if manifest.get("wave") == 2:
        from sensys_runs.energy_hardware_campaign import WAVE2_ARMS
        ordered = sorted(manifest["runs"], key=lambda r: (
            r["seed"], WAVE2_ARMS.index((r["method"], r["training_profile"])) ))
        return ordered[shard::count]
    ordered = sorted(manifest["runs"], key=lambda r: (
        r["seed"], ARMS.index((r["method"], r["training_profile"]))))
    return ordered[shard::count]


def validate(manifest: dict, console: Path) -> None:
    if manifest["schema_version"] != 1 or not manifest["runs"]:
        raise RunnerError("unsupported or empty manifest")
    if Path(manifest["cwd"]).resolve() != REPO_ROOT.resolve():
        raise RunnerError("manifest cwd must be this repository")
    if manifest["source"]["git_commit"] != git("rev-parse", "HEAD"):
        raise RunnerError("manifest commit does not equal HEAD")
    if git("status", "--porcelain"):
        raise RunnerError("repository is not porcelain-empty")
    if not os.environ.get("NTFY_TOPIC", "").strip():
        raise RunnerError("NTFY_TOPIC is required")
    if manifest["threads"] != 1:
        raise RunnerError("campaign threads must be 1")
    repo = REPO_ROOT.resolve()
    if console == repo or repo in console.parents:
        raise RunnerError("console directory must be outside the repository")
    targets = set()
    roots = set()
    for row in manifest["runs"]:
        argv = row["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise RunnerError("argv must be a nonempty string list")
        if option(argv, "--threads") != "1":
            raise RunnerError("every run must use threads 1")
        exp = option(argv, "--exp_name")
        directory = option(argv, "--dir_name")
        if any(not s or s in (".", "..") or "/" in s or "\\" in s for s in (exp, directory)):
            raise RunnerError("exp_name and dir_name must be directory components")
        target = Path(row["output_directory"]).resolve()
        expected = (repo / "lio/results" / exp / directory).resolve()
        if target != expected or target in targets:
            raise RunnerError("duplicate or inconsistent target directory")
        if (option(argv, "--method"), option(argv, "--cost_profile")) != (row["method"], row["training_profile"]):
            raise RunnerError("argv arm does not match manifest")
        if int(option(argv, "--seed")) != row["seed"]:
            raise RunnerError("argv seed does not match manifest")
        if "--energy_in_reward" in argv:
            from sensys_runs.energy_reward_runtime import REWARD_FORM, registered_beta
            cfg = row["effective_config"]
            if (argv.count("--energy_in_reward") != 1 or row["method"] != "lio_er"
                    or not cfg.get("energy_in_reward") or cfg.get("energy_reward_beta") != registered_beta()
                    or cfg.get("energy_reward_form") != REWARD_FORM):
                raise RunnerError("Invalid energy-reward manifest")
            from sensys_runs.energy_profiles import get_profile
            get_profile(row["training_profile"])
        elif row.get("effective_config", {}).get("energy_in_reward", False):
            raise RunnerError("Energy-reward config requires its recorded argv flag")
        # Extend acceptance per row only after validating the explicit objective.
        ARMS = globals()["ARMS"]
        if "--energy_in_reward" in argv:
            ARMS = (*ARMS, (row["method"], row["training_profile"]))
        if (row["method"], row["training_profile"]) not in ARMS:
            raise RunnerError("unknown campaign arm")
        if "--hardware_profile" in argv:
            from sensys_runs.energy_profiles import get_hardware_profile
            hardware = get_hardware_profile(option(argv, "--hardware_profile"))
            cfg = row["effective_config"]
            if (cfg["hardware_profile_lambda"] != list(hardware.vector(cfg["env"]["n_agents"]))
                    or cfg["hardware_profile"] != hardware.snapshot(cfg["env"]["n_agents"])):
                raise RunnerError("manifest hardware config differs from argv")
        roots.add(target.parent)
        targets.add(target)
        if target.exists() or target.is_symlink():
            raise RunnerError("target already exists: " + str(target))
    if len(roots) != 1 or manifest["n_runs"] != len(targets):
        raise RunnerError("inconsistent campaign root or run count")
    root = next(iter(roots))
    git("check-ignore", "--quiet", str(root) + os.sep)
    if console == root or root in console.parents:
        raise RunnerError("console directory must be outside results")


def measure(argv: list[str], console, timeout: float | None) -> tuple[int, float, float]:
    env = os.environ.copy()
    env.update(CHILD_ENV_ADDITIONS)
    started = time.monotonic()
    process = subprocess.Popen(argv, cwd=REPO_ROOT, env=env,
                               stdout=console, stderr=subprocess.STDOUT)
    timed_out = False
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            break
        if timeout is not None and time.monotonic() - started >= timeout:
            # Popen.kill() can poll/reap first on POSIX, losing wait4's rusage.
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(process.pid, 0)
            timed_out = True
            break
        time.sleep(0.05)
    process.returncode = os.waitstatus_to_exitcode(status)
    # Linux reports ru_maxrss in KiB; use MiB for the peak_rss_mb column.
    return (124 if timed_out else process.returncode,
            time.monotonic() - started, usage.ru_maxrss / 1024.0)


def execute(manifest: dict, rows: list[dict], console_dir: Path,
            shard: int, count: int, timeout: float | None, synchronize: str | None) -> int:
    console_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard-{shard}-of-{count}"
    log_path, timing_path = console_dir / (stem + ".log"), console_dir / (stem + ".csv")
    with log_path.open("x", encoding="utf-8", buffering=1) as log, timing_path.open("x", encoding="utf-8", newline="") as timing:
        def emit(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")

        emit("CHILD_ENV_ADDITIONS " + json.dumps(CHILD_ENV_ADDITIONS, sort_keys=True))
        writer = csv.DictWriter(timing, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        timing.flush()
        if synchronize:
            # All shards validate all absent targets before any shard may train.
            ready = console_dir / (synchronize + "-" + stem + ".ready")
            with ready.open("x", encoding="utf-8") as stream:
                stream.write(manifest["source"]["git_commit"])
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while not all((console_dir / f"{synchronize}-shard-{i}-of-{count}.ready").exists() for i in range(count)):
                if time.monotonic() >= deadline:
                    emit("CONFIG_ERROR: not all shards passed startup validation")
                    notify("Shard startup barrier timed out", title=stem, priority="high")
                    return 2
                time.sleep(0.1)
        failed = 0
        for index, row in enumerate(rows, 1):
            rc, wall, rss, error = 2, 0.0, 0.0, ""
            try:
                check_fresh(row)
                emit("ARGV " + json.dumps(row["argv"]))
                rc, wall, rss = measure(row["argv"], log, timeout)
            except (OSError, RunnerError) as exc:
                error = str(exc)
                emit("RUN_ERROR " + error)
            failed += int(rc != 0)
            stamp = datetime.now(timezone.utc).isoformat()
            exp = option(row["argv"], "--exp_name")
            progress = (f"{stamp} {exp} rc={rc} ({index}/{len(rows)}, "
                        f"{int(wall)//60:02d}:{int(wall)%60:02d}) {row['method']} "
                        f"{row['training_profile']} seed={row['seed']} peak_rss_mb={rss:.3f}")
            emit(progress)
            writer.writerow(dict(time=stamp, exp_name=exp, method=row["method"],
                                 profile=row["training_profile"], seed=row["seed"], rc=rc,
                                 wall_seconds=wall, peak_rss_mb=rss, error=error))
            timing.flush()
            notify(progress + ("\n" + error if error else ""), title=stem,
                   priority="high" if rc else "default")
        summary = f"SHARD_SUMMARY {stem} runs={len(rows)} failed={failed}"
        emit(summary)
        notify(summary, title=stem, priority="high" if failed else "default")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--console-dir", type=Path)
    parser.add_argument("--timeout", type=float, help="Per-run seconds; default: no timeout")
    parser.add_argument("--start-together", metavar="TOKEN", help="Internal launcher startup token")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        name = manifest["campaign"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise RunnerError("invalid campaign name")
        console = (args.console_dir or Path.home() / "iccps_runs" / (name + "_console")).resolve()
        if args.console_dir is None:
            console = default_console_dir(manifest).resolve()
        validate(manifest, console)
        rows = shard_plan(manifest, args.shard, args.of)
        if args.start_together and (len(args.start_together) != 32 or
                                   any(c not in "0123456789abcdef" for c in args.start_together)):
            raise RunnerError("invalid startup token")
        if args.timeout is not None and args.timeout <= 0:
            raise RunnerError("timeout must be positive")
        # Any old shard artifacts mean this console root is not a fresh launch.
        if not args.start_together and console.exists() and any(console.iterdir()):
            raise RunnerError("console directory is not fresh")
        for row in rows:
            print("PLAN " + json.dumps(row["argv"]), flush=True)
        if args.dry_run:
            return 0
        if not WAIT4_SUPPORTED:
            raise RunnerError("execution requires Linux os.wait4; dry-run is portable")
        return execute(manifest, rows, console, args.shard, args.of, args.timeout, args.start_together)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("CONFIG_ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
