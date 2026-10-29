"""Strict EIA manifest dispatch using the unchanged shard executor and gates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sensys_runs import energy_profile_runner as core
from sensys_runs.iccps27_followup import eia_campaign as campaign
from sensys_runs.iccps27_followup.eia_adapter import rebind


def expected_shards(manifest):
    return 2 if manifest.get("mode") == "smoke" else 5


def validate_eia(manifest, console, original_validate):
    """Extend method recognition while keeping every original gate in force."""
    original_validate(manifest, console)
    if manifest.get("mode") not in ("campaign", "smoke"):
        raise core.RunnerError("Unknown EIA mode")
    if type(manifest.get("fallback_declared")) is not bool:
        raise core.RunnerError("EIA fallback decision must be explicit")
    exact_tag = core.git("describe", "--tags", "--exact-match", "HEAD")
    if not exact_tag or exact_tag != manifest["source"].get("git_describe"):
        raise core.RunnerError("Manifest must name the exact checked-out release tag")
    hashes = manifest["source"].get("sha256", {})
    required = {"sensys_runs/iccps27_followup/" + name for name in
                ("eia_campaign.py", "eia_adapter.py", "eia_runner.py")}
    required.add("lio/alg/REFiNE_eia_er.py")
    if not required.issubset(hashes):
        raise core.RunnerError("EIA release source hashes are incomplete")
    for relative, digest in hashes.items():
        candidate = (REPO / relative).resolve()
        if (not candidate.is_relative_to(REPO) or not candidate.is_file()
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest):
            raise core.RunnerError("EIA source hash mismatch: " + relative)
    expected = campaign.build_campaign(manifest["campaign"], manifest["platform"],
                                       manifest["threads"], manifest["mode"] == "smoke",
                                       manifest["fallback_declared"])
    for field in ("n_runs", "n_seeds_per_arm", "weight_pairs", "planned_formal_runs", "methods"):
        if manifest.get(field) != expected[field]:
            raise core.RunnerError("EIA matrix declaration differs: " + field)
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    if canonical(manifest["runs"]) != canonical(expected["runs"]):
        raise core.RunnerError("EIA recorded run/config/argv differs from the registered matrix")


@contextmanager
def runner_scope():
    original_validate, original_plan = core.validate, core.shard_plan
    def validate(manifest, console):
        return validate_eia(manifest, console, original_validate)
    def plan(manifest, shard, count):
        if count != expected_shards(manifest):
            raise core.RunnerError("EIA requires K=2 smoke or K=5 formal")
        return original_plan(manifest, shard, count)
    with ExitStack() as stack:
        stack.enter_context(rebind(core, "ARMS", (*core.ARMS, *((method, "uniform_v1") for method in campaign.METHODS))))
        stack.enter_context(rebind(core, "validate", validate))
        stack.enter_context(rebind(core, "shard_plan", plan))
        yield


def tmux_plan(manifest_path, manifest, count, session, console, token):
    if count != expected_shards(manifest) or not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise core.RunnerError("EIA launch requires the registered K and a safe session name")
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise core.RunnerError("Invalid startup token")
    commands = []
    for shard in range(count):
        child = ["env", "NTFY_TOPIC=" + os.environ.get("NTFY_TOPIC", ""),
                 "NTFY_SERVER=" + os.environ.get("NTFY_SERVER", "https://ntfy.sh"),
                 "PYTHONDONTWRITEBYTECODE=1", sys.executable, "-B", str(Path(__file__)),
                 str(manifest_path), "--shard", str(shard), "--of", str(count),
                 "--console-dir", str(console), "--start-together", token]
        shell = shlex.join(["bash", "-c", "exec " + shlex.join(child)])
        prefix = ["tmux", "new-session", "-d", "-s", session] if shard == 0 else ["tmux", "new-window", "-t", session + ":"]
        commands.append([*prefix, "-n", "shard-" + str(shard), shell])
    return commands


def dispatch(args):
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    console = (args.console_dir or core.default_console_dir(manifest)).resolve()
    with runner_scope():
        core.validate(manifest, console)
        for shard in range(args.launch):
            core.shard_plan(manifest, shard, args.launch)
    if args.launch * manifest["threads"] > (os.cpu_count() or 1):
        raise core.RunnerError("K times threads exceeds available CPUs")
    if console.exists() or console.is_symlink():
        raise core.RunnerError("EIA console directory must be absent")
    if not shutil.which("tmux"):
        raise core.RunnerError("tmux is required")
    existing = subprocess.run(["tmux", "has-session", "-t", "=" + args.session], capture_output=True)
    if existing.returncode == 0:
        raise core.RunnerError("tmux session already exists")
    commands = tmux_plan(manifest_path, manifest, args.launch, args.session, console, uuid.uuid4().hex)
    for command in commands:
        print("LAUNCH_PLAN " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not core.WAIT4_SUPPORTED:
        raise core.RunnerError("Execution requires Linux os.wait4")
    console.parent.mkdir(parents=True, exist_ok=True)
    console.mkdir()
    for command in commands:
        subprocess.run(command, check=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--launch", type=int)
    pre.add_argument("--session")
    options, remaining = pre.parse_known_args(argv)
    try:
        if options.launch is None:
            if options.session is not None:
                raise core.RunnerError("--session requires --launch")
            with runner_scope():
                return core.main(remaining)
        if options.session is None or options.launch < 1:
            raise core.RunnerError("--launch requires positive K and --session")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("manifest", type=Path)
        parser.add_argument("--console-dir", type=Path)
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args(remaining)
        args.launch, args.session = options.launch, options.session
        return dispatch(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print("CONFIG_ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
