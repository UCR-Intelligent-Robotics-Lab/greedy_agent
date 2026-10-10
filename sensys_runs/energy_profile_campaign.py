"""Build the fixed ICCPS 2027 clean-ER cost campaign; never execute training.

The minimal energy plan fixes ER(4,2), four arms, and ten existing seed indices.
Learning settings come from live config factories; commands reuse run_batch's
run_single argument builder. Config files are loaded directly to avoid importing
lio.alg's eager TensorFlow dependencies during a dry run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sensys_runs import run_batch
from sensys_runs.run_single import METHOD_MAP
from sensys_runs.energy_profiles import get_profile
from sensys_runs.energy_profile_runtime import validate_run

# Explicit study decisions from REFiNE_Minimal_Energy_Code_Plan.md, Section 6.
CAMPAIGN = "iccps2027_energy_v1"
TASK = (4, 2)
SEED_INDICES = tuple(range(1, 11))
ARMS = (("lio_er", "uniform_v1"), ("refine_er", "uniform_v1"),
        ("refine_er", "low_contrast_v1"), ("refine_er", "paper_ratio_v1"))


class CampaignError(ValueError):
    """Invalid campaign configuration; the CLI exits with code 2."""


def load_config(method: str):
    """Execute the existing config factory without executing lio.alg.__init__."""
    basename = METHOD_MAP[method][0]
    path = REPO_ROOT / "lio" / "alg" / (basename + ".py")
    spec = importlib.util.spec_from_file_location("_energy_campaign_" + basename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def command_text(argv: list[str]) -> str:
    """Quote for the generator host: PowerShell on Windows, POSIX shell otherwise."""
    if os.name == "nt":
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in argv)
    return shlex.join(argv)


def source_identity() -> dict:
    """Pin generator inputs without requiring git for an exported source tree."""
    paths = ["sensys_runs/energy_profile_campaign.py", "sensys_runs/run_batch.py",
             "sensys_runs/run_single.py", "sensys_runs/energy_profiles.py",
             "sensys_runs/energy_profile_runtime.py", "sensys_runs/energy_profile_eval.py",
             "lio/alg/train_lio_er.py", "lio/alg/train_REFiNE_er.py",
             "lio/alg/config_room_lio.py", "lio/alg/config_room_REFiNE.py"]
    identity = {"sha256": {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in paths}}
    for label, args in (("git_commit", ["rev-parse", "HEAD"]),
                        ("git_describe", ["describe", "--tags", "--always"]),
                        ("git_status", ["status", "--porcelain"])):
        try:
            identity[label] = subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                                       stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            identity[label] = None
    return identity


def build_campaign(campaign: str, platform: str, threads: int,
                   use_gpu: bool = False, smoke: bool = False) -> dict:
    """Construct and validate the entire matrix before returning any commands."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", campaign):
        raise CampaignError("campaign must be one directory component using letters, digits, _ or -")
    if not platform.strip() or threads < 1:
        raise CampaignError("platform is required and threads must be positive")
    if TASK not in run_batch.SIZES:
        raise CampaignError("Requested ER task is no longer in the existing benchmark registry")
    runs = []
    indices = SEED_INDICES[:1] if smoke else SEED_INDICES
    exp_name = campaign + ("_smoke" if smoke else "")
    for method, profile in ARMS:
        for index in indices:
            cfg = load_config(method)
            seed = run_batch._seed(index)
            directory = f"{method}_{profile}_s{seed}"
            cfg.main.exp_name, cfg.main.dir_name = exp_name, directory
            cfg.main.seed, cfg.main.use_gpu = seed, use_gpu
            cfg.env.n_agents, cfg.env.min_at_lever = TASK
            cfg.lio.min_at_lever = TASK[1]
            if smoke:
                # Reuse the existing short-run duration and period; keep the
                # campaign's periodic n_eval and final n_test for ER(4,2).
                short = run_batch.build_matrix(method, "smoke")[0]
                cfg.alg.n_episodes, cfg.alg.period = short["n_episodes"], short["period"]
            spec = dict(method=method, exp_name=exp_name, dir_name=directory, seed=seed,
                        n_episodes=cfg.alg.n_episodes, period=cfg.alg.period, n_eval=cfg.alg.n_eval,
                        n_agents=cfg.env.n_agents, min_at_lever=cfg.env.min_at_lever)
            if method == "refine_er":
                cfg.lio.Fairness_multiplier = 0.0
                spec.update(fairness_mult=cfg.lio.Fairness_multiplier, energy_weight=cfg.lio.energy_weight)
            cfg.cost_profile = get_profile(profile).snapshot()
            argv = run_batch.spec_to_argv(spec, use_gpu, threads, platform)
            argv.extend(["--cost_profile", profile])
            output = REPO_ROOT / "lio" / "results" / exp_name / directory
            validate_run(profile, method, str(output))
            if cfg.alg.n_eval < TASK[0] or cfg.alg.n_test < TASK[0]:
                raise CampaignError("Existing evaluator requires n_eval and n_test >= n_agents")
            if cfg.alg.n_episodes <= 0 or cfg.alg.period <= 0 or cfg.alg.n_episodes % cfg.alg.period:
                raise CampaignError("Training duration must be a positive multiple of evaluation period")
            runs.append(dict(method=method, training_profile=profile, seed=seed,
                             argv=argv, command=command_text(argv), effective_config=cfg,
                             output_directory=str(output), expected_checkpoint=str(output / cfg.main.model_name),
                             checkpoint_episode=cfg.alg.n_episodes))
    if len({r["output_directory"] for r in runs}) != len(runs):
        raise CampaignError("Duplicate output paths in the campaign")
    # Both arms must use the same task, networks and evaluation schedule.
    reference = runs[0]["effective_config"]
    for row in runs:
        cfg = row["effective_config"]
        if (cfg.env != reference.env or cfg.nn != reference.nn
                or any(cfg.alg[k] != reference.alg[k] for k in ("n_episodes", "period", "n_eval", "n_test"))
                or any(cfg.lio[k] != reference.lio[k] for k in
                       ("gamma", "epsilon_start", "epsilon_end", "epsilon_div"))):
            raise CampaignError("Cross-arm task, network or schedule mismatch")
    return dict(schema_version=1, campaign=campaign, mode="smoke" if smoke else "campaign",
                execution_description="simulation with literature-informed cost sensitivity",
                platform=platform, threads=threads, use_gpu=use_gpu,
                generator_python=sys.executable, cwd=str(REPO_ROOT),
                command_shell="powershell" if os.name == "nt" else "posix",
                n_runs=len(runs), n_seeds_per_arm=len(indices),
                cost_units="normalized_modeled_cost", source=source_identity(),
                time_budget=dict(status="unmeasured", measured_serial_seconds=None,
                                 note="Measure on the execution host; short smoke duration is not a full-run estimate.",
                                 serial_formula="10 * sum(measured full-run seconds for each of the four arms)"),
                runs=runs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN)
    parser.add_argument("--platform", required=True, help="Actual execution platform label, e.g. nano or local")
    parser.add_argument("--threads", type=int, required=True, help="Explicit execution-host thread limit")
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--smoke", action="store_true", help="Generate four short checks instead of forty full runs")
    parser.add_argument("--dry-run", action="store_true", help="Print only; never create files or launch training")
    parser.add_argument("--format", choices=("json", "commands"), default="json")
    parser.add_argument("--out", type=Path, help="Optional fresh manifest/command file, outside lio/results")
    args = parser.parse_args()
    if args.dry_run and args.out:
        parser.error("--dry-run cannot be combined with --out")
    if args.out:
        output = args.out.resolve()
        results = (REPO_ROOT / "lio" / "results").resolve()
        if output == results or results in output.parents or output.exists() or args.out.is_symlink():
            parser.error("--out must be a fresh file outside lio/results")
    try:
        manifest = build_campaign(args.campaign, args.platform, args.threads, args.use_gpu, args.smoke)
        if args.format == "json":
            text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        else:
            text = (f"# Generated commands only; run manually in the recorded environment.\n"
                    f"# cwd: {manifest['cwd']}\n# shell: {manifest['command_shell']}\n"
                    + "\n".join(row["command"] for row in manifest["runs"]) + "\n")
        if args.out:
            with args.out.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
