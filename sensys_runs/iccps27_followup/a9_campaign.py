"""Build only registered A9 manifests; this module never launches training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensys_runs import energy_profile_campaign as base
from sensys_runs.energy_profiles import get_hardware_profile, get_profile
from sensys_runs.energy_profile_runtime import validate_run
from sensys_runs.energy_reward_runtime import configure

CAMPAIGN = "iccps27_a9_20261028"
METHOD = "lio_er"
PROFILE = "uniform_v1"
HARDWARE = "homogeneous_v1"
FORMAL_SEEDS = tuple(range(12341, 12351))
SMOKE_SEEDS = FORMAL_SEEDS[:2]
TASK = (4, 2)
INVENTORY_FIELDS = ("run_id", "arm", "method", "seed", "checkpoint_prefix", "evidence")
INVENTORY_ARM = "lio_er__uniform_v1__homogeneous_v1__energy_reward"


def source_identity() -> dict:
    """Preserve the existing identity fields and pin the additive generator."""
    identity = base.source_identity()
    for relative in (
        "sensys_runs/iccps27_followup/a9_campaign.py",
        "sensys_runs/energy_reward_runtime.py",
        "sensys_runs/energy_reward_campaign.py",
        "sensys_runs/energy_profile_runner.py",
    ):
        identity["sha256"][relative] = hashlib.sha256(
            (REPO_ROOT / relative).read_bytes()).hexdigest()
    return identity


def build_campaign(campaign: str, platform: str, threads: int,
                   use_gpu: bool = False, smoke: bool = False) -> dict:
    """Reuse A2's factories and argument builder with only the uniform profile."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", campaign):
        raise base.CampaignError("campaign must be one safe directory component")
    if not platform.strip() or threads != 1:
        raise base.CampaignError("A9 requires an explicit platform and threads 1")
    if TASK != base.TASK or TASK not in base.run_batch.SIZES:
        raise base.CampaignError("A9 requires the registered ER(4,2) task")
    registered_seeds = tuple(base.run_batch._seed(i) for i in base.SEED_INDICES)
    if registered_seeds != FORMAL_SEEDS:
        raise base.CampaignError("The existing campaign seed registry changed")
    seeds = SMOKE_SEEDS if smoke else FORMAL_SEEDS
    exp_name = campaign + ("_smoke" if smoke else "")
    hardware = get_hardware_profile(HARDWARE)
    runs = []
    for seed in seeds:
        cfg = base.load_config(METHOD)
        directory = f"{METHOD}_{PROFILE}_{HARDWARE}_energy_reward_s{seed}"
        cfg.main.exp_name, cfg.main.dir_name = exp_name, directory
        cfg.main.seed, cfg.main.use_gpu = seed, use_gpu
        cfg.env.n_agents, cfg.env.min_at_lever = TASK
        cfg.lio.min_at_lever = TASK[1]
        if smoke:
            short = base.run_batch.build_matrix(METHOD, "smoke")[0]
            cfg.alg.n_episodes = short["n_episodes"]
            cfg.alg.period = short["period"]
        cfg.cost_profile = get_profile(PROFILE).snapshot()
        cfg.hardware_profile = hardware.snapshot(cfg.env.n_agents)
        cfg.hardware_profile_lambda = list(hardware.vector(cfg.env.n_agents))
        configure(cfg)
        if (cfg.energy_reward_beta != 1.0 or cfg.energy_reward_form != "episode_mean"
                or cfg.lio.Fairness_multiplier != 0.0):
            raise base.CampaignError("Registered A9 energy objective changed")
        if (cfg.alg.n_eval < TASK[0] or cfg.alg.n_test < TASK[0]
                or cfg.alg.n_episodes <= 0 or cfg.alg.period <= 0
                or cfg.alg.n_episodes % cfg.alg.period):
            raise base.CampaignError("Invalid registered evaluation schedule")
        spec = dict(method=METHOD, exp_name=exp_name, dir_name=directory, seed=seed,
                    n_episodes=cfg.alg.n_episodes, period=cfg.alg.period,
                    n_eval=cfg.alg.n_eval, n_agents=cfg.env.n_agents,
                    min_at_lever=cfg.env.min_at_lever)
        argv = base.run_batch.spec_to_argv(spec, use_gpu, threads, platform)
        argv.extend(["--cost_profile", PROFILE, "--hardware_profile", HARDWARE,
                     "--energy_in_reward"])
        output = REPO_ROOT / "lio" / "results" / exp_name / directory
        validate_run(PROFILE, METHOD, str(output))
        runs.append(dict(arm="A9", method=METHOD, training_profile=PROFILE,
                         hardware_profile=HARDWARE, objective="energy_reward", seed=seed,
                         argv=argv, command=base.command_text(argv), effective_config=cfg,
                         output_directory=str(output),
                         expected_checkpoint=str(output / cfg.main.model_name),
                         checkpoint_episode=cfg.alg.n_episodes))
    if len({r["output_directory"] for r in runs}) != len(seeds):
        raise base.CampaignError("Duplicate A9 output directory")
    return dict(schema_version=1, campaign=campaign, wave=3, arm="A9",
                mode="smoke" if smoke else "campaign",
                execution_description="simulation with literature-informed cost sensitivity",
                platform=platform, threads=threads, use_gpu=use_gpu,
                generator_python=sys.executable, cwd=str(REPO_ROOT),
                command_shell="powershell" if os.name == "nt" else "posix",
                n_runs=len(runs), n_seeds_per_arm=len(seeds),
                cost_units="normalized_modeled_cost", source=source_identity(),
                time_budget=dict(status="unmeasured", measured_serial_seconds=None,
                                 note="Use the registered full-run median; smoke is not a full-run estimate.",
                                 serial_formula="n_seeds_per_arm * measured full-run seconds for A9"),
                runs=runs)


def inventory_rows(manifest: dict) -> list[dict]:
    """Use the frozen six-column inventory schema without importing analysis code."""
    rows = []
    for run in manifest["runs"]:
        main = run["effective_config"]["main"]
        run_id = f"{main['exp_name']}/{main['dir_name']}"
        rows.append(dict(run_id=run_id, arm=INVENTORY_ARM, method=METHOD,
                         seed=str(run["seed"]),
                         checkpoint_prefix=f"lio/results/{run_id}/model.ckpt",
                         evidence="[verified from file]"))
    return rows


def preflight_inventory_outputs(manifest_path: Path, inventory_path: Path) -> None:
    """Check both exclusive destinations before creating either output."""
    results = (REPO_ROOT / "lio" / "results").resolve()
    if manifest_path.resolve() == inventory_path.resolve():
        raise base.CampaignError("Manifest and inventory outputs must differ")
    for candidate in (manifest_path, inventory_path):
        path = candidate.resolve()
        if (path == results or results in path.parents or candidate.exists()
                or candidate.is_symlink() or not candidate.parent.is_dir()):
            raise base.CampaignError(
                "Both outputs must be fresh files in existing directories outside lio/results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Generate two short A9 rows, seeds 12341 and 12342")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inventory-out", type=Path,
                        help="Fresh six-column inventory; requires --out and forbids --dry-run")
    args = parser.parse_args(argv)
    if args.dry_run and args.out:
        parser.error("--dry-run cannot be combined with --out")
    if args.inventory_out and (not args.out or args.dry_run):
        parser.error("--inventory-out requires --out and cannot be combined with --dry-run")
    try:
        if args.inventory_out:
            preflight_inventory_outputs(args.out, args.inventory_out)
        if args.out:
            path = args.out.resolve()
            results = (REPO_ROOT / "lio" / "results").resolve()
            if (path == results or results in path.parents or path.exists()
                    or args.out.is_symlink()):
                raise base.CampaignError("--out must be a fresh file outside lio/results")
        manifest = build_campaign(args.campaign, args.platform, args.threads,
                                  args.use_gpu, args.smoke)
        text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            if args.inventory_out:
                with args.inventory_out.open("x", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=INVENTORY_FIELDS,
                                            lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(inventory_rows(manifest))
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
