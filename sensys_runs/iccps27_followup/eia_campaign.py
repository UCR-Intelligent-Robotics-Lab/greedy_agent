"""Create the preregistered EIA matrix and inventory without running training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sensys_runs import energy_profile_campaign as base
from sensys_runs.energy_profiles import get_profile, get_hardware_profile
from sensys_runs.iccps27_followup.eia_adapter import METHODS, WEIGHTS

SEEDS = tuple(range(12341, 12351))
FIELDS = ("run_id", "arm", "method", "seed", "checkpoint_prefix", "evidence")


def weight_label(pair):
    return ",".join(str(float(value)) for value in pair)


def source_identity():
    identity = base.source_identity()
    for relative in ("sensys_runs/iccps27_followup/eia_adapter.py",
                     "sensys_runs/iccps27_followup/eia_runner.py",
                     "sensys_runs/iccps27_followup/eia_campaign.py",
                     "lio/alg/REFiNE_eia_er.py", "lio/alg/lio_eia_er.py",
                     "lio/alg/train_lio_eia_er.py", "lio/alg/train_REFiNE_eia_er.py"):
        identity["sha256"][relative] = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
    return identity


def build_campaign(campaign, platform, threads, smoke=False, drop_mid_weight=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", campaign):
        raise base.CampaignError("Campaign must be one safe directory component")
    if not platform.strip() or threads != 1:
        raise base.CampaignError("EIA requires an explicit platform and one thread")
    selected = (WEIGHTS[0], WEIGHTS[2]) if drop_mid_weight else WEIGHTS
    pairs = (WEIGHTS[1],) if smoke else selected
    seeds = SEEDS[:1] if smoke else SEEDS
    experiment = campaign + ("_smoke" if smoke else "")
    runs = []
    for method in METHODS:
        for pair in pairs:
            label = weight_label(pair)
            for seed in seeds:
                config = base.load_config(method)
                directory = f"{method}_uniform_v1_homogeneous_v1_w{label.replace('.', 'p').replace(',', '_')}_s{seed}"
                config.main.exp_name, config.main.dir_name = experiment, directory
                config.main.seed, config.main.use_gpu = seed, False
                config.env.n_agents, config.env.min_at_lever = (4, 2)
                config.lio.min_at_lever = 2
                config.lio.eia_w_lever, config.lio.eia_w_door = pair
                config.cost_profile = get_profile("uniform_v1").snapshot()
                hardware = get_hardware_profile("homogeneous_v1")
                config.hardware_profile = hardware.snapshot(4)
                config.hardware_profile_lambda = list(hardware.vector(4))
                if smoke:
                    config.alg.n_episodes, config.alg.period = 10, 2
                spec = dict(method=method, exp_name=experiment, dir_name=directory,
                            seed=seed, n_agents=4, min_at_lever=2,
                            n_episodes=config.alg.n_episodes, n_eval=config.alg.n_eval,
                            period=config.alg.period, w_lever=pair[0], w_door=pair[1])
                if method == "refine_eia_er":
                    config.lio.Fairness_multiplier = 0.0
                    config.lio.energy_weight = 1.0
                    spec.update(fairness_mult=0.0, energy_weight=1.0)
                argv = base.run_batch.spec_to_argv(spec, False, threads, platform)
                argv[1] = str(Path(__file__).with_name("eia_adapter.py"))
                argv.extend(["--cost_profile", "uniform_v1", "--hardware_profile", "homogeneous_v1"])
                output = REPO / "lio/results" / experiment / directory
                if os.path.lexists(output):
                    raise base.CampaignError("EIA output already exists: " + str(output))
                arm = f"{method}__uniform_v1__homogeneous_v1__eia_w{label}"
                runs.append(dict(method=method, arm=arm, training_profile="uniform_v1",
                                 hardware_profile="homogeneous_v1", weight=label,
                                 weight_pair=list(pair), seed=seed, argv=argv,
                                 command=base.command_text(argv), effective_config=config,
                                 output_directory=str(output), expected_checkpoint=str(output / config.main.model_name),
                                 checkpoint_episode=config.alg.n_episodes))
    return dict(schema_version=1, campaign=campaign, mode="smoke" if smoke else "campaign",
                methods=list(METHODS), weight_pairs=[list(pair) for pair in selected],
                smoke_weight_pair=list(WEIGHTS[1]) if smoke else None,
                fallback_declared=bool(drop_mid_weight), planned_formal_runs=40 if drop_mid_weight else 60,
                n_runs=len(runs), n_seeds_per_arm=len(seeds), runs=runs,
                platform=platform, threads=threads, use_gpu=False, cwd=str(REPO),
                generator_python=sys.executable,
                command_shell="powershell" if os.name == "nt" else "posix",
                cost_units="normalized_modeled_cost", source=source_identity())


def inventory_rows(manifest):
    result = []
    for row in manifest["runs"]:
        main = row["effective_config"]["main"]
        run_id = main["exp_name"] + "/" + main["dir_name"]
        result.append(dict(run_id=run_id, arm=row["arm"], method=row["method"],
                           seed=str(row["seed"]), checkpoint_prefix=f"lio/results/{run_id}/model.ckpt",
                           evidence="[verified from file]"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="iccps27_eia_20261028")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--drop-mid-weight", action="store_true",
                        help="Declare the 40-run fallback before any formal launch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inventory-out", type=Path)
    args = parser.parse_args(argv)
    if args.dry_run and (args.out or args.inventory_out):
        parser.error("--dry-run cannot write output files")
    if args.inventory_out and not args.out:
        parser.error("--inventory-out requires --out")
    try:
        outputs = [path for path in (args.out, args.inventory_out) if path is not None]
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise base.CampaignError("Output paths must differ")
        results = (REPO / "lio/results").resolve()
        for path in outputs:
            resolved = path.resolve()
            if (path.exists() or path.is_symlink() or not path.parent.is_dir()
                    or resolved == results or results in resolved.parents):
                raise base.CampaignError("Output must be fresh, outside lio/results, with an existing parent")
        manifest = build_campaign(args.campaign, args.platform, args.threads,
                                  args.smoke, args.drop_mid_weight)
        text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            if args.inventory_out:
                with args.inventory_out.open("x", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(inventory_rows(manifest))
        else:
            print(text, end="")
    except (ValueError, OSError) as exc:
        print("CONFIG_ERROR: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
