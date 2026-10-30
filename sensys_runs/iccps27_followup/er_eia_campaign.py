"""Generate the registered 30-run ER-LIO+EIA matrix without launching training."""
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
from sensys_runs.energy_profiles import get_hardware_profile, get_profile
from sensys_runs.iccps27_followup import eia_campaign as existing
from sensys_runs.iccps27_followup import er_eia_adapter as adapter

METHODS = (adapter.METHOD,)
WEIGHTS = adapter.eia.WEIGHTS
SEEDS = tuple(range(12341, 12351))
FIELDS = existing.FIELDS
DEFAULT_CAMPAIGN = "iccps27_er_lio_eia_20261029"
RELEASE_TAG = "iccps2027-followup-er-lio-eia"
weight_label = existing.weight_label
inventory_rows = existing.inventory_rows


def source_identity():
    identity = base.source_identity()
    identity["sha256"].update(adapter.source_hashes())
    for relative in ("sensys_runs/iccps27_followup/er_eia_campaign.py",
                     "sensys_runs/iccps27_followup/er_eia_runner.py",
                     "sensys_runs/iccps27_followup/eia_campaign.py",
                     "sensys_runs/iccps27_followup/eia_runner.py",
                     "sensys_runs/energy_profile_runner.py", "sensys_runs/ntfy_notify.py"):
        identity["sha256"][relative] = hashlib.sha256((REPO/relative).read_bytes()).hexdigest()
    return identity


def build_campaign(campaign, platform, threads, smoke=False):
    if not isinstance(campaign, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", campaign):
        raise base.CampaignError("Campaign must be one safe directory component")
    if not isinstance(platform, str) or not platform.strip() or threads != 1:
        raise base.CampaignError("ER-LIO+EIA requires an explicit platform and one thread")
    pairs, seeds = ((WEIGHTS[1],), SEEDS[:2]) if smoke else (WEIGHTS, SEEDS)
    experiment = campaign+("_smoke" if smoke else "")
    runs = []
    for pair in pairs:
        label = weight_label(pair)
        for seed in seeds:
            cfg = base.load_config(adapter.METHOD)
            directory = (f"{adapter.METHOD}_uniform_v1_homogeneous_v1_energy_reward_"
                         f"w{label.replace('.', 'p').replace(',', '_')}_s{seed}")
            cfg.main.exp_name, cfg.main.dir_name = experiment, directory
            cfg.main.seed, cfg.main.use_gpu = seed, False
            cfg.env.n_agents, cfg.env.min_at_lever = 4, 2
            cfg.lio.min_at_lever = 2
            cfg.lio.eia_w_lever, cfg.lio.eia_w_door = pair
            cfg.cost_profile = get_profile("uniform_v1").snapshot()
            hardware = get_hardware_profile("homogeneous_v1")
            cfg.hardware_profile = hardware.snapshot(4)
            cfg.hardware_profile_lambda = list(hardware.vector(4))
            if smoke:
                cfg.alg.n_episodes, cfg.alg.period = 10, 2
            adapter.configure(cfg)
            spec = dict(method=adapter.METHOD, exp_name=experiment, dir_name=directory,
                        seed=seed, n_agents=4, min_at_lever=2, n_episodes=cfg.alg.n_episodes,
                        n_eval=cfg.alg.n_eval, period=cfg.alg.period,
                        w_lever=pair[0], w_door=pair[1])
            argv = base.run_batch.spec_to_argv(spec, False, threads, platform)
            argv[1] = str(Path(__file__).with_name("er_eia_adapter.py"))
            argv.extend(["--cost_profile", "uniform_v1", "--hardware_profile", "homogeneous_v1",
                         "--energy_in_reward"])
            output = REPO/"lio/results"/experiment/directory
            if os.path.lexists(output):
                raise base.CampaignError("ER-LIO+EIA output already exists: "+str(output))
            arm = f"{adapter.METHOD}__uniform_v1__homogeneous_v1__energy_reward__eia_w{label}"
            runs.append(dict(method=adapter.METHOD, arm=arm, objective=adapter.OBJECTIVE,
                             followup_arm=adapter.ARM, display_label=adapter.DISPLAY_LABEL,
                             training_profile="uniform_v1", hardware_profile="homogeneous_v1",
                             weight=label, weight_pair=list(pair), seed=seed, argv=argv,
                             command=base.command_text(argv), effective_config=cfg,
                             output_directory=str(output),
                             expected_checkpoint=str(output/cfg.main.model_name),
                             checkpoint_episode=cfg.alg.n_episodes))
    return dict(schema_version=1, campaign=campaign, mode="smoke" if smoke else "campaign",
                methods=list(METHODS), objective=adapter.OBJECTIVE, followup_arm=adapter.ARM,
                display_label=adapter.DISPLAY_LABEL, weight_pairs=[list(pair) for pair in WEIGHTS],
                smoke_weight_pair=list(WEIGHTS[1]) if smoke else None,
                fallback_declared=False, planned_formal_runs=30, n_runs=len(runs),
                n_seeds_per_arm=len(seeds), runs=runs, platform=platform, threads=threads,
                use_gpu=False, cwd=str(REPO), generator_python=sys.executable,
                command_shell="powershell" if os.name == "nt" else "posix",
                cost_units="normalized_modeled_cost", source=source_identity())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
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
        results = (REPO/"lio/results").resolve()
        for path in outputs:
            resolved = path.resolve()
            if (path.exists() or path.is_symlink() or not path.parent.is_dir()
                    or resolved == results or results in resolved.parents):
                raise base.CampaignError("Output must be fresh, outside lio/results, with an existing parent")
        manifest = build_campaign(args.campaign, args.platform, args.threads, args.smoke)
        text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n"
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
        print("CONFIG_ERROR: "+str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
