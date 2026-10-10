"""Wave 3: LIO episode-mean energy reward under two registered hardware designs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sensys_runs import energy_profile_campaign as base
from sensys_runs.energy_profiles import get_profile, get_hardware_profile
from sensys_runs.energy_profile_runtime import validate_run
from sensys_runs.energy_reward_runtime import configure

WAVE3_HARDWARE = ("homogeneous_v1", "hetero_2x1p5_v1")
TRAINING_PROFILE = "paper_ratio_v1"


def build_campaign(campaign: str, platform: str, threads: int,
                   use_gpu: bool = False, smoke: bool = False) -> dict:
    if threads != 1:
        raise base.CampaignError("Wave 3 requires threads 1")
    manifest = base.build_campaign(campaign + "_wave3", platform, threads, use_gpu, smoke)
    templates = [r for r in manifest["runs"] if r["method"] == "lio_er"]
    runs = []
    for hardware_id in WAVE3_HARDWARE:
        hardware = get_hardware_profile(hardware_id)
        for template in templates:
            row = deepcopy(template)
            cfg, argv = row["effective_config"], row["argv"]
            directory = f"lio_er_{TRAINING_PROFILE}_{hardware_id}_energy_reward_s{row['seed']}"
            cfg.main.dir_name = directory
            cfg.cost_profile = get_profile(TRAINING_PROFILE).snapshot()
            cfg.hardware_profile = hardware.snapshot(cfg.env.n_agents)
            cfg.hardware_profile_lambda = list(hardware.vector(cfg.env.n_agents))
            configure(cfg)
            argv[argv.index("--dir_name") + 1] = directory
            argv[argv.index("--cost_profile") + 1] = TRAINING_PROFILE
            argv.extend(["--hardware_profile", hardware_id, "--energy_in_reward"])
            output = base.REPO_ROOT / "lio/results" / cfg.main.exp_name / directory
            validate_run(TRAINING_PROFILE, "lio_er", str(output))
            row.update(training_profile=TRAINING_PROFILE, hardware_profile=hardware_id,
                       objective="energy_reward", command=base.command_text(argv),
                       output_directory=str(output), expected_checkpoint=str(output / cfg.main.model_name))
            runs.append(row)
    manifest.update(runs=runs, n_runs=len(runs), wave=3)
    manifest["time_budget"]["serial_formula"] = "n_seeds_per_arm * sum(measured full-run seconds for the two arms)"
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=base.CAMPAIGN)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.dry_run and args.out:
        parser.error("--dry-run cannot be combined with --out")
    try:
        if args.out:
            path, results = args.out.resolve(), (base.REPO_ROOT / "lio/results").resolve()
            if path == results or results in path.parents or path.exists() or args.out.is_symlink():
                raise base.CampaignError("--out must be a fresh file outside lio/results")
        manifest = build_campaign(args.campaign, args.platform, args.threads, args.use_gpu, args.smoke)
        text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
