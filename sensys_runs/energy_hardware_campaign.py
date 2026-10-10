"""Wave 2: two registered heterogeneous REFiNE arms, importing all P3 settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sensys_runs import energy_profile_campaign as base
from sensys_runs.energy_profiles import get_hardware_profile
from sensys_runs.energy_profile_runtime import validate_run

HARDWARE_ID = "hetero_2x1p5_v1"
WAVE2_ARMS = (("refine_er", "paper_ratio_v1"), ("refine_er", "uniform_v1"))


def build_campaign(campaign: str, platform: str, threads: int,
                   use_gpu: bool = False, smoke: bool = False) -> dict:
    if threads != 1:
        raise base.CampaignError("Wave 2 requires threads 1")
    # The suffix keeps both waves separate even when campaign names are shared.
    manifest = base.build_campaign(campaign + "_wave2", platform, threads, use_gpu, smoke)
    hardware = get_hardware_profile(HARDWARE_ID)
    runs = []
    for arm in WAVE2_ARMS:
        for row in manifest["runs"]:
            if (row["method"], row["training_profile"]) != arm:
                continue
            cfg = row["effective_config"]
            cfg.hardware_profile = hardware.snapshot(cfg.env.n_agents)
            cfg.hardware_profile_lambda = list(hardware.vector(cfg.env.n_agents))
            row["argv"].extend(["--hardware_profile", HARDWARE_ID])
            row["command"] = base.command_text(row["argv"])
            row["hardware_profile"] = HARDWARE_ID
            validate_run(row["training_profile"], row["method"], row["output_directory"])
            runs.append(row)
    manifest.update(runs=runs, n_runs=len(runs), wave=2)
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
            with args.out.open("x", encoding="utf-8", newline="\n") as f:
                f.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
