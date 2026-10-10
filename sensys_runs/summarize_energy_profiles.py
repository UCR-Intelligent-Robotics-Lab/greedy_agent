"""Validate final-policy sidecars and score each trajectory under all P1 tables.

Seed-level means include unsuccessful missions. Worst-agent cost is the maximum
of agent-wise episode means, not the mean of episode-wise maxima. No RNG is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sensys_runs.energy_profile_eval import (
    EPISODE_FIELDS, EPISODE_FILE, MANIFEST_FILE, SCHEMA_VERSION,
    EnergyEvaluationError, checkpoint_hashes, json_copy, sha256,
)
from sensys_runs.energy_profiles import PROFILES

METRICS = ("success_rate", "mean_mission_steps", "mean_agent_mission_cost",
           "worst_agent_mean_mission_cost", "mean_agent_env_return",
           "step_cost_component", "support_cost_component")
SEED_FIELDS = ("run_id", "seed", "method", "training_profile", "evaluation_profile",
               "n_episodes", "n_agents", *METRICS)
GROUP_FIELDS = ("method", "training_profile", "evaluation_profile")
SUMMARY_FIELDS = (*GROUP_FIELDS, "n_seeds", *METRICS)
PAIR_FIELDS = ("seed", *GROUP_FIELDS, "reference_method", "reference_training_profile",
               *("delta_" + metric for metric in METRICS))


def load_run(directory: Path) -> tuple[dict, list[dict]]:
    """Refuse missing, corrupt, mixed, duplicated or incomplete evaluation data."""
    with (directory / MANIFEST_FILE).open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if meta["schema_version"] != SCHEMA_VERSION:
        raise EnergyEvaluationError("Unsupported episode schema version")
    if meta["evaluation_profiles"] != json_copy({k: p.snapshot() for k, p in PROFILES.items()}):
        raise EnergyEvaluationError("Evaluation profiles differ from the recorded registry")
    if meta["training_profile_snapshot"] != meta["evaluation_profiles"][meta["training_profile"]]:
        raise EnergyEvaluationError("Training profile provenance is inconsistent")
    if meta["config"]["cost_profile"] != meta["training_profile_snapshot"]:
        raise EnergyEvaluationError("Config and training profile differ")
    prefix = meta["checkpoint_prefix"]
    if Path(prefix).name != prefix:
        raise EnergyEvaluationError("Checkpoint prefix must be a local filename")
    if checkpoint_hashes(directory / prefix) != meta["checkpoint_sha256"]:
        raise EnergyEvaluationError("Checkpoint hashes differ from the final-policy manifest")
    csv_path = directory / EPISODE_FILE
    if sha256(csv_path) != meta["episodes_sha256"]:
        raise EnergyEvaluationError("Episode file hash differs from the manifest")
    n_agents, n_eval = meta["n_agents"], meta["n_eval_episodes"]
    cfg = meta["config"]
    if (n_agents != cfg["env"]["n_agents"] or n_eval != cfg["alg"]["n_test"]
            or meta["checkpoint_episode"] != cfg["alg"]["n_episodes"]
            or meta["seed"] != cfg["main"]["seed"]):
        raise EnergyEvaluationError("Manifest and run config differ")
    rows = []
    seen = set()
    episode_state = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
            raise EnergyEvaluationError("Episode header differs from the registered schema")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise EnergyEvaluationError("Malformed episode row")
            for key in EPISODE_FIELDS[:6]:
                if row[key] != str(meta[key]):
                    raise EnergyEvaluationError(f"Mixed run identity in column {key}")
            for key in EPISODE_FIELDS[6:-1]:
                row[key] = int(row[key])
            row["env_return"] = float(row["env_return"])
            episode, agent = row["eval_episode"], row["agent_id"]
            identity = (episode, agent)
            if identity in seen or not (1 <= episode <= n_eval and 0 <= agent < n_agents):
                raise EnergyEvaluationError("Duplicate or out-of-range episode/agent")
            seen.add(identity)
            counts = [row[key] for key in ("n_lever", "n_start", "n_door")]
            if (row["steps"] <= 0 or min(counts) < 0 or sum(counts) != row["steps"]
                    or row["terminal_code"] not in (1, 2) or not math.isfinite(row["env_return"])):
                raise EnergyEvaluationError("Invalid action counts, return or terminal code")
            state = (row["steps"], row["terminal_code"])
            if episode_state.setdefault(episode, state) != state:
                raise EnergyEvaluationError("Agents disagree on episode termination")
            rows.append(row)
    if len(rows) != n_agents * n_eval or not rows:
        raise EnergyEvaluationError("Missing complete evaluation episodes")
    return meta, rows


def score_rows(rows: list[dict], costs) -> dict:
    """Aggregate complete validated episode/agent rows; failures remain included."""
    per_agent = defaultdict(list)
    episodes = {}
    for row in rows:
        cost = sum(c * row[k] for c, k in zip(costs, ("n_lever", "n_start", "n_door")))
        per_agent[row["agent_id"]].append(cost)
        episodes[row["eval_episode"]] = row
    b, d = costs[1], costs[0] - costs[1]
    if costs[1] != costs[2]:
        raise EnergyEvaluationError("Step/support decomposition requires equal start and door costs")
    return dict(
        n_episodes=len(episodes), n_agents=len(per_agent),
        success_rate=fmean(r["terminal_code"] == 1 for r in episodes.values()),
        mean_mission_steps=fmean(r["steps"] for r in episodes.values()),
        mean_agent_mission_cost=fmean(c for values in per_agent.values() for c in values),
        worst_agent_mean_mission_cost=max(fmean(values) for values in per_agent.values()),
        mean_agent_env_return=fmean(r["env_return"] for r in rows),
        step_cost_component=b * fmean(r["steps"] for r in rows),
        support_cost_component=d * fmean(r["n_lever"] for r in rows),
    )


def summarize_runs(runs: list[tuple[dict, list[dict]]]) -> tuple[list, list, list]:
    """Keep seeds independent and pair only identical seed IDs within one ER setup."""
    seed_rows, identities, setups, arm_configs = [], set(), set(), {}
    for meta, rows in runs:
        identity = (meta["method"], meta["training_profile"], meta["seed"])
        if identity in identities:
            raise EnergyEvaluationError("Duplicate training seed within an arm")
        identities.add(identity)
        arm = identity[:2]
        settings = json.dumps({k: meta["config"][k] for k in ("alg", "env", "lio", "nn")}, sort_keys=True)
        if arm_configs.setdefault(arm, settings) != settings:
            raise EnergyEvaluationError("Training settings differ across seeds within an arm")
        # Compare policies only when their task/evaluation protocol agrees.
        setups.add(json.dumps(dict(env=meta["config"]["env"],
                                   nn=meta["config"]["nn"],
                                   n_eval=meta["n_eval_episodes"],
                                   checkpoint_episode=meta["checkpoint_episode"]), sort_keys=True))
        for profile, definition in PROFILES.items():
            seed_rows.append(dict(run_id=meta["run_id"], seed=meta["seed"],
                                  method=meta["method"], training_profile=meta["training_profile"],
                                  evaluation_profile=profile, **score_rows(rows, definition.costs)))
    if len(setups) != 1:
        raise EnergyEvaluationError("Summary requires one common task/evaluation protocol")
    grouped = defaultdict(dict)
    for row in seed_rows:
        grouped[tuple(row[k] for k in GROUP_FIELDS)][row["seed"]] = row
    summary, pairs = [], []
    for key, seeds in sorted(grouped.items()):
        summary.append(dict(zip(GROUP_FIELDS, key), n_seeds=len(seeds),
                            **{m: fmean(row[m] for row in seeds.values()) for m in METRICS}))
        references = [ref for ref in grouped if ref[2] == key[2] and ref != key
                      and (ref[0] == "lio_er" or ref[:2] == ("refine_er", "uniform_v1"))]
        if key[0] != "refine_er":
            continue
        for reference in sorted(references):
            if seeds.keys() != grouped[reference].keys():
                raise EnergyEvaluationError("Paired comparisons require matching training seed sets")
            for seed, row in sorted(seeds.items()):
                baseline = grouped[reference][seed]
                pairs.append(dict(seed=seed, **dict(zip(GROUP_FIELDS, key)),
                                  reference_method=reference[0], reference_training_profile=reference[1],
                                  **{"delta_" + m: row[m] - baseline[m] for m in METRICS}))
    return seed_rows, summary, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Exact final run directories")
    parser.add_argument("--out", type=Path, required=True, help="Fresh summary directory")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("Summary output must not exist")
    runs, failures = [], []
    for directory in args.runs:
        try:
            runs.append(load_run(directory))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"{directory}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    try:
        seeds, summary, pairs = summarize_runs(runs)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=False)
    for name, fields, rows in (("seed_metrics.csv", SEED_FIELDS, seeds),
                                ("summary.csv", SUMMARY_FIELDS, summary),
                                ("paired_differences.csv", PAIR_FIELDS, pairs)):
        with (args.out / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (args.out / "inputs.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(inputs=[dict(directory=str(p.resolve()),
                                    manifest_sha256=sha256(p / MANIFEST_FILE)) for p in args.runs],
                       units="normalized_modeled_cost", delta_convention="method minus reference",
                       failed_episodes="included", intervals="not computed",
                       evaluation_profiles=json_copy({k: p.snapshot() for k, p in PROFILES.items()})),
                  stream, indent=2, sort_keys=True)
    print(f"Validated {len(runs)} runs; wrote {len(seeds)} seed/profile rows and {len(pairs)} paired rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
