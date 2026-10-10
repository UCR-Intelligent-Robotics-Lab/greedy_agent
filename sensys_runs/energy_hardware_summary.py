"""Registered lambda rescoring from validated action counts; seed-level only.

Historical P1-P3b source revisions below required energy_param=1. Their recorded
commit is mandatory evidence for homogeneous-by-construction interpretation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

from sensys_runs.energy_profile_eval import EnergyEvaluationError, json_copy, MANIFEST_FILE, sha256
from sensys_runs.energy_profiles import PROFILES, get_hardware_profile

HOMOGENEOUS_COMMITS = frozenset((
    "83a248e5cc64cab377dc3defee0dc1bc3b66769b",
    "06c4fc1aa734fd6bbbcf16591a82f2e45d5542d9",
    "9d704b3e8a5399e2fad71262dedb0de6368755ae",
    "df7522d4a896c42b555542f8a3af8dc3b0d67661",
    "5fea384386c86af6d01f131917b4689e96a3837c",
    "78971fe75c6314bf80264131a84048a81f0a3891",
))
HARDWARE_SEED_FIELDS = (
    "run_id", "seed", "method", "training_profile", "training_hardware_profile",
    "evaluation_profile", "evaluation_hardware_profile", "hardware_profile_lambda",
    "per_agent_mean_mission_cost", "mean_agent_mission_cost", "worst_agent_mean_mission_cost",
    "step_cost_component", "support_cost_component", "support_share",
    "support_share_reference", "support_share_status", "hardware_origin", "git_commit",
)


def record_hardware_origin(meta: dict, directory: Path) -> dict:
    """Never silently interpret absent lambda, partial metadata or unknown history."""
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    cfg, n_agents = meta["config"], meta["n_agents"]
    commit = provenance.get("git_commit")
    if "hardware_profile_lambda" not in cfg:
        if ("hardware_profile_lambda" in provenance or "hardware_profile" in cfg
                or "hardware_profile" in provenance or commit not in HOMOGENEOUS_COMMITS
                or provenance.get("git_dirty") is not False):
            raise EnergyEvaluationError("Missing lambda needs a clean, recorded pre-P4 commit")
        profile = get_hardware_profile("homogeneous_v1")
        origin = "homogeneous by construction; recorded pre-P4 commit " + commit
        print(f"HARDWARE_ORIGIN {meta['run_id']}: {origin}")
    else:
        profile = get_hardware_profile(cfg["hardware_profile"]["profile_id"])
        snapshot = json_copy(profile.snapshot(n_agents))
        if (cfg["hardware_profile"] != snapshot or provenance.get("hardware_profile") != snapshot
                or cfg["hardware_profile_lambda"] != list(profile.vector(n_agents))
                or provenance.get("hardware_profile_lambda") != cfg["hardware_profile_lambda"]):
            raise EnergyEvaluationError("Config/provenance lambda differs from registry")
        origin = "explicit registered hardware profile"
    record = dict(profile_id=profile.profile_id, hardware_profile_lambda=list(profile.vector(n_agents)),
                  origin=origin, git_commit=commit)
    meta["hardware_origin"] = record
    return record


def score_hardware_rows(rows: list[dict], costs, lambdas: tuple) -> dict:
    """Lowest-lambda share pools counts within this seed, never across seeds."""
    per_agent = [[] for _ in lambdas]
    for row in rows:
        i = row["agent_id"]
        cost = sum(c * row[k] for c, k in zip(costs, ("n_lever", "n_start", "n_door")))
        per_agent[i].append(cost * lambdas[i])
    means = [fmean(values) for values in per_agent]
    low = {i for i, value in enumerate(lambdas) if value == min(lambdas)}
    lever = sum(row["n_lever"] for row in rows)
    heterogeneous = len(low) < len(lambdas)
    share = (sum(row["n_lever"] for row in rows if row["agent_id"] in low) / lever
             if heterogeneous and lever else None)
    return dict(per_agent_mean_mission_cost=json.dumps(means),
                mean_agent_mission_cost=fmean(means), worst_agent_mean_mission_cost=max(means),
                step_cost_component=fmean(costs[1] * r["steps"] * lambdas[r["agent_id"]] for r in rows),
                support_cost_component=fmean((costs[0]-costs[1]) * r["n_lever"] * lambdas[r["agent_id"]] for r in rows),
                support_share=share, support_share_reference=len(low)/len(lambdas) if heterogeneous else None,
                support_share_status="defined" if share is not None else
                                     "no lever actions" if heterogeneous else "homogeneous: no lower-lambda group")


def write_hardware_summary(runs, directories, out: Path, evaluation_profile: str | None) -> int:
    """Separate hardware identities; no averaging seeds or confidence intervals."""
    records, identities, setups = [], set(), set()
    for meta, rows in runs:
        origin = meta["hardware_origin"]
        identity = (meta["method"], meta["training_profile"], origin["profile_id"], meta["seed"])
        if identity in identities:
            raise EnergyEvaluationError("Duplicate training seed within hardware arm")
        identities.add(identity)
        setups.add(json.dumps(dict(env=meta["config"]["env"], nn=meta["config"]["nn"],
                                   n_eval=meta["n_eval_episodes"], episode=meta["checkpoint_episode"]), sort_keys=True))
        hardware = get_hardware_profile(evaluation_profile or origin["profile_id"])
        lambdas = hardware.vector(meta["n_agents"])
        for profile_id, definition in PROFILES.items():
            records.append(dict(run_id=meta["run_id"], seed=meta["seed"], method=meta["method"],
                                training_profile=meta["training_profile"], training_hardware_profile=origin["profile_id"],
                                evaluation_profile=profile_id, evaluation_hardware_profile=hardware.profile_id,
                                hardware_profile_lambda=json.dumps(lambdas), hardware_origin=origin["origin"],
                                git_commit=origin["git_commit"], **score_hardware_rows(rows, definition.costs, lambdas)))
    if len(setups) != 1:
        raise EnergyEvaluationError("Hardware summary requires one common task/evaluation protocol")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "hardware_seed_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HARDWARE_SEED_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with (out / "inputs.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(inputs=[dict(directory=str(p.resolve()), hardware=meta["hardware_origin"],
                                    manifest_sha256=sha256(p / MANIFEST_FILE),
                                    evaluation_hardware=get_hardware_profile(evaluation_profile or
                                        meta["hardware_origin"]["profile_id"]).snapshot(meta["n_agents"]))
                               for p, (meta, _) in zip(directories, runs)],
                       evaluation_hardware_profile=evaluation_profile, intervals="not computed",
                       aggregation="seed-level only; failures included", units="normalized_modeled_cost"),
                  stream, indent=2, sort_keys=True)
    print(f"Validated {len(runs)} runs; wrote {len(records)} hardware seed/profile rows; no intervals")
    return 0
