"""Objective-aware, lambda-aware summaries with explicit paired-reference matching."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from statistics import fmean

from sensys_runs.energy_profile_eval import EnergyEvaluationError, MANIFEST_FILE, sha256
from sensys_runs.energy_profiles import PROFILES, get_hardware_profile
from sensys_runs.energy_hardware_summary import HARDWARE_SEED_FIELDS, score_hardware_rows
from sensys_runs.energy_reward_runtime import REWARD_FORM, registered_beta

ARM_FIELDS = ("method", "training_profile", "evaluation_profile", "objective",
              "training_hardware_profile", "evaluation_hardware_profile")
OBJECTIVE_HARDWARE_FIELDS = (*HARDWARE_SEED_FIELDS, "objective")
LIO_REFERENCE_RULE = "lio_env_reward_accounting_only"
LIO_REFERENCE_JUSTIFICATION = (
    "LIO actor update does not read buffer energy (lio/alg/lio_agent_er.py:339-373); "
    "fixed cost and lambda lookups consume no RNG "
    "(sensys_runs/energy_profile_runtime.py:48-52; "
    "sensys_runs/energy_hardware_runtime.py:32-37)."
)
LIO_REFERENCE_REQUIREMENTS = (
    "Only lio_er + env_reward reference arms ignore training cost and hardware profiles; "
    "evaluation table, evaluation hardware and complete seed sets must match. "
    "REFiNE and energy_reward arms receive no exemption."
)


def objective(meta: dict) -> str:
    cfg = meta["config"]
    flag = cfg.get("energy_in_reward", False)
    if not isinstance(flag, bool):
        raise EnergyEvaluationError("energy_in_reward must be a boolean")
    if flag and (meta["method"] != "lio_er" or meta["training_profile"] == "legacy"
                 or cfg.get("energy_reward_form") != REWARD_FORM
                 or cfg.get("energy_reward_beta") != registered_beta()):
        raise EnergyEvaluationError("Energy-reward objective differs from registered design")
    return "energy_reward" if flag else "env_reward"


def summarize_objectives(runs, evaluation_hardware=None):
    from sensys_runs.summarize_energy_profiles import METRICS, score_rows
    seeds, identities, setups, settings = [], set(), set(), {}
    for meta, rows in runs:
        cfg, label = meta["config"], objective(meta)
        training_hardware = meta.get("hardware_origin", {}).get("profile_id")
        if training_hardware is None:
            training_hardware = cfg.get("hardware_profile", {}).get("profile_id", "homogeneous_v1")
        hardware = get_hardware_profile(evaluation_hardware or training_hardware)
        vector = hardware.vector(meta["n_agents"])
        arm = (meta["method"], meta["training_profile"], label, training_hardware)
        identity = (*arm, meta["seed"])
        if identity in identities:
            raise EnergyEvaluationError("Duplicate training seed within objective/hardware arm")
        identities.add(identity)
        config_key = json.dumps({k: cfg[k] for k in ("alg", "env", "lio", "nn")}, sort_keys=True)
        if settings.setdefault(arm, config_key) != config_key:
            raise EnergyEvaluationError("Training settings differ across seeds within an arm")
        setups.add(json.dumps(dict(env=cfg["env"], nn=cfg["nn"], n_eval=meta["n_eval_episodes"],
                                   checkpoint_episode=meta["checkpoint_episode"]), sort_keys=True))
        for profile, definition in PROFILES.items():
            metrics = score_rows(rows, definition.costs)
            weighted = score_hardware_rows(rows, definition.costs, vector)
            metrics.update(weighted)
            seeds.append(dict(run_id=meta["run_id"], seed=meta["seed"], method=meta["method"],
                              training_profile=meta["training_profile"], evaluation_profile=profile,
                              objective=label, training_hardware_profile=training_hardware,
                              evaluation_hardware_profile=hardware.profile_id,
                              hardware_profile_lambda=json.dumps(vector),
                              hardware_origin=meta.get("hardware_origin", {}).get("origin"),
                              git_commit=meta.get("hardware_origin", {}).get("git_commit"), **metrics))
    if len(setups) != 1:
        raise EnergyEvaluationError("Summary requires one common task/evaluation protocol")
    grouped = defaultdict(dict)
    for row in seeds:
        grouped[tuple(row[k] for k in ARM_FIELDS)][row["seed"]] = row
    summaries, pairs = [], []
    for key, values in sorted(grouped.items()):
        summaries.append(dict(zip(ARM_FIELDS, key), n_seeds=len(values),
                              **{m: fmean(r[m] for r in values.values()) for m in METRICS}))
        if key[3] == "energy_reward":
            references = [r for r in grouped if r[3] == "env_reward" and r[0] in ("lio_er", "refine_er")
                          and r[2] == key[2] and r[5] == key[5]
                          and (r[0] == "lio_er" or (r[1] == key[1] and r[4] == key[4]))]
            for method in ("lio_er", "refine_er"):
                if not any(r[0] == method for r in references):
                    print(f"PAIR_MISSING objective=energy_reward profile={key[1]} hardware={key[4]} evaluation={key[2]} reference={method}")
        elif key[0] == "refine_er":
            # Retain the P2 comparisons among environment-reward controls.
            references = [r for r in grouped if r != key and r[2:] == key[2:]
                          and (r[0] == "lio_er" or r[:2] == ("refine_er", "uniform_v1"))]
        else:
            references = []
        for reference in sorted(references):
            if values.keys() != grouped[reference].keys():
                raise EnergyEvaluationError("Paired comparisons require matching training seed sets")
            for seed, row in sorted(values.items()):
                baseline = grouped[reference][seed]
                pairs.append(dict(seed=seed, **dict(zip(ARM_FIELDS, key)), reference_method=reference[0],
                                  reference_training_profile=reference[1], reference_objective=reference[3],
                                  reference_hardware_profile=reference[4],
                                  reference_training_match=(LIO_REFERENCE_RULE if reference[0] == "lio_er"
                                      and reference[3] == "env_reward" else "strict"),
                                  reference_training_match_justification=(LIO_REFERENCE_JUSTIFICATION
                                      if reference[0] == "lio_er" and reference[3] == "env_reward" else ""),
                                  **{"delta_" + m: row[m] - baseline[m] for m in METRICS}))
    return seeds, summaries, pairs


def write_objective_summary(runs, directories, out, evaluation_hardware=None) -> int:
    from sensys_runs.summarize_energy_profiles import SEED_FIELDS, SUMMARY_FIELDS, PAIR_FIELDS
    seeds, summaries, pairs = summarize_objectives(runs, evaluation_hardware)
    out.mkdir(parents=True, exist_ok=False)
    outputs = (("seed_metrics.csv", SEED_FIELDS, seeds), ("summary.csv", SUMMARY_FIELDS, summaries),
               ("paired_differences.csv", PAIR_FIELDS, pairs),
               ("hardware_seed_metrics.csv", OBJECTIVE_HARDWARE_FIELDS, seeds))
    for name, fields, rows in outputs:
        with (out / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in fields} for row in rows)
    with (out / "inputs.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(inputs=[dict(directory=str(p.resolve()), manifest_sha256=sha256(p / MANIFEST_FILE),
                                    hardware=meta["hardware_origin"], objective=objective(meta),
                                    evaluation_hardware=get_hardware_profile(evaluation_hardware or
                                        meta["hardware_origin"]["profile_id"]).snapshot(meta["n_agents"]))
                               for p, (meta, _) in zip(directories, runs)],
                       evaluation_hardware_profile=evaluation_hardware, intervals="not computed",
                       delta_convention="method minus reference", failed_episodes="included",
                       pairing="energy_reward versus env_reward controls; exact evaluation table/hardware and seed sets",
                       reference_training_match=dict(rule=LIO_REFERENCE_RULE,
                           requirements=LIO_REFERENCE_REQUIREMENTS,
                           justification=LIO_REFERENCE_JUSTIFICATION)),
                  stream, indent=2, sort_keys=True)
    print(f"Validated {len(runs)} runs; wrote {len(seeds)} objective seed/profile rows and {len(pairs)} paired rows")
    return 0
