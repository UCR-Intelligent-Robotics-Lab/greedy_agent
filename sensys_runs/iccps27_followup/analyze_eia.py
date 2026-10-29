"""Observed EIA comparisons and separately validated reset-probe summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sensys_runs.iccps27_followup import analyze_a9 as a9

METHODS = ("lio_eia_er", "refine_eia_er")
WEIGHTS = ((2.0, 0.2), (1.1, 0.9), (1.5, 1.5))
CSV_FIELDS = ("method", "weight", "seed", "observed_mean_cost", "mean_steps",
              "one_step_fraction", "success_rate")
ADAPTER = ("Numerical G1 adapter only: actual refine_eia_er occupies the existing "
           "refine_er/uniform_v1/env_reward/homogeneous_v1 method slot and actual "
           "lio_eia_er occupies the existing lio_er reference slot. The actual "
           "EIA method labels remain unchanged in every reported run. Unchanged "
           "score_rows and evaluate_gates.evaluate supply observed 100-episode "
           "mean-agent modeled costs, paired relative median and percentile "
           "bootstrap interval. Internal tests and gate decisions are not reported.")
PROBE_WORDING = "computed exactly from the deployed policy at the reset input, in floating point"


def require(condition, message):
    if not condition:
        raise a9.AnalysisError(message)


def safe_id(value):
    require(isinstance(value, str) and len(value.split("/")) == 2 and
            all(x not in ("", ".", "..") for x in value.split("/")) and
            "\\" not in value and ":" not in value, "Unsafe EIA run identity")
    return value


def weight_label(pair):
    return ",".join(str(float(v)) for v in pair)


def matrix(manifest, inventory_path, inputs):
    require(manifest.get("mode") == "campaign", "Formal EIA analysis cannot use a smoke manifest")
    require(type(manifest.get("fallback_declared")) is bool, "The pre-launch fallback declaration is required")
    weights = (WEIGHTS[0], WEIGHTS[2]) if manifest["fallback_declared"] else WEIGHTS
    require(manifest.get("weight_pairs") == [list(w) for w in weights], "Declared EIA weight matrix mismatch")
    expected = {(method, weight, seed) for method in METHODS for weight in weights for seed in a9.SEEDS}
    runs = manifest.get("runs", [])
    require(manifest.get("n_runs") == len(expected) == len(runs), "Expected the complete declared 60-run or 40-run matrix")
    identities, ids = {}, set()
    for row in runs:
        pair = tuple(row.get("weight_pair", []))
        key = row.get("method"), pair, row.get("seed")
        require(key in expected and key not in identities, "Missing, duplicate or unregistered EIA cell")
        config = row["effective_config"]
        rid = safe_id(config["main"]["exp_name"] + "/" + config["main"]["dir_name"])
        require(rid not in ids and config["main"]["seed"] == key[2], "Duplicate EIA run or config seed mismatch")
        require(row.get("weight") == weight_label(pair), "Weight label differs from authoritative pair")
        identities[key] = dict(run_id=rid, method=key[0], weight=row["weight"], weight_pair=list(pair),
                               seed=key[2], expected_config=config)
        ids.add(rid)
    require(set(identities) == expected, "The declared EIA matrix is incomplete")
    inventory = list(csv.DictReader(io.StringIO(inputs.read(inventory_path).decode("utf-8-sig"))))
    actual = [(safe_id(r["run_id"]), r["method"], int(r["seed"])) for r in inventory]
    required = {(r["run_id"], r["method"], r["seed"]) for r in identities.values()}
    require(len(actual) == len(required) and set(actual) == required, "Probe inventory does not match the full declared campaign")
    return weights, [identities[(m, w, s)] for w in weights for m in METHODS for s in a9.SEEDS]


def attack_provenance(config, provenance, pair):
    attack = provenance.get("eia_attack")
    require(isinstance(attack, dict) and attack == config.get("eia_attack"), "Config/provenance EIA attack mismatch")
    require(attack.get("schema_version") == 1 and attack.get("action_order") == ["lever", "start", "door"]
            and attack.get("exploitative_giver_id") == 1 and attack.get("applies_to") == "recipient_action"
            and attack.get("training_active") is True and attack.get("evaluation_active") is True,
            "Unexpected EIA agent/action or evaluation semantics")
    require(attack.get("configured_weights") == {"lever": pair[0], "door": pair[1]}, "Wrong configured EIA weights")
    require(config["lio"].get("eia_w_lever") == pair[0] and config["lio"].get("eia_w_door") == pair[1], "Config weight mismatch")
    effective = attack.get("effective_multipliers", [])
    require(len(effective) == 16 and {(r["giver_id"], r["recipient_id"]) for r in effective} ==
            {(g, r) for g in range(4) for r in range(4)}, "Incomplete effective EIA multiplier provenance")
    for record in effective:
        g, r = record["giver_id"], record["recipient_id"]
        expected = [pair[0], 1.0, pair[1]] if g == 1 and r != 1 else [1.0] * 3
        require(record["multipliers"] == expected and record.get("delivered") is (g != r), "Effective EIA multipliers differ from the registered mask")
    return attack


def add_one_step(metrics, path, inputs):
    rows = list(csv.DictReader(io.StringIO(inputs.read(path).decode("utf-8-sig"))))
    first_agent = [r for r in rows if int(r["agent_id"]) == 0]
    require(len(first_agent) == 100, "Expected 100 final episodes for the one-step cross-check")
    result = dict(metrics)
    result["one_step_fraction"] = statistics.fmean(int(r["steps"]) == 1 for r in first_agent)
    return result


def read_observed(identity, root, inputs, summary):
    rid = identity["run_id"]
    run = Path(root) / rid
    result = dict(identity, observed=None, observed_eligible=False, issues=[], probe=None,
                  probe_eligible=False, probe_issues=[], sources={"run": str(run.resolve())})
    try:
        cfg = inputs.json(run / "config.json")
        prov = inputs.json(run / "provenance.json")
        require({k: v for k, v in cfg.items() if k != "eia_attack"} == identity["expected_config"], "Saved config differs from the pre-launch manifest")
        require(prov["method"] == identity["method"] and prov["seed"] == identity["seed"], "True EIA method/seed mismatch")
        require(cfg["cost_profile"]["profile_id"] == "uniform_v1" and cfg["cost_profile"]["normalized_vector"] == [1, 1, 1]
                and cfg["hardware_profile"]["profile_id"] == "homogeneous_v1" and cfg["hardware_profile_lambda"] == [1, 1, 1, 1],
                "EIA uniform/homogeneous profile mismatch")
        if identity["method"] == "refine_eia_er":
            require(cfg["lio"]["Fairness_multiplier"] == 0.0 and cfg["lio"]["energy_weight"] == 1.0, "REFiNE EIA requires B=0 and beta=1")
        attack = attack_provenance(cfg, prov, identity["weight_pair"])
        completion = inputs.json(run / "eia_run_completion.json")
        csv_path = run / "energy_eval_episodes.csv"
        csv_digest = hashlib.sha256(inputs.read(csv_path)).hexdigest()
        require(completion.get("status") == "complete" and completion.get("run_id") == rid
                and completion.get("method") == identity["method"] and completion.get("seed") == identity["seed"]
                and completion.get("final_evaluation_episodes") == 100 and completion.get("final_evaluation_rows") == 400
                and completion.get("schema_version") == 1
                and completion.get("episodes_sha256") == csv_digest, "Missing or invalid EIA completion evidence")
        meta = inputs.json(run / "energy_eval_provenance.json")
        require(completion.get("final_evaluation_sha256") == inputs.hashes[str((run / "energy_eval_provenance.json").resolve())], "Completion metadata hash mismatch")
        require(meta["config"] == cfg and meta["episodes_sha256"] == csv_digest and meta["method"] == identity["method"]
                and meta["seed"] == identity["seed"] and meta["run_id"] == rid
                and meta["checkpoint_episode"] == 25000 and meta["n_eval_episodes"] == 100 and meta["n_agents"] == 4
                and meta.get("eia_attack") == attack
                and meta.get("effective_uniform_costs") == [dict(agent_id=i, costs=[1.0]*3) for i in range(4)],
                "Final evaluation metadata identity mismatch")
        checkpoint_hashes = {}
        for path in sorted(run.glob("model.ckpt.*")):
            checkpoint_hashes[path.name] = hashlib.sha256(inputs.read(path)).hexdigest()
        require("model.ckpt.index" in checkpoint_hashes and "model.ckpt.meta" in checkpoint_hashes
                and any(k.startswith("model.ckpt.data-") for k in checkpoint_hashes)
                and checkpoint_hashes == meta["checkpoint_sha256"], "Final checkpoint/evaluation hash mismatch")
        metrics, issue = a9.observed_metrics(csv_path, identity, inputs, summary)
        require(metrics is not None and issue is None, issue or "Observed evaluation unavailable")
        result.update(observed=add_one_step(metrics, csv_path, inputs), observed_eligible=True,
                      checkpoint_hashes=checkpoint_hashes, attack=attack,
                      config_provenance_hashes={name: inputs.hashes[str((run/name).resolve())]
                                               for name in ("config.json", "provenance.json")})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, FileNotFoundError) and exc.filename:
            inputs.missing.add(str(Path(exc.filename).resolve()))
        result["issues"].append("UNVERIFIED or failed run: " + str(exc))
    result.pop("expected_config", None)
    return result


def probe_context(args, identities, refs, inputs):
    from sensys_runs.iccps27_followup import reset_probe
    receipt = inputs.json(args.validation_receipt)
    manifest = inputs.json(Path(args.probe_root) / "manifest.json")
    require(receipt.get("status") == "PASS" and receipt.get("execution_kind") == "actual_tensorflow_checkpoint_restore"
            and receipt.get("tolerance") == 1e-5, "Six real clean-checkpoint validation receipt is required")
    checks, expected = receipt.get("checks", []), receipt.get("expected", [])
    keys = {(m, s) for m in ("lio_er", "refine_er") for s in (12341, 12342, 12343)}
    require(len(checks) == len(expected) == 6 and {(r["method"], r["seed"]) for r in checks} == keys
            and {(r["method"], r["seed"]) for r in expected} == keys, "Incomplete six-checkpoint receipt")
    expected_map = {r["run_id"]: r for r in expected}
    reference_map = {(r["method"], r["seed"]): r for r in refs}
    for check in checks:
        ref = reference_map[(check["method"], check["seed"])]
        original = expected_map.get(check["run_id"], {})
        require(check["run_id"] == ref["run_id"] and ref["eligible"] and
                check.get("passed") is True and math.isfinite(check["absolute_difference"])
                and check["absolute_difference"] <= 1e-5
                and abs(check["P1"] - check["expected_P_tau_1"]) == check["absolute_difference"]
                and check["expected_P_tau_1"] == ref["exact"]["P_tau_1"]
                and original.get("expected_P_tau_1") == ref["exact"]["P_tau_1"]
                and original.get("reference_result_sha256") == inputs.hashes[ref["sources"]["exact"]],
                "Clean validation value or frozen reference identity mismatch")
        meta = inputs.json(Path(args.reference_evaluation_root) / ref["run_id"] / "energy_eval_provenance.json")
        require(check["checkpoint_hashes"] == meta["checkpoint_sha256"], "Clean validation checkpoint differs from frozen reference")
    require(manifest.get("sources_unchanged") is True and manifest.get("inputs_unchanged") is True
            and not manifest.get("pending"), "Probe batch has unfinished or changed inputs")
    require(manifest.get("source_hashes") == receipt.get("source_hashes")
            and set(manifest.get("source_hashes", {})) == set(reset_probe.SOURCE_FILES), "Probe and validation sources differ or source coverage is incomplete")
    require(all(check.get("runtime") == manifest.get("runtime") for check in checks)
            and isinstance(manifest.get("runtime"), dict)
            and manifest["runtime"].get("device") == "CPU"
            and "tensorflow" in manifest["runtime"], "Probe and validation runtime mismatch")
    require(manifest.get("source_git_identity") == receipt.get("source_git_identity")
            == reset_probe.source_git_identity(), "Probe and validation Git source identity mismatch")
    for name in manifest["source_hashes"]:
        path = (REPO / name).resolve()
        require(path.is_relative_to(REPO), "Probe source path escapes repository")
        # Cross-platform identity uses committed blobs. Local raw bytes are still
        # retained and checked before/after this analysis without normalization.
        inputs.read(path)
    receipt_digest = inputs.hashes[str(Path(args.validation_receipt).resolve())]
    require(manifest.get("validation_receipt_sha256") == receipt_digest and
            set(manifest.get("input_hashes", {}).values()) == {receipt_digest, inputs.hashes[str(Path(args.eia_inventory).resolve())]},
            "Probe receipt or inventory hash mismatch")
    all_records = manifest.get("completed", []) + manifest.get("failed", [])
    require(len(all_records) == len(identities) and {r["run_id"] for r in all_records} == {r["run_id"] for r in identities}, "Probe coverage differs from the complete registered matrix")
    require(manifest.get("status") == ("INCOMPLETE" if manifest.get("failed") else "COMPLETE"), "Probe manifest status disagrees with failures")
    return manifest, receipt_digest


def attach_probe(row, root, context, inputs):
    manifest, receipt_digest = context
    completed = {r["run_id"]: r for r in manifest["completed"]}
    if row["run_id"] not in completed:
        failure = next((r.get("error", "UNVERIFIED") for r in manifest.get("failed", []) if r["run_id"] == row["run_id"]), "UNVERIFIED")
        row["probe_issues"].append("Reset probe failed or missing: " + failure)
        return
    try:
        path = Path(root) / (row["run_id"].replace("/", "__") + ".json")
        result = inputs.json(path)
        require(row["observed_eligible"], "Observed checkpoint binding is unavailable")
        require(result.get("run_id") == row["run_id"] and result.get("method") == row["method"]
                and result.get("seed") == row["seed"] and result.get("eia_attack") == row["attack"], "Probe true method/seed/attack mismatch")
        require(result.get("execution_kind") == "actual_tensorflow_checkpoint_restore" and result.get("no_payment_query") is True
                and result.get("no_training_or_optimizer") is True and result.get("wording") == PROBE_WORDING,
                "Probe execution contract mismatch")
        require(result.get("validation_receipt_sha256") == receipt_digest
                and result.get("runtime") == manifest["runtime"]
                and result.get("checkpoint_hashes") == row["checkpoint_hashes"]
                and result.get("config_provenance_hashes") == row["config_provenance_hashes"], "Probe checkpoint/config/provenance binding mismatch")
        p1 = result["P1"]
        require(isinstance(p1, (int, float)) and math.isfinite(p1) and 0 <= p1 <= 1, "Invalid reset P1")
        lower, upper = 2-p1, 1+4*(1-p1)
        require(result["E_T_lower"] == lower and result["E_T_upper"] == upper
                and result["E_T_bound_width"] == upper-lower, "Reset-derived bounds are inconsistent")
        record = completed[row["run_id"]]
        require(record.get("file") == path.name and all(record[k] == result[k] for k in ("P1", "E_T_lower", "E_T_upper", "E_T_bound_width")), "Probe manifest/result mismatch")
        row["probe"] = {k: result[k] for k in ("P1", "E_T_lower", "E_T_upper", "E_T_bound_width", "wording")}
        row["probe"]["mission_time_primary"] = "observed_100_episode_mean" if upper-lower > .1 else "reset_based_interval"
        row["probe_eligible"] = True
        row["sources"]["probe"] = str(path.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, FileNotFoundError) and exc.filename:
            inputs.missing.add(str(Path(exc.filename).resolve()))
        row["probe_issues"].append("UNVERIFIED reset probe: " + str(exc))


def paired_observed(method, reference, gates):
    complete = len(method) == len(reference) == 10 and all(r["observed_eligible"] for r in method+reference)
    if not complete:
        return dict(status="INCOMPLETE", reason="All ten valid paired observed runs are required; no replacement or partial effect estimate", adapter=ADAPTER)
    require([r["seed"] for r in method] == [r["seed"] for r in reference] == list(a9.SEEDS), "Paired seed ordering mismatch")
    def key(name):
        return name, "uniform_v1", "env_reward", "homogeneous_v1", "uniform_v1", "homogeneous_v1"
    wave = {key("refine_er"): {r["seed"]: r["observed"] for r in method},
            key("lio_er"): {r["seed"]: r["observed"] for r in reference}}
    cost = gates.evaluate({"wave1": wave})["gates"]["G1"]["cost"]
    left = [r["observed"]["mean_agent_mission_cost"] for r in method]
    right = [r["observed"]["mean_agent_mission_cost"] for r in reference]
    lower = sum(x < y for x, y in zip(left, right))
    higher = sum(x > y for x, y in zip(left, right))
    overlap = max(min(left), min(right)) <= min(max(left), max(right))
    wording = ("REFiNE+EIA has lower observed mean-agent modeled cost in 10 of 10 paired seeds, with non-overlapping ranges" if lower == 10 and not overlap else
               "LIO+EIA has lower observed mean-agent modeled cost in 10 of 10 paired seeds, with non-overlapping ranges" if higher == 10 and not overlap else
               "no consistent difference across 10 paired seeds")
    return dict(status="COMPLETE", adapter=ADAPTER, sign="(REFiNE+EIA - LIO+EIA) / LIO+EIA; negative is lower REFiNE+EIA cost",
                **{k: cost[k] for k in ("relative_median", "relative_interval", "relative_differences", "seeds")},
                refine_lower=lower, lio_lower=higher, ties=10-lower-higher,
                refine_range=[min(left), max(left)], lio_range=[min(right), max(right)],
                ranges_overlap=overlap, paper_wording=wording,
                source="unchanged summarize_energy_profiles.score_rows and evaluate_gates.evaluate G1")


def descriptive_shift(rows, refs):
    pairs = []
    for left, right in zip(rows, refs):
        require(left["seed"] == right["seed"], "E3 pair identity mismatch")
        good = left["observed_eligible"] and right["observed_eligible"]
        metrics = {name: left["observed"][name] - right["observed"][name] for name in
                   ("mean_agent_mission_cost", "mean_mission_steps", "one_step_fraction", "success_rate")} if good else None
        pairs.append(dict(seed=left["seed"], eligible=good, shift=metrics))
    seen = [pair["seed"] for pair in pairs]
    require(len(seen) == len(set(seen)) and set(seen).issubset(a9.SEEDS), "E3 duplicate or unregistered seed")
    pairs.extend(dict(seed=seed, eligible=False, shift=None) for seed in a9.SEEDS if seed not in seen)
    complete = len(pairs) == 10 and all(pair["eligible"] for pair in pairs)
    valid = [(left, right) for left, right in zip(rows, refs)
             if left["observed_eligible"] and right["observed_eligible"]]
    method_name = {"refine_eia_er": "REFiNE+EIA", "lio_eia_er": "LIO+EIA"}.get(
        rows[0].get("method") if rows else None, "EIA")
    reference_name = refs[0].get("arm", "the no-attack reference") if refs else "the no-attack reference"
    metric_labels = {"mean_agent_mission_cost": "observed mean-agent modeled cost",
                     "mean_mission_steps": "observed mean mission steps",
                     "one_step_fraction": "observed one-step fraction",
                     "success_rate": "observed success rate"}
    metric_comparisons = {}
    for name, label in metric_labels.items():
        left_values = [left["observed"][name] for left, right in valid]
        right_values = [right["observed"][name] for left, right in valid]
        lower = sum(left < right for left, right in zip(left_values, right_values))
        higher = sum(left > right for left, right in zip(left_values, right_values))
        left_range = [min(left_values), max(left_values)] if left_values else None
        right_range = [min(right_values), max(right_values)] if right_values else None
        overlap = max(left_range[0], right_range[0]) <= min(left_range[1], right_range[1]) if valid else None
        if not complete:
            wording = "INCOMPLETE: all 10 registered paired seeds are required"
        elif lower == 10 and not overlap:
            wording = method_name + " has lower " + label + " than " + reference_name + " in 10 of 10 paired seeds, with non-overlapping ranges"
        elif higher == 10 and not overlap:
            wording = method_name + " has higher " + label + " than " + reference_name + " in 10 of 10 paired seeds, with non-overlapping ranges"
        else:
            wording = "no consistent difference across 10 paired seeds"
        metric_comparisons[name] = dict(status="COMPLETE" if complete else "INCOMPLETE",
            registered_pairs=10, eligible_pairs=len(valid), eia_lower=lower, eia_higher=higher,
            ties=len(valid)-lower-higher, eia_method=method_name, reference_arm=reference_name,
            eia_range=left_range, reference_range=right_range, ranges_overlap=overlap,
            paper_wording=wording, direction="Lower and higher describe numerical direction only.")
    return dict(status="COMPLETE" if all(p["eligible"] for p in pairs) else "INCOMPLETE", pairs=pairs,
                sign="EIA minus the same-seed no-attack reference; descriptive shifts only",
                metrics=metric_comparisons,
                source="validated observed 100-episode final evaluations; A8 for REFiNE, A3 for LIO")


def threshold_summary(rows, *, reference=False):
    valid = [r for r in rows if r["probe_eligible"]]
    observed = [r for r in rows if r["observed_eligible"]]
    return dict(status="COMPLETE" if len(valid) == 10 else "INCOMPLETE", registered_seeds=10,
                reset_threshold=.95, reset_eligible_seeds=len(valid),
                reset_passing_seeds=sum(r["probe"]["P1"] >= .95 for r in valid),
                observed_status="COMPLETE" if len(observed) == 10 else "INCOMPLETE",
                observed_eligible_seeds=len(observed),
                observed_one_step_at_least_95_count=sum(r["observed"]["one_step_fraction"] >= .95 for r in observed),
                source="frozen full clean enumeration" if reference else PROBE_WORDING)


def analyze(args):
    inputs = a9.Inputs()
    out = Path(args.output_dir).resolve()
    require(re.fullmatch(r"\d{8}", args.date) and args.date in out.name and not out.exists(), "Output must be a fresh dated directory")
    protected = [REPO, Path(args.results_root).resolve(), Path(args.probe_root).resolve(),
                 Path(args.reference_evaluation_root).resolve(), Path(args.reference_enumeration_root).resolve()]
    require(not any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected), "Output overlaps repository or evidence")
    inputs.read(Path(__file__))
    inputs.read(Path(a9.__file__))
    gates, summary = a9.pinned_modules(args.source_manifest, args.workspace_root, inputs, args.protected_manifest)
    inputs.read(args.registration)
    campaign = inputs.json(args.campaign_manifest)
    weights, identities = matrix(campaign, args.eia_inventory, inputs)
    refs_inventory = a9.inventory(args.reference_inventory, ("A3", "A8"), inputs)
    refs = {}
    for arm in ("A3", "A8"):
        refs[arm] = a9.read_arm(arm, refs_inventory[arm], args.reference_evaluation_root,
                              args.reference_enumeration_root, inputs, summary, args.workspace_root)
        for row in refs[arm]:
            row["observed_eligible"] = row["eligible"]
            row["probe_eligible"] = row["eligible"]
            row["probe"] = {"P1": row["exact"]["P_tau_1"]} if row["eligible"] else None
            if row["eligible"]:
                row["observed"] = add_one_step(row["observed"], Path(row["sources"]["observed"]), inputs)
    runs = [read_observed(identity, args.results_root, inputs, summary) for identity in identities]
    probe_problem = None
    try:
        context = probe_context(args, identities, refs["A3"] + refs["A8"], inputs)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, FileNotFoundError) and exc.filename:
            inputs.missing.add(str(Path(exc.filename).resolve()))
        context = None
        probe_problem = "UNVERIFIED probe batch: " + str(exc)
    for row in runs:
        if context is None:
            row["probe_issues"].append(probe_problem)
        else:
            attach_probe(row, args.probe_root, context, inputs)
    results = {}
    for weight in weights:
        label = weight_label(weight)
        cells = {m: [r for r in runs if r["method"] == m and r["weight"] == label] for m in METHODS}
        results[label] = dict(E1={m: threshold_summary(cells[m]) for m in METHODS},
                              E2=paired_observed(cells["refine_eia_er"], cells["lio_eia_er"], gates),
                              E3={"refine_eia_er_vs_A8": descriptive_shift(cells["refine_eia_er"], refs["A8"]),
                                  "lio_eia_er_vs_A3": descriptive_shift(cells["lio_eia_er"], refs["A3"])})
    wide_refine = [dict(run_id=r["run_id"], seed=r["seed"], weight=r["weight"], width=r["probe"]["E_T_bound_width"])
                   for r in runs if r["method"] == "refine_eia_er" and r["probe_eligible"] and r["probe"]["E_T_bound_width"] > .1]
    report = dict(date=args.date, declared_run_count=len(runs), fallback_declared=campaign["fallback_declared"],
                  weights=[list(w) for w in weights], results=results, runs=runs,
                  no_attack_references={arm: dict(E1=threshold_summary(refs[arm], reference=True), runs=refs[arm]) for arm in refs},
                  mission_time_rule="Width > 0.1: observed 100-episode mean is primary. Otherwise the reset-based E[T] interval is primary. Neither interval midpoint nor bound is an exact E[T].",
                  deferred_decision=dict(deadline="2026-11-02", candidates=wide_refine,
                                         status="DECLARATION_REQUIRED_BY_DEADLINE" if wide_refine else "NO_VERIFIED_WIDE_REFINE_INTERVAL",
                                         action="Record the human decision by the deadline. This report never launches EIA export or enumeration. LIO interval width alone does not trigger this rule."),
                  failure_rule="All registered cells and failed or missing seeds are retained; no replacements. E2 needs all ten valid observed pairs. E1 remains INCOMPLETE without all ten validated probes.",
                  adapter=ADAPTER, registration_source=str(Path(args.registration).resolve()),
                  code_sources={name: str(Path(function.__code__.co_filename).resolve()) + ":" + str(function.__code__.co_firstlineno)
                                for name, function in (("score_rows", summary.score_rows), ("G1_evaluate", gates.evaluate),
                                                       ("observed_validation", a9.observed_metrics), ("EIA_analysis", analyze))})
    inputs.verify()
    out.mkdir(parents=True, exist_ok=False)
    stem = "REFiNE_ICCPS2027_EIA_Results_" + args.date
    with (out / (stem + ".json")).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (out / "per_run.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in runs:
            obs = row["observed"] or {}
            writer.writerow(dict(method=row["method"], weight=row["weight"], seed=row["seed"],
                                 observed_mean_cost=obs.get("mean_agent_mission_cost"), mean_steps=obs.get("mean_mission_steps"),
                                 one_step_fraction=obs.get("one_step_fraction"), success_rate=obs.get("success_rate")))
    prose = ["# Registered EIA results, " + args.date, "", "Sources: input_manifest.json and per-run evidence paths in the companion JSON.", "", ADAPTER, ""]
    for label, result in results.items():
        prose += ["Weight " + label, "E1: " + json.dumps(result["E1"], sort_keys=True),
                  "E2: " + json.dumps(result["E2"], sort_keys=True),
                  "E3: " + json.dumps(result["E3"], sort_keys=True),
                  "Source: matching per-run observed/probe records and unchanged G1 code path in the companion JSON.", ""]
    prose += [report["mission_time_rule"], "Source: registration and per-run probe interval widths.", "",
              json.dumps(report["deferred_decision"], sort_keys=True), "Source: registered deadline and only validated REFiNE+EIA interval widths.", "", report["failure_rule"], "Source: registration and per-run issues.", ""]
    with (out / (stem + ".md")).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(prose))
    inputs.verify()
    with (out / "input_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(input_sha256=inputs.hashes, missing_inputs=sorted(inputs.missing), inputs_unchanged=True,
                       source_manifests_checked_before_and_after=True, command=sys.argv), stream, indent=2)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-manifest", "eia-inventory", "results-root", "probe-root", "validation-receipt", "registration",
                 "reference-inventory", "reference-evaluation-root", "reference-enumeration-root",
                 "source-manifest", "protected-manifest", "workspace-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "EIA_ANALYSIS_ERROR: " + str(exc) + "\n")
    print("Wrote " + str(args.output_dir) + "; retained " + str(len(report["runs"])) + " registered cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
