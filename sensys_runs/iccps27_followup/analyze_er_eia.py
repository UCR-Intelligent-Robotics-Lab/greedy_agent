"""Registered ER-LIO+EIA analysis with historical probe source identities."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sensys_runs.iccps27_followup import analyze_a9 as a9
from sensys_runs.iccps27_followup import analyze_eia as ae
from sensys_runs.iccps27_followup.source_identity_v2 import source_identity_scope, verified_source_identity

ARM = "er_lio_eia_er"
LABEL = "ER-LIO+EIA"
METHOD = "lio_eia_er"
HISTORICAL_REF = "4ce4341d867cba0ceb4836e4e288a3c19812a0ce"
METRICS = ("mean_agent_mission_cost", "mean_mission_steps", "one_step_fraction", "success_rate")
CSV_FIELDS = ("arm", "method", "weight", "seed", "observed_mean_cost", "mean_steps",
              "one_step_fraction", "success_rate", "P1", "E_T_lower", "E_T_upper",
              "E_T_bound_width", "mission_time_primary", "observed_eligible", "probe_eligible")
ADAPTER = ("Numerical G1 adapter only: ER-LIO+EIA occupies the refine_er slot and "
           "REFiNE+EIA occupies the lio_er reference slot. Scientific labels remain "
           "unchanged. Unchanged score_rows and evaluate_gates.evaluate G1 supply "
           "the observed cost and median paired relative difference with its 95% "
           "percentile bootstrap interval, 10000 resamples, seed 20261010. No internal "
           "p-values or gate decisions are reported.")
require = ae.require


def energy_identity(config, provenance=None):
    expected = dict(energy_in_reward=True, energy_reward_beta=1.0,
                    energy_reward_form="episode_mean", followup_arm=ARM,
                    training_objective="energy_reward", display_label=LABEL)
    require(all(config.get(k) == v for k, v in expected.items()) and
            config.get("energy_in_reward") is True and
            config["lio"].get("Fairness_multiplier") == 0.0,
            "Missing or invalid registered ER-LIO+EIA objective")
    if provenance is None:
        return
    require(all(provenance.get(k) == expected[k] for k in
                ("followup_arm", "training_objective", "display_label")),
            "ER-LIO+EIA provenance scientific identity mismatch")
    energy = provenance.get("energy_reward", {})
    fixed = dict(schema_version=1, method=METHOD, arm=ARM, objective="energy_reward",
                 display_label=LABEL, energy_in_reward=True, energy_reward_beta=1.0,
                 energy_reward_form="episode_mean", agent_ids=[0, 1, 2, 3],
                 episode_kinds=["ordinary", "prime"], buffer_field="reward",
                 energy_field="total_energy", penalty="beta * buffer.total_energy / len(buffer.reward)",
                 total_energy_includes_lambda=True, consumers=["actor_update", "incentive_learner"],
                 training_only=True, environment_reward_unchanged=True,
                 implementation="sensys_runs.energy_reward_runtime.wrap_training")
    require(energy == fixed, "ER-LIO+EIA energy provenance differs from registered wrapper")
    require(provenance.get("er_eia_adapter") == "scoped_registered_er_lio_eia_v1" and
            provenance.get("er_eia_launch_argv", []).count("--energy_in_reward") == 1,
            "Missing ER-LIO+EIA adapter or explicit opt-in provenance")


def matrix(manifest, inventory_path, inputs):
    require(manifest.get("mode") == "campaign", "Formal analysis cannot use a smoke manifest")
    require(manifest.get("weight_pairs") == [list(w) for w in ae.WEIGHTS], "Wrong registered weight matrix")
    expected = {(w, s) for w in ae.WEIGHTS for s in a9.SEEDS}
    runs = manifest.get("runs", [])
    require(manifest.get("n_runs") == len(runs) == 30, "Expected exactly 30 registered ER-LIO+EIA runs")
    result, ids = {}, set()
    for row in runs:
        pair, seed = tuple(row.get("weight_pair", [])), row.get("seed")
        require((pair, seed) in expected and (pair, seed) not in result and row.get("method") == METHOD,
                "Duplicate or unregistered ER-LIO+EIA cell")
        config = row["effective_config"]
        energy_identity(config)
        rid = ae.safe_id(config["main"]["exp_name"] + "/" + config["main"]["dir_name"])
        require(rid not in ids and "energy_reward" in config["main"]["dir_name"] and
                config["main"]["seed"] == seed and row.get("weight") == ae.weight_label(pair),
                "ER-LIO+EIA run identity mismatch")
        require(config["lio"].get("eia_w_lever") == pair[0] and config["lio"].get("eia_w_door") == pair[1],
                "ER-LIO+EIA weight differs from declared matrix")
        result[pair, seed] = dict(run_id=rid, method=METHOD, arm=ARM, display_label=LABEL,
                                 weight=ae.weight_label(pair), weight_pair=list(pair), seed=seed,
                                 expected_config=config)
        ids.add(rid)
    require(set(result) == expected, "Incomplete ER-LIO+EIA matrix")
    rows = list(csv.DictReader(io.StringIO(inputs.read(inventory_path).decode("utf-8-sig"))))
    actual = [(ae.safe_id(r["run_id"]), r["method"], int(r["seed"])) for r in rows]
    require(len(actual) == 30 and set(actual) == {(r["run_id"], METHOD, r["seed"]) for r in result.values()},
            "ER-LIO+EIA inventory differs from the complete 30-run matrix")
    return [result[w, s] for w in ae.WEIGHTS for s in a9.SEEDS]


def read_observed(identity, root, inputs, summary):
    row = ae.read_observed(identity, root, inputs, summary)
    if row["observed_eligible"]:
        try:
            run = Path(root) / row["run_id"]
            config, prov = (inputs.json(run / name) for name in ("config.json", "provenance.json"))
            energy_identity(config, prov)
            require(row["observed"]["mean_agent_mission_cost"] == row["observed"]["mean_mission_steps"],
                    "Uniform observed cost differs from observed mean steps")
            row["energy_reward"] = prov["energy_reward"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["observed_eligible"] = False
            row["issues"].append("UNVERIFIED ER-LIO+EIA objective: " + str(exc))
    return row


def require_observed_bindings(rows):
    failed = [{"run_id": row["run_id"], "issues": row["issues"]}
              for row in rows if not row["observed_eligible"]]
    require(not failed, "Invalid ER-LIO+EIA observed bindings; analysis stopped before writing any result: "
            + json.dumps(failed, sort_keys=True))

def clean_references(args, inputs, summary):
    inventory = a9.inventory(args.reference_inventory, ("A3", "A8"), inputs)
    refs = {}
    for arm in ("A3", "A8"):
        refs[arm] = a9.read_arm(arm, inventory[arm], args.reference_evaluation_root,
                              args.reference_enumeration_root, inputs, summary, args.workspace_root)
    require(all(r["eligible"] for rows in refs.values() for r in rows), "Invalid frozen A3/A8 reference")
    return refs


def historical_probes(args, identities, rows, refs, inputs):
    from sensys_runs.iccps27_followup import reset_probe as rp
    receipt = inputs.json(args.validation_receipt)
    digest = hashlib.sha256(inputs.read(args.protected_manifest)).hexdigest()
    require(digest == rp.PROTECTED_MANIFEST_SHA256, "Protected manifest identity mismatch")
    audit = {}
    with source_identity_scope(rp, repo=REPO, source_ref=HISTORICAL_REF,
            source_files=rp.SOURCE_FILES, protected_pins=rp.PROTECTED_PINS,
            protected_manifest_sha256=digest, expected_identity=receipt["source_git_identity"], audit=audit):
        context = ae.probe_context(args, identities, refs["A3"] + refs["A8"], inputs)
        for row in rows:
            ae.attach_probe(row, args.probe_root, context, inputs)
    return audit


def load_historical(args, refs, inputs, summary):
    """Recover validated E1 without rewriting or recomputing historical E2/E3."""
    old = inputs.json(args.historical_report)
    manifest = inputs.json(args.historical_campaign_manifest)
    weights, identities = ae.matrix(manifest, args.historical_inventory, inputs)
    require(tuple(weights) == ae.WEIGHTS and len(identities) == 60 and
            old.get("declared_run_count") == 60 and len(old.get("runs", [])) == 60,
            "Historical reference must be the registered complete 60-run matrix")
    stored = {r["run_id"]: r for r in old["runs"]}
    require(len(stored) == 60, "Duplicate historical result row")
    rows = [ae.read_observed(i, args.historical_results_root, inputs, summary) for i in identities]
    keys = ("method", "weight", "weight_pair", "seed", "observed", "observed_eligible",
            "issues", "checkpoint_hashes", "attack", "config_provenance_hashes")
    for row in rows:
        prior = stored.get(row["run_id"], {})
        require(row["observed_eligible"] and not row["issues"] and
                all(row.get(k) == prior.get(k) for k in keys),
                "Historical observed input differs from accepted report: " + row["run_id"])
    old_args = SimpleNamespace(**vars(args))
    old_args.probe_root = args.historical_probe_root
    old_args.validation_receipt = args.historical_validation_receipt
    old_args.eia_inventory = args.historical_inventory
    audit = historical_probes(old_args, identities, rows, refs, inputs)
    require(all(r["probe_eligible"] and not r["probe_issues"] for r in rows),
            "Historical probe bindings failed; no recovered E1 may be reported")
    validated = {}
    for weight in ae.WEIGHTS:
        label = ae.weight_label(weight)
        validated[label] = {method: ae.threshold_summary([r for r in rows if r["method"] == method and r["weight"] == label])
                            for method in ae.METHODS}
    review = inputs.read(args.historical_review).decode("utf-8-sig")
    accepted = {}
    for match in re.finditer(r"READONLY_VALIDATED_E1 ([^\s]+) ([^\s]+) (\{[^\r\n]*\})", review):
        key = match.group(1), match.group(2)
        require(key not in accepted, "Duplicate E1 in accepted read-only review")
        accepted[key] = json.loads(match.group(3))
    require(len(accepted) == 6 and all(accepted.get((w, m)) == value for w, methods in validated.items()
                                      for m, value in methods.items()),
            "Recovered E1 differs from the accepted read-only review")
    return dict(runs=rows, validated_E1=validated, source_identity_audit=audit,
                historical_results=old["results"], historical_no_attack_references=old["no_attack_references"],
                source_report=str(Path(args.historical_report).resolve()),
                source_review=str(Path(args.historical_review).resolve()),
                source_report_sha256=inputs.hashes[str(Path(args.historical_report).resolve())],
                validation_receipt_sha256=inputs.hashes[str(Path(args.historical_validation_receipt).resolve())],
                validation_source_identity=inputs.json(args.historical_validation_receipt)["source_git_identity"],
                note="Historical E2/E3 are retained unchanged; E1 is freshly rebound to the accepted historical probes.")


def load_a9(path, inputs, summary, workspace):
    report = inputs.json(path)
    saved = [r for r in report.get("runs", []) if r.get("arm") == "A9"]
    require(len(saved) == 10 and {r["seed"] for r in saved} == set(a9.SEEDS), "Incomplete A9 reference")
    saved.sort(key=lambda r: r["seed"])
    inventory = {r["seed"]: r for r in saved}
    first = saved[0]
    root = Path(first["sources"]["observed"]).parents[2]
    enumeration = Path(first["sources"]["exact"]).parents[1]
    export = Path(first["sources"]["export"]).parent
    rows = a9.read_arm("A9", inventory, root, enumeration, inputs, summary, workspace, export)
    require(rows == saved and all(r["eligible"] for r in rows), "A9 scientific inputs differ from completed A9 result")
    for row in rows:
        row["observed"] = ae.add_one_step(row["observed"], row["sources"]["observed"], inputs)
        row.update(observed_eligible=True, probe_eligible=True, probe={"P1": row["exact"]["P_tau_1"]})
    return rows


def analysis_context(args, inputs, campaign, identities, rows):
    from sensys_runs.iccps27_followup import er_eia_probe as ep
    inventory = list(csv.DictReader(io.StringIO(inputs.read(args.eia_inventory).decode("utf-8-sig"))))
    records = ep.validate_matrix(campaign, inventory, 30)
    require(set(records) == {r["run_id"] for r in identities}, "Analysis/probe matrix identity mismatch")
    context = inputs.json(Path(args.probe_root) / "er_context.json")
    legacy = inputs.json(Path(args.probe_root) / "manifest.json")
    receipt = inputs.json(args.validation_receipt)
    required = dict(schema_version=1, objective="energy_reward", followup_arm=ARM, display_label=LABEL,
                    campaign=ep.CAMPAIGN, expected_count=30)
    require(all(context.get(k) == v for k, v in required.items()), "Invalid ER probe companion identity")
    expected_inputs = {name: hashlib.sha256(inputs.read(path)).hexdigest() for name, path in
        (("campaign_manifest", args.campaign_manifest), ("inventory", args.eia_inventory),
         ("validation_receipt", args.validation_receipt), ("protected_manifest", args.protected_manifest))}
    require(context.get("input_hashes") == expected_inputs and
            context.get("legacy_manifest_sha256") == inputs.hashes[str((Path(args.probe_root)/"manifest.json").resolve())] and
            context.get("status") == legacy.get("status"), "ER probe companion input/manifest binding mismatch")
    require(expected_inputs["protected_manifest"] == ep.rp.PROTECTED_MANIFEST_SHA256,
            "ER companion protected manifest differs from approved pin")
    release = context.get("training_release", {})
    require(release == {k: campaign["source"].get(k) for k in ("git_commit", "git_describe")} and
            release.get("git_describe") == args.release_ref, "ER campaign/probe release mismatch")
    audit = {}
    wrapper = verified_source_identity(REPO, args.release_ref, ep.WRAPPER_FILES, {},
        expected_inputs["protected_manifest"], expected_identity=context.get("wrapper_source_git_identity"), audit=audit)
    require(wrapper["commit"] == release.get("git_commit") and
            context.get("inference_source_git_identity") == receipt.get("source_git_identity"),
            "ER training/inference source identity mismatch")
    source_names = set(ep.adapter.source_hashes())
    expected_sources = {name: campaign["source"]["sha256"][name] for name in source_names}
    current = {}
    for row in rows:
        rid = row["run_id"]
        run = Path(args.results_root)/rid
        config, provenance = (inputs.json(run/name) for name in ("config.json", "provenance.json"))
        ep.validate_energy_run(config, provenance, records[rid], release["git_commit"], expected_sources)
        current[rid] = dict(config_sha256=inputs.hashes[str((run/"config.json").resolve())],
                           provenance_sha256=inputs.hashes[str((run/"provenance.json").resolve())],
                           checkpoint_hashes=row.get("checkpoint_hashes"))
    require(context.get("run_inputs") == current, "ER companion checkpoint/config/provenance bindings differ")
    for name in ep.WRAPPER_FILES:
        inputs.read(REPO/name)
    return dict(companion=context, wrapper_source_identity_audit=audit,
                companion_source=str((Path(args.probe_root)/"er_context.json").resolve()))

def metric_comparison(rows, refs, field):
    require([r["seed"] for r in rows] == [r["seed"] for r in refs] == list(a9.SEEDS), "Paired seed ordering mismatch")
    valid = [(x, y) for x, y in zip(rows, refs) if x["observed_eligible"] and y["observed_eligible"]]
    left, right = ([r[i]["observed"][field] for r in valid] for i in (0, 1))
    overlap = max(min(left), min(right)) <= min(max(left), max(right)) if valid else None
    lower, higher = sum(x < y for x, y in zip(left, right)), sum(x > y for x, y in zip(left, right))
    return dict(status="COMPLETE" if len(valid) == 10 else "INCOMPLETE", registered_pairs=10,
                eligible_pairs=len(valid), er_lower=lower, reference_lower=higher, ties=len(valid)-lower-higher,
                er_range=[min(left), max(left)] if left else None,
                reference_range=[min(right), max(right)] if right else None, ranges_overlap=overlap)


def paired_cost(rows, refs, gates):
    comparison = metric_comparison(rows, refs, "mean_agent_mission_cost")
    if comparison["status"] != "COMPLETE":
        return dict(comparison, adapter=ADAPTER, reason="All ten valid paired observed seeds are required")
    def key(method):
        return method, "uniform_v1", "env_reward", "homogeneous_v1", "uniform_v1", "homogeneous_v1"
    wave = {key("refine_er"): {r["seed"]: r["observed"] for r in rows},
            key("lio_er"): {r["seed"]: r["observed"] for r in refs}}
    cost = gates.evaluate({"wave1": wave})["gates"]["G1"]["cost"]
    return dict(comparison, adapter=ADAPTER,
                **{k: cost[k] for k in ("relative_median", "relative_interval", "relative_differences", "seeds")},
                sign="(ER-LIO+EIA - REFiNE+EIA) / REFiNE+EIA; negative means lower ER-LIO+EIA cost",
                source="unchanged summarize_energy_profiles.score_rows and evaluate_gates.evaluate G1")


def paper_wording(cost, er_e1, refine_e1, weight=None):
    """Amendment 1: completion match, then historical cost-only rules.

    Completion-only differences retain both E1 counts. They do not introduce
    a new priority over the historical E2 cost criterion. G1 estimates,
    intervals, observed completion frequencies and probe bounds do not decide.
    """
    if any(x["status"] != "COMPLETE" for x in (cost, er_e1, refine_e1)):
        return dict(rule=None, outcome=None,
                    text="INCOMPLETE: all ten valid paired seeds and reset probes are required")
    lead = "At this weight"
    if weight is not None:
        pair = tuple(float(value) for value in weight.split(",")) if isinstance(weight, str) else tuple(weight)
        require(pair in ae.WEIGHTS, "Paper wording requires a registered weight pair")
        lead = f"At weight ({pair[0]:.1f}, {pair[1]:.1f})"
    er_count, refine_count = er_e1["reset_passing_seeds"], refine_e1["reset_passing_seeds"]
    overlap_clause = "the observed mean-cost ranges overlap"
    if cost["ranges_overlap"]:
        winner = ("ER-LIO+EIA" if cost["er_lower"] == 10 else
                  "REFiNE+EIA" if cost["reference_lower"] == 10 else None)
        if winner is not None:
            overlap_clause = (winner + " has lower observed mean-agent modeled cost in 10 of 10 paired seeds, "
                              "and the observed mean-cost ranges overlap")
    if er_count >= 9 and refine_count >= 9 and cost["ranges_overlap"]:
        rule, outcome = 1, "a"
        text = (lead + ", ER-LIO+EIA reaches the same one-step completion as REFiNE+EIA, "
                "so at this weight preserving one-step completion under EIA does not depend on REFiNE's incentive-learning path; "
                + overlap_clause + ".")
    elif cost["reference_lower"] == 10 and not cost["ranges_overlap"]:
        rule, outcome = 2, "b"
        text = (lead + ", REFiNE+EIA has lower observed mean-agent modeled cost than ER-LIO+EIA "
                "in 10 of 10 paired seeds, with non-overlapping ranges.")
    elif cost["er_lower"] == 10 and not cost["ranges_overlap"]:
        rule, outcome = 3, "c"
        text = (lead + ", ER-LIO+EIA has lower observed mean-agent modeled cost than REFiNE+EIA "
                "in 10 of 10 paired seeds, with non-overlapping ranges.")
    else:
        rule, outcome = 4, "d"
        text = lead + ", there is no consistent difference in observed mean-agent modeled cost across 10 paired seeds."
        if cost["ranges_overlap"] and (cost["er_lower"] == 10 or cost["reference_lower"] == 10):
            text += " " + overlap_clause[0].upper() + overlap_clause[1:] + "."
    text += (f" Reset P1 >= 0.95 holds for {er_count}/10 ER-LIO+EIA seeds "
             f"and {refine_count}/10 REFiNE+EIA seeds.")
    return dict(rule=rule, outcome=outcome, text=text, er_reset_count=er_count,
                refine_reset_count=refine_count,
                er_lower=cost["er_lower"], refine_lower=cost["reference_lower"],
                er_observed_cost_range=cost["er_range"], refine_observed_cost_range=cost["reference_range"],
                ranges_overlap=cost["ranges_overlap"],
                interpretation="Registered categorical completion match is not a statistical equivalence test; "
                               "completion-only differences report counts without a new superiority rule.")


def descriptive(rows, refs, label):
    metrics = {field: metric_comparison(rows, refs, field) for field in METRICS}
    pairs = []
    for x, y in zip(rows, refs):
        valid = x["observed_eligible"] and y["observed_eligible"]
        pairs.append(dict(seed=x["seed"], eligible=valid, er_run_id=x["run_id"], reference_run_id=y["run_id"],
                          shift={k: x["observed"][k] - y["observed"][k] for k in METRICS} if valid else None,
                          er_probe=x.get("probe"), reference_probe=y.get("probe")))
    return dict(status="COMPLETE" if all(p["eligible"] for p in pairs) else "INCOMPLETE", pairs=pairs,
                metrics=metrics, reference=label, sign="ER-LIO+EIA minus " + label,
                direction="Numerical lower/higher only; lower one-step fraction or success rate is not an improvement.")


def write_results(out, date, report, inputs):
    inputs.verify()
    out.mkdir(parents=True, exist_ok=False)
    stem = "REFiNE_ICCPS2027_ER_LIO_EIA_Results_" + date
    with (out / (stem + ".json")).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (out / "per_run.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in report["runs"]:
            obs, probe = row.get("observed") or {}, row.get("probe") or {}
            writer.writerow(dict(arm=ARM, method=row["method"], weight=row["weight"], seed=row["seed"],
                observed_mean_cost=obs.get("mean_agent_mission_cost"), mean_steps=obs.get("mean_mission_steps"),
                one_step_fraction=obs.get("one_step_fraction"), success_rate=obs.get("success_rate"),
                **{k: probe.get(k) for k in ("P1", "E_T_lower", "E_T_upper", "E_T_bound_width", "mission_time_primary")},
                observed_eligible=row["observed_eligible"], probe_eligible=row["probe_eligible"]))
    prose = ["# Registered ER-LIO+EIA results, " + date, "", "Status: " + report["status"], "", ADAPTER, "",
             "Sources: input_manifest.json, registration, and per-run source paths in the companion JSON.", ""]
    for label, cell in report["results"].items():
        prose.extend(["Weight " + label, json.dumps(cell, sort_keys=True),
                      "Source: the paired observed and reset-probe records in the companion JSON.", ""])
    prose.extend([report["mission_time_rule"], "Source: registered reporting rule and per-run interval widths.", ""])
    with (out / (stem + ".md")).open("x", encoding="utf-8") as handle:
        handle.write("\n".join(prose))
    inputs.verify()
    with (out / "input_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(input_sha256=inputs.hashes, missing_inputs=sorted(inputs.missing), inputs_unchanged=True,
                       source_manifests_checked_before_and_after=True, command=sys.argv), handle, indent=2)
        handle.write("\n")


def analyze(args):
    inputs = a9.Inputs()
    out = Path(args.output_dir).resolve()
    require(re.fullmatch(r"\d{8}", args.date) and args.date in out.name and not out.exists(), "Output must be a fresh dated directory")
    forbidden = [REPO] + [Path(getattr(args, k)).resolve() for k in
                 ("results_root", "probe_root", "historical_results_root", "historical_probe_root",
                  "reference_evaluation_root", "reference_enumeration_root")]
    require(not any(out.is_relative_to(p) or p.is_relative_to(out) for p in forbidden), "Output overlaps repository or evidence")
    for path in (__file__, a9.__file__, ae.__file__, args.registration):
        inputs.read(path)
    gates, summary = a9.pinned_modules(args.source_manifest, args.workspace_root, inputs, args.protected_manifest)
    campaign = inputs.json(args.campaign_manifest)
    identities = matrix(campaign, args.eia_inventory, inputs)
    refs = clean_references(args, inputs, summary)
    historical = load_historical(args, refs, inputs, summary)
    a9_rows = load_a9(args.a9_report, inputs, summary, args.workspace_root)
    runs = [read_observed(identity, args.results_root, inputs, summary) for identity in identities]
    require_observed_bindings(runs)
    # The companion validator binds the new wrapper/training release separately
    # from the unchanged inference kernel and historical six-clean receipt.
    context = analysis_context(args, inputs, campaign, identities, runs)
    audit = historical_probes(args, identities, runs, refs, inputs)
    results = {}
    for weight in ae.WEIGHTS:
        label = ae.weight_label(weight)
        er = [r for r in runs if r["weight"] == label]
        cells = {m: [r for r in historical["runs"] if r["weight"] == label and r["method"] == m] for m in ae.METHODS}
        e1 = {ARM: ae.threshold_summary(er), **historical["validated_E1"][label], "A9": ae.threshold_summary(a9_rows, reference=True)}
        cost = paired_cost(er, cells["refine_eia_er"], gates)
        results[label] = dict(E1=e1, E2=cost,
            E3={"ER_LIO_EIA_vs_LIO_EIA": descriptive(er, cells["lio_eia_er"], "LIO+EIA"),
                "ER_LIO_EIA_vs_A9": descriptive(er, a9_rows, "A9")},
            paper_wording=paper_wording(cost, e1[ARM], e1["refine_eia_er"], label))
    complete = all(r["observed_eligible"] and r["probe_eligible"] for r in runs)
    report = dict(date=args.date, status="COMPLETE" if complete else "INCOMPLETE", declared_run_count=30,
                  arm=ARM, display_label=LABEL, base_method=METHOD, results=results, runs=runs,
                  historical_eia=historical, a9_reference=dict(runs=a9_rows, source_report=str(Path(args.a9_report).resolve()),
                      source_report_sha256=inputs.hashes[str(Path(args.a9_report).resolve())],
                      historical_results={k: inputs.json(args.a9_report)[k] for k in ("M1", "M2", "M3", "paper_wording")}),
                  probe_context=context, historical_inference_audit=audit,
                  mission_time_rule="Width > 0.1: observed 100-episode mean steps is primary and labelled observed. Otherwise report the two-sided reset-based E[T] interval. P1 is " + ae.PROBE_WORDING + ".",
                  failure_rule="All 30 registered cells are retained. No seed replacement. Invalid observed/checkpoint/companion bindings stop before writing and list affected run IDs. Bound failed probes yield INCOMPLETE. Wide intervals do not make a run ineligible.",
                  registration_source=str(Path(args.registration).resolve()), adapter=ADAPTER,
                  code_sources={name: str(Path(fn.__code__.co_filename).resolve()) + ":" + str(fn.__code__.co_firstlineno)
                                for name, fn in (("score_rows", summary.score_rows), ("G1_evaluate", gates.evaluate),
                                                 ("observed_validation", ae.read_observed), ("probe_binding", ae.attach_probe))})
    write_results(out, args.date, report, inputs)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-manifest", "eia-inventory", "results-root", "probe-root", "validation-receipt", "registration",
                 "reference-inventory", "reference-evaluation-root", "reference-enumeration-root",
                 "historical-report", "historical-campaign-manifest", "historical-inventory", "historical-results-root",
                 "historical-probe-root", "historical-validation-receipt", "historical-review", "a9-report",
                 "source-manifest", "protected-manifest", "workspace-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--release-ref", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "ER_EIA_ANALYSIS_ERROR: " + str(exc) + "\n")
    print("ER_EIA_ANALYSIS " + report["status"] + " retained=30 output=" + str(args.output_dir))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
