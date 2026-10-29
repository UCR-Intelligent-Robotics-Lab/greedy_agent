"""Analyze registered A9 inputs without training, inference, or new estimators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(12341, 12351))
ARMS = {
    "A9": "lio_er__uniform_v1__homogeneous_v1__energy_reward",
    "A3": "lio_er__uniform_v1__homogeneous_v1__env_reward",
    "A8": "refine_er__uniform_v1__homogeneous_v1__env_reward",
}
CRITERION = "v6_probability_payment_1e-5_float32_forward_error"
PINNED_IMPORTS = (
    "sensys_runs/evaluate_gates.py", "sensys_runs/summarize_energy_profiles.py",
    "sensys_runs/energy_profile_eval.py", "sensys_runs/energy_profiles.py",
)
ADAPTER = (
    "Numerical G1 adapter only: actual A9 (energy-reward LIO) occupies the existing "
    "G1 method slot keyed refine_er/uniform_v1/env_reward/homogeneous_v1; actual "
    "A3 occupies the G1 reference slot keyed lio_er/uniform_v1/env_reward/"
    "homogeneous_v1. These temporary keys select unchanged evaluate_gates.evaluate "
    "code and do not relabel A9 scientifically. Only observed 100-episode uniform "
    "mean-agent costs and success rates enter that adapter. Existing internal "
    "p-values, gate decisions, and other tests are not reported."
)


class AnalysisError(ValueError):
    pass


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.missing = set()

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise AnalysisError(f"Input changed while reading: {path}")
        self.hashes[str(path)] = digest
        return raw

    def json(self, path, optional=False):
        path = Path(path).resolve()
        if optional and not path.exists():
            self.missing.add(str(path))
            return None
        return json.loads(self.read(path).decode("utf-8-sig"))

    def verify(self):
        for name, digest in self.hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise AnalysisError(f"Input changed: {name}")
        if any(Path(p).exists() for p in self.missing):
            raise AnalysisError("A previously missing input appeared during analysis")


def pinned_modules(manifest, workspace, inputs, protected_manifest=None):
    """Verify all R5 entries before importing any existing analysis module."""
    rows = list(csv.DictReader(io.StringIO(inputs.read(manifest).decode("utf-8-sig")), delimiter="\t"))
    roots = {"repo": REPO, "workspace": Path(workspace).resolve()}
    found = set()
    for row in rows:
        root = roots.get(row["source_root"])
        if root is None:
            raise AnalysisError("Unknown source-manifest root")
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise AnalysisError("Source-manifest path escapes its root")
        digest = hashlib.sha256(inputs.read(path)).hexdigest()
        if digest != row["sha256"].lower():
            raise AnalysisError(f"Pinned source mismatch: {path}")
        found.add((row["source_root"], row["relative_path"].replace("\\", "/")))
    if not all(("repo", name) in found for name in PINNED_IMPORTS):
        raise AnalysisError("R5 source manifest lacks a required imported module")
    if protected_manifest is not None:
        protected = list(csv.DictReader(io.StringIO(inputs.read(protected_manifest).decode("utf-8-sig")), delimiter="\t"))
        if len(protected) != 23:
            raise AnalysisError("Protected analysis manifest must contain all 23 files")
        for row in protected:
            path = (REPO / row["relative_path"]).resolve()
            if not path.is_relative_to(REPO / "sensys_runs/iccps27_analysis"):
                raise AnalysisError("Protected manifest path escapes analysis directory")
            if hashlib.sha256(inputs.read(path)).hexdigest() != row["sha256"].lower():
                raise AnalysisError(f"Protected source mismatch: {path}")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    gates = importlib.import_module("sensys_runs.evaluate_gates")
    summary = importlib.import_module("sensys_runs.summarize_energy_profiles")
    return gates, summary


def inventory(path, labels, inputs):
    rows = list(csv.DictReader(io.StringIO(inputs.read(path).decode("utf-8-sig"))))
    selected = {label: {} for label in labels}
    all_ids = set()
    for row in rows:
        rid = row.get("run_id", "")
        parts = rid.replace("\\", "/").split("/")
        if not rid or any(p in ("", ".", "..") for p in parts) or ":" in rid or rid in all_ids:
            raise AnalysisError("Unsafe or duplicate inventory run_id")
        all_ids.add(rid)
        for label in labels:
            if row.get("arm") != ARMS[label]:
                continue
            seed = int(row["seed"])
            method = "refine_er" if label == "A8" else "lio_er"
            if seed not in SEEDS or seed in selected[label] or row.get("method") != method:
                raise AnalysisError(f"Invalid or duplicate {label} identity")
            selected[label][seed] = row
    if labels == ("A9",) and len(rows) != 10:
        raise AnalysisError("A9 inventory must contain exactly ten A9 runs")
    for label, runs in selected.items():
        if set(runs) != set(SEEDS):
            raise AnalysisError(f"Inventory must retain all ten registered {label} seeds")
    return selected


def observed_metrics(path, identity, inputs, summary):
    """Validate copied episode rows, then use unchanged score_rows."""
    if not path.exists():
        inputs.missing.add(str(path.resolve()))
        return None, "UNVERIFIED: observed 100-episode CSV missing"
    rows = list(csv.DictReader(io.StringIO(inputs.read(path).decode("utf-8-sig"))))
    if len(rows) != 400:
        raise AnalysisError(f"Expected 400 episode/agent rows: {path}")
    seen, episodes = set(), {}
    for row in rows:
        if row["run_id"] != identity["run_id"] or int(row["seed"]) != int(identity["seed"]) or row["method"] != identity["method"] or row["training_profile"] != "uniform_v1":
            raise AnalysisError(f"Observed identity mismatch: {path}")
        for key in ("schema_version", "seed", "checkpoint_episode", "eval_episode", "agent_id", "terminal_code", "steps", "n_lever", "n_start", "n_door"):
            row[key] = int(row[key])
        row["env_return"] = float(row["env_return"])
        pair = row["eval_episode"], row["agent_id"]
        counts = [row[k] for k in ("n_lever", "n_start", "n_door")]
        state = row["steps"], row["terminal_code"]
        if pair in seen or pair[0] not in range(1, 101) or pair[1] not in range(4) or row["schema_version"] != 1 or row["checkpoint_episode"] != 25000:
            raise AnalysisError(f"Invalid observed episode/agent identity: {path}")
        if not 1 <= row["steps"] <= 5 or min(counts) < 0 or sum(counts) != row["steps"] or row["terminal_code"] not in (1, 2) or not math.isfinite(row["env_return"]):
            raise AnalysisError(f"Invalid observed action counts or outcome: {path}")
        if row["terminal_code"] == 2 and row["steps"] != 5:
            raise AnalysisError(f"Timeout must consume five steps: {path}")
        if pair[0] in episodes and episodes[pair[0]] != state:
            raise AnalysisError(f"Inconsistent joint episode outcome: {path}")
        seen.add(pair)
        episodes[pair[0]] = state
    return summary.score_rows(rows, (1.0, 1.0, 1.0)), None


def exact_metrics(path, inputs):
    result = inputs.json(path, optional=True)
    if result is None:
        return None, "UNVERIFIED: exact enumeration result missing"
    tau = result.get("P_tau", [])
    timeout, et = result.get("P_timeout"), result.get("E_T")
    if result.get("stub") or result.get("H") != 5 or result.get("node_count") != 12326281 or len(tau) != 5:
        raise AnalysisError(f"Incomplete or synthetic enumeration: {path}")
    if not all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in [*tau, timeout, et]):
        raise AnalysisError(f"Invalid exact quantities: {path}")
    expected = sum((i + 1) * x for i, x in enumerate(tau)) + 5 * timeout
    if abs(sum(tau) + timeout - 1) > 1e-10 or abs(expected - et) > 1e-10:
        raise AnalysisError(f"Inconsistent exact distribution: {path}")
    return {"E_T": et, "P_tau_1": tau[0], "modeled_cost": et}, None


def recorded_input_digest(manifest, path, workspace_root):
    """Resolve old workspace-relative hash keys and new absolute keys explicitly."""
    target = Path(path).resolve()
    matches = []
    for name, digest in manifest.get("input_hashes", {}).items():
        recorded = Path(name.replace("\\", "/"))
        resolved = recorded.resolve() if recorded.is_absolute() else (Path(workspace_root) / recorded).resolve()
        if resolved == target:
            matches.append(digest)
    if len(set(matches)) > 1:
        raise AnalysisError("Conflicting recorded input hashes for " + str(target))
    return matches[0] if matches else None


def read_arm(label, runs, evidence_root, enumeration_root, inputs, summary, workspace_root, export_root=None):
    enumeration_root, evidence_root = Path(enumeration_root), Path(evidence_root)
    manifest_path = enumeration_root / "v3_manifest.json"
    manifest = inputs.json(manifest_path, optional=True) or {}
    completed = set(manifest.get("completed", []))
    parity = manifest.get("parity_by_run", {})
    exclusions = {r["run_id"]: r for r in manifest.get("excluded", [])}
    exports = {}
    export_manifest_path = None
    if export_root is not None:
        export_manifest_path = Path(export_root) / "manifest.json"
        export_manifest = inputs.json(export_manifest_path, optional=True) or {}
        exports = {r["run_id"]: r for field in ("computed", "excluded") for r in export_manifest.get(field, [])}
    rows = []
    for seed in SEEDS:
        identity = runs[seed]
        rid = identity["run_id"]
        obs_path = evidence_root / rid / "energy_eval_episodes.csv"
        result_path = enumeration_root / rid.replace("/", "__") / "result.json"
        validation_path = enumeration_root / "validation" / (rid.replace("/", "__") + ".json")
        try:
            obs, obs_issue = observed_metrics(obs_path, identity, inputs, summary)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            obs, obs_issue = None, "INVALID observed evidence: " + str(exc)
        try:
            exact, exact_issue = exact_metrics(result_path, inputs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            exact, exact_issue = None, "INVALID exact evidence: " + str(exc)
        record = exclusions.get(rid, exports.get(rid, {}))
        verdict = parity.get(rid, record.get("parity", record)).get("amended_status", "UNVERIFIED")
        issues = [x for x in (obs_issue, exact_issue) if x]
        validation = inputs.json(validation_path, optional=True)
        if validation is None or validation.get("claim_eligible") is not True:
            issues.append("UNVERIFIED or flagged: original model-validation claim_eligible is not true")
        if record.get("reason"):
            issues.append(record["reason"])
        if obs is not None and rid in completed:
            recorded = recorded_input_digest(manifest, obs_path, workspace_root)
            if recorded != inputs.hashes[str(obs_path.resolve())]:
                issues.append("UNVERIFIED: observed CSV is not bound to the v3 input hash")
        if verdict != "pass":
            issues.append(f"Amended parity verdict: {verdict}")
        if rid not in completed:
            issues.append("UNVERIFIED: run not completed in v3 manifest")
        if manifest.get("dry_run") is not False or manifest.get("acceptance_criterion") != CRITERION or manifest.get("input_hashes_unchanged") is not True or manifest.get("source_hashes_unchanged") is not True:
            issues.append("UNVERIFIED: incomplete or invalid v3 provenance")
        rows.append(dict(arm=label, scientific_identity=ARMS[label], method=identity["method"], seed=seed,
                         run_id=rid, parity_verdict=verdict, eligible=not issues, issues=issues,
                         observed=obs, exact=exact, sources={"observed": str(obs_path.resolve()),
                         "exact": str(result_path.resolve()), "parity": str(manifest_path.resolve()),
                         "model_validation": str(validation_path.resolve()),
                         "export": str(export_manifest_path.resolve()) if export_manifest_path else None}))
    return rows


def m1_estimate(a9, a3, gates):
    if len(a9) != 10 or len(a3) != 10 or not all(r["eligible"] for r in a9 + a3):
        return {"status": "INCOMPLETE", "reason": "All ten eligible paired seeds are required by the unchanged G1 path", "adapter": ADAPTER}
    def key(method):
        return (method, "uniform_v1", "env_reward", "homogeneous_v1", "uniform_v1", "homogeneous_v1")
    wave = {key("refine_er"): {r["seed"]: r["observed"] for r in a9},
            key("lio_er"): {r["seed"]: r["observed"] for r in a3}}
    cost = gates.evaluate({"wave1": wave})["gates"]["G1"]["cost"]
    return dict(status="COMPLETE", adapter=ADAPTER, seeds=cost["seeds"],
                metric="observed 100-episode mean-agent modeled cost, uniform homogeneous",
                sign="(A9-A3)/A3; negative means lower A9 cost",
                relative_median=cost["relative_median"], relative_interval=cost["relative_interval"],
                relative_differences=cost["relative_differences"], estimator_source="sensys_runs/evaluate_gates.py:evaluate -> G1 -> bootstrap_median")


def m2_m3(a9, a8):
    paired = []
    for left, right in zip(a9, a8):
        if left["seed"] != right["seed"]:
            raise AnalysisError("M2 pair seed mismatch")
        paired.append({"seed": left["seed"], "A9": left["exact"], "A8": right["exact"],
                       "eligible_pair": left["eligible"] and right["eligible"]})
    eligible = [p for p in paired if p["eligible_pair"]]
    metrics = {}
    for metric in ("E_T", "P_tau_1", "modeled_cost"):
        left, right = ([p[arm][metric] for p in eligible] for arm in ("A9", "A8"))
        metrics[metric] = dict(n_pairs=len(eligible), A9_lower=sum(x < y for x, y in zip(left, right)),
                               A8_lower=sum(y < x for x, y in zip(left, right)), ties=sum(x == y for x, y in zip(left, right)),
                               A9_range=[min(left), max(left)] if left else None,
                               A8_range=[min(right), max(right)] if right else None,
                               ranges_overlap=max(min(left), min(right)) <= min(max(left), max(right)) if left else None)
    available = [r for r in a9 if r["eligible"]]
    count = sum(r["exact"]["P_tau_1"] >= .95 for r in available)
    complete = len(eligible) == 10 and len(available) == 10
    et = metrics["E_T"]
    if not complete:
        wording = "INCOMPLETE: no whole-ten-seed outcome claim; missing or failed runs remain reported."
    elif count >= 9 and et["ranges_overlap"]:
        wording = "energy-reward LIO reaches the same one-step completion as REFiNE under the uniform profile"
    elif et["A8_lower"] == 10 and not et["ranges_overlap"]:
        wording = "REFiNE's incentive-learning path contributes beyond the shared actor update"
    elif et["A9_lower"] == 10 and not et["ranges_overlap"]:
        wording = "energy-reward LIO has lower exact E[T] than REFiNE in 10 of 10 paired seeds, with non-overlapping ranges"
    else:
        wording = "no consistent difference across 10 paired seeds"
    return ({"status": "COMPLETE" if complete else "INCOMPLETE", "pairs": paired, "metrics": metrics,
             "interpretation_metric": "E_T; lower P_tau_1 is reported descriptively and is not an improvement"},
            {"status": "COMPLETE" if len(available) == 10 else "INCOMPLETE", "threshold": .95,
             "passing_seeds": count, "eligible_seeds": len(available), "registered_seeds": 10}, wording)


def analyze(args):
    inputs = Inputs()
    out = Path(args.output_dir).resolve()
    if not re.fullmatch(r"\d{8}", args.date) or args.date not in out.name or out.exists():
        raise AnalysisError("Output must be a fresh directory whose name contains the YYYYMMDD date")
    protected = [REPO, *(Path(getattr(args, k)).resolve() for k in (
        "a9_evaluation_root", "a9_enumeration_root", "reference_evaluation_root", "reference_enumeration_root"))]
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        raise AnalysisError("Output directory overlaps the repository or an input root")
    gates, summary = pinned_modules(args.source_manifest, args.workspace_root, inputs, args.protected_manifest)
    new = inventory(args.a9_inventory, ("A9",), inputs)
    refs = inventory(args.reference_inventory, ("A3", "A8"), inputs)
    a9 = read_arm("A9", new["A9"], args.a9_evaluation_root, args.a9_enumeration_root, inputs, summary, args.workspace_root, args.a9_export_root)
    a3 = read_arm("A3", refs["A3"], args.reference_evaluation_root, args.reference_enumeration_root, inputs, summary, args.workspace_root)
    a8 = read_arm("A8", refs["A8"], args.reference_evaluation_root, args.reference_enumeration_root, inputs, summary, args.workspace_root)
    m1 = m1_estimate(a9, a3, gates)
    m2, m3, wording = m2_m3(a9, a8)
    report = dict(date=args.date, M1=m1, M2=m2, M3=m3, paper_wording=wording, runs=a9+a3+a8,
                  cost_identity="Exact uniform homogeneous mean-agent modeled cost equals E[T]; team cost is four times this value.",
                  failure_rule="Retain every registered seed and failure; no replacements. Incomplete inputs do not support a whole-ten-seed claim.")
    inputs.verify()
    out.mkdir(parents=True, exist_ok=False)
    stem = "REFiNE_ICCPS2027_A9_Results_" + args.date
    (out / (stem + ".json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    input_record = dict(input_sha256=inputs.hashes, missing_inputs=sorted(inputs.missing),
                        inputs_unchanged=True, command=sys.argv, adapter=ADAPTER)
    (out / "input_manifest.json").write_text(json.dumps(input_record, indent=2)+"\n", encoding="utf-8")
    with (out / "per_run.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ("arm", "method", "seed", "run_id", "E_T", "P_tau_1", "modeled_cost", "observed_mean_agent_cost", "parity_verdict", "eligible", "issues", "sources")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow({**{k: row[k] for k in ("arm", "method", "seed", "run_id", "parity_verdict", "eligible")},
                             **(row["exact"] or {}), "observed_mean_agent_cost": (row["observed"] or {}).get("mean_agent_mission_cost"),
                             "issues": "; ".join(row["issues"]), "sources": json.dumps(row["sources"], sort_keys=True)})
    lines = ["# Registered A9 results, " + args.date, "", "Inputs and source hashes: input_manifest.json. Per-run values and evidence paths: per_run.csv.", "", ADAPTER, "",
             "M1: " + json.dumps(m1, sort_keys=True), "Source: unchanged evaluate_gates.evaluate G1 path and per_run.csv observed values.", "",
             "M2: " + json.dumps({k:v for k,v in m2.items() if k != "pairs"}, sort_keys=True),
             "Source: per_run.csv exact values and M2.pairs in the companion JSON.", "",
             "M3: " + json.dumps(m3, sort_keys=True), "Source: per_run.csv eligible A9 P_tau_1 >= 0.95.", "",
             "Predeclared paper wording: " + wording, "Source: M2 E_T paired directions/ranges and M3 threshold count.", "",
             report["cost_identity"], "Source: uniform scoring identity in REFiNE_ICCPS2027_Results_Supplement_R1_20261019.md:30-47.", "",
             report["failure_rule"], "Source: A9 registration and this report's per_run.csv eligibility/issues.", ""]
    (out / (stem + ".md")).write_text("\n".join(lines), encoding="utf-8")
    inputs.verify()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a9-inventory", "a9-export-root", "a9-evaluation-root", "a9-enumeration-root", "reference-inventory", "reference-evaluation-root", "reference-enumeration-root", "source-manifest", "protected-manifest", "workspace-root", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "ANALYSIS_ERROR: " + str(exc) + "\n")
    print("Wrote " + str(args.output_dir) + "; M1="+report["M1"]["status"]+", M2="+report["M2"]["status"]+", M3="+report["M3"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
