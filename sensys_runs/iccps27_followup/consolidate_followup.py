"""Write one fresh consolidated paper-evidence Markdown from completed ER analysis.

No training, probing, enumeration, estimator rerun, or historical-file change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sensys_runs.iccps27_followup import sampling_diagnostics as sampling

WEIGHTS = ("2.0,0.2", "1.1,0.9", "1.5,1.5")
SEEDS = set(range(12341, 12351))
METHODS = ("lio_eia_er", "refine_eia_er")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_inputs(manifest, additional):
    require(manifest.get("missing_inputs") == [] and manifest.get("inputs_unchanged") is True
            and manifest.get("source_manifests_checked_before_and_after") is True,
            "Analysis input manifest is incomplete or unverified")
    pins = manifest.get("input_sha256")
    require(isinstance(pins, dict) and pins, "Analysis input hashes are empty")
    for name, expected in {**pins, **additional}.items():
        path = Path(name)
        require(path.is_absolute() and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "Invalid absolute input path or SHA256")
        require(path.is_file() and sha(path) == expected, "Input hash mismatch: " + name)


def validate_rows(rows, expected, label):
    require(len(rows) == len(expected) and len({r["run_id"] for r in rows}) == len(expected),
            label + " run count or run identities are incomplete")
    keys = [(r["method"], r.get("weight"), r["seed"]) for r in rows]
    require(len(set(keys)) == len(expected) and set(keys) == expected, label + " matrix is incomplete or duplicated")
    for row in rows:
        require(row.get("observed_eligible") is True and row.get("probe_eligible") is True
                and not row.get("issues") and not row.get("probe_issues"),
                "Genuine observed/probe problem: " + row["run_id"])
        obs, probe = row["observed"], row["probe"]
        require(obs.get("n_episodes") == 100 and obs.get("n_agents") == 4
                and all(isinstance(obs.get(k), (int, float)) and not isinstance(obs.get(k), bool)
                        and math.isfinite(obs[k]) for k in
                        ("mean_agent_mission_cost", "mean_mission_steps", "one_step_fraction", "success_rate"))
                and obs["mean_agent_mission_cost"] >= 0 and 0 <= obs["success_rate"] <= 1,
                "Invalid observed 100-episode metrics: " + row["run_id"])
        diagnostic = sampling.diagnose(probe["P1"], obs["mean_mission_steps"], obs["one_step_fraction"], obs["n_episodes"])
        if label == "A9":
            require(isinstance(row.get("exact", {}).get("E_T"), (int, float))
                    and math.isfinite(row["exact"]["E_T"]) and 1 <= row["exact"]["E_T"] <= 5,
                    "A9 exact completion time is missing or invalid")
        if label != "A9":
            lo, hi = diagnostic["population_E_T_bounds"]
            require(probe.get("E_T_lower") == lo and probe.get("E_T_upper") == hi
                    and probe.get("E_T_bound_width") == hi-lo
                    and probe.get("mission_time_primary") == ("observed_100_episode_mean" if hi-lo > .1 else "reset_based_interval"),
                    "Inconsistent reset-bound or mission-time description: " + row["run_id"])
    return rows


def bound_report(section, pins):
    path = Path(section["source_report"]).resolve()
    require(pins.get(str(path)) == section.get("source_report_sha256") == sha(path),
            "Historical source report is not bound to the analysis inputs")
    return path, read_json(path)


def check_report(report, pins):
    require(report.get("status") == "COMPLETE" and report.get("declared_run_count") == 30
            and report.get("arm") == "er_lio_eia_er" and report.get("base_method") == "lio_eia_er",
            "Consolidation requires a COMPLETE registered 30-run ER analysis")
    new = validate_rows(report["runs"], {("lio_eia_er", w, s) for w in WEIGHTS for s in SEEDS}, "ER-LIO+EIA")
    historical = report["historical_eia"]
    old = validate_rows(historical["runs"], {(m, w, s) for m in METHODS for w in WEIGHTS for s in SEEDS}, "historical EIA")
    reference = report["a9_reference"]
    a9 = validate_rows(reference["runs"], {("lio_er", None, s) for s in SEEDS}, "A9")
    require(len({r["run_id"] for r in new+old+a9}) == 100, "Run identity overlaps among scientific arms")
    old_path, old_report = bound_report(historical, pins)
    a9_path, a9_report = bound_report(reference, pins)
    require(historical["historical_results"] == old_report["results"], "Historical E2/E3 were changed")
    require(reference["historical_results"] == {k: a9_report[k] for k in ("M1", "M2", "M3", "paper_wording")},
            "Historical A9 results were changed")
    require(all(a9_report[k]["status"] == "COMPLETE" for k in ("M1", "M2", "M3")), "A9 comparison is incomplete")
    require(set(report["results"]) == set(historical["validated_E1"]) == set(WEIGHTS), "Weight summaries are incomplete")
    for weight in WEIGHTS:
        cell = report["results"][weight]
        require(set(cell["E1"]) == {"er_lio_eia_er", "A9", *METHODS}
                and set(historical["validated_E1"][weight]) == set(METHODS), "E1 method cells are incomplete or extra")
        require(cell["E2"]["status"] == "COMPLETE" and all(x["status"] == "COMPLETE" for x in cell["E3"].values())
                and cell["paper_wording"].get("rule") in (1, 2, 3, 4), "ER comparison summary is incomplete")
        require(old_report["results"][weight]["E2"]["status"] == "COMPLETE"
                and all(x["status"] == "COMPLETE" for x in old_report["results"][weight]["E3"].values()),
                "Historical observed E2/E3 are incomplete")
        for arm, rows in [("er_lio_eia_er", [r for r in new if r["weight"] == weight]),
                          *((m, [r for r in old if r["weight"] == weight and r["method"] == m]) for m in METHODS),
                          ("A9", a9)]:
            e1 = cell["E1"][arm]
            require(e1["status"] == "COMPLETE" and e1["registered_seeds"] == e1["reset_eligible_seeds"] == 10
                    and e1["reset_threshold"] == .95 and e1["reset_passing_seeds"] == sum(r["probe"]["P1"] >= .95 for r in rows),
                    "Validated E1 count differs from bound probe records")
            if arm in METHODS:
                require(e1 == historical["validated_E1"][weight][arm], "Historical validated E1 differs across report sections")
    return new, old, a9, old_path, a9_path


def citation(path, marker=None):
    text = Path(path).read_text(encoding="utf-8-sig")
    position = 0 if marker is None else text.find(marker)
    require(position >= 0, "Citation marker missing in source: " + str(path))
    return str(Path(path).resolve()) + ":" + str(text.count("\n", 0, position)+1)


def compact(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def diagnostic_rows(rows):
    return [(row, sampling.diagnose(row["probe"]["P1"], row["observed"]["mean_mission_steps"],
                                  row["observed"]["one_step_fraction"], row["observed"]["n_episodes"])) for row in rows]


def render(report, report_path, manifest_path, date, pins):
    new, old, a9, old_path, a9_path = check_report(report, pins)
    src = citation(report_path, '"status"')
    lines = ["# Consolidated REFiNE follow-up results, " + date, "",
        "Complete evidence covers A9 (10 runs), the original EIA matrix (60 runs), and ER-LIO+EIA (30 runs). Source: " + src + ".", "",
        "All historical inputs were hash-checked before rendering. Analysis input manifest SHA256: " + sha(manifest_path)
        + ". Source: " + citation(manifest_path, '"input_sha256"') + ".", "",
        "ER analysis SHA256: " + sha(report_path) + ". Source: " + src + ".", "",
        "## A9", ""]
    result = report["a9_reference"]["historical_results"]
    m1, m2, m3 = (result[k] for k in ("M1", "M2", "M3"))
    a9_rule = "Registered wording rule 1" if m3["passing_seeds"] >= 9 and m2["metrics"]["E_T"]["ranges_overlap"] else "Registered wording"
    lines += ["M1 stored relative median " + repr(m1["relative_median"]) + ", 95% percentile-bootstrap interval "
        + compact(m1["relative_interval"]) + "; A9 lower in " + str(sum(x < 0 for x in m1["relative_differences"]))
        + "/10 paired seeds. Negative means lower A9 cost; no positive-reduction conversion is used. Source: " + citation(a9_path, '"M1"') + ".", "",
        "M2 exact E[T] paired direction and ranges: " + compact(m2["metrics"]["E_T"]) + ". Source: " + citation(a9_path, '"M2"') + ".", "",
        "M3: " + str(m3["passing_seeds"]) + "/" + str(m3["registered_seeds"]) + " at P1 >= 0.95. " + a9_rule + ": "
        + result["paper_wording"] + ". Source: " + citation(a9_path, '"M3"') + ".", "",
        "## Original 60-run EIA matrix", "",
        "E1 below is recovered from the validated probe bindings. Historical E2/E3 numerical objects are retained unchanged. Sources: "
        + citation(report_path, '"validated_E1"') + "; " + citation(old_path, '"results"') + ".", ""]
    for weight in WEIGHTS:
        old_cell = report["historical_eia"]["historical_results"][weight]
        lines += ["Weight " + weight + ": validated E1 " + compact(report["historical_eia"]["validated_E1"][weight])
                  + ". Source: " + citation(report_path, '"validated_E1"') + ".", "",
                  "Stored E2: " + compact(old_cell["E2"]) + ". Source: " + citation(old_path, json.dumps(weight)) + ".", ""]
        for name, value in old_cell["E3"].items():
            lines += ["Stored E3 " + name + ": " + compact(value) + ". Source: " + citation(old_path, json.dumps(weight)) + ".", ""]
    lines += ["## ER-LIO+EIA", ""]
    for weight, cell in report["results"].items():
        weight_src = citation(report_path, json.dumps(weight))
        lines += ["Weight " + weight + ": E1 " + compact(cell["E1"]) + ". Source: " + weight_src + ".", "",
                  "Primary E2 versus REFiNE+EIA: " + compact(cell["E2"]) + ". Source: " + weight_src + ".", "",
                  "Registered wording: " + compact(cell["paper_wording"]) + ". Source: " + weight_src + ".", ""]
        for name, value in cell["E3"].items():
            lines += ["Descriptive E3 " + name + ": " + compact(value) + ". Source: " + weight_src + ".", ""]
    rows = diagnostic_rows(a9+old+new)
    outside = [(r, d) for r, d in rows if d["population_bound_side"] != "inside"]
    diagnostic_src = str(Path(sampling.__file__).resolve()) + ":" + str(sampling.diagnose.__code__.co_firstlineno)
    lines += ["## Duration descriptions and sampling diagnostics", "",
        report["mission_time_rule"] + " Source: " + citation(report_path, '"mission_time_rule"') + ".", "",
        "Applied the registered separate sampling diagnostic to all 100 runs; " + str(len(outside))
        + " observed 100-episode means fall outside their population reset bounds. Sources: " + citation(report_path, '"runs"')
        + "; " + diagnostic_src + ".", "",
        "The central binomial count interval and conservative mean envelope are pointwise 95% prediction diagnostics. They are not confidence intervals for E[T], statistical equivalence claims, or simultaneous guarantees. Observations and eligibility are unchanged. Source: "
        + diagnostic_src + ".", ""]
    for row, diagnostic in outside:
        row_src = citation(report_path, json.dumps(row["run_id"]))
        lines += ["- " + row["run_id"] + ": " + compact(diagnostic) + ". Sources: " + row_src + "; " + diagnostic_src + "."]
    lines += ["", "## Per-run duration evidence", "",
        "A9 retains its exact completion time as primary. Each EIA row explicitly names either its reset-based interval or observed 100-episode mean as primary; all unadjusted quantities remain visible. Sources: "
        + citation(report_path, '"mission_time_rule"') + "; " + citation(a9_path, '"M2"') + ".", ""]
    a9_ids = {row["run_id"] for row in a9}
    for row, diagnostic in rows:
        is_a9 = row["run_id"] in a9_ids
        value = dict(P1=row["probe"]["P1"], E_T_bounds=diagnostic["population_E_T_bounds"],
                     observed_mean_steps=row["observed"]["mean_mission_steps"],
                     mission_time_primary="exact_A9_full_enumeration" if is_a9 else row["probe"]["mission_time_primary"])
        if is_a9:
            value["exact_E_T"] = row["exact"]["E_T"]
        lines += ["- " + row["run_id"] + ": " + compact(value) + ". Source: "
                  + citation(report_path, json.dumps(row["run_id"])) + "."]
    lines += ["", "All historical values above are copied from their bound reports; no G1 estimator is rerun by this consolidator. Sources: "
              + citation(old_path, '"results"') + "; " + citation(a9_path, '"M1"') + "; " + src + ".", ""]
    text = "\n".join(lines)
    require("\u2014" not in text, "English report must not contain em dashes")
    return text


def consolidate(args):
    output = Path(args.output_file).resolve()
    require(re.fullmatch(r"\d{8}", args.date) is not None and args.date in output.name and output.suffix == ".md"
            and not output.exists() and not output.is_symlink() and output.parent.is_dir(), "Output must be a fresh dated Markdown file")
    manifest_path, report_path = Path(args.input_manifest).resolve(), Path(args.er_report).resolve()
    require(not output.is_relative_to(REPO) and output not in (manifest_path, report_path), "Output overlaps repository or inputs")
    manifest, report = read_json(manifest_path), read_json(report_path)
    evidence_roots = [report_path.parent,
        Path(report["historical_eia"]["source_report"]).resolve().parent,
        Path(report["a9_reference"]["source_report"]).resolve().parent]
    evidence_roots.extend(Path(r["sources"]["run"]).resolve() for section in
        (report["runs"], report["historical_eia"]["runs"], report["a9_reference"]["runs"])
        for r in section if r.get("sources", {}).get("run"))
    require(not any(output.is_relative_to(p) for p in evidence_roots)
            and str(output) not in manifest.get("input_sha256", {}),
            "Output must be outside input evidence directories")
    additional = {str(manifest_path): sha(manifest_path), str(report_path): sha(report_path),
                  str(Path(__file__).resolve()): sha(__file__), str(Path(sampling.__file__).resolve()): sha(sampling.__file__)}
    verify_inputs(manifest, additional)
    text = render(report, report_path, manifest_path, args.date, manifest["input_sha256"])
    verify_inputs(manifest, additional)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    verify_inputs(manifest, additional)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("er-report", "input-manifest", "output-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    try:
        path = consolidate(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "CONSOLIDATION_ERROR: " + str(exc) + "\n")
    print("CONSOLIDATED_COMPLETE " + str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
