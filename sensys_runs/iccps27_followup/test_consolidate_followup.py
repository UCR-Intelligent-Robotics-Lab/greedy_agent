"""Synthetic-only consolidation tests; no new scientific output is generated."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from sensys_runs.iccps27_followup import consolidate_followup as c


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def row(method, weight, seed, arm):
    p1 = .99
    lower, upper = 2-p1, 1+4*(1-p1)
    return dict(run_id=arm + "/" + (weight or "uniform") + "_" + str(seed), method=method,
        weight=weight, seed=seed, observed_eligible=True, probe_eligible=True, issues=[], probe_issues=[],
        observed=dict(n_episodes=100, n_agents=4, mean_agent_mission_cost=1.0,
                      mean_mission_steps=1.0, one_step_fraction=1.0, success_rate=1.0),
        probe=dict(P1=p1, E_T_lower=lower, E_T_upper=upper, E_T_bound_width=upper-lower,
                   mission_time_primary="reset_based_interval"))


def e1():
    return dict(status="COMPLETE", registered_seeds=10, reset_eligible_seeds=10,
                reset_threshold=.95, reset_passing_seeds=10)


@pytest.fixture
def fixture(tmp_path):
    old_path = tmp_path/"historical"/"old.json"
    a9_path = tmp_path/"a9"/"old_a9.json"
    old_results = {w: dict(E2=dict(status="COMPLETE", relative_median=-.1, relative_interval=[-.2, -.01]),
                          E3={"old_comparison": dict(status="COMPLETE", pairs=[], metrics={})}) for w in c.WEIGHTS}
    old_report = dict(results=old_results)
    write(old_path, old_report)
    a9_report = dict(M1=dict(status="COMPLETE", relative_median=-.1175, relative_interval=[-.2161,-.0648],
                            relative_differences=[-.1]*10),
                     M2=dict(status="COMPLETE", metrics={"E_T": dict(A9_lower=10,A8_lower=0,ties=0,
                            A9_range=[1.0077,1.016],A8_range=[1.0087,1.0168],ranges_overlap=True)}),
                     M3=dict(status="COMPLETE", passing_seeds=10,registered_seeds=10),
                     paper_wording="energy-reward LIO reaches the same one-step completion as REFiNE under the uniform profile")
    write(a9_path, a9_report)
    new_rows = [row("lio_eia_er",w,s,"synthetic_er") for w in c.WEIGHTS for s in sorted(c.SEEDS)]
    old_rows = [row(m,w,s,"synthetic_"+m) for m in c.METHODS for w in c.WEIGHTS for s in sorted(c.SEEDS)]
    a9_rows = [row("lio_er",None,s,"synthetic_a9") for s in sorted(c.SEEDS)]
    for item in a9_rows:
        item["exact"] = {"E_T": 1.02}
    validated = {w: {m:e1() for m in c.METHODS} for w in c.WEIGHTS}
    results = {w: dict(E1={"er_lio_eia_er":e1(),"A9":e1(),**deepcopy(validated[w])},
                      E2=dict(status="COMPLETE", relative_median=0.0, relative_interval=[0.0,0.0]),
                      E3={"new_comparison":dict(status="COMPLETE",pairs=[],metrics={})},
                      paper_wording=dict(rule=1,text="SYNTHETIC WORDING ONLY")) for w in c.WEIGHTS}
    report = dict(status="COMPLETE", declared_run_count=30, arm="er_lio_eia_er", base_method="lio_eia_er",
        runs=new_rows, results=results,
        historical_eia=dict(runs=old_rows, validated_E1=validated, historical_results=old_results,
            source_report=str(old_path),source_report_sha256=c.sha(old_path)),
        a9_reference=dict(runs=a9_rows,historical_results=a9_report,source_report=str(a9_path),source_report_sha256=c.sha(a9_path)),
        mission_time_rule="Width > 0.1 uses observed 100-episode mean steps as primary description.")
    report_path=tmp_path/"new_analysis"/"er.json"
    write(report_path,report)
    evidence=tmp_path/"evidence"/"evidence.txt"
    evidence.parent.mkdir()
    evidence.write_bytes(b"SYNTHETIC EVIDENCE")
    manifest=dict(input_sha256={str(p):c.sha(p) for p in (old_path,a9_path,evidence)},
                  missing_inputs=[],inputs_unchanged=True,source_manifests_checked_before_and_after=True)
    manifest_path=tmp_path/"new_analysis"/"input_manifest.json"
    write(manifest_path,manifest)
    args=SimpleNamespace(er_report=report_path,input_manifest=manifest_path,date="20261029",
                         output_file=tmp_path/"Consolidated_SYNTHETIC_20261029.md")
    return args,report,manifest


def test_complete_synthetic_consolidation_preserves_all_inputs(fixture):
    args,report,manifest=fixture
    before={name:c.sha(name) for name in manifest["input_sha256"]}
    report_hash=c.sha(args.er_report)
    output=c.consolidate(args)
    text=output.read_text()
    assert "A9 lower in 10/10 paired seeds" in text
    assert '"A9_lower":10' in text and '"ranges_overlap":true' in text
    assert "Original 60-run EIA matrix" in text and "ER-LIO+EIA" in text
    assert "Applied the registered separate sampling diagnostic to all 100 runs; 100" in text
    assert '"observed_values_adjusted":false' in text
    assert "\u2014" not in text
    assert c.sha(args.er_report)==report_hash
    assert {name:c.sha(name) for name in before}==before
    assert len(list(output.parent.glob("Consolidated*.md")))==1


@pytest.mark.parametrize("fault", ["incomplete", "duplicate", "missing", "issue", "probe_issue", "probe_flag", "bounds", "old_change", "a9_change", "e1"])
def test_incomplete_or_genuine_scientific_problem_stops_without_output(fixture,fault):
    args,report,_=fixture
    if fault=="incomplete": report["status"]="INCOMPLETE"
    elif fault=="duplicate": report["runs"][1]=deepcopy(report["runs"][0])
    elif fault=="missing": report["historical_eia"]["runs"].pop()
    elif fault=="issue": report["runs"][0]["issues"].append("genuine problem")
    elif fault=="probe_issue": report["runs"][0]["probe_issues"].append("genuine problem")
    elif fault=="probe_flag": report["runs"][0]["probe_eligible"]=False
    elif fault=="bounds": report["runs"][0]["probe"]["E_T_upper"]=2.0
    elif fault=="old_change": report["historical_eia"]["historical_results"]["2.0,0.2"]["E2"]["relative_median"]=-.2
    elif fault=="a9_change": report["a9_reference"]["historical_results"]["M1"]["relative_median"]=-.2
    else: report["results"]["2.0,0.2"]["E1"]["er_lio_eia_er"]["reset_passing_seeds"]=9
    write(args.er_report,report)
    with pytest.raises(ValueError): c.consolidate(args)
    assert not args.output_file.exists()


def test_input_content_tamper_stops_without_output(fixture):
    args,_,manifest=fixture
    evidence=next(p for p in manifest["input_sha256"] if p.endswith("evidence.txt"))
    from pathlib import Path
    Path(evidence).write_bytes(b"GENUINE CHANGE")
    with pytest.raises(ValueError,match="Input hash mismatch"): c.consolidate(args)
    assert not args.output_file.exists()


@pytest.mark.parametrize("fault", ["missing_inputs","empty_hashes","not_checked"])
def test_invalid_manifest_stops(fixture,fault):
    args,_,manifest=fixture
    if fault=="missing_inputs":manifest["missing_inputs"]=["missing"]
    elif fault=="empty_hashes":manifest["input_sha256"]={}
    else:manifest["inputs_unchanged"]=False
    write(args.input_manifest,manifest)
    with pytest.raises(ValueError):c.consolidate(args)
    assert not args.output_file.exists()


def test_outside_population_bounds_sampling_compatible_and_incompatible_are_reported_unadjusted(fixture):
    args,report,_=fixture
    r=report["runs"][0]
    p1=.999
    r["probe"].update(P1=p1,E_T_lower=2-p1,E_T_upper=1+4*(1-p1),
                      E_T_bound_width=(1+4*(1-p1))-(2-p1))
    r["observed"].update(mean_agent_mission_cost=1.1,mean_mission_steps=1.1,one_step_fraction=.9)
    write(args.er_report,report)
    original=c.sha(args.er_report)
    text=c.consolidate(args).read_text()
    assert '"mean_inside_prediction_envelope":false' in text
    assert '"mean_inside_prediction_envelope":true' in text
    assert '"observed_mean_steps":1.1' in text
    assert "Not accounted for by the conservative pointwise mean envelope" in text
    assert c.sha(args.er_report)==original


def test_output_never_overwrites_or_enters_evidence(fixture):
    args,_,_=fixture
    args.output_file.write_bytes(b"UNCHANGED")
    with pytest.raises(ValueError,match="fresh dated"):c.consolidate(args)
    assert args.output_file.read_bytes()==b"UNCHANGED"
    args.output_file=args.er_report.parent/"Bad_20261029.md"
    with pytest.raises(ValueError,match="outside input evidence"):c.consolidate(args)
    assert not args.output_file.exists()


def test_wide_interval_with_mean_inside_bounds_still_reports_observed_primary(fixture):
    args,report,_=fixture
    record=report["runs"][0]
    p1=.94
    lo,hi=2-p1,1+4*(1-p1)
    record["probe"].update(P1=p1,E_T_lower=lo,E_T_upper=hi,E_T_bound_width=hi-lo,
                           mission_time_primary="observed_100_episode_mean")
    record["observed"].update(mean_agent_mission_cost=1.1,mean_mission_steps=1.1,one_step_fraction=.94)
    report["results"][record["weight"]]["E1"]["er_lio_eia_er"]["reset_passing_seeds"]=9
    write(args.er_report,report)
    text=c.consolidate(args).read_text()
    lines=[line for line in text.splitlines() if line.startswith("- "+record["run_id"]+":")]
    assert len(lines)==1
    assert '"observed_mean_steps":1.1' in lines[0]
    assert '"mission_time_primary":"observed_100_episode_mean"' in lines[0]
    assert '"mission_time_primary":"exact_A9_full_enumeration"' in text
    assert '"exact_E_T":1.02' in text
