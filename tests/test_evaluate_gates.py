"""Synthetic gate fixtures only; fix gate logic, never relax registered boundaries."""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensys_runs import evaluate_gates as g


def arm(method="refine_er", profile="uniform_v1", hardware=g.HOMO,
        evaluation=None, table=None, objective="env_reward", cost=95, success=1, share=0.7):
    """Ten seed-level records, never synthetic episodes masquerading as seeds."""
    key = (method, profile, objective, hardware, table or profile, evaluation or hardware)
    rows = {seed: {g.COST: str(cost), g.WORST: str(cost * 2), "success_rate": str(success),
                   "support_cost_component": str(cost / 2), g.SUPPORT: str(share),
                   "support_share_status": "defined"} for seed in g.setting("seeds")}
    return {key: rows}


def primary_waves():
    return {"wave1": {**arm(), **arm("lio_er", cost=100)},
            "wave2": {**arm(profile="paper_ratio_v1", hardware=g.HETERO, share=0.7),
                      **arm(hardware=g.HETERO, table="paper_ratio_v1", share=0.5)}}


def write_summary(path, records):
    path.mkdir()
    fields = (*g.IDENTITY, "seed", *g.METRICS, "support_share_status")
    target = path / "seed_metrics.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, seeds in records.items():
            for seed, values in seeds.items():
                writer.writerow({**dict(zip(g.IDENTITY, key)), "seed": seed, **values})
    return target


def test_exact_wilcoxon_known_cases():
    """Enumerated tails must stay exact even with ties or all-zero differences."""
    result = g.wilcoxon(-np.arange(1, 11), "less")
    assert result["p"] == 1 / 1024 and result["sign_assignments"] == 1024
    assert g.wilcoxon([-3, -2, -1, 1, 2, 3], "less")["p"] >= 0.5
    tied = g.wilcoxon([-1, -1, 2, 0], "less")
    assert tied["p"] == 5 / 8
    assert tied["zero_count"] == 1 and tied["positive_rank_sum"] == 3
    assert g.wilcoxon([0, 0], "greater")["p"] == 1
    assert g.wilcoxon([1, 1, 0], "greater")["p"] == 1 / 4


def test_holm_known_triple_and_missing_slots():
    """Missing waves must not silently shrink the pre-registered primary family."""
    assert g.holm({"G1": 0.01, "G3a": 0.04, "G3b": 0.03}) == {
        "G1": 0.03, "G3a": 0.06, "G3b": 0.06}
    assert g.holm({"G1": 0.01}) == {"G1": 0.03, "G3a": None, "G3b": None}


def test_bootstrap_fixed_generator_and_improvement_denominator():
    """Zeros stay in the improvement denominator and bootstrap sample."""
    d = np.arange(-5, 5)
    np.random.seed(7)
    first = g.bootstrap_median(d)
    np.random.seed(99)
    assert first == g.bootstrap_median(d)
    rng = np.random.default_rng(20261010)
    expected = np.quantile(np.median(rng.choice(d, (10000, 10)), axis=1), [0.025, 0.975])
    assert first == expected.tolist()
    report = g.statistics([-1, 0, 1], "less", [1, 2, 3])
    assert report["improved_seed_count"] == 1 and report["probability_of_improvement"] == 1 / 3


@pytest.mark.parametrize("cost,success,status", [(95, 0.95, "PASS"), (95.001, 0.95, "NOT_DETECTED"),
                                                  (95, 0.94999, "NOT_DETECTED"), (94.99, 0.951, "PASS")])
def test_g1_inclusive_effect_and_success_boundaries(cost, success, status):
    """Inclusive decimal boundaries must pass exactly, not via a tolerance."""
    waves = primary_waves()
    waves["wave1"] = {**arm(cost=cost, success=success), **arm("lio_er", cost=100)}
    result = g.evaluate(waves)["gates"]["G1"]
    assert result["status"] == status
    if cost == 95 and success == 0.95:
        assert result["cost"]["relative_median"] == -0.05
        assert result["success"]["interval"] == [-0.05, -0.05]


def test_p_boundary_is_strict():
    """An adjusted p equal to alpha does not pass."""
    assert not g.g1_conditions({"relative_median": -0.05}, {"interval": [-0.05, 0]}, 0.05)["test"]


@pytest.mark.parametrize("excluded,inconclusive", [(2, False), (3, True)])
def test_g3_exclusion_and_pairwise_intersection(excluded, inconclusive):
    """Exclusions are per arm before intersection; G3 must expose excluded seeds."""
    waves = primary_waves()
    a, b = list(waves["wave2"].values())
    for seed in g.setting("seeds")[:excluded]:
        a[seed].update(support_share="", support_share_status="no lever actions")
    b[g.setting("seeds")[-1]].update(support_share="", support_share_status="no lever actions")
    report = g.evaluate(waves)
    for name in ("G3a", "G3b"):
        assert (report["gates"][name]["status"] == "INCONCLUSIVE") is inconclusive
    if not inconclusive:
        assert report["gates"]["G3a"]["n_seeds"] == 8
        assert report["gates"]["G3b"]["n_seeds"] == 7


@pytest.mark.parametrize("share,expected", [(0.65, "PASS"), (0.64999, "NOT_DETECTED")])
def test_g3_share_effect_boundary(share, expected):
    """A significant directional effect alone cannot pass G3a."""
    waves = primary_waves()
    waves["wave2"].update(arm(profile="paper_ratio_v1", hardware=g.HETERO, share=share))
    assert g.evaluate(waves)["gates"]["G3a"]["status"] == expected


@pytest.mark.parametrize("statuses,text", [
    (("PASS", "PASS", "PASS"), "main line stands"),
    (("PASS", "NOT_DETECTED", "PASS"), "Narrow the claim"),
    (("NOT_DETECTED", "PASS", "PASS"), "re-scoping"),
    (("PASS", "INCONCLUSIVE", "INCONCLUSIVE"), "no claim"),
    (("PASS", "NOT_EVALUATED", "NOT_EVALUATED"), "NOT_EVALUATED")])
def test_october_decisions(statuses, text):
    """Decision wording must distinguish absent, inconclusive and undetected effects."""
    assert text in g.decision(*statuses)


def test_absent_waves():
    """No input must never produce a gate failure or a positive claim."""
    report = g.evaluate({})
    assert {r["status"] for r in report["gates"].values()} == {"NOT_EVALUATED"}


def test_mismatched_seed_sets_refused():
    """Never silently intersect missing training seeds; only no-lever exclusions are allowed."""
    waves = primary_waves()
    next(iter(waves["wave1"].values())).pop(g.setting("seeds")[0])
    with pytest.raises(g.GateError, match="different seed sets"):
        g.evaluate(waves)


def test_sanity_reference_uses_fixed_groups_via_rescoring():
    """Homogeneous training needs heterogeneous evaluation shares, not all-agent share 1."""
    waves = primary_waves()
    waves["wave1"].update(arm("lio_er", evaluation=g.HETERO, share=0.7))
    waves["wave1"].update(arm(evaluation=g.HETERO, share=0.5))
    sanity = g.evaluate(waves)["sanity_reference"]
    assert sanity["lio_er"]["agent_index_asymmetry"] is True
    assert sanity["refine_er"]["agent_index_asymmetry"] is False


def test_g2_and_g4_reference_matching_and_unspecified_direction():
    """G4 uses P5 accounting-only LIO matching but never pools hardware or objectives."""
    waves = primary_waves()
    waves["wave1"].update(arm(profile="paper_ratio_v1"))
    waves["wave1"].update(arm(table="paper_ratio_v1", cost=100))
    waves["wave1"].update(arm("lio_er", table="paper_ratio_v1", cost=100))
    waves["wave1"].update(arm("lio_er", table="paper_ratio_v1", evaluation=g.HETERO, cost=100))
    waves["wave3"] = {**arm("lio_er", "paper_ratio_v1", objective="energy_reward"),
                       **arm("lio_er", "paper_ratio_v1", hardware=g.HETERO, objective="energy_reward")}
    report = g.evaluate(waves)
    assert report["gates"]["G2"]["p"] == 1 / 1024
    g4 = report["gates"]["G4"]
    assert g4["status"] == "REPORTED" and "Not pre-registered" in g4["direction"]
    for metrics in g4["comparisons"].values():
        assert metrics[g.COST]["less"]["status"] == "REPORTED"
        assert metrics[g.COST]["greater"]["status"] == "REPORTED"


def test_input_hashes_fresh_output_and_actual_registry_commit(tmp_path):
    """End-to-end reports must trace every consumed synthetic CSV and refuse overwrite."""
    paths = []
    argv = []
    for wave, records in primary_waves().items():
        directory = tmp_path / wave
        paths.append(write_summary(directory, records))
        argv += ["--" + wave, str(directory)]
    out = tmp_path / "report"
    assert g.main([*argv, "--out", str(out)]) == 0
    raw = (out / "gates_report.json").read_bytes()
    report = json.loads(raw)
    assert report["registration"]["commit"] == "4755ea5bd00c10ff937fde8b5efc3648d55dc497"
    assert report["input_sha256"] == {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert report["gates"]["G1"]["status"] == "PASS"
    assert g.main([*argv, "--out", str(out)]) == 2
    assert (out / "gates_report.json").read_bytes() == raw
    assert "sanity_reference" in (out / "gates_report.md").read_text()


def test_loader_rejects_conflict_and_duplicate(tmp_path):
    """Conflicting or duplicated seed rows must not be averaged or overwritten."""
    path = write_summary(tmp_path / "input", arm())
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[1] + "\n")
    with pytest.raises(g.GateError, match="Duplicate"):
        g.load_summaries([path.parent], {})


def test_hardware_columns_merge_without_episode_reads(tmp_path):
    """Split P4 hardware and P5 seed files must supply one seed record per identity."""
    records = arm(hardware=g.HETERO)
    path = write_summary(tmp_path / "input", records)
    hardware = path.parent / "hardware_seed_metrics.csv"
    hardware.write_bytes(path.read_bytes())
    hashes = {}
    assert g.load_summaries([path.parent], hashes) == records
    assert len(hashes) == 2
    hardware.write_text(hardware.read_text().replace(",95,", ",96,"))
    with pytest.raises(g.GateError, match="Conflicting"):
        g.load_summaries([path.parent], {})


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_cost_inputs_refused(tmp_path, value):
    """Malformed costs must stop evaluation before statistical inference."""
    records = arm()
    next(iter(records.values()))[g.setting("seeds")[0]][g.COST] = value
    path = write_summary(tmp_path / "input", records)
    with pytest.raises(g.GateError):
        g.load_summaries([path.parent], {})


def test_cli_absent_waves_and_bad_seed_sets_leave_no_output(tmp_path):
    """Invalid input must not leave an apparently valid report directory."""
    out = tmp_path / "absent"
    assert g.main(["--out", str(out)]) == 0
    report = json.loads((out / "gates_report.json").read_text())
    assert all(r["status"] == "NOT_EVALUATED" for r in report["gates"].values())
    records = primary_waves()["wave1"]
    next(iter(records.values())).pop(g.setting("seeds")[0])
    path = write_summary(tmp_path / "bad", records)
    bad_out = tmp_path / "bad_report"
    assert g.main(["--wave1", str(path.parent), "--out", str(bad_out)]) == 2
    assert not bad_out.exists()
