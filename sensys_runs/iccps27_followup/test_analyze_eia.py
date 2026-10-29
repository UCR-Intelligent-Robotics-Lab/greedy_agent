"""Synthetic analysis-contract tests. No TensorFlow restore, training or probing."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sensys_runs import evaluate_gates
from sensys_runs.iccps27_followup import analyze_a9 as a9
from sensys_runs.iccps27_followup import analyze_eia as eia
from sensys_runs.iccps27_followup import reset_probe
from sensys_runs.iccps27_followup.test_analyze_a9 import synthetic_inputs, write_csv, write_json


SYNTHETIC_GIT = dict(commit="SYNTHETIC_NOT_A_COMMIT", blobs={name: "SYNTHETIC_BLOB" for name in reset_probe.SOURCE_FILES})


def run_analysis(args):
    with patch.object(reset_probe, "source_git_identity", return_value=SYNTHETIC_GIT, create=True):
        return eia.analyze(args)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attack(pair):
    return dict(schema_version=1, action_order=["lever", "start", "door"], exploitative_giver_id=1,
                applies_to="recipient_action", training_active=True, evaluation_active=True,
                configured_weights=dict(lever=pair[0], door=pair[1]),
                effective_multipliers=[dict(giver_id=g, recipient_id=r, delivered=g != r,
                    multipliers=[pair[0], 1.0, pair[1]] if g == 1 and r != 1 else [1.0]*3)
                    for g in range(4) for r in range(4)])


def fixture(root, fallback=False):
    old, _ = synthetic_inputs(root)
    args = argparse.Namespace(**vars(old), campaign_manifest=root/"campaign.json", eia_inventory=root/"eia.csv",
                              results_root=root/"eia_observed", probe_root=root/"probe",
                              validation_receipt=root/"six_validation.json", registration=root/"registration.md")
    args.output_dir = root/"eia_analysis_20261028"
    args.registration.write_text("Synthetic pre-launch registration fixture only.\n", encoding="utf-8")
    weights = (eia.WEIGHTS[0], eia.WEIGHTS[2]) if fallback else eia.WEIGHTS
    campaign = dict(schema_version=1, mode="campaign", fallback_declared=fallback,
                    weight_pairs=[list(p) for p in weights], n_runs=len(weights)*20, runs=[])
    inventory = []
    for pair in weights:
        for method in eia.METHODS:
            for seed in a9.SEEDS:
                label = eia.weight_label(pair)
                directory = method+"_w"+label+"_s"+str(seed)
                rid = "synthetic_eia/"+directory
                cfg = dict(main=dict(exp_name="synthetic_eia", dir_name=directory, seed=seed),
                           lio=dict(eia_w_lever=pair[0], eia_w_door=pair[1]),
                           cost_profile=dict(profile_id="uniform_v1", normalized_vector=[1.0]*3),
                           hardware_profile=dict(profile_id="homogeneous_v1"), hardware_profile_lambda=[1.0]*4)
                if method == "refine_eia_er":
                    cfg["lio"].update(Fairness_multiplier=0.0, energy_weight=1.0)
                campaign["runs"].append(dict(method=method, weight=label, weight_pair=list(pair), seed=seed,
                                               effective_config=copy.deepcopy(cfg)))
                cfg["eia_attack"] = attack(pair)
                run = args.results_root/rid
                write_json(run/"config.json", cfg)
                write_json(run/"provenance.json", dict(method=method, seed=seed, eia_attack=cfg["eia_attack"]))
                for name in ("model.ckpt.index", "model.ckpt.meta", "model.ckpt.data-00000-of-00001"):
                    (run/name).write_bytes(("SYNTHETIC NOT A CHECKPOINT "+rid+name).encode())
                episodes = []
                for episode in range(1, 101):
                    steps = 2 if method == "lio_eia_er" and episode <= 40 else 1
                    for agent in range(4):
                        episodes.append(dict(schema_version=1, run_id=rid, seed=seed, method=method,
                            training_profile="uniform_v1", checkpoint_episode=25000, eval_episode=episode,
                            agent_id=agent, terminal_code=1, steps=steps, n_lever=steps if agent < 2 else 0,
                            n_start=0, n_door=steps if agent >= 2 else 0, env_return=0.0))
                write_csv(run/"energy_eval_episodes.csv", episodes)
                csv_hash = sha(run/"energy_eval_episodes.csv")
                write_json(run/"energy_eval_provenance.json", dict(config=cfg, episodes_sha256=csv_hash,
                    method=method, seed=seed, run_id=rid, checkpoint_episode=25000, n_eval_episodes=100, n_agents=4,
                    eia_attack=cfg["eia_attack"], effective_uniform_costs=[dict(agent_id=i, costs=[1.0]*3) for i in range(4)],
                    checkpoint_sha256={p.name: sha(p) for p in run.glob("model.ckpt.*")}))
                write_json(run/"eia_run_completion.json", dict(schema_version=1, status="complete", method=method,
                    seed=seed, run_id=rid, final_evaluation_episodes=100, final_evaluation_rows=400,
                    episodes_sha256=csv_hash, final_evaluation_sha256=sha(run/"energy_eval_provenance.json")))
                inventory.append(dict(run_id=rid, arm="SYNTHETIC EIA", method=method, seed=seed,
                    checkpoint_prefix="lio/results/"+rid+"/model.ckpt", evidence="SYNTHETIC TEST ONLY"))
    write_json(args.campaign_manifest, campaign)
    write_csv(args.eia_inventory, inventory)
    return args, campaign, inventory


def add_synthetic_probes(args, inventory, *, refine_p=.99, lio_p=.6):
    """Serialize fake records to test validation logic, never execute actor code."""
    runtime = dict(tensorflow="SYNTHETIC_FIXTURE_NOT_EXECUTED", device="CPU")
    sources = {name: sha(a9.REPO/name) for name in reset_probe.SOURCE_FILES}
    expected, checks = [], []
    for arm, method, p1 in (("A3", "lio_er", .8), ("A8", "refine_er", .99)):
        for seed in (12341, 12342, 12343):
            rid = "synthetic_"+arm+"/seed_"+str(seed)
            exact = args.reference_enumeration_root/rid.replace("/", "__")/"result.json"
            checkpoints = {"model.ckpt.index": "SYNTHETIC_"+rid}
            write_json(args.reference_evaluation_root/rid/"energy_eval_provenance.json", dict(checkpoint_sha256=checkpoints))
            expected.append(dict(run_id=rid, method=method, seed=seed, expected_P_tau_1=p1,
                reference_result_sha256=sha(exact)))
            checks.append(dict(run_id=rid, method=method, seed=seed, expected_P_tau_1=p1, P1=p1,
                absolute_difference=0.0, passed=True, checkpoint_hashes=checkpoints, runtime=runtime))
    write_json(args.validation_receipt, dict(status="PASS", execution_kind="actual_tensorflow_checkpoint_restore",
        tolerance=1e-5, expected=expected, checks=checks, source_hashes=sources, source_git_identity=SYNTHETIC_GIT))
    receipt_hash = sha(args.validation_receipt)
    completed = []
    for identity in inventory:
        rid = identity["run_id"]
        run = args.results_root/rid
        cfg = json.loads((run/"config.json").read_text())
        meta = json.loads((run/"energy_eval_provenance.json").read_text())
        p1 = refine_p if identity["method"] == "refine_eia_er" else lio_p
        lower, upper = 2-p1, 1+4*(1-p1)
        filename = rid.replace("/", "__")+".json"
        result = dict(run_id=rid, method=identity["method"], seed=identity["seed"], eia_attack=cfg["eia_attack"],
            P1=p1, E_T_lower=lower, E_T_upper=upper, E_T_bound_width=upper-lower, wording=eia.PROBE_WORDING,
            runtime=runtime, checkpoint_hashes=meta["checkpoint_sha256"],
            config_provenance_hashes={name: sha(run/name) for name in ("config.json", "provenance.json")},
            validation_receipt_sha256=receipt_hash, no_payment_query=True, no_training_or_optimizer=True,
            execution_kind="actual_tensorflow_checkpoint_restore")
        write_json(args.probe_root/filename, result)
        completed.append(dict(run_id=rid, file=filename, **{k: result[k] for k in ("P1", "E_T_lower", "E_T_upper", "E_T_bound_width")}))
    write_json(args.probe_root/"manifest.json", dict(completed=completed, failed=[], pending=[],
        source_hashes=sources, source_git_identity=SYNTHETIC_GIT, input_hashes={"receipt": receipt_hash, "inventory": sha(args.eia_inventory)},
        validation_receipt_sha256=receipt_hash, runtime=runtime, sources_unchanged=True, inputs_unchanged=True, status="COMPLETE"))


class EIAAnalysisTests(unittest.TestCase):
    def test_e2_reuses_exact_g1_output_and_sign(self):
        def rows(cost):
            return [dict(seed=s, observed_eligible=True, observed=dict(mean_agent_mission_cost=cost+i/100,
                success_rate=1.0)) for i, s in enumerate(a9.SEEDS)]
        left, right = rows(1), rows(2)
        result = eia.paired_observed(left, right, evaluate_gates)
        keys = [(m, "uniform_v1", "env_reward", "homogeneous_v1", "uniform_v1", "homogeneous_v1") for m in ("refine_er", "lio_er")]
        wave = {k: {r["seed"]: r["observed"] for r in rows} for k, rows in zip(keys, (left, right))}
        expected = evaluate_gates.evaluate({"wave1": wave})["gates"]["G1"]["cost"]
        for name in ("relative_median", "relative_interval", "relative_differences", "seeds"):
            self.assertEqual(result[name], expected[name])
        self.assertLess(result["relative_median"], 0)
        self.assertEqual(result["refine_lower"], 10)
        self.assertIn("non-overlapping", result["paper_wording"])
        self.assertTrue(eia.paired_observed(right, left, evaluate_gates)["paper_wording"].startswith("LIO+EIA"))
        self.assertEqual(eia.paired_observed(left, left, evaluate_gates)["paper_wording"], "no consistent difference across 10 paired seeds")

    def test_complete60_observed_without_probe_retained_and_csv_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = fixture(Path(folder))
            report = run_analysis(args)
            self.assertEqual(len(report["runs"]), 60)
            result = report["results"]["2.0,0.2"]
            self.assertEqual(result["E2"]["status"], "COMPLETE")
            self.assertEqual(result["E1"]["refine_eia_er"]["status"], "INCOMPLETE")
            self.assertEqual(result["E3"]["lio_eia_er_vs_A3"]["status"], "COMPLETE")
            self.assertAlmostEqual(result["E3"]["lio_eia_er_vs_A3"]["pairs"][0]["shift"]["mean_mission_steps"], .2)
            with (args.output_dir/"per_run.csv").open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), eia.CSV_FIELDS)
                self.assertEqual(len(list(reader)), 60)
            raw = (args.output_dir/"REFiNE_ICCPS2027_EIA_Results_20261028.json").read_text()
            for forbidden in ('"p":', '"holm_p":', '"conditions":'):
                self.assertNotIn(forbidden, raw)
            with self.assertRaisesRegex(a9.AnalysisError, "fresh dated"):
                run_analysis(args)

    def test_declared40_and_unregistered_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args, campaign, _ = fixture(Path(folder), fallback=True)
            weights, identities = eia.matrix(campaign, args.eia_inventory, a9.Inputs())
            self.assertEqual(len(identities), 40)
            self.assertEqual(weights, (eia.WEIGHTS[0], eia.WEIGHTS[2]))
            campaign["fallback_declared"] = False
            with self.assertRaisesRegex(a9.AnalysisError, "weight matrix"):
                eia.matrix(campaign, args.eia_inventory, a9.Inputs())
            campaign["fallback_declared"] = True
            campaign["runs"][-1] = copy.deepcopy(campaign["runs"][0])
            with self.assertRaisesRegex(a9.AnalysisError, "duplicate"):
                eia.matrix(campaign, args.eia_inventory, a9.Inputs())

    def test_c04_mask_keeps_start_action_one(self):
        cfg = dict(lio=dict(eia_w_lever=1.5, eia_w_door=1.5), eia_attack=attack((1.5, 1.5)))
        eia.attack_provenance(cfg, dict(eia_attack=cfg["eia_attack"]), (1.5, 1.5))
        cfg["eia_attack"]["effective_multipliers"][4]["multipliers"][1] = 1.5
        with self.assertRaisesRegex(a9.AnalysisError, "registered mask"):
            eia.attack_provenance(cfg, dict(eia_attack=cfg["eia_attack"]), (1.5, 1.5))

    def test_validated_probe_and_only_refine_width_triggers_declaration(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, inventory = fixture(Path(folder), fallback=True)
            add_synthetic_probes(args, inventory)
            report = run_analysis(args)
            self.assertTrue(all(r["probe_eligible"] for r in report["runs"]))
            self.assertEqual(report["results"]["2.0,0.2"]["E1"]["refine_eia_er"]["reset_passing_seeds"], 10)
            self.assertEqual(report["deferred_decision"]["candidates"], [])
            self.assertEqual(report["runs"][0]["probe"]["mission_time_primary"], "observed_100_episode_mean")
            args.output_dir = Path(folder)/"wide_refine_20261028"
            add_synthetic_probes(args, inventory, refine_p=.94)
            report = run_analysis(args)
            self.assertEqual(len(report["deferred_decision"]["candidates"]), 20)
            self.assertEqual(report["deferred_decision"]["status"], "DECLARATION_REQUIRED_BY_DEADLINE")

    def test_missing_completion_and_tampered_probe_retained_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, inventory = fixture(Path(folder), fallback=True)
            add_synthetic_probes(args, inventory)
            first = inventory[0]
            (args.results_root/first["run_id"]/"eia_run_completion.json").unlink()
            second = inventory[1]
            path = args.probe_root/(second["run_id"].replace("/", "__")+".json")
            data = json.loads(path.read_text())
            data["checkpoint_hashes"]["model.ckpt.index"] = "0"*64
            write_json(path, data)
            report = run_analysis(args)
            self.assertEqual(len(report["runs"]), 40)
            self.assertFalse(report["runs"][0]["observed_eligible"])
            self.assertFalse(report["runs"][1]["probe_eligible"])
            self.assertTrue(report["runs"][1]["observed_eligible"])
            self.assertEqual(report["results"]["2.0,0.2"]["E2"]["status"], "INCOMPLETE")

    def test_invalid_validation_receipt_does_not_suppress_observed(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, inventory = fixture(Path(folder), fallback=True)
            add_synthetic_probes(args, inventory)
            receipt = json.loads(args.validation_receipt.read_text())
            receipt["checks"][0]["passed"] = False
            write_json(args.validation_receipt, receipt)
            report = run_analysis(args)
            self.assertTrue(all(r["observed_eligible"] for r in report["runs"]))
            self.assertFalse(any(r["probe_eligible"] for r in report["runs"]))
            self.assertEqual(report["results"]["2.0,0.2"]["E2"]["status"], "COMPLETE")

    def test_cross_platform_git_identity_mismatch_invalidates_probe_only(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, inventory = fixture(Path(folder), fallback=True)
            add_synthetic_probes(args, inventory)
            path = args.probe_root/"manifest.json"
            data = json.loads(path.read_text())
            data["source_git_identity"]["commit"] = "WRONG_COMMIT"
            write_json(path, data)
            report = run_analysis(args)
            self.assertTrue(all(r["observed_eligible"] for r in report["runs"]))
            self.assertFalse(any(r["probe_eligible"] for r in report["runs"]))
            self.assertIn("Git source identity mismatch", report["runs"][0]["probe_issues"][0])

    def test_e3_direction_counts_ranges_and_neutral_success_wording(self):
        def rows(method, arm, cost, success):
            return [dict(method=method, arm=arm, seed=seed, observed_eligible=True,
                         observed=dict(mean_agent_mission_cost=cost, mean_mission_steps=cost,
                                       one_step_fraction=success, success_rate=success)) for seed in a9.SEEDS]
        left = rows("refine_eia_er", "EIA", 1.0, .8)
        right = rows("refine_er", "A8", 2.0, 1.0)
        report = eia.descriptive_shift(left, right)
        self.assertEqual(report["status"], "COMPLETE")
        cost = report["metrics"]["mean_agent_mission_cost"]
        self.assertEqual((cost["eia_lower"], cost["eia_higher"], cost["ties"]), (10, 0, 0))
        self.assertEqual(cost["eia_range"], [1.0, 1.0])
        self.assertEqual(cost["reference_range"], [2.0, 2.0])
        self.assertFalse(cost["ranges_overlap"])
        self.assertEqual(cost["paper_wording"], "REFiNE+EIA has lower observed mean-agent modeled cost than A8 in 10 of 10 paired seeds, with non-overlapping ranges")
        success = report["metrics"]["success_rate"]
        self.assertEqual(success["paper_wording"], "REFiNE+EIA has lower observed success rate than A8 in 10 of 10 paired seeds, with non-overlapping ranges")
        self.assertNotIn("better", success["paper_wording"])
        self.assertNotIn("improv", success["paper_wording"])
        for row in left:
            row["observed"]["success_rate"] = 1.0
        for row in right:
            row["observed"]["success_rate"] = .8
        success = eia.descriptive_shift(left, right)["metrics"]["success_rate"]
        self.assertEqual((success["eia_lower"], success["eia_higher"], success["ties"]), (0, 10, 0))
        self.assertEqual(success["paper_wording"], "REFiNE+EIA has higher observed success rate than A8 in 10 of 10 paired seeds, with non-overlapping ranges")
        same = eia.descriptive_shift(left, left)["metrics"]["mean_agent_mission_cost"]
        self.assertEqual(same["ties"], 10)
        self.assertTrue(same["ranges_overlap"])
        self.assertEqual(same["paper_wording"], "no consistent difference across 10 paired seeds")

    def test_e3_incomplete_never_makes_ten_pair_claim(self):
        def rows(method, arm, value):
            return [dict(method=method, arm=arm, seed=seed, observed_eligible=True,
                         observed=dict(mean_agent_mission_cost=value, mean_mission_steps=value,
                                       one_step_fraction=.9, success_rate=.9)) for seed in a9.SEEDS]
        left, right = rows("lio_eia_er", "EIA", 1.0), rows("lio_er", "A3", 2.0)
        left[0]["observed_eligible"] = False
        result = eia.descriptive_shift(left, right)
        self.assertEqual(result["status"], "INCOMPLETE")
        for metric in result["metrics"].values():
            self.assertEqual(metric["status"], "INCOMPLETE")
            self.assertEqual(metric["eligible_pairs"], 9)
            self.assertTrue(metric["paper_wording"].startswith("INCOMPLETE"))
            self.assertNotIn("10 of 10", metric["paper_wording"])
        result = eia.descriptive_shift(left[1:], right[1:])
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(result["pairs"]), 10)
        self.assertTrue(all(metric["status"] == "INCOMPLETE" for metric in result["metrics"].values()))

    def test_source_manifest_tamper_stops_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = fixture(Path(folder), fallback=True)
            text = args.source_manifest.read_text()
            original = text.splitlines()[1].split("\t")[-1]
            args.source_manifest.write_text(text.replace(original, "0"*64))
            with self.assertRaisesRegex(a9.AnalysisError, "Pinned source mismatch"):
                run_analysis(args)
            self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
