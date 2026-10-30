"""Synthetic contract tests for the additional arm; no actor inference or training."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sensys_runs import evaluate_gates, summarize_energy_profiles
from sensys_runs.iccps27_followup import analyze_er_eia as er
from sensys_runs.iccps27_followup import analyze_a9 as a9
from sensys_runs.iccps27_followup import analyze_eia as ae
from sensys_runs.iccps27_followup import test_analyze_eia as old_fixture
from sensys_runs.iccps27_followup.test_analyze_a9 import write_csv, write_json


def rows(cost, p1=.99, method=er.METHOD):
    return [dict(seed=s, run_id="synthetic/"+method+str(s), method=method,
                 observed_eligible=True, probe_eligible=True, issues=[], probe_issues=[],
                 observed=dict(n_episodes=100, mean_agent_mission_cost=cost+i/100,
                    mean_mission_steps=cost+i/100, success_rate=1.0, one_step_fraction=.99),
                 probe=dict(P1=p1, E_T_lower=2-p1, E_T_upper=1+4*(1-p1),
                    E_T_bound_width=3*(1-p1), mission_time_primary="reset_based_interval"))
            for i, s in enumerate(a9.SEEDS)]



def decision_inputs(er_passes, refine_passes, pattern):
    """Build ten observed pairs and reset probes, including the .95 boundary."""
    er_cost, refine_cost = {
        "refine_lower": (2.0, 1.0),
        "er_lower": (1.0, 2.0),
        "overlap": (1.1, 1.1),
    }[pattern]
    left = rows(er_cost)
    right = rows(refine_cost, method="refine_eia_er")
    for arm, passing in ((left, er_passes), (right, refine_passes)):
        for index, row in enumerate(arm):
            p1 = .95 if index < passing else .949
            row["probe"].update(P1=p1, E_T_lower=2-p1,
                                E_T_upper=1+4*(1-p1), E_T_bound_width=3*(1-p1))
    return left, right


CELL_SENTENCES = {
    "a": "At weight (1.1, 0.9), ER-LIO+EIA reaches the same one-step completion as REFiNE+EIA, "
         "so at this weight preserving one-step completion under EIA does not depend on REFiNE's incentive-learning path; "
         "the observed mean-cost ranges overlap.",
    "b": "At weight (1.1, 0.9), REFiNE+EIA has lower observed mean-agent modeled cost than ER-LIO+EIA "
         "in 10 of 10 paired seeds, with non-overlapping ranges.",
    "c": "At weight (1.1, 0.9), ER-LIO+EIA has lower observed mean-agent modeled cost than REFiNE+EIA "
         "in 10 of 10 paired seeds, with non-overlapping ranges.",
    "d": "At weight (1.1, 0.9), there is no consistent difference in observed mean-agent modeled cost "
         "across 10 paired seeds.",
}


def campaign_fixture(root):
    manifest = dict(mode="campaign", n_runs=30, weight_pairs=[list(w) for w in ae.WEIGHTS], runs=[])
    inventory = []
    for pair in ae.WEIGHTS:
        for seed in a9.SEEDS:
            directory = "energy_reward_"+ae.weight_label(pair)+"_"+str(seed)
            cfg = dict(main=dict(exp_name="synthetic", dir_name=directory, seed=seed),
                lio=dict(Fairness_multiplier=0.0, eia_w_lever=pair[0], eia_w_door=pair[1]),
                energy_in_reward=True, energy_reward_beta=1.0, energy_reward_form="episode_mean",
                followup_arm=er.ARM, training_objective="energy_reward", display_label=er.LABEL)
            manifest["runs"].append(dict(method=er.METHOD, weight_pair=list(pair), weight=ae.weight_label(pair),
                                         seed=seed, effective_config=cfg))
            inventory.append(dict(run_id="synthetic/"+directory, method=er.METHOD, seed=seed))
    path = root/"inventory.csv"
    write_csv(path, inventory)
    return manifest, path


class ERAnalysisTests(unittest.TestCase):
    def test_30row_matrix_and_base_method_arm_distinction(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, path = campaign_fixture(Path(folder))
            actual = er.matrix(manifest, path, a9.Inputs())
            self.assertEqual(len(actual), 30)
            self.assertTrue(all(r["method"] == "lio_eia_er" and r["arm"] == er.ARM for r in actual))
            self.assertEqual([r["seed"] for r in actual[:10]], list(a9.SEEDS))
            for mutate in (lambda m: m["runs"].pop(),
                           lambda m: m["runs"][0]["effective_config"].update(energy_in_reward=False),
                           lambda m: m["runs"].__setitem__(1, deepcopy(m["runs"][0]))):
                bad = deepcopy(manifest)
                mutate(bad)
                with self.assertRaises(a9.AnalysisError):
                    er.matrix(bad, path, a9.Inputs())

    def test_inventory_wrong_count_or_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, path = campaign_fixture(Path(folder))
            with path.open(encoding="utf-8") as handle:
                content = list(csv.DictReader(handle))
            write_csv(path, content[:-1])
            with self.assertRaisesRegex(a9.AnalysisError, "inventory"):
                er.matrix(manifest, path, a9.Inputs())
            content[0]["method"] = "refine_eia_er"
            write_csv(path, content)
            with self.assertRaisesRegex(a9.AnalysisError, "inventory"):
                er.matrix(manifest, path, a9.Inputs())

    def test_primary_is_exact_unchanged_g1_with_correct_sign(self):
        left, right = rows(1), rows(2, method="refine_eia_er")
        actual = er.paired_cost(left, right, evaluate_gates)
        keys = [(m, "uniform_v1", "env_reward", "homogeneous_v1", "uniform_v1", "homogeneous_v1")
                for m in ("refine_er", "lio_er")]
        wave = {k: {r["seed"]: r["observed"] for r in rr} for k, rr in zip(keys, (left, right))}
        expected = evaluate_gates.evaluate({"wave1": wave})["gates"]["G1"]["cost"]
        for field in ("relative_median", "relative_interval", "relative_differences", "seeds"):
            self.assertEqual(actual[field], expected[field])
        self.assertLess(actual["relative_median"], 0)
        self.assertEqual(actual["er_lower"], 10)
        self.assertNotIn("p", actual)
        left[0]["observed_eligible"] = False
        incomplete = er.paired_cost(left, right, evaluate_gates)
        self.assertEqual(incomplete["status"], "INCOMPLETE")
        self.assertNotIn("relative_median", incomplete)

    def test_wording_match_preserves_ten_of_ten_direction(self):
        left, right = rows(1), rows(1.001, method="refine_eia_er")
        cost = er.paired_cost(left, right, evaluate_gates)
        text = er.paper_wording(cost, ae.threshold_summary(left), ae.threshold_summary(right))
        self.assertEqual(text["rule"], 1)
        self.assertEqual(text["er_lower"], 10)
        self.assertTrue(text["ranges_overlap"])
        self.assertIn("not a statistical equivalence", text["interpretation"])

    def test_wording_direction_rules_and_incomplete(self):
        left, right = rows(1, .8), rows(2, .8, "refine_eia_er")
        e1 = ae.threshold_summary(left)
        self.assertEqual(er.paper_wording(er.paired_cost(left, right, evaluate_gates), e1, e1)["rule"], 3)
        self.assertEqual(er.paper_wording(er.paired_cost(right, left, evaluate_gates), e1, e1)["rule"], 2)
        self.assertEqual(er.paper_wording(er.paired_cost(left, left, evaluate_gates), e1, e1)["rule"], 4)
        e1["status"] = "INCOMPLETE"
        self.assertIsNone(er.paper_wording(er.paired_cost(left, right, evaluate_gates), e1, e1)["rule"])

    def _assert_decision_cell(self, er_passes, refine_passes, pattern, outcome):
        left, right = decision_inputs(er_passes, refine_passes, pattern)
        cost = er.metric_comparison(left, right, "mean_agent_mission_cost")
        er_e1, refine_e1 = ae.threshold_summary(left), ae.threshold_summary(right)
        self.assertEqual(er_e1["reset_passing_seeds"], er_passes)
        self.assertEqual(refine_e1["reset_passing_seeds"], refine_passes)
        actual = er.paper_wording(cost, er_e1, refine_e1, "1.1,0.9")
        expected = CELL_SENTENCES[outcome] + (
            " Reset P1 >= 0.95 holds for {}/10 ER-LIO+EIA seeds and {}/10 REFiNE+EIA seeds."
            .format(er_passes, refine_passes))
        self.assertEqual(actual["outcome"], outcome)
        self.assertEqual(actual["rule"], {"a": 1, "b": 2, "c": 3, "d": 4}[outcome])
        self.assertEqual(actual["text"], expected)
        self.assertEqual(actual["er_reset_count"], er_passes)
        self.assertEqual(actual["refine_reset_count"], refine_passes)
        self.assertEqual(actual["er_lower"], 10 if pattern == "er_lower" else 0)
        self.assertEqual(actual["refine_lower"], 10 if pattern == "refine_lower" else 0)

    def test_cell_both_complete_refine_cost_b(self):
        self._assert_decision_cell(9, 9, "refine_lower", "b")

    def test_cell_both_complete_er_cost_c(self):
        self._assert_decision_cell(9, 9, "er_lower", "c")

    def test_cell_both_complete_overlap_a(self):
        self._assert_decision_cell(9, 9, "overlap", "a")

    def test_cell_only_er_complete_refine_cost_b(self):
        self._assert_decision_cell(9, 8, "refine_lower", "b")

    def test_cell_only_er_complete_er_cost_c(self):
        self._assert_decision_cell(9, 8, "er_lower", "c")

    def test_cell_only_er_complete_overlap_d(self):
        self._assert_decision_cell(9, 8, "overlap", "d")

    def test_cell_only_refine_complete_refine_cost_b(self):
        self._assert_decision_cell(8, 9, "refine_lower", "b")

    def test_cell_only_refine_complete_er_cost_c(self):
        self._assert_decision_cell(8, 9, "er_lower", "c")

    def test_cell_only_refine_complete_overlap_d(self):
        self._assert_decision_cell(8, 9, "overlap", "d")

    def test_cell_neither_complete_refine_cost_b(self):
        self._assert_decision_cell(8, 8, "refine_lower", "b")

    def test_cell_neither_complete_er_cost_c(self):
        self._assert_decision_cell(8, 8, "er_lower", "c")

    def test_cell_neither_complete_overlap_d(self):
        self._assert_decision_cell(8, 8, "overlap", "d")

    def test_registered_weight_format_and_observational_cost_sentence(self):
        left, right = decision_inputs(9, 9, "refine_lower")
        cost = er.metric_comparison(left, right, "mean_agent_mission_cost")
        for raw, formatted in (("2.0,0.2", "(2.0, 0.2)"),
                               ("1.1,0.9", "(1.1, 0.9)"),
                               ("1.5,1.5", "(1.5, 1.5)")):
            with self.subTest(weight=raw):
                actual = er.paper_wording(cost, ae.threshold_summary(left), ae.threshold_summary(right), raw)
                expected = ("At weight " + formatted + ", REFiNE+EIA has lower observed mean-agent modeled cost "
                            "than ER-LIO+EIA in 10 of 10 paired seeds, with non-overlapping ranges. "
                            "Reset P1 >= 0.95 holds for 9/10 ER-LIO+EIA seeds and 9/10 REFiNE+EIA seeds.")
                self.assertEqual(actual["text"], expected)
                self.assertNotIn("robustness", actual["text"])

    def test_match_sentence_contains_each_ten_of_ten_direction_and_overlap(self):
        for lower_arm in ("ER-LIO+EIA", "REFiNE+EIA"):
            with self.subTest(lower_arm=lower_arm):
                left, right = decision_inputs(9, 9, "overlap")
                higher = right if lower_arm == "ER-LIO+EIA" else left
                for row in higher:
                    row["observed"]["mean_agent_mission_cost"] += .001
                cost = er.metric_comparison(left, right, "mean_agent_mission_cost")
                actual = er.paper_wording(cost, ae.threshold_summary(left), ae.threshold_summary(right), "1.1,0.9")
                expected = (
                    "At weight (1.1, 0.9), ER-LIO+EIA reaches the same one-step completion as REFiNE+EIA, "
                    "so at this weight preserving one-step completion under EIA does not depend on REFiNE's incentive-learning path; "
                    + lower_arm + " has lower observed mean-agent modeled cost in 10 of 10 paired seeds, "
                    "and the observed mean-cost ranges overlap. "
                    "Reset P1 >= 0.95 holds for 9/10 ER-LIO+EIA seeds and 9/10 REFiNE+EIA seeds.")
                self.assertEqual(actual["outcome"], "a")
                self.assertEqual(actual["text"], expected)
                self.assertNotIn("energy-regularized actor", actual["text"])

    def test_observed_one_step_counts_do_not_replace_reset_counts(self):
        left, right = decision_inputs(8, 9, "overlap")
        cost = er.metric_comparison(left, right, "mean_agent_mission_cost")
        initial = er.paper_wording(cost, ae.threshold_summary(left), ae.threshold_summary(right), "1.1,0.9")
        self.assertEqual(ae.threshold_summary(left)["observed_one_step_at_least_95_count"], 10)
        self.assertEqual(initial["outcome"], "d")
        for row in left + right:
            row["observed"]["one_step_fraction"] = .01
        self.assertEqual(ae.threshold_summary(right)["observed_one_step_at_least_95_count"], 0)
        actual = er.paper_wording(cost, ae.threshold_summary(left), ae.threshold_summary(right), "1.1,0.9")
        self.assertEqual(actual, initial)

    def test_g1_summary_and_probe_intervals_are_descriptive(self):
        left, right = decision_inputs(9, 9, "overlap")
        cost = er.paired_cost(left, right, evaluate_gates)
        initial = er.paper_wording(cost, ae.threshold_summary(left), ae.threshold_summary(right), "1.1,0.9")
        for median, interval in ((.8, [.7, .9]), (-.8, [-.9, -.7])):
            with self.subTest(relative_median=median):
                descriptive = dict(cost, relative_median=median, relative_interval=interval)
                self.assertEqual(er.paper_wording(descriptive, ae.threshold_summary(left),
                    ae.threshold_summary(right), "1.1,0.9"), initial)
        old_bounds = [row["probe"]["E_T_bound_width"] for row in left + right]
        for row in left + right:
            p1 = .999 if row["probe"]["P1"] >= .95 else .4
            row["probe"].update(P1=p1, E_T_lower=2-p1, E_T_upper=1+4*(1-p1),
                                E_T_bound_width=3*(1-p1), mission_time_primary="observed_100_episode_mean")
        self.assertNotEqual(old_bounds, [row["probe"]["E_T_bound_width"] for row in left + right])
        self.assertEqual(er.paper_wording(cost, ae.threshold_summary(left), ae.threshold_summary(right), "1.1,0.9"), initial)

    def test_wide_bounds_are_not_an_eligibility_failure(self):
        left = rows(1.2, .8)
        summary = ae.threshold_summary(left)
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["reset_passing_seeds"], 0)
        self.assertEqual(summary["reset_eligible_seeds"], 10)

    def test_descriptive_shift_does_not_call_lower_success_improvement(self):
        left, right = rows(1), rows(2)
        for r in left:
            r["observed"]["success_rate"] = .9
        result = er.descriptive(left, right, "LIO+EIA")
        self.assertEqual(result["metrics"]["success_rate"]["er_lower"], 10)
        self.assertAlmostEqual(result["pairs"][0]["shift"]["success_rate"], -.1)
        self.assertIn("not an improvement", result["direction"])
        right.reverse()
        with self.assertRaisesRegex(a9.AnalysisError, "ordering"):
            er.descriptive(left, right, "A9")

    def test_observed_wrapper_requires_er_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            args, campaign, _ = old_fixture.fixture(Path(folder))
            _, identities = ae.matrix(campaign, args.eia_inventory, a9.Inputs())
            identity = identities[0]
            plain = ae.read_observed(identity, args.results_root, a9.Inputs(), summarize_energy_profiles)
            self.assertTrue(plain["observed_eligible"])
            checked = er.read_observed(identity, args.results_root, a9.Inputs(), summarize_energy_profiles)
            self.assertFalse(checked["observed_eligible"])
            self.assertIn("objective", checked["issues"][0])

    def test_historical_60_rebound_without_modifying_e2_e3(self):
        from sensys_runs.iccps27_followup import reset_probe as rp
        with tempfile.TemporaryDirectory() as folder:
            args, _, inventory = old_fixture.fixture(Path(folder))
            old_fixture.add_synthetic_probes(args, inventory)
            report = old_fixture.run_analysis(args)
            old_path = args.output_dir/"REFiNE_ICCPS2027_EIA_Results_20261028.json"
            old_bytes = old_path.read_bytes()
            review = Path(folder)/"accepted_readonly.txt"
            review.write_text("\n".join("READONLY_VALIDATED_E1 "+weight+" "+method+" "+json.dumps(value)
                for weight, cell in report["results"].items() for method, value in cell["E1"].items()))
            for name, source in (("historical_report", old_path), ("historical_campaign_manifest", args.campaign_manifest),
                  ("historical_inventory", args.eia_inventory), ("historical_results_root", args.results_root),
                  ("historical_probe_root", args.probe_root), ("historical_validation_receipt", args.validation_receipt),
                  ("historical_review", review)):
                setattr(args, name, source)
            inputs = a9.Inputs()
            refs = er.clean_references(args, inputs, summarize_energy_profiles)
            @contextmanager
            def synthetic_scope(module, **kwargs):
                with patch.object(module, "source_git_identity", return_value=old_fixture.SYNTHETIC_GIT):
                    yield
            digest = hashlib.sha256(args.protected_manifest.read_bytes()).hexdigest()
            with patch.object(er, "source_identity_scope", synthetic_scope), patch.object(rp, "PROTECTED_MANIFEST_SHA256", digest):
                recovered = er.load_historical(args, refs, inputs, summarize_energy_profiles)
            self.assertEqual(len(recovered["runs"]), 60)
            self.assertTrue(all(r["probe_eligible"] for r in recovered["runs"]))
            self.assertEqual(recovered["historical_results"], report["results"])
            self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertEqual(recovered["validated_E1"]["2.0,0.2"]["refine_eia_er"]["reset_passing_seeds"], 10)
            review.write_text(review.read_text().replace('"reset_passing_seeds": 10', '"reset_passing_seeds": 9'))
            with patch.object(er, "source_identity_scope", synthetic_scope), patch.object(rp, "PROTECTED_MANIFEST_SHA256", digest):
                with self.assertRaisesRegex(a9.AnalysisError, "accepted read-only"):
                    fresh = a9.Inputs()
                    fresh_refs = er.clean_references(args, fresh, summarize_energy_profiles)
                    er.load_historical(args, fresh_refs, fresh, summarize_energy_profiles)

    def test_complete30_observed_and_probe_contract_integration(self):
        from lio.utils.configdict import ConfigDict
        from sensys_runs.iccps27_followup import er_eia_adapter
        from sensys_runs.iccps27_followup import reset_probe as rp
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args, old_campaign, _ = old_fixture.fixture(root)
            new_campaign = dict(mode="campaign", n_runs=30, weight_pairs=[list(w) for w in ae.WEIGHTS], runs=[])
            new_inventory = []
            old_root = args.results_root
            args.results_root = root/"new_er"
            args.probe_root = root/"new_er_probes"
            for previous in old_campaign["runs"]:
                if previous["method"] != er.METHOD:
                    continue
                cfg = deepcopy(previous["effective_config"])
                old_rid = cfg["main"]["exp_name"]+"/"+cfg["main"]["dir_name"]
                cfg["main"]["exp_name"] = "synthetic_er"
                cfg["main"]["dir_name"] = "energy_reward_"+cfg["main"]["dir_name"]
                cfg["lio"]["Fairness_multiplier"] = 0.0
                cfg.update(energy_in_reward=True, energy_reward_beta=1.0, energy_reward_form="episode_mean",
                           followup_arm=er.ARM, training_objective="energy_reward", display_label=er.LABEL)
                record = dict(previous, effective_config=deepcopy(cfg))
                new_campaign["runs"].append(record)
                rid = cfg["main"]["exp_name"]+"/"+cfg["main"]["dir_name"]
                source, run = old_root/old_rid, args.results_root/rid
                cfg["eia_attack"] = old_fixture.attack(record["weight_pair"])
                write_json(run/"config.json", cfg)
                energy = er_eia_adapter.energy_provenance(ConfigDict(deepcopy(cfg)))
                write_json(run/"provenance.json", dict(method=er.METHOD, seed=record["seed"],
                    eia_attack=cfg["eia_attack"], followup_arm=er.ARM, training_objective="energy_reward",
                    display_label=er.LABEL, energy_reward=energy, er_eia_adapter=er_eia_adapter.ADAPTER_ID,
                    er_eia_launch_argv=["synthetic", "--energy_in_reward"]))
                for path in source.glob("model.ckpt.*"):
                    (run/path.name).write_bytes(path.read_bytes())
                with (source/"energy_eval_episodes.csv").open(encoding="utf-8") as handle:
                    episodes = list(csv.DictReader(handle))
                for row in episodes:
                    row["run_id"] = rid
                write_csv(run/"energy_eval_episodes.csv", episodes)
                meta = json.loads((source/"energy_eval_provenance.json").read_text())
                meta.update(config=cfg, run_id=rid, episodes_sha256=old_fixture.sha(run/"energy_eval_episodes.csv"))
                write_json(run/"energy_eval_provenance.json", meta)
                completion = json.loads((source/"eia_run_completion.json").read_text())
                completion.update(run_id=rid, episodes_sha256=meta["episodes_sha256"],
                                  final_evaluation_sha256=old_fixture.sha(run/"energy_eval_provenance.json"))
                write_json(run/"eia_run_completion.json", completion)
                new_inventory.append(dict(run_id=rid, method=er.METHOD, seed=record["seed"]))
            args.eia_inventory = root/"er_inventory.csv"
            write_csv(args.eia_inventory, new_inventory)
            old_fixture.add_synthetic_probes(args, new_inventory, lio_p=.99)
            # Unrelated JSON records must never be scanned as per-run probes.
            write_json(args.probe_root/"posttraining_review_20261029.json", {"not_a_probe": True})
            inputs = a9.Inputs()
            identities = er.matrix(new_campaign, args.eia_inventory, inputs)
            actual = [er.read_observed(r, args.results_root, inputs, summarize_energy_profiles) for r in identities]
            refs = er.clean_references(args, inputs, summarize_energy_profiles)
            @contextmanager
            def synthetic_scope(module, **kwargs):
                with patch.object(module, "source_git_identity", return_value=old_fixture.SYNTHETIC_GIT):
                    yield
            with patch.object(er, "source_identity_scope", synthetic_scope), patch.object(rp,
                 "PROTECTED_MANIFEST_SHA256", old_fixture.sha(args.protected_manifest)):
                er.historical_probes(args, identities, actual, refs, inputs)
            self.assertEqual(len(actual), 30)
            self.assertTrue(all(r["observed_eligible"] and r["probe_eligible"] and not r["issues"] and not r["probe_issues"] for r in actual))
            self.assertTrue(all(r["arm"] == er.ARM and r["display_label"] == er.LABEL for r in actual))
            inputs.verify()
            first = actual[0]
            path = args.probe_root/(first["run_id"].replace("/", "__")+".json")
            tampered = json.loads(path.read_text())
            tampered["checkpoint_hashes"]["model.ckpt.index"] = "0"*64
            write_json(path, tampered)
            fresh = a9.Inputs()
            fresh.read(args.eia_inventory)
            context = (fresh.json(args.probe_root/"manifest.json"), old_fixture.sha(args.validation_receipt))
            first.update(probe_eligible=False, probe=None)
            ae.attach_probe(first, args.probe_root, context, fresh)
            self.assertFalse(first["probe_eligible"])
            self.assertIn("binding mismatch", first["probe_issues"][0])

    def test_invalid_observed_binding_names_run_and_stops_before_output(self):
        bad = rows(1)
        bad[2].update(observed_eligible=False, issues=["Missing final checkpoint"])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"results_20261029"
            with self.assertRaisesRegex(a9.AnalysisError, bad[2]["run_id"]):
                er.require_observed_bindings(bad)
                er.write_results(output, "20261029", {}, a9.Inputs())
            self.assertFalse(output.exists())

    def test_write_retains_all30_and_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"results_20261029"
            selected = []
            for weight in ae.WEIGHTS:
                for row in rows(1):
                    row["weight"] = ae.weight_label(weight)
                    selected.append(row)
            selected[0]["probe_eligible"] = False
            selected[0]["probe"] = None
            report = dict(status="INCOMPLETE", runs=selected, results={}, mission_time_rule="SYNTHETIC TEST ONLY")
            er.write_results(output, "20261029", report, a9.Inputs())
            with (output/"per_run.csv").open() as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), er.CSV_FIELDS)
                written = list(reader)
            self.assertEqual(len(written), 30)
            self.assertEqual(written[0]["probe_eligible"], "False")
            self.assertEqual(written[0]["arm"], er.ARM)
            with self.assertRaises(FileExistsError):
                er.write_results(output, "20261029", report, a9.Inputs())


if __name__ == "__main__":
    unittest.main()
