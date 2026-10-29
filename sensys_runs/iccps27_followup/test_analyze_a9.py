"""Synthetic A9 analysis contracts; no training, inference, or enumeration."""
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

from sensys_runs.iccps27_followup import analyze_a9 as a9
from sensys_runs import evaluate_gates, summarize_energy_profiles


def metric_rows(arm, et, p=.99):
    return [dict(arm=arm, seed=s, eligible=True, exact=dict(E_T=et, P_tau_1=p, modeled_cost=et),
                 observed=dict(mean_agent_mission_cost=et + i / 100, success_rate=1.0))
            for i, s in enumerate(a9.SEEDS)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_inputs(root):
    args = argparse.Namespace(a9_inventory=root/"a9.csv", a9_export_root=root/"export",
                              a9_evaluation_root=root/"new_observed", a9_enumeration_root=root/"new_exact",
                              reference_inventory=root/"reference.csv", reference_evaluation_root=root/"ref_observed",
                              reference_enumeration_root=root/"ref_exact", source_manifest=root/"source.tsv",
                              protected_manifest=root/"protected.tsv", workspace_root=root,
                              output_dir=root/"analysis_20261028", date="20261028")
    source_rows = [dict(source_root="repo", relative_path=name,
                        sha256=hashlib.sha256((a9.REPO/name).read_bytes()).hexdigest()) for name in a9.PINNED_IMPORTS]
    write_csv(args.source_manifest, source_rows, "\t")
    protected = sorted(p for p in (a9.REPO/"sensys_runs/iccps27_analysis").rglob("*") if p.suffix in (".py", ".csv"))
    write_csv(args.protected_manifest, [dict(relative_path=p.relative_to(a9.REPO).as_posix(),
                                             sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in protected], "\t")
    manifests = {}
    inventory = {"A9": [], "reference": []}
    for arm in ("A9", "A3", "A8"):
        observed = args.a9_evaluation_root if arm == "A9" else args.reference_evaluation_root
        exact = args.a9_enumeration_root if arm == "A9" else args.reference_enumeration_root
        manifest = manifests.setdefault(exact, dict(dry_run=False, acceptance_criterion=a9.CRITERION,
                input_hashes_unchanged=True, source_hashes_unchanged=True, completed=[], parity_by_run={}, input_hashes={}, excluded=[]))
        for seed in a9.SEEDS:
            rid = "synthetic_"+arm+"/seed_"+str(seed)
            method = "refine_er" if arm == "A8" else "lio_er"
            identity = dict(run_id=rid, arm=a9.ARMS[arm], method=method, seed=str(seed), checkpoint_prefix="lio/results/"+rid+"/model.ckpt", evidence="SYNTHETIC TEST ONLY")
            inventory["A9" if arm == "A9" else "reference"].append(identity)
            episodes = []
            for episode in range(1, 101):
                steps = 2 if arm == "A3" and episode <= 20 else 1
                for agent in range(4):
                    episodes.append(dict(schema_version=1, run_id=rid, seed=seed, method=method,
                        training_profile="uniform_v1", checkpoint_episode=25000, eval_episode=episode,
                        agent_id=agent, terminal_code=1, steps=steps, n_lever=steps if agent < 2 else 0,
                        n_start=0, n_door=steps if agent >= 2 else 0, env_return=0.0))
            csv_path = observed/rid/"energy_eval_episodes.csv"
            write_csv(csv_path, episodes)
            manifest["input_hashes"][str(csv_path.resolve())] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            manifest["completed"].append(rid)
            manifest["parity_by_run"][rid] = dict(criterion=a9.CRITERION, amended_status="pass")
            # Serialized distributions exercise report logic only. No history engine runs.
            p = .8 if arm == "A3" else .99
            write_json(exact/rid.replace("/", "__")/"result.json",
                       dict(H=5, node_count=12326281, P_tau=[p, 1-p, 0, 0, 0], P_timeout=0, E_T=2-p))
            write_json(exact/"validation"/(rid.replace("/", "__")+".json"), dict(claim_eligible=True))
    write_csv(args.a9_inventory, inventory["A9"])
    write_csv(args.reference_inventory, inventory["reference"])
    for path, manifest in manifests.items():
        write_json(path/"v3_manifest.json", manifest)
    write_json(args.a9_export_root/"manifest.json", dict(computed=[], excluded=[]))
    return args, manifests


class A9AnalysisTests(unittest.TestCase):
    def test_m1_matches_direct_unchanged_evaluate_g1(self):
        method, reference = metric_rows("A9", 1.02), metric_rows("A3", 1.20)
        actual = a9.m1_estimate(method, reference, evaluate_gates)
        keys = [(m, "uniform_v1", "env_reward", "homogeneous_v1", "uniform_v1", "homogeneous_v1") for m in ("refine_er", "lio_er")]
        wave = {key: {r["seed"]: r["observed"] for r in rows} for key, rows in zip(keys, (method, reference))}
        expected = evaluate_gates.evaluate({"wave1": wave})["gates"]["G1"]["cost"]
        for field in ("relative_median", "relative_interval", "relative_differences"):
            self.assertEqual(actual[field], expected[field])
        self.assertLess(actual["relative_median"], 0)
        self.assertIn("actual A9 (energy-reward LIO)", actual["adapter"])
        self.assertNotIn("p", actual)

    def test_predeclared_wording_priority_and_incomplete(self):
        cases = [(1.01, 1.01, "energy-reward LIO reaches the same one-step completion"),
                 (1.03, 1.01, "REFiNE's incentive-learning path contributes"),
                 (1.01, 1.03, "energy-reward LIO has lower exact E[T]")]
        for x, y, text in cases:
            m2, m3, wording = a9.m2_m3(metric_rows("A9", x), metric_rows("A8", y))
            self.assertTrue(wording.startswith(text))
            self.assertEqual(m3["passing_seeds"], 10)
            self.assertEqual(m2["metrics"]["modeled_cost"]["A9_lower"], 10 if x < y else 0)
        rows = metric_rows("A9", 1.01, .94)
        self.assertEqual(a9.m2_m3(rows, metric_rows("A8", 1.01))[2], "no consistent difference across 10 paired seeds")
        rows[0]["eligible"] = False
        self.assertTrue(a9.m2_m3(rows, metric_rows("A8", 1.01))[2].startswith("INCOMPLETE"))
        self.assertEqual(a9.m1_estimate(rows, metric_rows("A3", 1.2), evaluate_gates)["status"], "INCOMPLETE")

    def test_end_to_end_synthetic_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = synthetic_inputs(Path(folder))
            report = a9.analyze(args)
            self.assertEqual(report["M1"]["status"], "COMPLETE")
            self.assertEqual(report["M3"]["passing_seeds"], 10)
            self.assertEqual(len(report["runs"]), 30)
            raw = (args.output_dir/"REFiNE_ICCPS2027_A9_Results_20261028.json").read_text()
            for forbidden in ('"p":', '"holm_p":', '"conditions":', '"decision":'):
                self.assertNotIn(forbidden, raw)
            self.assertEqual(report["runs"][10]["observed"]["mean_agent_mission_cost"], 1.2)
            with self.assertRaisesRegex(a9.AnalysisError, "fresh directory"):
                a9.analyze(args)

    def test_failed_and_missing_seed_remains_in_report(self):
        with tempfile.TemporaryDirectory() as folder:
            args, manifests = synthetic_inputs(Path(folder))
            manifest = manifests[args.a9_enumeration_root]
            rid = manifest["completed"].pop(0)
            del manifest["parity_by_run"][rid]
            manifest["excluded"] = [dict(run_id=rid, amended_status="fail", reason="Synthetic parity failure", parity=dict(amended_status="fail"))]
            write_json(args.a9_enumeration_root/"v3_manifest.json", manifest)
            (args.a9_enumeration_root/rid.replace("/", "__")/"result.json").unlink()
            report = a9.analyze(args)
            self.assertEqual(len(report["runs"]), 30)
            self.assertEqual(report["runs"][0]["parity_verdict"], "fail")
            self.assertEqual(report["M1"]["status"], "INCOMPLETE")
            self.assertEqual(report["M3"]["eligible_seeds"], 9)
            self.assertTrue(report["paper_wording"].startswith("INCOMPLETE"))

    def test_source_pin_mismatch_stops_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = synthetic_inputs(Path(folder))
            with args.source_manifest.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            rows[0]["sha256"] = "0" * 64
            write_csv(args.source_manifest, rows, "\t")
            with patch.object(a9.importlib, "import_module") as importer:
                with self.assertRaisesRegex(a9.AnalysisError, "Pinned source mismatch"):
                    a9.pinned_modules(args.source_manifest, args.workspace_root, a9.Inputs(), args.protected_manifest)
                importer.assert_not_called()

    def test_observed_hash_tampering_invalidates_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = synthetic_inputs(Path(folder))
            path = args.a9_evaluation_root/"synthetic_A9/seed_12341/energy_eval_episodes.csv"
            path.write_text(path.read_text().replace("0.0", "1.0"), encoding="utf-8")
            report = a9.analyze(args)
            self.assertFalse(report["runs"][0]["eligible"])
            self.assertIn("not bound", " ".join(report["runs"][0]["issues"]))

    def test_workspace_relative_hash_keys_and_validation_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args, manifests = synthetic_inputs(root)
            for exact, manifest in manifests.items():
                manifest["input_hashes"] = {str(Path(name).relative_to(root)).replace("/", "\\"): digest
                                            for name, digest in manifest["input_hashes"].items()}
                write_json(exact/"v3_manifest.json", manifest)
            validation = args.a9_enumeration_root/"validation/synthetic_A9__seed_12341.json"
            write_json(validation, dict(claim_eligible=False))
            report = a9.analyze(args)
            self.assertEqual(sum(r["eligible"] for r in report["runs"] if r["arm"] in ("A3", "A8")), 20)
            self.assertFalse(report["runs"][0]["eligible"])
            self.assertIn("claim_eligible", " ".join(report["runs"][0]["issues"]))
            self.assertEqual(report["M1"]["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
