"""Local inventory/identity regressions without TensorFlow or policy inference."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline_common import A9_ARM, FIELDS, check_identity, load_inventory
from sensys_runs.energy_profiles import get_profile, get_hardware_profile
from enumerate_a9 import preflight as pc_preflight
from export_a9 import main as export_main


def row(seed=12341):
    rid = "a9_test/lio_er_uniform_v1_homogeneous_v1_energy_reward_s%d" % seed
    return dict(run_id=rid, arm=A9_ARM, method="lio_er", seed=str(seed),
                checkpoint_prefix="lio/results/" + rid + "/model.ckpt", evidence="test fixture")


def write_inventory(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class InventoryContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = self.root / "inventory.csv"
        write_inventory(self.inventory, [row()])

    def test_explicit_count_rejects_missing_seed(self):
        with self.assertRaisesRegex(ValueError, "count"):
            load_inventory(self.inventory, 10)

    def test_duplicate_run_identity_rejected(self):
        write_inventory(self.inventory, [row(), row()])
        with self.assertRaisesRegex(ValueError, "identity"):
            load_inventory(self.inventory, 2)

    def test_wrong_objective_arm_rejected(self):
        changed = row()
        changed["arm"] = changed["arm"].replace("energy_reward", "env_reward")
        write_inventory(self.inventory, [changed])
        with self.assertRaisesRegex(ValueError, "A9"):
            load_inventory(self.inventory, 1)

    def test_checkpoint_alias_and_path_traversal_rejected(self):
        for field, value in (("checkpoint_prefix", "lio/results/other/model.ckpt"),
                             ("run_id", "../elsewhere")):
            with self.subTest(field=field):
                changed = row()
                changed[field] = value
                write_inventory(self.inventory, [changed])
                with self.assertRaises(ValueError):
                    load_inventory(self.inventory, 1)

    def test_config_and_provenance_must_bind_same_seed_and_profile(self):
        selected = row()
        common = dict(cost_profile=json.loads(json.dumps(get_profile("uniform_v1").snapshot())),
                      hardware_profile=json.loads(json.dumps(get_hardware_profile("homogeneous_v1").snapshot(4))),
                      hardware_profile_lambda=[1.0] * 4, energy_in_reward=True,
                      energy_reward_beta=1.0, energy_reward_form="episode_mean")
        config = dict(copy.deepcopy(common), main=dict(exp_name="a9_test", dir_name=selected["run_id"].split("/")[1], seed=12341),
                      lio={"Fairness_multiplier": 0.0})
        provenance = dict(copy.deepcopy(common), method="lio_er", seed=12341)
        check_identity(selected, config, provenance)
        bad = copy.deepcopy(provenance)
        bad["seed"] = 12342
        with self.assertRaisesRegex(ValueError, "identity"):
            check_identity(selected, config, bad)
        bad = copy.deepcopy(provenance)
        bad["cost_profile"] = json.loads(json.dumps(get_profile("paper_ratio_v1").snapshot()))
        with self.assertRaisesRegex(ValueError, "profile"):
            check_identity(selected, config, bad)

    def test_export_wrong_count_stops_before_inference_or_output(self):
        import pipeline_common
        target = self.root / "new_export"
        with patch("export_a9.export_one", side_effect=AssertionError("Inference must not run")):
            with self.assertRaisesRegex(ValueError, "count"):
                export_main(["--repo-root", str(pipeline_common.REPO), "--results-root", str(self.root),
                             "--inventory", str(self.inventory), "--expected-count", "10",
                             "--output-dir", str(target)])
        self.assertFalse(target.exists())

    def test_subset_cannot_relabel_a_frozen_row(self):
        from pipeline_common import ANALYSIS, sha
        from parity_v6 import CRITERION
        source = self.root / "source.csv"
        source_row = row()
        source_row["arm"] = "lio_er__uniform_v1__homogeneous_v1__env_reward"
        write_inventory(source, [source_row])
        export = self.root / "export"
        export.mkdir()
        source_file = ANALYSIS / "numpy_policy.py"
        manifest = dict(smoke=False, stub=False, acceptance_criterion=CRITERION,
                        source_hashes_unchanged=True,
                        source_hashes={str(source_file): sha(source_file)},
                        inventory_sha256=sha(source), expected_count=1)
        (export / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "differs from original"):
            pc_preflight(export, self.root, self.inventory, 1, source)

    def test_smoke_export_cannot_enter_pc_pipeline(self):
        export = self.root / "smoke_export"
        export.mkdir()
        (export / "manifest.json").write_text(json.dumps({"smoke": True}))
        with self.assertRaisesRegex(ValueError, "completed real"):
            pc_preflight(export, self.root, self.inventory, 1)


if __name__ == "__main__":
    unittest.main()
