"""Local functional contracts, never the six-real-Nano validation gate."""
import argparse
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sensys_runs.iccps27_followup import reset_probe as probe


def configuration(seed=12341, method="lio_er", name="validation/run"):
    campaign, folder = name.split("/")
    return dict(env=probe.ENVIRONMENT.copy(), nn=dict(n_h1=64, n_h2=32),
                main=dict(seed=seed, exp_name=campaign, dir_name=folder),
                lio=dict(eia_w_lever=1.1, eia_w_door=.9))


def attack():
    return dict(schema_version=1, configured_weights=dict(lever=1.1, door=.9),
                action_order=["lever", "start", "door"], exploitative_giver_id=1,
                training_active=True, evaluation_active=True, location="outside_differentiable_reward_graph",
                effective_multipliers=[dict(giver_id=g, recipient_id=r,
                    multipliers=[1.1,1,.9] if g == 1 and g != r else [1,1,1], delivered=g != r)
                    for g in range(4) for r in range(4)])


class FakeActor:
    calls = []
    def __init__(self, run, config):
        self.run, self.config = Path(run), config
        self.variable_shapes = probe.expected_actor_shapes()
        self.runtime = {"execution": "TEST_DOUBLE"}
        type(self).calls.append(str(run))

    def probabilities_at_reset(self, observations):
        np.testing.assert_array_equal(observations[:,12:], np.zeros((4,3)))
        return np.ones((4,3))/3

    def close(self):
        pass


def make_validation(root, fail_first=False):
    rows = []
    for method in ("lio_er", "refine_er"):
        for seed in probe.VALIDATION_SEEDS:
            rid = "synthetic/"+method+"_"+str(seed)
            run = root/rid
            run.mkdir(parents=True)
            (run/"model.ckpt.index").write_bytes(b"SYNTHETIC TEST INDEX")
            (run/"model.ckpt.data-00000-of-00001").write_bytes(b"SYNTHETIC TEST DATA")
            (run/"config.json").write_text(json.dumps(configuration(seed,method,rid)))
            (run/"provenance.json").write_text(json.dumps(dict(method=method,seed=seed)))
            rows.append(dict(run_id=rid,method=method,seed=seed,arm="A3" if method == "lio_er" else "A8",
                             expected_P_tau_1=22/81,reference_result_path="SYNTHETIC TEST ONLY",
                             reference_result_sha256="0"*64))
    if fail_first:
        rows[0]["expected_P_tau_1"] = .99
    table = root/"references.json"
    probe.write_json(table,dict(runs=rows,tolerance=1e-5))
    return argparse.Namespace(results_root=root,reference_table=table,output_dir=root/"validation_out")


class ResetProbeTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("gym") is None:
            # Isolate the optional gym baseline package initializer only.
            # The real room_symmetric.Env and room_agent.Actor files still run.
            package=ModuleType("lio.env")
            package.__path__=[str(probe.REPO/"lio/env")]
            guard=patch.dict(sys.modules,{"lio.env":package})
            guard.start()
            self.addCleanup(guard.stop)

    def test_static_actor_subgraphs_match_all_four_classes(self):
        pieces = []
        for name in ("lio_agent_er.py", "lio_eia_er.py", "REFiNE_er.py", "REFiNE_eia_er.py"):
            tree = ast.parse((probe.REPO/"lio/alg"/name).read_text(encoding="utf-8-sig"))
            fn = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == "create_networks")
            outer = next(n for n in fn.body if isinstance(n,ast.With))
            main = outer.body[0]
            actor = main.body[0]
            mixture = next(n for n in main.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == "self.probs")
            pieces.append((ast.dump(actor),ast.dump(mixture)))
        self.assertTrue(all(p == pieces[0] for p in pieces))
        shapes = probe.expected_actor_shapes()
        self.assertEqual(len(shapes),24)
        self.assertEqual(shapes["agent_1/policy_main/policy/actor_h1/kernel"],[15,64])
        self.assertTrue(all("/policy_main/policy/" in n for n in shapes))

    def test_reset_uses_environment_and_zero_payment_coordinates(self):
        obs = probe.reset_observations(configuration())
        np.testing.assert_array_equal(obs,np.tile([0,1,0]*4+[0,0,0],(4,1)))
        cfg = configuration()
        cfg["env"]["randomize"] = True
        with self.assertRaisesRegex(probe.ProbeError,"fixed-reset"):
            probe.reset_observations(cfg)

    def test_imported_success_mask_and_time_bounds(self):
        from finite_histories import ACTIONS, SUCCESS
        self.assertIs(probe.ACTIONS,ACTIONS)
        self.assertIs(probe.SUCCESS,SUCCESS)
        result = probe.reset_statistics(np.ones((4,3))/3)
        self.assertAlmostEqual(result["P1"],22/81)
        self.assertAlmostEqual(result["E_T_bound_width"],3*(1-22/81))
        success = np.array([[1,0,0],[1,0,0],[0,0,1],[0,1,0]])
        self.assertEqual(probe.reset_statistics(success)["E_T_lower"],1)
        failure = np.tile([0,1,0],(4,1))
        self.assertEqual(probe.reset_statistics(failure)["E_T_upper"],5)
        self.assertEqual(probe.reset_statistics(failure)["E_T_lower"],2)

    def test_invalid_probabilities_are_not_normalized_or_clipped(self):
        bad = np.ones((4,3))/2
        with self.assertRaisesRegex(probe.ProbeError,"mass"):
            probe.reset_statistics(bad)
        with self.assertRaises(probe.ProbeError):
            probe.reset_statistics(np.full((4,3),np.nan))

    def test_double_cannot_satisfy_real_six_checkpoint_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            args = make_validation(Path(folder))
            receipt = probe.validate_clean(args,backend_class=FakeActor)
            self.assertEqual(len(receipt["checks"]),6)
            self.assertEqual(receipt["status"],"UNVERIFIED_TEST_DOUBLE_ONLY")
            with self.assertRaisesRegex(probe.ProbeError,"Six real"):
                probe.require_validation(args.output_dir/"validation_receipt.json",args.results_root,{"execution":"TEST_DOUBLE"})

    def test_validation_first_failure_stops_without_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            args = make_validation(Path(folder),fail_first=True)
            receipt = probe.validate_clean(args,backend_class=FakeActor)
            self.assertEqual(receipt["status"],"FAIL")
            self.assertEqual(len(receipt["checks"]),1)
            self.assertNotIn("excluded",receipt)

    def test_true_eia_identity_and_effective_provenance_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); args=make_validation(root)
            rid="synthetic/lio_er_12341"; run=root/rid
            cfg=configuration(12341,"lio_eia_er",rid)
            cfg["eia_attack"]=attack()
            (run/"config.json").write_text(json.dumps(cfg))
            (run/"provenance.json").write_text(json.dumps(dict(method="lio_eia_er",seed=12341,eia_attack=attack())))
            result=probe.probe_one(root,dict(run_id=rid,method="lio_eia_er",seed=12341),eia=True,backend_class=FakeActor)
            self.assertEqual(result["method"],"lio_eia_er")
            self.assertEqual(result["eia_attack"],attack())
            self.assertTrue(result["no_payment_query"])
            self.assertNotIn("payments",result)
            with self.assertRaisesRegex(probe.ProbeError,"identity"):
                probe.probe_one(root,dict(run_id=rid,method="lio_er",seed=12341),eia=True,backend_class=FakeActor)

    def test_recorded_effective_attack_mismatch_is_rejected(self):
        cfg=configuration(); cfg["eia_attack"]=attack()
        provenance=dict(eia_attack=attack())
        provenance["eia_attack"]["effective_multipliers"][4]["multipliers"][1]=1.5
        with self.assertRaisesRegex(probe.ProbeError,"records differ"):
            probe.verify_attack_provenance(cfg,provenance)

    def test_receipt_source_and_checkpoint_bindings_are_rechecked(self):
        identity = dict(commit="SYNTHETIC_TEST_ONLY", blobs={})
        identity_guard = patch.object(probe, "source_git_identity", return_value=identity)
        identity_guard.start()
        self.addCleanup(identity_guard.stop)
        with tempfile.TemporaryDirectory() as folder:
            args=make_validation(Path(folder))
            receipt=probe.validate_clean(args,backend_class=FakeActor)
            # Synthetic record mutation tests the reader only, never authorizes a real probe.
            receipt["status"]="PASS"; receipt["execution_kind"]="actual_tensorflow_checkpoint_restore"
            receipt["source_git_identity"] = identity
            path=Path(folder)/"synthetic_receipt_reader_fixture.json"
            probe.write_json(path,receipt)
            probe.require_validation(path,args.results_root,{"execution":"TEST_DOUBLE"})
            first=receipt["expected"][0]["run_id"]
            shard=args.results_root/first/"model.ckpt.data-00000-of-00001"
            shard.write_bytes(b"CHANGED SYNTHETIC TEST DATA")
            with self.assertRaisesRegex(probe.ProbeError,"checkpoint/config/provenance changed"):
                probe.require_validation(path,args.results_root,{"execution":"TEST_DOUBLE"})
            receipt["source_hashes"]["sensys_runs/iccps27_followup/reset_probe.py"]="0"*64
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(probe.ProbeError,"source hashes"):
                probe.require_validation(path,args.results_root,{"execution":"TEST_DOUBLE"})

    def test_git_identity_uses_read_only_canonical_blobs_and_rejects_dirty_source(self):
        commands = []
        def fake_run(command, **kwargs):
            commands.append(command)
            value = "a"*40 if command[-1] == "HEAD" else "b"*40
            return SimpleNamespace(returncode=0, stdout=value+"\n", stderr="")
        with patch.object(probe.subprocess, "run", side_effect=fake_run):
            identity = probe.source_git_identity()
        self.assertEqual(identity["commit"], "a"*40)
        self.assertEqual(set(identity["blobs"]), set(probe.SOURCE_FILES)-set(probe.PROTECTED_PINS))
        self.assertEqual(identity["protected_sha256"], probe.PROTECTED_PINS)
        self.assertEqual(identity["protected_manifest_sha256"], probe.PROTECTED_MANIFEST_SHA256)
        self.assertFalse(any("HEAD:"+name in command for name in probe.PROTECTED_PINS for command in commands))
        self.assertTrue(all(command[:3] == ["git", "-C", str(probe.REPO)] for command in commands))
        self.assertTrue(all("-w" not in command for command in commands))
        def changed_run(command, **kwargs):
            result = fake_run(command, **kwargs)
            if "hash-object" in command:
                result.stdout = "c"*40+"\n"
            return result
        with patch.object(probe.subprocess, "run", side_effect=changed_run):
            with self.assertRaisesRegex(probe.ProbeError, "committed canonical Git blob"):
                probe.source_git_identity()

    def test_missing_or_changed_protected_source_is_rejected_without_git(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(probe, "REPO", Path(folder)), patch.object(probe.subprocess, "run") as git:
                with self.assertRaisesRegex(ValueError, "Protected source missing"):
                    probe.source_git_identity()
                first = Path(folder)/next(iter(probe.PROTECTED_PINS))
                first.parent.mkdir(parents=True)
                first.write_bytes(b"CHANGED SYNTHETIC PROTECTED SOURCE")
                with self.assertRaisesRegex(ValueError, "approved G2 SHA256"):
                    probe.source_git_identity()
                git.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"),"TensorFlow unavailable; real actor restore remains unverified locally")
    def test_real_tf_actor_only_matches_original_classes_on_synthetic_checkpoints(self):
        # Windows functional test only. This never creates a six-clean validation receipt.
        tf,_=probe.tensorflow_runtime()
        from lio.alg.config_room_lio import get_config
        from lio.alg.config_room_REFiNE import get_config as refine_config
        variants=(("lio_agent_er","LIO",get_config),("lio_eia_er","ExploitativeLIO",get_config),
                  ("REFiNE_er","REFiNE",refine_config),("REFiNE_eia_er","REFiNEExploitative",refine_config))
        for module,name,config_factory in variants:
            with self.subTest(method=module), tempfile.TemporaryDirectory() as folder:
                cfg=config_factory(); cfg.env.update(probe.ENVIRONMENT); cfg.lio.min_at_lever=2
                cls=getattr(importlib.import_module("lio.alg."+module),name)
                graph=tf.Graph()
                with graph.as_default():
                    tf.set_random_seed(12341)
                    agents=[cls(cfg.lio,15,3,cfg.nn,"agent_%d"%i,2.,4,i,1.) for i in range(4)]
                    init=tf.global_variables_initializer(); saver=tf.train.Saver()
                obs=probe.reset_observations(cfg)
                with tf.Session(graph=graph,config=tf.ConfigProto(device_count={"GPU":0},intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)) as sess:
                    sess.run(init)
                    feed={a.obs:np.asarray([obs[i]]) for i,a in enumerate(agents)}
                    feed.update({a.epsilon:0. for a in agents})
                    expected=np.asarray(sess.run([a.probs for a in agents],feed_dict=feed))[:,0,:]
                    saver.save(sess,str(Path(folder)/"model.ckpt"))
                restored=probe.TFActorOnly(folder,cfg)
                try:
                    actual=restored.probabilities_at_reset(obs)
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(restored.variable_shapes,probe.expected_actor_shapes())
                finally:
                    restored.close()


if __name__ == "__main__":
    unittest.main()
