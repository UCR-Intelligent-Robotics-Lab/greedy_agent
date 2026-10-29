"""Read-only actor reset probe with a mandatory six-clean-checkpoint gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "sensys_runs/iccps27_analysis"
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
# Approved G2 source: W/iccps27_analysis_manifest_20261028.txt, rows 2, 12, 15.
# Protected files remain outside Git. Never normalize them or require a commit.
PROTECTED_MANIFEST_SHA256 = "3032a99d7b0ba38ffe44602001b9c7a58c709bd73484cd9b509a6c37ca4d8e18"
PROTECTED_PINS = {
    "sensys_runs/iccps27_analysis/analysis_core.py": "ecff94bb9035515ff856f9cfb7df75130eda63f3ec1a8871962a9f5962192ba0",
    "sensys_runs/iccps27_analysis/finite_histories.py": "00fbc5c6c388eea8bcbffb37209852deb7101dfffd81356e9b304aa04d0e38f7",
    "sensys_runs/iccps27_analysis/numpy_policy.py": "7ff03e0272afcbe29c4720acfcdfb3efb9544877c22faa85226bec3071dabe2d",
}


def verified_protected_hashes():
    for name, expected in PROTECTED_PINS.items():
        path = REPO / name
        if not path.is_file():
            raise ValueError("Protected source missing: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Protected source differs from approved G2 SHA256: " + name)
    return PROTECTED_PINS.copy()


verified_protected_hashes()
# Import the existing action catalog and success mask; never duplicate the rule.
from finite_histories import ACTIONS, SUCCESS

WORDING = "computed exactly from the deployed policy at the reset input, in floating point"
VALIDATION_SEEDS = (12341, 12342, 12343)
CLEAN_ARMS = {
    "A3": "lio_er__uniform_v1__homogeneous_v1__env_reward",
    "A8": "refine_er__uniform_v1__homogeneous_v1__env_reward",
}
SOURCE_FILES = (
    "sensys_runs/iccps27_followup/reset_probe.py",
    "sensys_runs/iccps27_analysis/finite_histories.py",
    "sensys_runs/iccps27_analysis/analysis_core.py",
    "sensys_runs/iccps27_analysis/numpy_policy.py",
    "lio/alg/networks.py", "lio/env/room_symmetric.py", "lio/env/room_agent.py",
    "lio/utils/configdict.py",
)
ENVIRONMENT = dict(max_steps=5, min_at_lever=2, n_agents=4, name="er",
                   r_multiplier=2.0, randomize=False, reward_sanity_check=False)


class ProbeError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def safe_id(rid):
    parts = PurePosixPath(rid).parts
    if len(parts) != 2 or "\\" in rid or ":" in rid or rid.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ProbeError("Unsafe run_id")
    return rid


def source_hashes():
    return {name: sha(REPO / name) for name in SOURCE_FILES}


def source_git_identity():
    """Bind canonical Git blobs across platforms while retaining raw local hashes."""
    protected = verified_protected_hashes()
    def read(*arguments):
        result = subprocess.run(["git", "-C", str(REPO), *arguments],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise ProbeError("Final committed source identity is unavailable: " + result.stderr.strip())
        return result.stdout.strip()
    commit = read("rev-parse", "HEAD")
    blobs = {}
    for name in SOURCE_FILES:
        if name in protected:
            continue
        blob = read("rev-parse", "HEAD:" + name)
        working = read("hash-object", "--path=" + name, str(REPO / name))
        if working != blob:
            raise ProbeError("Working source differs from its committed canonical Git blob: " + name)
        blobs[name] = blob
    if read("rev-parse", "HEAD") != commit:
        raise ProbeError("Git HEAD changed while recording source identity")
    return dict(commit=commit, blobs=blobs, protected_sha256=protected,
                protected_manifest_sha256=PROTECTED_MANIFEST_SHA256)


def checkpoint_hashes(run):
    files = sorted(Path(run).glob("model.ckpt.*"))
    if not (Path(run)/"model.ckpt.index").is_file() or not any(p.name.startswith("model.ckpt.data-") for p in files):
        raise ProbeError("Missing checkpoint index or data shards: " + str(run))
    return {p.name: sha(p) for p in files}


def expected_actor_shapes(n_h1=64, n_h2=32):
    shapes = {}
    for agent in range(4):
        prefix = f"agent_{agent}/policy_main/policy/"
        for layer, left, right in (("actor_h1", 15, n_h1), ("actor_h2", n_h1, n_h2), ("actor_out", n_h2, 3)):
            shapes[prefix+layer+"/kernel"] = [left, right]
            shapes[prefix+layer+"/bias"] = [right]
    return shapes


def reset_observations(config):
    """Use exactly the environment.reset entry point called by evaluate.test_room."""
    if config.get("env") != ENVIRONMENT:
        raise ProbeError("Reset probe is registered only for fixed-reset ER(4,2), H=5")
    from lio.env.room_symmetric import Env
    from lio.utils.configdict import ConfigDict
    env = Env(ConfigDict(config["env"]))
    observations = np.asarray(env.reset())
    if observations.shape != (4, 15) or not np.array_equal(observations[:, 12:], np.zeros((4, 3))):
        raise ProbeError("Reset observation includes nonzero payment memory or unexpected dimensions")
    if env.state != [1, 1, 1, 1]:
        raise ProbeError("Unexpected reset positions")
    return observations


def reset_statistics(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (4, 3) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ProbeError("Invalid actor probabilities")
    masses = p.sum(axis=1)
    if np.max(np.abs(masses-1)) > 1e-6:
        raise ProbeError("Actor probability mass differs from one")
    joint = np.ones(len(ACTIONS), dtype=np.float64)
    for agent in range(4):
        joint *= p[agent, ACTIONS[:, agent]]
    p1 = math.fsum(joint[SUCCESS].tolist())
    if not 0 <= p1 <= 1:
        raise ProbeError("Reset success probability outside [0,1]; no silent clipping")
    lower, upper = 2-p1, 1+4*(1-p1)
    return dict(P1=p1, E_T_lower=lower, E_T_upper=upper, E_T_bound_width=upper-lower,
                actor_probability_mass=masses.tolist(), joint_probability_mass=float(joint.sum()),
                reset_success_action_count=int(np.count_nonzero(SUCCESS)),
                epsilon=0.0, wording=WORDING,
                bound_definition="For capped T in [1,5], P(T=1)=P1 gives 2-P1 <= E[T] <= 1+4*(1-P1).",
                bounds_are_not_an_exact_E_T=True)


def tensorflow_runtime():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    return tf, dict(tensorflow=tf.__version__, numpy=np.__version__, python=platform.python_version(),
                    device="CPU", intra_op_threads=1, inter_op_threads=1, onednn="0")


class TFActorOnly:
    """Build the shared actor_mlp and restore only its 24 main-actor variables."""
    def __init__(self, run, config, *, tf_module=None, networks_module=None):
        if config.get("nn", {}).get("n_h1") != 64 or config.get("nn", {}).get("n_h2") != 32:
            raise ProbeError("Unexpected actor layer sizes")
        self.run = Path(run)
        if tf_module is None:
            tf, self.runtime = tensorflow_runtime()
        else:
            tf, self.runtime = tf_module, {"execution": "TEST_DOUBLE_NOT_A_REAL_CHECKPOINT_VALIDATION"}
        if networks_module is None:
            from lio.alg import networks as networks_module
        self.graph = tf.Graph()
        self.observations, self.epsilons, self.probabilities = [], [], []
        self.session = None
        with self.graph.as_default():
            for agent in range(4):
                obs = tf.placeholder(tf.float32, [None, 15], "l_obs")
                epsilon = tf.placeholder(tf.float32, None, "epsilon")
                with tf.variable_scope(f"agent_{agent}"):
                    with tf.variable_scope("policy_main"):
                        with tf.variable_scope("policy"):
                            probs = networks_module.actor_mlp(obs, 3, SimpleNamespace(**config["nn"]))
                        probs = (1-epsilon)*probs + epsilon/3
                self.observations.append(obs)
                self.epsilons.append(epsilon)
                self.probabilities.append(probs)
            variables = tf.global_variables()
            expected = expected_actor_shapes()
            actual = {v.name.removesuffix(":0"): v.shape.as_list() for v in variables}
            if actual != expected:
                raise ProbeError("Actor-only graph names/shapes differ from the static gate")
            stored = dict(tf.train.list_variables(str(self.run/"model.ckpt")))
            if any(stored.get(name) != shape for name, shape in expected.items()):
                raise ProbeError("Checkpoint actor name/shape mismatch; no fallback restore")
            saver = tf.train.Saver(var_list=variables)
            self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(device_count={"GPU": 0},
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
            try:
                saver.restore(self.session, str(self.run/"model.ckpt"))
            except Exception:
                self.session.close()
                raise
        self.variable_shapes = actual

    def probabilities_at_reset(self, observations):
        feed = {}
        for agent in range(4):
            feed[self.observations[agent]] = np.asarray([observations[agent]])
            feed[self.epsilons[agent]] = 0.0
        return np.asarray(self.session.run(self.probabilities, feed_dict=feed))[:, 0, :]

    def close(self):
        if self.session is not None:
            self.session.close()


def verify_attack_provenance(config, provenance):
    attack = provenance.get("eia_attack")
    if not isinstance(attack, dict) or attack.get("schema_version") != 1:
        raise ProbeError("Missing runtime effective EIA provenance")
    if config.get("eia_attack") != attack:
        raise ProbeError("Config and provenance effective EIA records differ")
    if attack.get("action_order") != ["lever", "start", "door"] or attack.get("exploitative_giver_id") != 1:
        raise ProbeError("Unexpected EIA agent/action provenance")
    if attack.get("training_active") is not True or attack.get("evaluation_active") is not True:
        raise ProbeError("Expected EIA active in both training and evaluation")
    weights = attack.get("configured_weights", {})
    if weights != {"lever": config["lio"]["eia_w_lever"], "door": config["lio"]["eia_w_door"]}:
        raise ProbeError("Configured EIA weights disagree with runtime provenance")
    effective = attack.get("effective_multipliers", [])
    if len(effective) != 16 or {(r["giver_id"], r["recipient_id"]) for r in effective} != {(g,r) for g in range(4) for r in range(4)}:
        raise ProbeError("Incomplete per-giver/per-recipient effective EIA provenance")
    for record in effective:
        if len(record["multipliers"]) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in record["multipliers"]):
            raise ProbeError("Invalid recorded effective multipliers")
        giver, recipient = record["giver_id"], record["recipient_id"]
        expected = [weights["lever"], 1.0, weights["door"]] if giver == 1 and giver != recipient else [1.0, 1.0, 1.0]
        if record["multipliers"] != expected or record.get("delivered") is not (giver != recipient):
            raise ProbeError("Runtime effective EIA record disagrees with configured agent/action semantics")
    return attack


def probe_one(results_root, row, *, eia, backend_class=TFActorOnly):
    rid = safe_id(row["run_id"])
    run = Path(results_root)/rid
    config, provenance = json_read(run/"config.json"), json_read(run/"provenance.json")
    method = row["method"]
    allowed = ("lio_eia_er", "refine_eia_er") if eia else ("lio_er", "refine_er")
    if method not in allowed or provenance.get("method") != method or int(provenance["seed"]) != int(row["seed"]) or int(config["main"]["seed"]) != int(row["seed"]):
        raise ProbeError("True method or seed identity mismatch")
    if config["main"]["exp_name"]+"/"+config["main"]["dir_name"] != rid:
        raise ProbeError("Config/run identity mismatch")
    attack = verify_attack_provenance(config, provenance) if eia else None
    before = {name: sha(run/name) for name in ("config.json", "provenance.json")}
    checkpoints = checkpoint_hashes(run)
    obs = reset_observations(config)
    backend = backend_class(run, config)
    try:
        probabilities = backend.probabilities_at_reset(obs)
        result = reset_statistics(probabilities)
        result.update(run_id=rid, method=method, seed=int(row["seed"]), eia_attack=attack,
                      reset_observations=obs.tolist(), actor_probabilities=np.asarray(probabilities).tolist(),
                      actor_variable_shapes=backend.variable_shapes, runtime=backend.runtime,
                      checkpoint_hashes=checkpoints, config_provenance_hashes=before,
                      no_payment_query=True, no_training_or_optimizer=True,
                      execution_kind="actual_tensorflow_checkpoint_restore" if backend_class is TFActorOnly else "TEST_DOUBLE",
                      evidence={"checkpoint": str(run/"model.ckpt"), "config": str(run/"config.json"),
                                "provenance": str(run/"provenance.json"), "success_rule": "finite_histories.ACTIONS/SUCCESS",
                                "reset": "lio/env/room_symmetric.py:89-103", "actor": "lio/alg/networks.py:106-115"})
    finally:
        backend.close()
    if checkpoint_hashes(run) != checkpoints or any(sha(run/name) != digest for name, digest in before.items()):
        raise ProbeError("Checkpoint/config/provenance changed during read-only probe")
    return result


def prepare_validation(args):
    with Path(args.reference_inventory).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for label, arm in CLEAN_ARMS.items():
        chosen = sorted((r for r in rows if r["arm"] == arm and int(r["seed"]) in VALIDATION_SEEDS), key=lambda r:int(r["seed"]))
        if [int(r["seed"]) for r in chosen] != list(VALIDATION_SEEDS):
            raise ProbeError("Clean reference inventory lacks exact registered validation seeds")
        for row in chosen:
            path = Path(args.reference_enumeration_root)/safe_id(row["run_id"]).replace("/", "__")/"result.json"
            result = json_read(path)
            if result.get("node_count") != 12326281 or len(result.get("P_tau", [])) != 5:
                raise ProbeError("Expected stored full clean-reference results")
            selected.append(dict(run_id=row["run_id"], method=row["method"], seed=int(row["seed"]),
                                 arm=label, expected_P_tau_1=result["P_tau"][0],
                                 reference_result_path=str(path.resolve()), reference_result_sha256=sha(path)))
    path = Path(args.output_file)
    if path.exists():
        raise ProbeError("Reference table must be a new file")
    table = dict(schema_version=1, role="six frozen clean reset validation references", runs=selected,
                 inventory_sha256=sha(args.reference_inventory), tolerance=1e-5)
    write_json(path, table)
    return table


def check_reference_table(table):
    rows = table.get("runs", [])
    expected = {(m, s) for m in ("lio_er", "refine_er") for s in VALIDATION_SEEDS}
    if len(rows) != 6 or {(r["method"], int(r["seed"])) for r in rows} != expected or table.get("tolerance") != 1e-5:
        raise ProbeError("Validation requires exactly 3 A3 and 3 A8 fixed seeds at tolerance 1e-5")
    if len({r["run_id"] for r in rows}) != 6 or any(not 0 <= r["expected_P_tau_1"] <= 1 or len(r["reference_result_sha256"]) != 64 for r in rows):
        raise ProbeError("Invalid frozen expected-value references")
    if any(r.get("arm") != ("A3" if r["method"] == "lio_er" else "A8") for r in rows):
        raise ProbeError("Clean reference arm labels disagree with actual methods")
    return rows


def validate_clean(args, backend_class=TFActorOnly):
    table = json_read(args.reference_table)
    rows = check_reference_table(table)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    receipt = dict(schema_version=1, status="INCOMPLETE", tolerance=1e-5, expected=rows, checks=[],
                   reference_table=str(Path(args.reference_table).resolve()), reference_table_sha256=sha(args.reference_table),
                   source_hashes=sources, results_root=str(Path(args.results_root).resolve()),
                   execution_kind="actual_tensorflow_checkpoint_restore" if backend_class is TFActorOnly else "TEST_DOUBLE")
    receipt["source_git_identity"] = source_git_identity() if backend_class is TFActorOnly else None
    try:
        for row in rows:
            result = probe_one(args.results_root, row, eia=False, backend_class=backend_class)
            delta = abs(result["P1"]-row["expected_P_tau_1"])
            check = dict(run_id=row["run_id"], method=row["method"], seed=row["seed"], expected_P_tau_1=row["expected_P_tau_1"],
                         P1=result["P1"], absolute_difference=delta, passed=delta <= 1e-5,
                         checkpoint_hashes=result["checkpoint_hashes"], config_provenance_hashes=result["config_provenance_hashes"], runtime=result["runtime"])
            receipt["checks"].append(check)
            write_json(out/(row["run_id"].replace("/", "__")+".json"), result)
            if not check["passed"]:
                raise ProbeError("Clean reset validation failed; stop with no seed exclusion: " + row["run_id"])
        if source_hashes() != sources or sha(args.reference_table) != receipt["reference_table_sha256"]:
            raise ProbeError("Validation source or reference changed")
        if backend_class is TFActorOnly and source_git_identity() != receipt["source_git_identity"]:
            raise ProbeError("Validation Git source identity changed")
        receipt["status"] = "PASS" if backend_class is TFActorOnly else "UNVERIFIED_TEST_DOUBLE_ONLY"
    except Exception as exc:
        receipt.update(status="FAIL", error=str(exc))
    write_json(out/"validation_receipt.json", receipt)
    return receipt


def require_validation(path, results_root, runtime):
    receipt = json_read(path)
    expected = check_reference_table(dict(runs=receipt.get("expected", []), tolerance=receipt.get("tolerance")))
    if receipt.get("status") != "PASS" or receipt.get("execution_kind") != "actual_tensorflow_checkpoint_restore" or len(receipt.get("checks", [])) != 6:
        raise ProbeError("Six real clean TF checkpoint validations must pass before any EIA probe")
    if receipt.get("source_hashes") != source_hashes():
        raise ProbeError("Validation receipt source hashes no longer match")
    if receipt.get("source_git_identity") != source_git_identity():
        raise ProbeError("Validation receipt canonical Git source identity no longer matches")
    reference_path = Path(receipt["reference_table"])
    if sha(reference_path) != receipt["reference_table_sha256"] or json_read(reference_path)["runs"] != expected:
        raise ProbeError("Validation reference table changed")
    checks = {r["run_id"]: r for r in receipt["checks"]}
    if len(checks) != 6:
        raise ProbeError("Duplicate clean validation record")
    for row in expected:
        check = checks.get(row["run_id"], {})
        if (check.get("method") != row["method"] or check.get("seed") != row["seed"] or
                check.get("expected_P_tau_1") != row["expected_P_tau_1"] or check.get("passed") is not True or
                not math.isfinite(check.get("absolute_difference", math.inf)) or check["absolute_difference"] > 1e-5 or
                abs(check["P1"]-row["expected_P_tau_1"]) != check["absolute_difference"] or check.get("runtime") != runtime):
            raise ProbeError("Validation identity, runtime, or error criterion mismatch")
        run = Path(results_root)/safe_id(row["run_id"])
        if checkpoint_hashes(run) != check["checkpoint_hashes"] or any(sha(run/name) != digest for name,digest in check["config_provenance_hashes"].items()):
            raise ProbeError("Clean validation checkpoint/config/provenance changed")
    return receipt


def probe_batch(args):
    _, runtime = tensorflow_runtime()
    require_validation(args.validation_receipt, args.results_root, runtime)
    with Path(args.inventory).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r["run_id"] for r in rows}) != len(rows):
        raise ProbeError("Empty or duplicate EIA inventory")
    if any(r["method"] not in ("lio_eia_er", "refine_eia_er") or int(r["seed"]) not in range(12341,12351) for r in rows):
        raise ProbeError("Probe inventory must retain true EIA methods and registered seeds")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    inputs = {str(Path(p).resolve()): sha(p) for p in (args.inventory, args.validation_receipt)}
    manifest = dict(schema_version=1, wording=WORDING, source_hashes=sources, input_hashes=inputs,
                    completed=[], failed=[], pending=[r["run_id"] for r in rows],
                    validation_receipt_sha256=sha(args.validation_receipt), runtime=runtime)
    manifest["source_git_identity"] = source_git_identity()
    for row in rows:
        rid = safe_id(row["run_id"])
        try:
            result = probe_one(args.results_root, row, eia=True)
            result["validation_receipt_sha256"] = manifest["validation_receipt_sha256"]
            filename = rid.replace("/", "__")+".json"
            write_json(out/filename, result)
            manifest["completed"].append(dict(run_id=rid, file=filename, P1=result["P1"], E_T_lower=result["E_T_lower"], E_T_upper=result["E_T_upper"], E_T_bound_width=result["E_T_bound_width"]))
        except Exception as exc:
            manifest["failed"].append(dict(run_id=rid, method=row["method"], seed=int(row["seed"]), error=str(exc)))
        manifest["pending"].remove(rid)
    manifest["sources_unchanged"] = source_hashes() == sources
    manifest["sources_unchanged"] = manifest["sources_unchanged"] and source_git_identity() == manifest["source_git_identity"]
    manifest["inputs_unchanged"] = all(sha(p) == d for p,d in inputs.items())
    manifest["status"] = "COMPLETE" if not manifest["failed"] and manifest["sources_unchanged"] and manifest["inputs_unchanged"] else "INCOMPLETE"
    write_json(out/"manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-validation")
    for name in ("reference-inventory", "reference-enumeration-root", "output-file"):
        prepare.add_argument("--"+name, type=Path, required=True)
    validate = commands.add_parser("validate-clean")
    for name in ("results-root", "reference-table", "output-dir"):
        validate.add_argument("--"+name, type=Path, required=True)
    probe = commands.add_parser("probe")
    for name in ("results-root", "inventory", "validation-receipt", "output-dir"):
        probe.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = {"prepare-validation": prepare_validation, "validate-clean": validate_clean, "probe": probe_batch}[args.command](args)
        print(json.dumps({"command":args.command, "status":result.get("status", "PREPARED"), "output":str(getattr(args,"output_dir",getattr(args,"output_file","")))}))
        return 0 if result.get("status", "PREPARED") in ("PASS", "COMPLETE", "PREPARED") else 2
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, "RESET_PROBE_ERROR: "+str(exc)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())
