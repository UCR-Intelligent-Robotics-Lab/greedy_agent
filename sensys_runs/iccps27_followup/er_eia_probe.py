"""ER-LIO+EIA reset probes using the unchanged validated actor-only engine."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from lio.utils.configdict import ConfigDict
from sensys_runs.iccps27_followup import er_eia_adapter as adapter
from sensys_runs.iccps27_followup import reset_probe as rp
from sensys_runs.iccps27_followup.er_eia_campaign import RELEASE_TAG
from sensys_runs.iccps27_followup.source_identity_v2 import verified_source_identity, source_identity_scope

CAMPAIGN = "iccps27_er_lio_eia_20261029"
INFERENCE_REF = "4ce4341d867cba0ceb4836e4e288a3c19812a0ce"
WEIGHTS = ((2.0, 0.2), (1.1, 0.9), (1.5, 1.5))
SEEDS = tuple(range(12341, 12351))
WRAPPER_FILES = tuple("sensys_runs/iccps27_followup/" + name for name in
    ("er_eia_probe.py", "source_identity_v2.py", "er_eia_adapter.py",
     "er_eia_campaign.py", "er_eia_runner.py"))
WRAPPER_FILES = tuple(dict.fromkeys((*WRAPPER_FILES, *adapter.source_hashes())))
ENERGY_FIELDS = dict(energy_in_reward=True, energy_reward_beta=1.0,
                    energy_reward_form="episode_mean", followup_arm=adapter.ARM,
                    training_objective=adapter.OBJECTIVE, display_label=adapter.DISPLAY_LABEL)


def require(condition, message):
    if not condition:
        raise rp.ProbeError(message)


def git(*arguments):
    result = subprocess.run(["git", "-C", str(REPO), *arguments],
                            capture_output=True, text=True, check=False)
    require(result.returncode == 0, "Release Git gate failed: " + result.stderr.strip())
    return result.stdout.strip()


def validate_energy(config):
    require(all(config.get(key) == value for key, value in ENERGY_FIELDS.items())
            and config.get("energy_in_reward") is True,
            "ER-LIO+EIA energy objective/config mismatch")
    adapter.validate_energy_config(ConfigDict(deepcopy(config)))
    require(config.get("env") == rp.ENVIRONMENT, "ER-LIO+EIA fixed reset environment mismatch")
    require(config.get("cost_profile", {}).get("profile_id") == "uniform_v1"
            and config.get("hardware_profile", {}).get("profile_id") == "homogeneous_v1"
            and config.get("hardware_profile_lambda") == [1.0] * 4,
            "ER-LIO+EIA requires uniform homogeneous costs")


def validate_matrix(manifest, inventory, expected_count):
    """Reject every incomplete or extra cell before any actor is constructed."""
    require(expected_count == 30, "The registered ER-LIO+EIA probe count is exactly 30")
    require(manifest.get("campaign") == CAMPAIGN and manifest.get("mode") == "campaign",
            "Only the registered formal ER-LIO+EIA campaign can be probed")
    require(manifest.get("n_runs") == manifest.get("planned_formal_runs") == 30
            and len(manifest.get("runs", [])) == 30
            and manifest.get("n_seeds_per_arm") == 10
            and manifest.get("methods") == [adapter.METHOD]
            and manifest.get("weight_pairs") == [list(pair) for pair in WEIGHTS],
            "The ER-LIO+EIA matrix must contain all 30 registered runs")
    expected = {(pair, seed) for pair in WEIGHTS for seed in SEEDS}
    seen, records = set(), {}
    for row in manifest["runs"]:
        pair = tuple(row.get("weight_pair", []))
        seed = row.get("seed")
        key = pair, seed
        require(type(seed) is int and key in expected and key not in seen,
                "Missing, duplicate or unregistered ER-LIO+EIA seed/weight")
        require(row.get("method") == adapter.METHOD and row.get("objective") == adapter.OBJECTIVE
                and row.get("followup_arm") == adapter.ARM,
                "Manifest row must retain the ER energy objective and true base method")
        require(row.get("weight") == ",".join(str(float(v)) for v in pair), "Weight label mismatch")
        config = row["effective_config"]
        validate_energy(config)
        rid = rp.safe_id(config["main"]["exp_name"] + "/" + config["main"]["dir_name"])
        require(config["main"]["exp_name"] == CAMPAIGN and config["main"]["seed"] == seed
                and "energy_reward" in config["main"]["dir_name"] and rid not in records,
                "ER-LIO+EIA run/config identity mismatch")
        require(config["lio"].get("eia_w_lever") == pair[0]
                and config["lio"].get("eia_w_door") == pair[1], "ER-LIO+EIA weight/config mismatch")
        require(config["alg"].get("n_episodes") == 25000 and config["alg"].get("n_eval") == 10
                and config["alg"].get("period") == 500, "Formal episode settings mismatch")
        seen.add(key)
        records[rid] = row
    require(seen == expected, "Incomplete ER-LIO+EIA matrix")
    require(len(inventory) == 30 and len({row["run_id"] for row in inventory}) == 30,
            "Missing, extra or duplicate probe inventory row")
    require({rp.safe_id(row["run_id"]) for row in inventory} == set(records),
            "Probe inventory does not match the complete campaign")
    for row in inventory:
        expected_row = records[row["run_id"]]
        require(row["method"] == adapter.METHOD and int(row["seed"]) == expected_row["seed"]
                and row["arm"] == expected_row["arm"]
                and row["checkpoint_prefix"] == "lio/results/" + row["run_id"] + "/model.ckpt",
                "Probe inventory identity differs from the training manifest")
    return records


def validate_energy_run(config, provenance, row, release_head, expected_source_hashes):
    """Pure shared training-record check; no TensorFlow and no checkout byte comparison."""
    rid = rp.safe_id(row["effective_config"]["main"]["exp_name"] + "/"
                     + row["effective_config"]["main"]["dir_name"])
    validate_energy(config)
    require(all(provenance.get(key) == value for key, value in ENERGY_FIELDS.items())
            and provenance.get("energy_in_reward") is True
            and provenance.get("er_eia_adapter") == adapter.ADAPTER_ID,
            "ER-LIO+EIA energy provenance mismatch: " + rid)
    require(provenance.get("energy_reward") == adapter.energy_provenance(ConfigDict(deepcopy(config))),
            "ER-LIO+EIA full energy provenance mismatch: " + rid)
    require(provenance.get("method") == adapter.METHOD and int(provenance["seed"]) == row["seed"]
            and config["main"]["seed"] == row["seed"]
            and config["main"]["exp_name"] + "/" + config["main"]["dir_name"] == rid,
            "ER-LIO+EIA config/provenance run identity mismatch")
    expected = row["effective_config"]
    for field in ("main", "alg", "env", "lio", "nn", "cost_profile", "hardware_profile", "hardware_profile_lambda"):
        require(config.get(field) == expected.get(field), "Stored config differs from formal manifest: " + field)
    require(provenance.get("git_commit") == release_head, "Training provenance has the wrong release HEAD")
    require(set(expected_source_hashes) == set(adapter.source_hashes())
            and provenance.get("er_eia_source_sha256") == expected_source_hashes,
            "Training adapter/source hashes differ from current release")
    require(provenance.get("er_eia_launch_argv") == row["argv"], "Training argv differs from manifest")
    require(row["argv"].count("--energy_in_reward") == 1, "Explicit energy flag must occur exactly once")
    rp.verify_attack_provenance(config, provenance)
    return provenance["energy_reward"]


def validate_run(results_root, rid, row, release_head):
    run = Path(results_root) / rp.safe_id(rid)
    config, provenance = rp.json_read(run/"config.json"), rp.json_read(run/"provenance.json")
    validate_energy_run(config, provenance, row, release_head, adapter.source_hashes())
    return dict(config_sha256=rp.sha(run/"config.json"), provenance_sha256=rp.sha(run/"provenance.json"),
                checkpoint_hashes=rp.checkpoint_hashes(run))


def run_probe(args):
    out = Path(args.output_dir)
    require(not out.exists() and not out.is_symlink(), "ER probe output directory must be new")
    require(not out.resolve().is_relative_to(Path(args.results_root).resolve())
            and not out.resolve().is_relative_to(rp.ANALYSIS.resolve()),
            "ER probe outputs must not enter training results or the protected directory")
    manifest = rp.json_read(args.campaign_manifest)
    with Path(args.inventory).open(encoding="utf-8-sig") as handle:
        inventory = list(csv.DictReader(handle))
    records = validate_matrix(manifest, inventory, args.expected_count)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", args.release_tag) is not None,
            "An explicit safe ER release tag is required")
    require(args.release_tag == RELEASE_TAG, "Only the approved ER-LIO+EIA release tag is allowed")
    require(git("status", "--porcelain") == "",
            "Nano release checkout must be clean (protected directory uses its approved local exclusion)")
    head = git("rev-parse", "HEAD")
    require(head == git("rev-parse", "--verify", "--end-of-options", args.release_tag + "^{commit}")
            and git("describe", "--tags", "--exact-match", "HEAD") == args.release_tag
            and manifest["source"].get("git_commit") == head
            and manifest["source"].get("git_describe") == args.release_tag,
            "Current HEAD, exact tag and training manifest release must agree")
    require(rp.sha(args.protected_manifest) == rp.PROTECTED_MANIFEST_SHA256,
            "Protected manifest file hash differs from the registered pin")
    receipt = rp.json_read(args.validation_receipt)
    inference_arguments = dict(repo=REPO, source_ref=INFERENCE_REF, source_files=rp.SOURCE_FILES,
        protected_pins=rp.PROTECTED_PINS, protected_manifest_sha256=rp.PROTECTED_MANIFEST_SHA256,
        expected_identity=receipt.get("source_git_identity"))
    require(receipt.get("source_git_identity", {}).get("commit") == INFERENCE_REF,
            "The six-clean receipt must retain its approved historical inference commit")
    inference_identity = verified_source_identity(**inference_arguments)
    wrapper_arguments = dict(repo=REPO, source_ref=args.release_tag, source_files=WRAPPER_FILES,
        protected_pins={}, protected_manifest_sha256=rp.PROTECTED_MANIFEST_SHA256)
    wrapper_identity = verified_source_identity(**wrapper_arguments)
    require(all(manifest["source"].get("sha256", {}).get(name) == digest
                for name, digest in adapter.source_hashes().items()),
            "Training manifest source hashes differ from the actual release")
    inputs = {name: rp.sha(getattr(args, name)) for name in
              ("campaign_manifest", "inventory", "validation_receipt", "protected_manifest")}
    run_inputs = {rid: validate_run(args.results_root, rid, row, head) for rid, row in records.items()}
    context = dict(schema_version=1, objective=adapter.OBJECTIVE, followup_arm=adapter.ARM,
        display_label=adapter.DISPLAY_LABEL, campaign=CAMPAIGN, expected_count=30,
        training_release=dict(git_commit=head, git_describe=args.release_tag),
        wrapper_source_git_identity=wrapper_identity, inference_source_git_identity=inference_identity,
        input_hashes=inputs, run_inputs=run_inputs,
        note="Payments are not evaluated because they do not enter the reset-state action distribution.")
    with source_identity_scope(rp, **inference_arguments):
        result = rp.probe_batch(args)
    require(git("rev-parse", "HEAD") == head, "Release HEAD changed during ER probing")
    require(verified_source_identity(**wrapper_arguments) == wrapper_identity,
            "ER wrapper source changed during probing")
    require(verified_source_identity(**inference_arguments) == inference_identity,
            "Historical inference source changed during probing")
    require(all(rp.sha(getattr(args, name)) == value for name, value in inputs.items()),
            "ER probe input changed during probing")
    require({rid: validate_run(args.results_root, rid, row, head) for rid, row in records.items()} == run_inputs,
            "ER training checkpoint/config/provenance changed during probing")
    context["status"] = result["status"]
    context["legacy_manifest_sha256"] = rp.sha(out/"manifest.json")
    rp.write_json(out/"er_context.json", context)
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-manifest", "inventory", "results-root", "validation-receipt",
                 "protected-manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args)
        print(json.dumps(dict(status=result["status"], expected_count=30, output=str(args.output_dir))))
        return 0 if result["status"] == "COMPLETE" else 2
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, "ER_EIA_PROBE_ERROR: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
