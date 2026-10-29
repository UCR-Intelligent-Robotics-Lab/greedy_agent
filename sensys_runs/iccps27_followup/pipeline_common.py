"""Strict inventory and identity contracts for the separate A9 pipeline."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "sensys_runs" / "iccps27_analysis"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ANALYSIS))
from analysis_core import sha, write_json

A9_ARM = "lio_er__uniform_v1__homogeneous_v1__energy_reward"
FIELDS = ("run_id", "arm", "method", "seed", "checkpoint_prefix", "evidence")
EVALUATION_FILES = ("energy_eval_episodes.csv", "energy_eval_provenance.json", "log.csv")


def load_inventory(path, expected_count, *, a9=True):
    if isinstance(expected_count, bool) or expected_count < 1:
        raise ValueError("Expected count must be positive")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Inventory must use the frozen six-column schema")
        rows = list(reader)
    if len(rows) != expected_count or len({r["run_id"] for r in rows}) != expected_count:
        raise ValueError("Inventory count or unique run identity mismatch")
    for row in rows:
        rid = row["run_id"]
        parts = PurePosixPath(rid).parts
        if len(parts) != 2 or any(x in ("", ".", "..") for x in parts) or "\\" in rid or ":" in rid or rid.startswith("/"):
            raise ValueError("Unsafe run identity")
        if row["method"] not in ("lio_er", "refine_er"):
            raise ValueError("Only clean inference methods are supported")
        if int(row["seed"]) not in range(12341, 12351):
            raise ValueError("Unregistered seed")
        if row["checkpoint_prefix"] != "lio/results/" + rid + "/model.ckpt":
            raise ValueError("Checkpoint prefix does not match run identity")
        if a9 and (row["arm"] != A9_ARM or row["method"] != "lio_er"):
            raise ValueError("Expected the A9 arm only")
    if a9 and len({int(r["seed"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate A9 seed")
    return rows


def check_identity(row, config, provenance, *, a9=True):
    if (config["main"]["exp_name"] + "/" + config["main"]["dir_name"] != row["run_id"]
            or provenance["method"] != row["method"]
            or int(provenance["seed"]) != int(row["seed"])
            or int(config["main"]["seed"]) != int(row["seed"])):
        raise ValueError("Run/config/provenance identity mismatch")
    if not a9:
        return
    from sensys_runs.energy_profiles import get_profile, get_hardware_profile
    from sensys_runs.energy_reward_runtime import REWARD_FORM, registered_beta
    profile = json.loads(json.dumps(get_profile("uniform_v1").snapshot()))
    hardware = json.loads(json.dumps(get_hardware_profile("homogeneous_v1").snapshot(4)))
    for record in (config, provenance):
        if (record.get("cost_profile") != profile or record.get("hardware_profile") != hardware
                or record.get("hardware_profile_lambda") != [1.0] * 4
                or record.get("energy_in_reward") is not True
                or record.get("energy_reward_beta") != registered_beta()
                or record.get("energy_reward_form") != REWARD_FORM):
            raise ValueError("A9 profile, hardware or objective mismatch")
    if config["lio"].get("Fairness_multiplier") != 0.0:
        raise ValueError("A9 requires B=0")


def verify_analysis_sources(source_hashes):
    """Preserve v3 byte-hash checks and additionally check follow-up sources."""
    matched = set()
    for name, digest in source_hashes.items():
        normalized = name.replace("\\", "/")
        for directory in ("iccps27_analysis", "iccps27_followup"):
            marker = "/sensys_runs/" + directory + "/"
            if marker in normalized:
                relative = normalized.split(marker, 1)[1]
                if ".." in PurePosixPath(relative).parts:
                    raise ValueError("Unsafe analysis source identity")
                local = REPO / "sensys_runs" / directory / relative
                if not local.is_file() or sha(local) != digest:
                    raise ValueError("Analysis source hash mismatch: " + name)
                matched.add(directory)
    if "iccps27_analysis" not in matched:
        raise ValueError("Export is missing protected analysis source hashes")


def local_source_hashes():
    return {str(p): sha(p) for root in (ANALYSIS, Path(__file__).parent)
            for p in sorted(root.rglob("*.py"))}


def unchanged(hashes):
    return all(Path(p).is_file() and sha(p) == digest for p, digest in hashes.items())
