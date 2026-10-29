"""Inventory-aware A9 enumeration with unchanged numerical and validation cores.

--reference-inventory is an explicit equivalence-check mode. It verifies the
complete original export against its original inventory before selecting rows.
It never treats a subset CSV hash as the original export's inventory hash.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

from pipeline_common import (FIELDS, load_inventory, check_identity, verify_analysis_sources,
                             local_source_hashes, sha, unchanged, write_json)
from numpy_policy import NumpyPolicy, POLICY_DEFINITION
from parity_v6 import CRITERION, require_amended
from model_validation_v3 import read_observations, run_tests, adjust_across_runs, WORDING
from enumerate_histories import one
import numpy as np


def preflight(export_root, evaluation_root, inventory, expected_count, reference_inventory=None):
    export_root, evaluation_root, inventory = map(Path, (export_root, evaluation_root, inventory))
    subset_mode = reference_inventory is not None
    selected = load_inventory(inventory, expected_count, a9=not subset_mode)
    manifest_path = export_root / "manifest.json"
    exported = json.loads(manifest_path.read_text())
    if (exported.get("smoke") or exported.get("stub") or exported.get("dry_run")
            or exported.get("acceptance_criterion") != CRITERION
            or not exported.get("source_hashes_unchanged")):
        raise ValueError("Expected completed real amended export")
    verify_analysis_sources(exported["source_hashes"])
    source_inventory = Path(reference_inventory) if subset_mode else inventory
    if sha(source_inventory) != exported["inventory_sha256"]:
        raise ValueError("Original export inventory hash mismatch")
    source_count = exported.get("expected_count", 80 if subset_mode else expected_count)
    source_rows = load_inventory(source_inventory, source_count, a9=not subset_mode)
    source_by_id = {r["run_id"]: r for r in source_rows}
    for row in selected:
        if row != source_by_id.get(row["run_id"]):
            raise ValueError("Selected inventory row differs from original inventory")
    if (exported["pending"] or exported["missing"] or exported.get("fatal_error")
            or len(exported["excluded"]) > 1):
        raise ValueError("Export incomplete or suspect")
    accepted = exported["computed"]
    excluded = exported["excluded"]
    ids = [r["run_id"] for r in accepted + excluded]
    if len(ids) != len(set(ids)) or set(ids) != set(source_by_id):
        raise ValueError("Export coverage or duplicate identity mismatch")
    for record in accepted + excluded:
        if any(record.get(k) != source_by_id[record["run_id"]][k] for k in FIELDS):
            raise ValueError("Export record does not match inventory identity")
    for record in excluded:
        if (record["parity"].get("criterion") != CRITERION
                or record["parity"].get("amended_status") != "fail"):
            raise ValueError("Malformed exclusion")
    inputs = {str(p.resolve()): sha(p) for p in (inventory, source_inventory, manifest_path)}
    observed, parity_by_run = {}, {}
    selected_ids = {r["run_id"] for r in selected}
    # Preserve the original full-export integrity gate even in subset mode.
    for record in accepted:
        rid = record["run_id"]
        meta_path = export_root / rid.replace("/", "__") / "policies.json"
        inputs[str(meta_path.resolve())] = sha(meta_path)
        meta = json.loads(meta_path.read_text())
        require_amended(meta)
        if record["parity"] != meta["parity"]:
            raise ValueError("Manifest/metadata parity mismatch")
        check_identity(source_by_id[rid], meta["config"], meta["provenance"], a9=not subset_mode)
        groups = meta["parity"]["groups"]
        if set(groups) != {"reset_queries", "random_inputs", "depth_4_histories"} or not all(g["pass"] for g in groups.values()):
            raise ValueError("Incomplete amended group gate")
        if meta["parity"]["cases"] != {"reset_joint_action_queries": 81, "random_inputs": 10000, "depth_4_histories": 100}:
            raise ValueError("Parity query count mismatch")
        archive = meta_path.with_name("policies.npz")
        NumpyPolicy.load(archive, meta_path)
        inputs[str(archive.resolve())] = sha(archive)
        evidence = read_observations(evaluation_root / rid, meta)
        for name, digest in evidence["source_hashes"].items():
            inputs[str((evaluation_root / rid / name).resolve())] = digest
        if rid in selected_ids:
            observed[rid], parity_by_run[rid] = evidence, meta["parity"]
    approved = [r for r in selected if r["run_id"] in observed]
    selected_exclusions = [r for r in excluded if r["run_id"] in selected_ids]
    if not approved:
        raise ValueError("No selected approved export")
    if not unchanged(inputs):
        raise ValueError("Input changed during preflight")
    return dict(rows=approved, excluded=selected_exclusions, observed=observed,
                parity_by_run=parity_by_run, input_hashes=inputs,
                source_count=source_count, source_inventory_sha256=sha(source_inventory),
                subset_mode=subset_mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--reference-inventory", type=Path,
                        help="Explicit subset-equivalence mode, with full original export verification")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-cap-mib", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    checked = preflight(args.export_root, args.evaluation_root, args.inventory,
                        args.expected_count, args.reference_inventory)
    sources = local_source_hashes()
    approved = [r["run_id"] for r in checked["rows"]]
    manifest = dict(evidence="[verified from file]", enumeration_wording=WORDING,
                    policy_definition=POLICY_DEFINITION, acceptance_criterion=CRITERION,
                    input_hashes=checked["input_hashes"], source_hashes=sources,
                    inventory_sha256=sha(args.inventory), expected_count=args.expected_count,
                    full_export_count=checked["source_count"],
                    original_inventory_sha256=checked["source_inventory_sha256"],
                    subset_equivalence_mode=checked["subset_mode"], dry_run=args.dry_run,
                    parity_by_run=checked["parity_by_run"], approved=approved,
                    excluded=checked["excluded"], completed=[], pending=list(approved),
                    numpy=np.__version__, python=platform.python_version(),
                    selection="explicit validated inventory subset" if checked["subset_mode"] else "all approved A9 runs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "v3_manifest.json", manifest)
    if args.dry_run:
        manifest["input_hashes_unchanged"] = unchanged(checked["input_hashes"])
        manifest["source_hashes_unchanged"] = unchanged(sources)
        write_json(args.output_dir / "v3_manifest.json", manifest)
        if not manifest["input_hashes_unchanged"] or not manifest["source_hashes_unchanged"]:
            raise ValueError("Preflight input/source changed")
        return
    write_json(args.output_dir / "manifest.json", manifest)
    validated = {}
    try:
        with (args.output_dir / "summary.csv").open("x", newline="") as stream:
            writer = None
            for row in checked["rows"]:
                rid = row["run_id"]
                meta_path = args.export_root / rid.replace("/", "__") / "policies.json"
                model = NumpyPolicy.load(meta_path.with_name("policies.npz"), meta_path)
                result = one(model, args.output_dir / rid.replace("/", "__"),
                             args.chunk_size, args.memory_cap_mib,
                             model.manifest.get("observed"), rid.split("/")[-1])
                summary = dict(run_id=rid, arm=row["arm"], seed=row["seed"],
                               L_reset=result["L_reset"], L_all=result["L_all"],
                               L_all_t=json.dumps(result["L_all_by_depth"]), E_T=result["E_T"],
                               P_timeout=result["P_timeout"], K_delta=json.dumps(result["K_delta"]),
                               two_level_upper=result["two_level_E_upper"],
                               depthwise_upper=result["depthwise_E_upper"],
                               observed_100=model.manifest.get("observed", {}).get("final_mean"),
                               observed_10=model.manifest.get("observed", {}).get("periodic_mean"),
                               node_count=result["node_count"], seconds=result["seconds"],
                               stub=result["stub"], policy_definition=POLICY_DEFINITION,
                               evidence="[your judgment]", **result["reference_summary"])
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(summary))
                    writer.writeheader()
                writer.writerow(summary)
                stream.flush()
                validated[rid] = run_tests(result, checked["observed"][rid])
                validated[rid].update(acceptance_criterion=CRITERION,
                                      original_parity_status=checked["parity_by_run"][rid]["original_status"],
                                      amended_parity_status=checked["parity_by_run"][rid]["amended_status"])
                manifest["completed"].append(rid)
                manifest["pending"].remove(rid)
                write_json(args.output_dir / "manifest.json", manifest)
        across = adjust_across_runs(validated)
        across["legacy_80_primary_nominal_reference"] = across.pop("registered_80_primary_nominal_expected")
        across["selected_primary_nominal_expected"] = {"0.05": len(validated) * .05, "0.01": len(validated) * .01}
        target = args.output_dir / "validation"
        target.mkdir()
        write_json(target / "across_runs.json", across)
        test_rows = []
        for rid, record in validated.items():
            write_json(target / (rid.replace("/", "__") + ".json"), record)
            for test in record["tests"]:
                test_rows.append(dict(run_id=rid, episodes=test["episodes"], metric=test["metric"],
                                      agent=test["agent"], role=test["role"], family=test["family"],
                                      observed_total=test["observed_total"], p=test["p_two_sided"],
                                      holm_p=test["holm_p"], model_reality_mismatch=record["model_reality_mismatch"],
                                      claim_eligible=record["claim_eligible"], evidence="[your judgment]"))
        with (target / "all_pvalues.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
            writer.writeheader()
            writer.writerows(test_rows)
        manifest["input_hashes_unchanged"] = unchanged(checked["input_hashes"])
        manifest["source_hashes_unchanged"] = unchanged(sources)
        if not manifest["input_hashes_unchanged"] or not manifest["source_hashes_unchanged"]:
            raise ValueError("Enumeration input/source changed")
    except Exception as exc:
        manifest["fatal_error"] = repr(exc)
        raise
    finally:
        write_json(args.output_dir / "manifest.json", manifest)
        write_json(args.output_dir / "v3_manifest.json", manifest)


if __name__ == "__main__":
    main()
