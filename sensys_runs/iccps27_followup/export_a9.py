"""Export A9 with unchanged clean-policy inference and v6 parity functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

from pipeline_common import (REPO, EVALUATION_FILES, check_identity, load_inventory,
                             sha, unchanged, write_json)
from analysis_core import checkpoint_files
from export_policies import export_one
from numpy_policy import POLICY_DEFINITION
from parity_v6 import CRITERION
import numpy as np


def preflight(repo, results_root, inventory, expected_count):
    if Path(repo).resolve() != REPO:
        raise ValueError("The executable and --repo-root must identify the same checkout")
    rows = load_inventory(inventory, expected_count)
    inputs = {str(Path(inventory).resolve()): sha(inventory)}
    for row in rows:
        run = Path(results_root) / row["run_id"]
        inputs.update(checkpoint_files(run / "model.ckpt"))
        config = json.loads((run / "config.json").read_text())
        provenance = json.loads((run / "provenance.json").read_text())
        check_identity(row, config, provenance)
        for name in ("config.json", "provenance.json", *EVALUATION_FILES):
            path = run / name
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs[str(path)] = sha(path)
    return rows, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and identities without TensorFlow")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.smoke and "smoke" not in args.output_dir.name.lower():
        raise ValueError("Smoke directory required")
    rows, inputs = preflight(args.repo_root, args.results_root, args.inventory, args.expected_count)
    selected = rows[:1] if args.smoke else rows
    sources = {str(p): sha(p) for root in (REPO / "lio", REPO / "sensys_runs")
               for p in sorted(root.rglob("*.py"))}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(evidence="[verified from file]", policy_definition=POLICY_DEFINITION,
                    acceptance_criterion=CRITERION, inventory_sha256=sha(args.inventory),
                    expected_count=args.expected_count, source_hashes=sources, input_hashes=inputs,
                    numpy=np.__version__, python=platform.python_version(), stub=False,
                    smoke=args.smoke, dry_run=args.dry_run,
                    result_label="SMOKE: not a registered result" if args.smoke else "A9 follow-up",
                    computed=[], excluded=[], missing=[], pending=[r["run_id"] for r in selected])
    write_json(args.output_dir / "manifest.json", manifest)
    if args.dry_run:
        manifest["source_hashes_unchanged"] = unchanged(sources)
        manifest["input_hashes_unchanged"] = unchanged(inputs)
        write_json(args.output_dir / "manifest.json", manifest)
        if not manifest["source_hashes_unchanged"] or not manifest["input_hashes_unchanged"]:
            raise ValueError("Preflight input/source changed")
        return
    failures = 0
    try:
        for row in selected:
            started = time.perf_counter()
            run = args.results_root / row["run_id"]
            status, parity = export_one(args.repo_root, run,
                                       args.output_dir / row["run_id"].replace("/", "__"),
                                       smoke=args.smoke)
            record = dict(row, original_status=parity["original_status"],
                          amended_status=parity["amended_status"], parity=parity,
                          seconds=time.perf_counter() - started)
            if status == "pass":
                manifest["computed"].append(record)
            else:
                failures += 1
                record["reason"] = "Amended parity failed; excluded without repair"
                manifest["excluded"].append(record)
            evidence = args.output_dir / "evaluation_inputs" / row["run_id"]
            evidence.mkdir(parents=True, exist_ok=False)
            for name in EVALUATION_FILES:
                source = run / name
                with (evidence / name).open("xb") as stream:
                    stream.write(source.read_bytes())
                if sha(evidence / name) != inputs[str(source)]:
                    raise ValueError("Evaluation copy mismatch")
            manifest["pending"].remove(row["run_id"])
            write_json(args.output_dir / "manifest.json", manifest)
            print(json.dumps({"run_id": row["run_id"], "status": status,
                              "seconds": record["seconds"]}), flush=True)
            if failures > 1:
                raise RuntimeError("More than one parity failure; port suspect")
        manifest["source_hashes_unchanged"] = unchanged(sources)
        manifest["input_hashes_unchanged"] = unchanged(inputs)
        if not manifest["source_hashes_unchanged"] or not manifest["input_hashes_unchanged"]:
            raise ValueError("Export input/source changed")
    except Exception as exc:
        manifest["fatal_error"] = repr(exc)
        raise
    finally:
        write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
