"""Compare A3 JSON, excluding exactly two root telemetry fields."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED_RESULT_PATHS = ("/seconds", "/peak_process_memory_bytes")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def first_difference(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return path + " (object keys differ)"
        for key in left:
            found = first_difference(left[key], right[key], path + "/" + str(key))
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path + " (array length differs)"
        for index, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, path + "/" + str(index))
            if found is not None:
                return found
        return None
    return None if left == right else path

def compare(stored, recomputed):
    files = [folder / name for folder in (stored, recomputed)
             for name in ("result.json", "references.json")]
    before = {str(path): digest(path) for path in files}
    old_result, old_references, new_result, new_references = [
        json.loads(path.read_text(encoding="utf-8")) for path in files]
    telemetry = {}
    for pointer in EXCLUDED_RESULT_PATHS:
        key = pointer[1:]
        if key not in old_result or key not in new_result:
            raise ValueError("Missing required root telemetry field: " + pointer)
        telemetry[pointer] = {"stored": old_result.pop(key), "recomputed": new_result.pop(key)}
    differences = {}
    for name, left, right in (("result.json", old_result, new_result),
                              ("references.json", old_references, new_references)):
        difference = first_difference(left, right)
        if difference is not None:
            differences[name] = difference
    if before != {str(path): digest(path) for path in files}:
        raise ValueError("Input changed during equivalence comparison")
    return {"status": "PASS" if not differences else "FAIL",
            "comparison": "Parsed JSON equality; float ==; no tolerance",
            "excluded_result_json_paths": list(EXCLUDED_RESULT_PATHS),
            "excluded_references_json_paths": [], "telemetry": telemetry,
            "first_differences": differences, "input_sha256": before}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stored", type=Path, required=True)
    parser.add_argument("--recomputed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a fresh file")
    result = compare(args.stored, args.recomputed)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
