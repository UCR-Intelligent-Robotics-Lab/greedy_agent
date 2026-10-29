"""Tests for the exact two-field telemetry exclusion contract."""
import json
import pytest
from sensys_runs.iccps27_followup.verify_equivalence import compare

def fixture_pair(tmp_path):
    stored, recomputed = tmp_path / "stored", tmp_path / "recomputed"
    for folder, seconds in ((stored, 1.0), (recomputed, 2.0)):
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps({
            "seconds": seconds, "peak_process_memory_bytes": int(seconds * 100),
            "E_T": 1.25, "nested": {"seconds": 9.0}}), encoding="utf-8")
        (folder / "references.json").write_text(json.dumps({"L_min0": 0.5}), encoding="utf-8")
    return stored, recomputed

def test_only_root_telemetry_is_excluded(tmp_path):
    old, new = fixture_pair(tmp_path)
    result = compare(old, new)
    assert result["status"] == "PASS"
    assert result["excluded_result_json_paths"] == ["/seconds", "/peak_process_memory_bytes"]
    assert result["excluded_references_json_paths"] == []
    assert result["telemetry"]["/seconds"] == {"stored": 1.0, "recomputed": 2.0}

@pytest.mark.parametrize("filename,key,value", [
    ("result.json", "E_T", 1.2500000000000002),
    ("result.json", "nested", {"seconds": 10.0}),
    ("references.json", "L_min0", 0.5000000000000001),
])
def test_any_other_difference_fails(tmp_path, filename, key, value):
    old, new = fixture_pair(tmp_path)
    path = new / filename
    content = json.loads(path.read_text(encoding="utf-8"))
    content[key] = value
    path.write_text(json.dumps(content), encoding="utf-8")
    assert compare(old, new)["status"] == "FAIL"

def test_missing_telemetry_fails(tmp_path):
    old, new = fixture_pair(tmp_path)
    path = new / "result.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    del content["seconds"]
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required root telemetry"):
        compare(old, new)
