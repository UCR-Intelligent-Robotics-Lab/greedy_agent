"""A9 manifest contracts and static isolation from existing A1-A8 sources."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensys_runs import energy_profile_campaign as wave1
from sensys_runs import energy_hardware_campaign as wave2
from sensys_runs import energy_reward_campaign as wave3
from sensys_runs import energy_profile_runner as runner
from sensys_runs.iccps27_followup import a9_campaign as a9

HEAD = "0561acb207f06beaac91e8a3bce77925412516bd"
FOLLOWUP = "sensys_runs/iccps27_followup/"


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    configs = {m: wave1.load_config(m) for m in ("lio_er", "refine_er")}
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(wave1, "REPO_ROOT", root)
    monkeypatch.setattr(a9, "REPO_ROOT", root)
    monkeypatch.setattr(wave1, "load_config", lambda m: deepcopy(configs[m]))
    identity = lambda: dict(git_commit=HEAD, git_describe="iccps2027-energy-p7",
                            git_status="", sha256={})
    monkeypatch.setattr(wave1, "source_identity", identity)
    monkeypatch.setattr(a9, "source_identity", identity)
    return root


def normalized_config(row):
    cfg = deepcopy(row["effective_config"])
    for key in ("exp_name", "dir_name"):
        cfg["main"].pop(key)
    cfg.pop("cost_profile")
    return cfg


@pytest.mark.parametrize("smoke,seeds,k", [
    (False, tuple(range(12341, 12351)), 5), (True, (12341, 12342), 2),
])
def test_exact_a9_rows_and_existing_runner_schema(isolated, monkeypatch, tmp_path,
                                                 smoke, seeds, k):
    manifest = a9.build_campaign("a9_contract", "local", 1, smoke=smoke)
    assert manifest["n_runs"] == manifest["n_seeds_per_arm"] == len(seeds)
    assert tuple(r["seed"] for r in manifest["runs"]) == seeds
    assert len({r["output_directory"] for r in manifest["runs"]}) == len(seeds)
    for row in manifest["runs"]:
        cfg, argv = row["effective_config"], row["argv"]
        assert (row["arm"], row["method"], row["training_profile"],
                row["hardware_profile"], row["objective"]) == (
                    "A9", "lio_er", "uniform_v1", "homogeneous_v1", "energy_reward")
        expected_options = {"--method": "lio_er", "--cost_profile": "uniform_v1",
                            "--hardware_profile": "homogeneous_v1", "--threads": "1",
                            "--n_agents": "4", "--min_at_lever": "2",
                            "--seed": str(row["seed"]), "--n_eval": "10"}
        for option, value in expected_options.items():
            assert argv.count(option) == 1
            assert argv[argv.index(option) + 1] == value
        assert argv.count("--energy_in_reward") == 1
        assert cfg.energy_in_reward is True
        assert cfg.energy_reward_beta == 1.0 and cfg.energy_reward_form == "episode_mean"
        assert cfg.lio.Fairness_multiplier == 0.0
        assert cfg.hardware_profile_lambda == [1.0] * 4
        assert cfg.cost_profile.normalized_vector == (1.0, 1.0, 1.0)
        assert (cfg.alg.n_episodes, cfg.alg.period, cfg.alg.n_test) == (
            (10, 2, 100) if smoke else (25000, 500, 100))
        assert cfg.main.exp_name == "a9_contract" + ("_smoke" if smoke else "")
        assert row["command"] == wave1.command_text(argv)
        assert row["expected_checkpoint"] == str(Path(row["output_directory"]) / "model.ckpt")
    monkeypatch.setattr(runner, "REPO_ROOT", isolated)
    monkeypatch.setenv("NTFY_TOPIC", "test-no-notification-sent")
    monkeypatch.setattr(runner, "git", lambda *args: HEAD if args == ("rev-parse", "HEAD") else "")
    runner.validate(manifest, tmp_path / "console")
    shards = [runner.shard_plan(manifest, index, k) for index in range(k)]
    assert all(shards)
    assert sorted(r["seed"] for shard in shards for r in shard) == list(seeds)


def test_a9_matches_a2_except_profile_and_output_identity(isolated):
    old = wave3.build_campaign("reference", "local", 1)
    a2 = {r["seed"]: r for r in old["runs"] if r["hardware_profile"] == "homogeneous_v1"}
    new = a9.build_campaign("a9_contract", "local", 1)
    assert len(a2) == len(new["runs"]) == 10
    for row in new["runs"]:
        assert normalized_config(row) == normalized_config(a2[row["seed"]])
        assert row["checkpoint_episode"] == a2[row["seed"]]["checkpoint_episode"]
        before, after = list(a2[row["seed"]]["argv"]), list(row["argv"])
        for option in ("--exp_name", "--dir_name", "--cost_profile"):
            before[before.index(option) + 1] = after[after.index(option) + 1] = "IDENTITY_OR_PROFILE"
        assert before == after


def test_a1_a8_manifests_byte_identical_around_a9_generation(isolated):
    def snapshots():
        return [json.dumps(module.build_campaign("unchanged", "local", 1, smoke=smoke),
                           indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
                for module in (wave1, wave2, wave3) for smoke in (False, True)]
    before = snapshots()
    a9.build_campaign("a9_contract", "local", 1)
    a9.build_campaign("a9_contract", "local", 1, smoke=True)
    assert snapshots() == before


@pytest.mark.parametrize("campaign,platform,threads", [
    ("../stale", "local", 1), ("bad/name", "local", 1),
    ("valid", "", 1), ("valid", "local", 2),
])
def test_invalid_campaign_settings_refused_before_source_capture(isolated, monkeypatch,
                                                               campaign, platform, threads):
    monkeypatch.setattr(a9, "source_identity", lambda: pytest.fail("identity must not be captured"))
    with pytest.raises(ValueError):
        a9.build_campaign(campaign, platform, threads)
    assert not list(isolated.iterdir())


def test_existing_run_directory_is_rejected(isolated):
    manifest = a9.build_campaign("a9_contract", "local", 1)
    Path(manifest["runs"][0]["output_directory"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        a9.build_campaign("a9_contract", "local", 1)


def test_cli_manifest_is_exclusive_and_never_under_results(isolated, tmp_path):
    path = tmp_path / "a9_manifest.json"
    args = ["--platform", "local", "--threads", "1", "--campaign", "a9_cli"]
    assert a9.main(args + ["--out", str(path)]) == 0
    original = path.read_bytes()
    assert json.loads(original)["n_runs"] == 10
    assert a9.main(args + ["--out", str(path)]) == 2
    assert path.read_bytes() == original
    refused = isolated / "lio" / "results" / "manifest.json"
    assert a9.main(args + ["--out", str(refused)]) == 2
    assert not refused.exists()


def test_dry_run_prints_only_and_rejects_output_argument(isolated, tmp_path, capsys):
    args = ["--platform", "local", "--threads", "1", "--smoke", "--dry-run"]
    assert a9.main(args) == 0
    assert json.loads(capsys.readouterr().out)["n_runs"] == 2
    assert not list(isolated.iterdir())
    with pytest.raises(SystemExit) as exc:
        a9.main(args + ["--out", str(tmp_path / "forbidden.json")])
    assert exc.value.code == 2
    assert not (tmp_path / "forbidden.json").exists()


def static_isolation_report(root: Path = ROOT) -> dict:
    """Check diff scope and reject followup references in tracked source files."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).split("\0")
    changed = [p for p in git("diff", "HEAD", "--name-only", "-z") if p]
    tracked = [p for p in git("ls-files", "-z") if p]
    outside_changes = [p for p in changed if not p.startswith(FOLLOWUP)]
    imports = []
    for relative in tracked:
        path = root / relative
        if relative.startswith(FOLLOWUP) or path.suffix.lower() not in (".py", ".sh", ".ps1"):
            continue
        for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if "iccps27_followup" in text:
                imports.append(f"{relative}:{line}:{text.strip()}")
    return dict(tracked_change_paths=changed, outside_followup_changes=outside_changes,
                tracked_source_followup_references=imports)


def test_existing_tracked_sources_do_not_import_followup():
    report = static_isolation_report()
    assert report["outside_followup_changes"] == []
    assert report["tracked_source_followup_references"] == []


@pytest.mark.parametrize("smoke,expected_count", [(False, 10), (True, 2)])
def test_optional_inventory_schema_and_fresh_destinations(isolated, tmp_path, smoke, expected_count):
    manifest_path = tmp_path / "paired_manifest.json"
    inventory_path = tmp_path / "paired_inventory.csv"
    args = ["--platform", "local", "--threads", "1", "--campaign", "a9_inventory",
            "--out", str(manifest_path), "--inventory-out", str(inventory_path)]
    if smoke:
        args.append("--smoke")
    assert a9.main(args) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(a9.INVENTORY_FIELDS)
        rows = list(reader)
    assert len(rows) == expected_count
    for record, run in zip(rows, manifest["runs"]):
        main = run["effective_config"]["main"]
        run_id = main["exp_name"] + "/" + main["dir_name"]
        assert record == dict(run_id=run_id,
                              arm="lio_er__uniform_v1__homogeneous_v1__energy_reward",
                              method="lio_er", seed=str(run["seed"]),
                              checkpoint_prefix="lio/results/" + run_id + "/model.ckpt",
                              evidence="[verified from file]")
    assert b"\r" not in inventory_path.read_bytes()
    before_manifest, before_inventory = manifest_path.read_bytes(), inventory_path.read_bytes()
    assert a9.main(args) == 2
    assert manifest_path.read_bytes() == before_manifest
    assert inventory_path.read_bytes() == before_inventory


@pytest.mark.parametrize("case", ["stale_inventory", "same_output", "missing_parent", "inside_results"])
def test_inventory_preflight_refuses_before_writing_manifest(isolated, tmp_path, case):
    manifest_path = tmp_path / "new_manifest.json"
    inventory_path = tmp_path / "inventory.csv"
    if case == "stale_inventory":
        inventory_path.write_text("retained data", encoding="utf-8")
    elif case == "same_output":
        inventory_path = manifest_path
    elif case == "missing_parent":
        inventory_path = tmp_path / "absent" / "inventory.csv"
    else:
        inventory_path = isolated / "lio" / "results" / "inventory.csv"
    args = ["--platform", "local", "--threads", "1", "--out", str(manifest_path),
            "--inventory-out", str(inventory_path)]
    assert a9.main(args) == 2
    assert not manifest_path.exists()
    if case == "stale_inventory":
        assert inventory_path.read_text(encoding="utf-8") == "retained data"
    else:
        assert not inventory_path.exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_inventory_requires_manifest_output(isolated, tmp_path, dry_run):
    inventory_path = tmp_path / "forbidden_inventory.csv"
    args = ["--platform", "local", "--threads", "1", "--inventory-out", str(inventory_path)]
    if dry_run:
        args.append("--dry-run")
    with pytest.raises(SystemExit) as exc:
        a9.main(args)
    assert exc.value.code == 2
    assert not inventory_path.exists()
