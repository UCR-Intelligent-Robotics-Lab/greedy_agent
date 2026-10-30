"""Registered ER-LIO+EIA matrix, strict release gates and unchanged execution."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sensys_runs.iccps27_followup import er_eia_adapter as adapter
from sensys_runs.iccps27_followup import er_eia_campaign as campaign
from sensys_runs.iccps27_followup import er_eia_runner as runner


def test_formal_matrix_has_exact_30_energy_eia_rows_and_pinned_sources():
    manifest = campaign.build_campaign("er_eia_contract", "local", 1)
    actual = {(row["method"], tuple(row["weight_pair"]), row["seed"]) for row in manifest["runs"]}
    expected = {("lio_eia_er", pair, seed) for pair in campaign.WEIGHTS for seed in campaign.SEEDS}
    assert manifest["n_runs"] == len(manifest["runs"]) == 30
    assert actual == expected and manifest["planned_formal_runs"] == 30
    assert manifest["fallback_declared"] is False
    for row in manifest["runs"]:
        cfg, argv = row["effective_config"], row["argv"]
        assert row["objective"] == "energy_reward" and row["followup_arm"] == "er_lio_eia_er"
        assert cfg.display_label == row["display_label"] == "ER-LIO+EIA"
        assert cfg.energy_in_reward is True and cfg.energy_reward_beta == 1.0
        assert cfg.energy_reward_form == "episode_mean"
        assert cfg.hardware_profile_lambda == [1.0]*4
        assert cfg.cost_profile.normalized_vector == (1.0, 1.0, 1.0)
        assert cfg.alg.n_episodes == 25000 and cfg.alg.n_test == 100
        assert cfg.env.n_agents == 4 and cfg.env.min_at_lever == 2
        assert argv.count("--energy_in_reward") == 1 and argv[1].endswith("er_eia_adapter.py")
        assert runner.core.option(argv, "--method") == "lio_eia_er"
        assert "energy_reward" in cfg.main.dir_name
    assert {key: manifest["source"]["sha256"][key] for key in adapter.source_hashes()} == adapter.source_hashes()
    rows = campaign.inventory_rows(manifest)
    assert len(rows) == 30 and len({row["run_id"] for row in rows}) == 30
    assert all(tuple(row) == campaign.FIELDS for row in rows)


def test_smoke_is_two_seeds_at_nondefault_weight():
    manifest = campaign.build_campaign("er_eia_contract", "local", 1, smoke=True)
    assert manifest["n_runs"] == 2 and manifest["n_seeds_per_arm"] == 2
    assert [(row["seed"], row["weight_pair"]) for row in manifest["runs"]] == [
        (12341, [1.1, 0.9]), (12342, [1.1, 0.9])]
    assert all(row["effective_config"].alg.n_episodes == 10 for row in manifest["runs"])
    assert all(row["effective_config"].alg.period == 2 for row in manifest["runs"])
    assert manifest["planned_formal_runs"] == 30


def test_generator_outputs_are_exclusive_and_existing_bytes_preserved(tmp_path):
    manifest, inventory = tmp_path/"manifest.json", tmp_path/"inventory.csv"
    args = ["--campaign", "er_eia_contract", "--platform", "local", "--threads", "1",
            "--out", str(manifest), "--inventory-out", str(inventory)]
    assert campaign.main(args) == 0
    before = manifest.read_bytes(), inventory.read_bytes()
    assert campaign.main(args) == 2
    assert before == (manifest.read_bytes(), inventory.read_bytes())
    assert len(json.loads(before[0])["runs"]) == 30


@pytest.fixture
def release(monkeypatch):
    identity = campaign.source_identity()
    identity.update(git_describe=campaign.RELEASE_TAG, git_status="")
    monkeypatch.setattr(campaign, "source_identity", lambda: deepcopy(identity))
    manifest = campaign.build_campaign("er_eia_runner_contract", "local", 1)
    def git(*args):
        if args == ("rev-parse", "HEAD"):
            return identity["git_commit"]
        if args == ("describe", "--tags", "--exact-match", "HEAD"):
            return campaign.RELEASE_TAG
        return ""
    monkeypatch.setattr(runner.core, "git", git)
    monkeypatch.setenv("NTFY_TOPIC", "test-never-send")
    return manifest, git


def test_gate_projects_only_private_copy_and_keeps_executed_argv(release, tmp_path):
    manifest, _ = release
    before = deepcopy(manifest)
    received = []
    original = runner.core.validate
    def validate(projected, console):
        received.append(deepcopy(projected))
        original(projected, console)
    runner.validate_er_eia(manifest, tmp_path/"console", validate)
    assert manifest == before
    assert all(row["method"] == "lio_er" and runner.core.option(row["argv"], "--method") == "lio_er"
               for row in received[0]["runs"])
    assert all(row["method"] == "lio_eia_er" and runner.core.option(row["argv"], "--method") == "lio_eia_er"
               for row in manifest["runs"])
    projected = deepcopy(before)
    for row in projected["runs"]:
        row["method"] = "lio_er"
        row["argv"][row["argv"].index("--method")+1] = "lio_er"
    assert received == [projected]


def test_shards_and_all_bindings_restore(release, tmp_path):
    manifest, _ = release
    arms, validate, plan = runner.core.ARMS, runner.core.validate, runner.core.shard_plan
    with runner.runner_scope():
        runner.core.validate(manifest, tmp_path/"console")
        shards = [runner.core.shard_plan(manifest, index, 5) for index in range(5)]
        assert [len(rows) for rows in shards] == [6]*5
        assert len({row["output_directory"] for rows in shards for row in rows}) == 30
        assert all(row["method"] == "lio_eia_er" for rows in shards for row in rows)
        with pytest.raises(ValueError, match="K=2 smoke or K=5 formal"):
            runner.core.shard_plan(manifest, 0, 2)
    assert runner.core.ARMS is arms and runner.core.validate is validate and runner.core.shard_plan is plan
    with pytest.raises(RuntimeError):
        with runner.runner_scope():
            raise RuntimeError("injected")
    assert runner.core.ARMS is arms and runner.core.validate is validate and runner.core.shard_plan is plan


@pytest.mark.parametrize("mutation", ["method", "entrypoint", "missing_flag", "beta", "form", "weight",
                                     "objective", "followup_arm", "count", "hash", "missing_hash", "fallback"])
def test_runner_rejects_mutated_registered_contract(release, tmp_path, mutation):
    manifest, _ = release
    changed = deepcopy(manifest)
    row = changed["runs"][0]
    if mutation == "method":
        row["method"] = "lio_er"
        row["argv"][row["argv"].index("--method")+1] = "lio_er"
    elif mutation == "entrypoint":
        row["argv"][1] = str(REPO/"sensys_runs/iccps27_followup/eia_adapter.py")
    elif mutation == "missing_flag":
        row["argv"].remove("--energy_in_reward")
    elif mutation == "beta":
        row["effective_config"]["energy_reward_beta"] = 0.5
    elif mutation == "form":
        row["effective_config"]["energy_reward_form"] = "per_step"
    elif mutation == "weight":
        row["weight_pair"] = [1.5, 1.5]
    elif mutation in ("objective", "followup_arm"):
        row[mutation] = "plain"
    elif mutation == "count":
        changed["n_runs"] = 29
    elif mutation == "hash":
        changed["source"]["sha256"]["sensys_runs/energy_reward_runtime.py"] = "0"*64
    elif mutation == "missing_hash":
        del changed["source"]["sha256"]["sensys_runs/energy_reward_runtime.py"]
    else:
        changed["fallback_declared"] = True
    with runner.runner_scope(), pytest.raises(ValueError):
        runner.core.validate(changed, tmp_path/"console")


@pytest.mark.parametrize("gate", ["dirty", "head", "tag", "no_ntfy", "console_inside_repo", "source_dirty"])
def test_inherited_release_and_location_gates_remain(release, monkeypatch, tmp_path, gate):
    manifest, git = release
    console = tmp_path/"console"
    if gate == "dirty":
        monkeypatch.setattr(runner.core, "git", lambda *a: "?? unexpected/" if a == ("status", "--porcelain") else git(*a))
    elif gate == "head":
        monkeypatch.setattr(runner.core, "git", lambda *a: "wrong-head" if a == ("rev-parse", "HEAD") else git(*a))
    elif gate == "tag":
        monkeypatch.setattr(runner.core, "git", lambda *a: "iccps2027-followup-eia" if a == ("describe", "--tags", "--exact-match", "HEAD") else git(*a))
    elif gate == "no_ntfy":
        monkeypatch.delenv("NTFY_TOPIC")
    elif gate == "source_dirty":
        manifest["source"]["git_status"] = "?? unexpected/"
    else:
        console = REPO/"console"
    with runner.runner_scope(), pytest.raises(ValueError):
        runner.core.validate(manifest, console)


def test_tmux_commands_use_new_entrypoint_and_restore_existing_module(release, tmp_path):
    manifest, _ = release
    original_file = runner.existing.__file__
    token = "1234567890abcdef"*2
    commands = runner.tmux_plan(tmp_path/"manifest.json", manifest, 5,
                                "er_eia_formal", tmp_path/"console", token)
    assert len(commands) == 5 and commands[0][:2] == ["tmux", "new-session"]
    assert all(command[:2] == ["tmux", "new-window"] for command in commands[1:])
    assert runner.existing.__file__ == original_file
    for index, command in enumerate(commands):
        payload = command[-1]
        assert "er_eia_runner.py" in payload and str(Path(original_file)) not in payload
        assert "--shard "+str(index) in payload and "--of 5" in payload
        assert token in payload and "NTFY_TOPIC=test-never-send" in payload
        assert "PYTHONDONTWRITEBYTECODE=1" in payload
    with pytest.raises(ValueError):
        runner.tmux_plan(tmp_path/"manifest.json", manifest, 2,
                         "er_eia_formal", tmp_path/"console", token)
    assert runner.existing.__file__ == original_file


def test_main_passes_original_energy_eia_rows_to_existing_executor(release, monkeypatch, tmp_path):
    manifest, _ = release
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    received = []
    def execute(actual, rows, console, shard, count, timeout, synchronize):
        received.extend(rows)
        return 0
    monkeypatch.setattr(runner.core, "execute", execute)
    monkeypatch.setattr(runner.core, "WAIT4_SUPPORTED", True)
    original_scope, original_plan = runner.existing.runner_scope, runner.existing.tmux_plan
    assert runner.main([str(path), "--shard", "0", "--of", "5",
                        "--console-dir", str(tmp_path/"console")]) == 0
    assert len(received) == 6
    source = {row["output_directory"]: row for row in manifest["runs"]}
    for row in received:
        assert row["argv"] == source[row["output_directory"]]["argv"]
        assert row["method"] == "lio_eia_er" and "--energy_in_reward" in row["argv"]
    assert runner.existing.runner_scope is original_scope and runner.existing.tmux_plan is original_plan
