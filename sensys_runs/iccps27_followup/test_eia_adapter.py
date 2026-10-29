"""Exercise actual EIA payment functions, scoped integration and matrix contracts."""
from __future__ import annotations

import ast
from copy import deepcopy
import csv
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from lio.utils import util
from sensys_runs import energy_profile_campaign as base
from sensys_runs.energy_profiles import get_profile, get_hardware_profile
from sensys_runs.iccps27_followup import eia_adapter as adapter
from sensys_runs.iccps27_followup import eia_campaign as campaign


def source_class(relative, name, source=None):
    source = (REPO / relative).read_text(encoding="utf-8-sig") if source is None else source
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(np=np, util=util, PolicyNew=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    cls = namespace[name]
    def networks(agent):
        agent.obs, agent.action_others, agent.reward_function = object(), object(), object()
    cls.create_networks = networks
    return cls


def classes(method):
    return tuple(source_class(module.replace(".", "/") + ".py", name)
                 for module, name in adapter.CLASS_MAP[method])


def config(method, pair=(1.1, 0.9)):
    cfg = base.load_config(method)
    cfg.env.n_agents, cfg.env.min_at_lever = 4, 2
    cfg.lio.min_at_lever = 2
    cfg.lio.eia_w_lever, cfg.lio.eia_w_door = pair
    if method == "refine_eia_er":
        cfg.lio.Fairness_multiplier, cfg.lio.energy_weight = 0.0, 1.0
    cfg.cost_profile = get_profile("uniform_v1").snapshot()
    hardware = get_hardware_profile("homogeneous_v1")
    cfg.hardware_profile, cfg.hardware_profile_lambda = hardware.snapshot(4), list(hardware.vector(4))
    return cfg


class FixedSession:
    def __init__(self, dtype=np.float32):
        self.values = np.array([[0.125, 0.33333334, 0.75, 0.999]], dtype=dtype)
    def run(self, operation, feed_dict):
        return self.values.copy()


def instance(cls, cfg, giver=1):
    return cls(cfg.lio, 15, 3, cfg.nn, f"agent_{giver}", 2.0, 4, giver, 1.0)


@pytest.mark.parametrize("pair", [(1.1, 0.9), (1.5, 1.5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_actual_nondefault_payment_functions_agree(pair, dtype):
    lio = instance(classes("lio_eia_er")[1], config("lio_eia_er", pair))
    refine = instance(classes("refine_eia_er")[1], config("refine_eia_er", pair))
    session = FixedSession(dtype)
    for actions in itertools.product(range(3), repeat=4):
        left = lio.give_reward(np.zeros(15), actions, session)
        right = refine.give_reward(np.zeros(15), actions, session)
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
        expected = session.values.flatten()
        for recipient, action in enumerate(actions):
            if recipient != 1 and action in (0, 2):
                expected[recipient] = pair[0 if action == 0 else 1] * expected[recipient]
        expected = expected.flatten() * 2.0
        assert right.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_default_refine_output_bit_identical_to_base_source(dtype):
    original = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "08b95744ee6ba21e51ee8011a5a872a78a842efa:lio/alg/REFiNE_eia_er.py"],
        text=True, encoding="utf-8")
    old_cls = source_class("lio/alg/REFiNE_eia_er.py", "REFiNEExploitative", original)
    new_cls = classes("refine_eia_er")[1]
    cfg = config("refine_eia_er", (2.0, 0.2))
    old, new = instance(old_cls, cfg), instance(new_cls, cfg)
    session = FixedSession(dtype)
    for actions in itertools.product(range(3), repeat=4):
        before = old.give_reward(np.zeros(15), actions, session)
        after = new.give_reward(np.zeros(15), actions, session)
        assert after.dtype == before.dtype and after.tobytes() == before.tobytes()


@pytest.mark.parametrize("method", adapter.METHODS)
@pytest.mark.parametrize("pair", adapter.WEIGHTS)
def test_provenance_probes_effective_per_giver_recipient_action(monkeypatch, method, pair):
    reached = classes(method)
    monkeypatch.setattr(adapter, "agent_classes", lambda _: reached)
    result = adapter.probe_attack(method, config(method, pair))
    assert result["configured_weights"] == dict(lever=pair[0], door=pair[1])
    assert result["applies_to"] == "recipient_action"
    assert result["training_active"] and result["evaluation_active"]
    assert len(result["effective_multipliers"]) == 16
    for row in result["effective_multipliers"]:
        active = row["giver_id"] == 1 and row["recipient_id"] != 1
        assert row["multipliers"] == ([pair[0], 1.0, pair[1]] if active else [1.0] * 3)
        assert row["delivered"] == (row["giver_id"] != row["recipient_id"])


@pytest.mark.parametrize("method", adapter.METHODS)
def test_uniform_scope_reaches_all_four_agents_and_restores(monkeypatch, method):
    reached = classes(method)
    monkeypatch.setattr(adapter, "agent_classes", lambda _: reached)
    original = [cls.calculate_energy_cost for cls in reached]
    cfg = config(method)
    agents = [instance(reached[1] if i == 1 else reached[0], cfg, i) for i in range(4)]
    with adapter.uniform_cost_scope(method):
        assert [[a.calculate_energy_cost(np.zeros(15), action) for action in range(3)] for a in agents] == [[1.0] * 3] * 4
    assert [cls.calculate_energy_cost for cls in reached] == original
    with pytest.raises(RuntimeError):
        with adapter.uniform_cost_scope(method):
            raise RuntimeError("injected")
    assert [cls.calculate_energy_cost for cls in reached] == original


@pytest.fixture
def isolated_campaign(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "REPO", tmp_path / "repo")
    monkeypatch.setattr(campaign, "source_identity", lambda: dict(git_commit="fixture", sha256={}))
    return tmp_path


@pytest.mark.parametrize("fallback,count", [(False, 60), (True, 40)])
def test_exact_matrix_inventory_and_objectives(isolated_campaign, fallback, count):
    manifest = campaign.build_campaign("eia_contract", "local", 1, drop_mid_weight=fallback)
    pairs = (adapter.WEIGHTS[0], adapter.WEIGHTS[2]) if fallback else adapter.WEIGHTS
    expected = {(method, pair, seed) for method in adapter.METHODS for pair in pairs for seed in range(12341, 12351)}
    actual = {(row["method"], tuple(row["weight_pair"]), row["seed"]) for row in manifest["runs"]}
    assert len(manifest["runs"]) == manifest["n_runs"] == count
    assert actual == expected and manifest["fallback_declared"] is fallback
    for row in manifest["runs"]:
        argv, cfg = row["argv"], row["effective_config"]
        assert argv[1].endswith("eia_adapter.py") and "--energy_in_reward" not in argv
        assert cfg.hardware_profile_lambda == [1.0] * 4
        assert cfg.cost_profile.normalized_vector == (1.0, 1.0, 1.0)
        if row["method"] == "refine_eia_er":
            assert cfg.lio.Fairness_multiplier == 0.0 and cfg.lio.energy_weight == 1.0
        assert cfg.alg.n_episodes == 25000 and cfg.alg.n_test == 100
    rows = campaign.inventory_rows(manifest)
    assert len(rows) == count and len({r["run_id"] for r in rows}) == count
    assert all(tuple(r) == campaign.FIELDS and r["checkpoint_prefix"] == "lio/results/" + r["run_id"] + "/model.ckpt" for r in rows)


def test_smoke_uses_nondefault_weights_for_both_methods(isolated_campaign):
    manifest = campaign.build_campaign("eia_contract", "local", 1, smoke=True)
    assert manifest["n_runs"] == 2
    assert {r["method"] for r in manifest["runs"]} == set(adapter.METHODS)
    assert all(r["weight_pair"] == [1.1, 0.9] and r["seed"] == 12341 for r in manifest["runs"])


def test_campaign_outputs_are_exclusive(isolated_campaign):
    out, inventory = isolated_campaign / "matrix.json", isolated_campaign / "inventory.csv"
    args = ["--platform", "local", "--threads", "1", "--out", str(out), "--inventory-out", str(inventory)]
    inventory.write_text("retained", encoding="utf-8")
    assert campaign.main(args) == 2 and not out.exists()
    inventory.unlink()
    assert campaign.main(args) == 0
    old = (out.read_bytes(), inventory.read_bytes())
    assert campaign.main(args) == 2
    assert (out.read_bytes(), inventory.read_bytes()) == old


@pytest.mark.parametrize("method", adapter.METHODS)
def test_launcher_seed_code_alias_identity_and_scoped_restoration(monkeypatch, tmp_path, method):
    import lio
    reached = classes(method)
    monkeypatch.setattr(adapter, "agent_classes", lambda _: reached)
    cfg = config(method)
    cfg.alg.n_episodes, cfg.alg.period = 2, 2
    graph, order = SimpleNamespace(seed=None), []
    def tf_seed(value):
        graph.seed = value
        order.append(("seed", value))
    tf = SimpleNamespace(__version__="test-double", set_random_seed=tf_seed,
                         compat=SimpleNamespace(v1=SimpleNamespace(set_random_seed=tf_seed, get_default_graph=lambda: graph)))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    package = ModuleType("lio.alg")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "lio.alg", package)
    monkeypatch.setattr(lio, "alg", package, raising=False)
    for module_name, cls_name in (("lio_agent_er", "LIO"), ("REFiNE_er", "REFiNE"),
                                 ("lio_eia_er", "ExploitativeLIO"), ("REFiNE_eia_er", "REFiNEExploitative")):
        module = ModuleType("lio.alg." + module_name)
        setattr(module, cls_name, reached[1] if "eia" in module_name else reached[0])
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(package, module_name, module, raising=False)
    config_name, trainer_name = adapter.run_single.METHOD_MAP[method]
    config_module = SimpleNamespace(get_config=lambda: deepcopy(cfg))
    trainer_module = SimpleNamespace(train=lambda c: None)
    old_import = adapter.run_single.importlib
    def importer(name):
        if name == "lio.alg." + config_name:
            return config_module
        if name == "lio.alg." + trainer_name:
            order.append(("trainer_import", name))
            return trainer_module
        return old_import.import_module(name)
    monkeypatch.setattr(adapter.run_single, "importlib", SimpleNamespace(import_module=importer))
    def wrap(train, profile, selected, results_root):
        def run(c):
            order.append(("train", graph.seed))
            target = adapter.CLASS_MAP[selected][1][0]
            assert sys.modules[target.rsplit(".", 1)[-1]] is sys.modules[target]
        return run
    monkeypatch.setattr(adapter, "wrap_eia_training", wrap)
    monkeypatch.setattr(adapter.run_single, "_git_field", lambda cmd: "fixture")
    monkeypatch.setattr(adapter.run_single, "_git_dirty", lambda: False)
    monkeypatch.setattr(adapter.run_single, "_set_thread_env", lambda threads: None)
    original_validate = adapter.run_single.validate_run
    args = ["--method", method, "--exp_name", "local_test", "--dir_name", method,
            "--seed", "12341", "--platform", "local", "--threads", "1",
            "--n_agents", "4", "--min_at_lever", "2", "--cost_profile", "uniform_v1",
            "--hardware_profile", "homogeneous_v1", "--w_lever", "1.1", "--w_door", "0.9",
            "--n_episodes", "2", "--period", "2", "--results-root", str(tmp_path / "results")]
    if method == "refine_eia_er":
        args += ["--fairness_mult", "0", "--energy_weight", "1"]
    before_np, before_random = np.random.get_state(), __import__("random").getstate()
    try:
        with pytest.raises(SystemExit) as exc:
            adapter.main(args)
        assert exc.value.code == 0
    finally:
        np.random.set_state(before_np)
        __import__("random").setstate(before_random)
    assert order == [("seed", 12341), ("trainer_import", "lio.alg." + trainer_name),
                     ("seed", 12341), ("train", 12341)]
    assert adapter.run_single.validate_run is original_validate
    provenance = json.loads((tmp_path / "results/local_test" / method / "provenance.json").read_text())
    assert provenance["tf_graph_seed"] == 12341 and provenance["eia_attack"]["configured_weights"] == dict(lever=1.1, door=0.9)


@pytest.mark.skipif(os.environ.get("REFINE_EIA_REAL_SMOKE") != "1", reason="Explicit real-TF smoke opt-in required")
@pytest.mark.parametrize("method", adapter.METHODS)
def test_real_tensorflow_two_episode_functional_smoke(tmp_path, method):
    output = tmp_path / "real_results"
    args = [sys.executable, "-B", str(Path(adapter.__file__)), "--method", method,
            "--exp_name", "eia_functional_smoke", "--dir_name", method,
            "--seed", "12341", "--platform", "windows_functional", "--threads", "1",
            "--n_agents", "4", "--min_at_lever", "2", "--cost_profile", "uniform_v1",
            "--hardware_profile", "homogeneous_v1", "--w_lever", "1.1", "--w_door", "0.9",
            "--n_episodes", "2", "--period", "2", "--n_eval", "10", "--results-root", str(output)]
    if method == "refine_eia_er":
        args += ["--fairness_mult", "0", "--energy_weight", "1"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TF_ENABLE_ONEDNN_OPTS="0",
               CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(REPO))
    result = subprocess.run(args, cwd=REPO, env=env, text=True, capture_output=True, timeout=180)
    (tmp_path / "smoke_console.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    run = output / "eia_functional_smoke" / method
    meta = json.loads((run / "energy_eval_provenance.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    completion = json.loads((run / "eia_run_completion.json").read_text())
    with (run / "energy_eval_episodes.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 400 and len({r["eval_episode"] for r in records}) == 100
    assert {r["method"] for r in records} == {method}
    assert meta["method"] == method and meta["n_eval_episodes"] == 100
    assert meta["eia_attack"] == provenance["eia_attack"] == meta["config"]["eia_attack"]
    assert meta["effective_uniform_costs"] == [{"agent_id": i, "costs": [1.0] * 3} for i in range(4)]
    assert completion["status"] == "complete" and completion["final_evaluation_rows"] == 400
    assert provenance["tf_graph_seed"] == 12341
    assert provenance["eia_attack"]["configured_weights"] == dict(lever=1.1, door=0.9)


@pytest.fixture
def runner_contract(monkeypatch):
    from sensys_runs.iccps27_followup import eia_runner
    manifest = campaign.build_campaign("iccps27_eia_runner_contract_20261028", "local", 1)
    def git(*args):
        if args == ("rev-parse", "HEAD"):
            return manifest["source"]["git_commit"]
        if args == ("describe", "--tags", "--exact-match", "HEAD"):
            return manifest["source"]["git_describe"]
        return ""
    monkeypatch.setattr(eia_runner.core, "git", git)
    monkeypatch.setenv("NTFY_TOPIC", "test-never-send")
    return eia_runner, manifest, git


def test_eia_runner_preserves_original_gates_and_restores_scope(runner_contract, monkeypatch, tmp_path):
    eia_runner, manifest, git = runner_contract
    original_arms, original_validate = eia_runner.core.ARMS, eia_runner.core.validate
    with eia_runner.runner_scope():
        eia_runner.core.validate(manifest, tmp_path / "console")
        rows = [eia_runner.core.shard_plan(manifest, shard, 5) for shard in range(5)]
        assert all(len(part) == 12 for part in rows)
        assert len({row["output_directory"] for part in rows for row in part}) == 60
        with pytest.raises(ValueError, match="K=2 smoke or K=5 formal"):
            eia_runner.core.shard_plan(manifest, 0, 1)
        monkeypatch.setattr(eia_runner.core, "git", lambda *args: "?? untracked/" if args == ("status", "--porcelain") else git(*args))
        with pytest.raises(ValueError, match="porcelain-empty"):
            eia_runner.core.validate(manifest, tmp_path / "console")
    assert eia_runner.core.ARMS is original_arms
    assert eia_runner.core.validate is original_validate


@pytest.mark.parametrize("mutation", ["hash", "matrix", "argv", "tag"])
def test_eia_runner_refuses_tampered_release_contract(runner_contract, monkeypatch, tmp_path, mutation):
    eia_runner, original, git = runner_contract
    manifest = deepcopy(original)
    if mutation == "hash":
        manifest["source"]["sha256"]["lio/alg/REFiNE_eia_er.py"] = "0" * 64
    elif mutation == "matrix":
        manifest["runs"][0]["weight_pair"] = [1.5, 1.5]
    elif mutation == "argv":
        manifest["runs"][0]["argv"][1] = str(REPO / "sensys_runs/run_single.py")
    else:
        monkeypatch.setattr(eia_runner.core, "git", lambda *args: "wrong-release" if args == ("describe", "--tags", "--exact-match", "HEAD") else git(*args))
    with eia_runner.runner_scope(), pytest.raises(ValueError):
        eia_runner.core.validate(manifest, tmp_path / "console")


def test_eia_tmux_plan_has_fixed_shards_barrier_and_notifications(runner_contract, tmp_path):
    eia_runner, manifest, _ = runner_contract
    token = "1234567890abcdef" * 2
    commands = eia_runner.tmux_plan(tmp_path / "manifest.json", manifest, 5,
                                   "eia_formal", tmp_path / "console", token)
    assert len(commands) == 5 and commands[0][:2] == ["tmux", "new-session"]
    assert all(command[:2] == ["tmux", "new-window"] for command in commands[1:])
    for index, command in enumerate(commands):
        payload = command[-1]
        assert "--shard " + str(index) in payload and "--of 5" in payload
        assert token in payload and "NTFY_TOPIC=test-never-send" in payload
        assert "PYTHONDONTWRITEBYTECODE=1" in payload and "eia_runner.py" in payload
    with pytest.raises(ValueError):
        eia_runner.tmux_plan(tmp_path / "manifest.json", manifest, 2,
                            "eia_formal", tmp_path / "console", token)
