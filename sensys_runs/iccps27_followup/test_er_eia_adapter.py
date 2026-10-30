"""ER-LIO+EIA objective reachability, isolation and functional smoke contracts."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sensys_runs.iccps27_followup import er_eia_adapter as adapter
from sensys_runs.iccps27_followup import eia_adapter as eia
from sensys_runs.iccps27_followup.test_eia_adapter import (
    classes, config, FixedSession, instance,
)


def configured():
    cfg = config("lio_eia_er")
    adapter.configure(cfg)
    return cfg


def fixture_buffers(prime):
    return [SimpleNamespace(reward=[10.0+i, 20.0+i][:1 if i == 3 else 2],
                            total_energy=float((i+1)*(3 if prime else 2)))
            for i in range(4)]


def test_config_uses_exact_a9_configuration_and_explicit_arm():
    cfg, a9 = config("lio_eia_er"), config("lio_eia_er")
    adapter.energy.configure(a9)
    adapter.configure(cfg)
    for field in adapter.energy.REWARD_FIELDS:
        assert cfg[field] == a9[field]
    assert cfg.energy_reward_beta == 1.0
    assert cfg.energy_reward_form == "episode_mean"
    assert cfg.lio == a9.lio
    assert cfg.followup_arm == "er_lio_eia_er"
    assert cfg.training_objective == "energy_reward"
    assert cfg.display_label == "ER-LIO+EIA"
    record = adapter.energy_provenance(cfg)
    assert record["agent_ids"] == [0, 1, 2, 3]
    assert record["episode_kinds"] == ["ordinary", "prime"]
    assert record["consumers"] == ["actor_update", "incentive_learner"]


@pytest.mark.parametrize("fails", [False, True])
def test_unchanged_a9_wrapper_charges_all_four_normal_and_prime_buffers(monkeypatch, fails):
    trainer = ModuleType("er_eia_contract_trainer")
    calls, received = [], []
    def run_episode(*args, prime=False, **kwargs):
        calls.append(prime)
        return fixture_buffers(prime), 1
    trainer.run_episode = run_episode
    def train(cfg):
        for prime in (False, True):
            result = trainer.run_episode(prime=prime)
            buffers, done = result
            assert done == 1
            expected = fixture_buffers(prime)
            for agent, (buf, before) in enumerate(zip(buffers, expected)):
                penalty = before.total_energy/len(before.reward)
                assert buf.reward == [reward-penalty for reward in before.reward]
                assert buf.total_energy == before.total_energy
                received.append((agent, prime, tuple(buf.reward)))
        if fails:
            raise RuntimeError("injected training failure")
        return "complete"
    train.__module__ = trainer.__name__
    monkeypatch.setitem(sys.modules, trainer.__name__, trainer)
    eia_calls = []
    def eia_wrap(function, profile, method, output):
        eia_calls.append((function, profile, method, output))
        return function
    monkeypatch.setattr(eia, "wrap_eia_training", eia_wrap)
    wrapped = adapter.wrap_er_eia_training(train, "uniform_v1", "lio_eia_er")
    if fails:
        with pytest.raises(RuntimeError, match="injected training failure"):
            wrapped(configured())
    else:
        assert wrapped(configured()) == "complete"
    assert len(received) == 8 and calls == [False, True]
    assert trainer.run_episode is run_episode
    assert trainer.run_episode(prime=False)[0][1].reward == [11.0, 21.0]
    assert eia_calls == [(train, "uniform_v1", "lio_eia_er", None)]


@pytest.mark.parametrize("pair", eia.WEIGHTS)
def test_energy_objective_preserves_actual_eia_transform_and_16_channels(monkeypatch, pair):
    reached = classes("lio_eia_er")
    monkeypatch.setattr(eia, "agent_classes", lambda _: reached)
    before = config("lio_eia_er", pair)
    after = deepcopy(before)
    adapter.configure(after)
    original = eia.probe_attack("lio_eia_er", before)
    charged = eia.probe_attack("lio_eia_er", after)
    assert charged == original and len(charged["effective_multipliers"]) == 16
    for dtype in (np.float32, np.float64):
        clean = instance(reached[1], before)
        energy = instance(reached[1], after)
        fixed = FixedSession(dtype)
        for actions in ([0, 1, 2, 0], [2, 0, 1, 2], [1, 2, 0, 1]):
            left = clean.give_reward(np.zeros(15), actions, fixed)
            right = energy.give_reward(np.zeros(15), actions, fixed)
            assert left.dtype == right.dtype and left.tobytes() == right.tobytes()


def test_adapter_scope_restores_all_launcher_bindings_on_exception():
    names = ("_parse_args", "_write_provenance", "validate_run", "wrap_training", "importlib", "os")
    originals = {name: getattr(adapter.run_single, name) for name in names}
    aliases = {name: sys.modules.get(name) for name in ("lio_eia_er", "REFiNE_eia_er")}
    with pytest.raises(RuntimeError, match="injected scope failure"):
        with adapter.adapter_scope():
            assert adapter.run_single.wrap_training is not originals["wrap_training"]
            raise RuntimeError("injected scope failure")
    assert all(getattr(adapter.run_single, name) is value for name, value in originals.items())
    assert all(sys.modules.get(name) is value for name, value in aliases.items())


@pytest.mark.parametrize("flags", [[], ["--energy_in_reward", "--energy_in_reward"]])
def test_explicit_energy_opt_in_is_required_exactly_once(flags):
    with pytest.raises(SystemExit) as exc:
        adapter.main(flags)
    assert exc.value.code == 2


@pytest.mark.parametrize("root", ["relative", str(REPO), str(REPO/"scratch")])
def test_explicit_results_root_must_be_absolute_and_outside_repository(root):
    with pytest.raises(SystemExit) as exc:
        adapter.main(["--energy_in_reward", "--results-root", root])
    assert exc.value.code == 2


def test_existing_plain_eia_parser_still_rejects_energy_flag():
    with adapter.eia.adapter_scope():
        previous = sys.argv
        try:
            sys.argv = ["eia_adapter.py", "--method", "lio_eia_er", "--exp_name", "plain",
                        "--dir_name", "plain", "--seed", "12341", "--threads", "1",
                        "--n_agents", "4", "--min_at_lever", "2", "--w_lever", "1.1",
                        "--w_door", "0.9", "--cost_profile", "uniform_v1",
                        "--hardware_profile", "homogeneous_v1", "--energy_in_reward"]
            with pytest.raises(adapter.EnergyProfileError):
                adapter.run_single._parse_args()
        finally:
            sys.argv = previous


def test_run_single_seed_code_and_energy_provenance_are_reused(monkeypatch, tmp_path):
    import lio
    reached = classes("lio_eia_er")
    monkeypatch.setattr(eia, "agent_classes", lambda _: reached)
    cfg = config("lio_eia_er")
    cfg.alg.n_episodes, cfg.alg.period = 2, 2
    graph, order = SimpleNamespace(seed=None), []
    def tf_seed(value):
        graph.seed = value
        order.append(("seed", value))
    tf = SimpleNamespace(__version__="test-double", set_random_seed=tf_seed,
                         compat=SimpleNamespace(v1=SimpleNamespace(set_random_seed=tf_seed,
                                                                  get_default_graph=lambda: graph)))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    package = ModuleType("lio.alg")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "lio.alg", package)
    monkeypatch.setattr(lio, "alg", package, raising=False)
    for short, cls_name in (("lio_agent_er", "LIO"), ("REFiNE_er", "REFiNE"),
                            ("lio_eia_er", "ExploitativeLIO"), ("REFiNE_eia_er", "REFiNEExploitative")):
        module = ModuleType("lio.alg."+short)
        setattr(module, cls_name, reached[1] if "eia" in short else reached[0])
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(package, short, module, raising=False)
    trainer = ModuleType("lio.alg.train_lio_eia_er")
    original_episode = lambda prime=False: (fixture_buffers(prime), 1)
    trainer.run_episode = original_episode
    def train(c):
        order.append(("train", graph.seed))
        assert sys.modules["lio_eia_er"] is sys.modules["lio.alg.lio_eia_er"]
        adapter.validate_energy_config(c)
        assert trainer.run_episode(prime=False)[0][1].reward == [9.0, 19.0]
    train.__module__ = trainer.__name__
    trainer.train = train
    monkeypatch.setitem(sys.modules, trainer.__name__, trainer)
    old_import = adapter.run_single.importlib
    def importer(name):
        if name == "lio.alg.config_room_lio":
            return SimpleNamespace(get_config=lambda: deepcopy(cfg))
        if name == trainer.__name__:
            order.append(("trainer_import", name))
            return trainer
        return old_import.import_module(name)
    monkeypatch.setattr(adapter.run_single, "importlib", SimpleNamespace(import_module=importer))
    monkeypatch.setattr(eia, "wrap_eia_training", lambda fn, *args: fn)
    monkeypatch.setattr(adapter.run_single, "_git_field", lambda cmd: "fixture")
    monkeypatch.setattr(adapter.run_single, "_git_dirty", lambda: False)
    monkeypatch.setattr(adapter.run_single, "_set_thread_env", lambda threads: None)
    args = ["--method", "lio_eia_er", "--exp_name", "local_test", "--dir_name", "er_lio_eia",
            "--seed", "12341", "--platform", "local", "--threads", "1", "--n_agents", "4",
            "--min_at_lever", "2", "--cost_profile", "uniform_v1", "--hardware_profile", "homogeneous_v1",
            "--w_lever", "1.1", "--w_door", "0.9", "--n_episodes", "2", "--period", "2",
            "--energy_in_reward", "--results-root", str(tmp_path/"results")]
    old_np, old_random = np.random.get_state(), random.getstate()
    old_argv, old_cwd = sys.argv, os.getcwd()
    try:
        with pytest.raises(SystemExit) as exc:
            adapter.main(args)
        assert exc.value.code == 0
    finally:
        np.random.set_state(old_np)
        random.setstate(old_random)
    assert order == [("seed", 12341), ("trainer_import", trainer.__name__),
                     ("seed", 12341), ("train", 12341)]
    assert trainer.run_episode is original_episode
    assert sys.argv is old_argv and os.getcwd() == old_cwd
    record = json.loads((tmp_path/"results/local_test/er_lio_eia/provenance.json").read_text())
    assert record["method"] == "lio_eia_er" and record["display_label"] == "ER-LIO+EIA"
    assert record["energy_in_reward"] is True and record["energy_reward_beta"] == 1.0
    assert record["energy_reward"]["agent_ids"] == [0, 1, 2, 3]
    assert record["tf_graph_seed"] == 12341
    assert record["er_eia_launch_argv"][2:] == args
    assert record["er_eia_launch_argv"].count("--energy_in_reward") == 1
    assert record["er_eia_source_sha256"] == adapter.source_hashes()
    assert len(record["eia_attack"]["effective_multipliers"]) == 16


@pytest.mark.skipif(os.environ.get("REFINE_ER_EIA_REAL_SMOKE") != "1", reason="Explicit real-TF functional smoke opt-in required")
def test_real_tensorflow_two_episode_energy_eia_functional_smoke(tmp_path):
    output = tmp_path/"real_results"
    args = [sys.executable, "-B", str(Path(adapter.__file__)), "--method", "lio_eia_er",
            "--exp_name", "er_eia_functional_smoke", "--dir_name", "energy_lio_eia",
            "--seed", "12341", "--platform", "windows_functional", "--threads", "1",
            "--n_agents", "4", "--min_at_lever", "2", "--cost_profile", "uniform_v1",
            "--hardware_profile", "homogeneous_v1", "--w_lever", "1.1", "--w_door", "0.9",
            "--n_episodes", "2", "--period", "2", "--n_eval", "10", "--energy_in_reward",
            "--results-root", str(output)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TF_ENABLE_ONEDNN_OPTS="0",
               CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(REPO))
    result = subprocess.run(args, cwd=REPO, env=env, text=True, capture_output=True, timeout=180)
    (tmp_path/"smoke_console.txt").write_text(result.stdout+result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout+result.stderr
    run = output/"er_eia_functional_smoke/energy_lio_eia"
    meta = json.loads((run/"energy_eval_provenance.json").read_text())
    provenance = json.loads((run/"provenance.json").read_text())
    completion = json.loads((run/"eia_run_completion.json").read_text())
    with (run/"energy_eval_episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400 and len({row["eval_episode"] for row in rows}) == 100
    assert {row["method"] for row in rows} == {"lio_eia_er"}
    assert meta["config"]["display_label"] == provenance["display_label"] == "ER-LIO+EIA"
    assert meta["config"]["energy_in_reward"] is provenance["energy_in_reward"] is True
    assert meta["config"]["energy_reward_beta"] == provenance["energy_reward_beta"] == 1.0
    assert meta["config"]["energy_reward_form"] == "episode_mean"
    assert meta["eia_attack"] == provenance["eia_attack"] == meta["config"]["eia_attack"]
    assert meta["effective_uniform_costs"] == [{"agent_id": i, "costs": [1.0]*3} for i in range(4)]
    assert completion["status"] == "complete" and completion["final_evaluation_rows"] == 400
    assert provenance["er_eia_source_sha256"] == adapter.source_hashes()
    assert len(provenance["eia_attack"]["effective_multipliers"]) == 16


def test_energy_objective_does_not_change_reset_observation_or_payment_memory():
    from sensys_runs.iccps27_followup import reset_probe
    before = config("lio_eia_er")
    after = deepcopy(before)
    adapter.configure(after)
    reset_before = reset_probe.reset_observations(before)
    reset_after = reset_probe.reset_observations(after)
    assert reset_before.shape == reset_after.shape == (4, 15)
    assert np.array_equal(reset_before, reset_after)
    assert np.array_equal(reset_after[:, 12:], np.zeros((4, 3)))
