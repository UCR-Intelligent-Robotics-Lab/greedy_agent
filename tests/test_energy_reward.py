"""Scripted P5 probes execute unchanged training bodies; no TensorFlow required."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensys_runs import energy_reward_runtime as reward
from sensys_runs import energy_profile_runtime as cost
from sensys_runs import energy_profile_campaign as wave1
from sensys_runs import energy_hardware_campaign as wave2
from sensys_runs import energy_reward_campaign as wave3
from sensys_runs import energy_profile_runner as runner
from sensys_runs import summarize_energy_profiles as summary
from sensys_runs import energy_reward_summary as objective_summary
from sensys_runs.energy_profiles import PROFILES, get_profile, get_hardware_profile
from sensys_runs.energy_profile_eval import evaluate_final_policy, EPISODE_FILE, MANIFEST_FILE
from test_energy_profiles import source_objects
from test_energy_profile_eval import fixture_model, saved_fixture

BASE = "e1fbd8cb31f167d5ad993c1cf877c04c417151b8"


@pytest.fixture
def model(monkeypatch):
    import numpy as np
    cfg = wave1.load_config("lio_er")
    cfg.env.n_agents, cfg.env.min_at_lever = wave1.TASK
    cfg.cost_profile = get_profile("paper_ratio_v1").snapshot()
    cfg.hardware_profile = get_hardware_profile("hetero_2x1p5_v1").snapshot(cfg.env.n_agents)
    cfg.hardware_profile_lambda = cfg.hardware_profile.hardware_profile_lambda
    ns = source_objects("lio/alg/train_lio_er.py", {"run_episode", "Buffer"}, {"np": np})
    trainer = SimpleNamespace(run_episode=ns["run_episode"])
    actor = source_objects("lio/env/room_agent.py", {"Actor"}, {"np": np})["Actor"]
    env_cls = source_objects("lio/env/room_symmetric.py", {"Env"}, {"np": np, "room_agent": SimpleNamespace(Actor=actor)})["Env"]
    class Agent:
        energy_param, can_give, include_cost_in_chain_rule = 1.0, False, False
        def calculate_energy_cost(self, state, action): raise AssertionError("P1 scope missing")
    agents = [Agent() for _ in range(cfg.env.n_agents)]
    for i, agent in enumerate(agents): agent.agent_id = i
    real_import = cost.importlib.import_module
    monkeypatch.setattr(cost.importlib, "import_module", lambda name: SimpleNamespace(LIO=Agent)
                        if name == "lio.alg.lio_agent_er" else real_import(name))
    def episode(prime):
        for agent in agents:
            lever = agent.agent_id in ((1, 3) if prime else (0, 2))
            sequence = iter([1, 0 if lever else 2])
            agent.run_actor = lambda *a, sequence=sequence: next(sequence)
        env = env_cls(cfg.env)
        before = env.step
        result = trainer.run_episode(None, env, agents, 0.0, prime=prime)
        assert env.step == before and "step" not in vars(env)
        return result
    def scoped(fn): return cost.wrap_training(lambda c: fn(), "paper_ratio_v1", "lio_er")(cfg)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("prime", [False, True])
def test_each_agent_reward_uses_its_lambda_weighted_total(model, prime):
    raw, raw_done = model.scoped(lambda: model.episode(prime))
    original = model.trainer.run_episode
    identities = []
    def recording(*a, **kw):
        result = original(*a, **kw)
        identities.extend(id(b.reward) for b in result[0])
        return result
    model.trainer.run_episode = recording
    with reward.energy_reward_scope(model.trainer, reward.registered_beta()):
        shaped, done = model.scoped(lambda: model.episode(prime))
    assert model.trainer.run_episode is recording and done == raw_done
    assert identities == [id(b.reward) for b in shaped]
    assert raw[2].total_energy == pytest.approx(1.5 * raw[0].total_energy)
    for before, after in zip(raw, shaped):
        assert before.total_energy > 0
        penalty = reward.registered_beta() * before.total_energy / len(before.reward)
        assert after.reward == [r - penalty for r in before.reward]
        assert after.reward != before.reward
        assert after.total_energy == before.total_energy and after.energy_cost == before.energy_cost


@pytest.mark.parametrize("where", ["body", "episode"])
def test_scope_restored_on_exception(model, where):
    def failing(*a, **kw): raise RuntimeError("episode failed")
    if where == "episode": model.trainer.run_episode = failing
    original = model.trainer.run_episode
    with pytest.raises(RuntimeError):
        with reward.energy_reward_scope(model.trainer, reward.registered_beta()):
            if where == "episode": model.scoped(lambda: model.episode(False))
            else: raise RuntimeError("body failed")
    assert model.trainer.run_episode is original


def test_final_sidecar_keeps_raw_reward_while_training_is_shaped(model, tmp_path):
    raw_dir, shaped_dir = tmp_path / "raw", tmp_path / "shaped"
    raw_dir.mkdir(); shaped_dir.mkdir()
    saved_fixture(raw_dir, "paper_ratio_v1", "lio_er")
    cfg, env, agents, evaluator = fixture_model("paper_ratio_v1", "lio_er")
    reward.configure(cfg)
    prefix = shaped_dir / cfg.main.model_name
    for suffix in (".index", ".meta", ".data-00000-of-00001"):
        Path(str(prefix) + suffix).write_bytes(b"synthetic checkpoint fixture")
    raw_buffers, _ = model.scoped(lambda: model.episode(False))
    with reward.energy_reward_scope(model.trainer, reward.registered_beta()):
        shaped_buffers, _ = model.scoped(lambda: model.episode(False))
        assert shaped_buffers[0].reward != raw_buffers[0].reward
        evaluate_final_policy(None, env, agents, cfg, prefix, "lio_er", evaluator)
    assert (raw_dir / EPISODE_FILE).read_bytes() == (shaped_dir / EPISODE_FILE).read_bytes()
    assert json.loads((shaped_dir / MANIFEST_FILE).read_text())["config"]["energy_in_reward"] is True


@pytest.mark.parametrize("method,profile", [("refine_er", "paper_ratio_v1"), ("lio_er", "legacy")])
def test_flag_refusal_before_tensorflow(monkeypatch, tmp_path, method, profile):
    from sensys_runs import run_single
    monkeypatch.setattr(run_single, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run_single", "--method", method, "--cost_profile", profile,
                                     "--exp_name", "unused", "--dir_name", "unused", "--seed", "1", "--energy_in_reward"])
    with pytest.raises(SystemExit) as exc: run_single.main()
    assert exc.value.code == 2 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_config_provenance_and_scoped_dispatch(monkeypatch, tmp_path, enabled):
    from sensys_runs import run_single
    cfg = wave1.load_config("lio_er")
    class Agent:
        def calculate_energy_cost(self, state, action): return -1
    def original(*args, **kw): return [SimpleNamespace(reward=[4.0, 2.0], total_energy=3.0)], 1
    trainer = SimpleNamespace(run_episode=original)
    def train(config):
        for prime in (False, True):
            buffers, _ = trainer.run_episode(prime=prime)
            expected = [4.0, 2.0]
            if enabled: expected = [r-reward.registered_beta()*3/2 for r in expected]
            assert buffers[0].reward == expected
        path = tmp_path / "lio/results/p5/probe/config.json"
        path.write_text(json.dumps(config))
    trainer.train = train
    modules = {"lio.alg.config_room_lio": SimpleNamespace(get_config=lambda: cfg),
               "lio.alg.train_lio_er": trainer, "lio.alg.lio_agent_er": SimpleNamespace(LIO=Agent)}
    real_import = cost.importlib.import_module
    monkeypatch.setattr(cost.importlib, "import_module", lambda name: modules[name] if name in modules else real_import(name))
    graph = SimpleNamespace(seed=None)
    def seed(value): graph.seed = value
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(__version__="stub", set_random_seed=seed,
                        compat=SimpleNamespace(v1=SimpleNamespace(set_random_seed=seed, get_default_graph=lambda: graph))))
    for key in ("REPO_ROOT", "LIO_ALG"): monkeypatch.setattr(run_single, key, str(tmp_path))
    monkeypatch.setattr(run_single, "_git_field", lambda cmd: "fixture")
    monkeypatch.setattr(run_single, "_git_dirty", lambda: False)
    monkeypatch.setattr(run_single, "_set_thread_env", lambda n: None)
    monkeypatch.chdir(tmp_path)
    argv = ["run_single", "--method", "lio_er", "--exp_name", "p5", "--dir_name", "probe", "--seed", "1", "--cost_profile", "paper_ratio_v1"]
    monkeypatch.setattr(sys, "argv", argv + (["--energy_in_reward"] if enabled else []))
    with pytest.raises(SystemExit) as exc: run_single.main()
    assert exc.value.code == 0 and trainer.run_episode is original
    for name in ("config.json", "provenance.json"):
        record = json.loads((tmp_path / "lio/results/p5/probe" / name).read_text())
        if enabled:
            assert record["energy_in_reward"] is True
            assert record["energy_reward_beta"] == reward.registered_beta()
            assert record["energy_reward_form"] == "episode_mean"
        else: assert not set(reward.REWARD_FIELDS) & record.keys()


def test_wave3_pinned_and_earlier_waves_unchanged(monkeypatch, tmp_path):
    configs = {m: wave1.load_config(m) for m in ("lio_er", "refine_er")}
    monkeypatch.setattr(wave1, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(wave1, "load_config", lambda m: deepcopy(configs[m]))
    monkeypatch.setattr(wave1, "source_identity", lambda: {"git_commit": BASE})
    for current, name in [(wave1, "energy_profile_campaign"), (wave2, "energy_hardware_campaign")]:
        source = subprocess.check_output(["git", "show", BASE + ":sensys_runs/" + name + ".py"], cwd=ROOT, text=True)
        old = ModuleType("frozen_" + name); old.__file__ = current.__file__
        exec(compile(source, old.__file__, "exec"), old.__dict__)
        if current is wave1:
            old.REPO_ROOT, old.load_config, old.source_identity = wave1.REPO_ROOT, wave1.load_config, wave1.source_identity
        for smoke in (False, True):
            a = old.build_campaign("same", "local", 1, smoke=smoke)
            b = current.build_campaign("same", "local", 1, smoke=smoke)
            assert json.dumps(a["runs"], sort_keys=True) == json.dumps(b["runs"], sort_keys=True)
    for smoke in (False, True):
        m = wave3.build_campaign("same", "local", 1, smoke=smoke)
        indices = wave1.SEED_INDICES[:1] if smoke else wave1.SEED_INDICES
        expected = [(h, wave1.run_batch._seed(i)) for h in wave3.WAVE3_HARDWARE for i in indices]
        assert [(r["hardware_profile"], r["seed"]) for r in m["runs"]] == expected
        assert m["n_runs"] == len(expected) and len({r["output_directory"] for r in m["runs"]}) == len(expected)
        for r in m["runs"]:
            cfg, argv = r["effective_config"], r["argv"]
            assert cfg.energy_in_reward is True and cfg.energy_reward_beta == configs["refine_er"].lio.energy_weight
            assert cfg.lio.Fairness_multiplier == 0 and r["method"] == "lio_er"
            assert r["training_profile"] == "paper_ratio_v1" and argv.count("--energy_in_reward") == 1
            assert cfg.main.exp_name == "same_wave3" + ("_smoke" if smoke else "")
            assert argv[argv.index("--hardware_profile")+1] == r["hardware_profile"]
        monkeypatch.setattr(runner, "REPO_ROOT", wave1.REPO_ROOT)
        monkeypatch.setenv("NTFY_TOPIC", "tests-never-send")
        monkeypatch.setattr(runner, "git", lambda *args: BASE if args == ("rev-parse", "HEAD") else "")
        runner.validate(m, tmp_path / "console")
        assert [r["hardware_profile"] for r in runner.shard_plan(m, 0, 1)[:2]] == list(wave3.WAVE3_HARDWARE)


def objective_fixture(directory, method, hardware, enabled=False, profile="paper_ratio_v1"):
    directory.mkdir()
    saved_fixture(directory, profile, method)
    meta_path = directory / MANIFEST_FILE
    meta = json.loads(meta_path.read_text())
    snapshot = get_hardware_profile(hardware).snapshot(meta["n_agents"])
    meta["config"].update(hardware_profile=snapshot, hardware_profile_lambda=snapshot["hardware_profile_lambda"])
    meta["config"]["lio"]["Fairness_multiplier"] = 0.0
    prov = dict(git_commit=BASE, git_dirty=False, hardware_profile=snapshot,
                hardware_profile_lambda=snapshot["hardware_profile_lambda"])
    if enabled:
        fields = dict(energy_in_reward=True, energy_reward_beta=reward.registered_beta(), energy_reward_form=reward.REWARD_FORM)
        meta["config"].update(fields); prov.update(fields)
    meta_path.write_text(json.dumps(meta))
    (directory / "provenance.json").write_text(json.dumps(prov))
    return summary.load_run(directory)


def test_summary_objective_columns_and_matching_pairs(tmp_path, monkeypatch):
    directories = []
    for hardware in wave3.WAVE3_HARDWARE:
        for method, enabled in [("lio_er", False), ("refine_er", False), ("lio_er", True)]:
            directory = tmp_path / f"{hardware}_{method}_{enabled}"
            objective_fixture(directory, method, hardware, enabled)
            directories.append(directory)
    out = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", ["summary", "--runs", *map(str, directories), "--out", str(out)])
    assert summary.main() == 0
    for name in ("seed_metrics.csv", "summary.csv", "paired_differences.csv"):
        with (out / name).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert {r["objective"] for r in rows} == {"env_reward", "energy_reward"}
        if name == "paired_differences.csv":
            energy = [r for r in rows if r["objective"] == "energy_reward"]
            assert len(energy) == len(wave3.WAVE3_HARDWARE) * len(PROFILES) * 2
            assert {r["reference_method"] for r in energy} == {"lio_er", "refine_er"}
            assert all(r["reference_objective"] == "env_reward" and r["training_profile"] == r["reference_training_profile"]
                       and r["training_hardware_profile"] == r["reference_hardware_profile"] for r in energy)


def test_pairing_refuses_seed_mismatch_and_does_not_substitute_profile(tmp_path, capsys):
    energy = objective_fixture(tmp_path / "energy", "lio_er", "homogeneous_v1", True)
    baseline = objective_fixture(tmp_path / "baseline", "lio_er", "homogeneous_v1")
    changed = deepcopy(baseline)
    changed[0]["seed"] += 1
    with pytest.raises(ValueError, match="matching training seed sets"):
        objective_summary.summarize_objectives([energy, changed])
    changed = objective_fixture(tmp_path / "refine_uniform", "refine_er", "homogeneous_v1", profile="uniform_v1")
    _, _, pairs = objective_summary.summarize_objectives([energy, changed])
    assert not pairs
    assert "PAIR_MISSING" in capsys.readouterr().out


@pytest.mark.parametrize("hardware", wave3.WAVE3_HARDWARE)
def test_wave1_lio_reference_accounting_exemption(tmp_path, monkeypatch, hardware):
    baseline_dir, energy_dir = tmp_path / "wave1", tmp_path / "wave3"
    baseline_dir.mkdir()
    saved_fixture(baseline_dir, "uniform_v1", "lio_er")
    objective_fixture(energy_dir, "lio_er", hardware, True)
    out = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", ["summary", "--runs", str(baseline_dir), str(energy_dir),
        "--evaluation_hardware_profile", hardware, "--out", str(out)])
    assert summary.main() == 0
    with (out / "paired_differences.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == len(PROFILES)
    paper = next(r for r in pairs if r["evaluation_profile"] == "paper_ratio_v1")
    assert paper["objective"] == "energy_reward"
    assert paper["training_profile"] == "paper_ratio_v1"
    assert paper["reference_method"] == "lio_er" and paper["reference_objective"] == "env_reward"
    assert paper["reference_training_profile"] == "uniform_v1"
    assert paper["reference_hardware_profile"] == "homogeneous_v1"
    assert paper["evaluation_hardware_profile"] == hardware
    assert all(r["reference_training_match"] == objective_summary.LIO_REFERENCE_RULE for r in pairs)
    assert all(r["reference_training_match_justification"] == objective_summary.LIO_REFERENCE_JUSTIFICATION
               for r in pairs)
    provenance = json.loads((out / "inputs.json").read_text())
    rule = provenance["reference_training_match"]
    assert rule["rule"] == "lio_env_reward_accounting_only"
    assert rule["justification"] == paper["reference_training_match_justification"]
    assert "lio/alg/lio_agent_er.py:339-373" in rule["justification"]
    assert "evaluation table, evaluation hardware and complete seed sets must match" in rule["requirements"]


def test_accounting_exemption_keeps_evaluation_hardware_and_seed_gates(tmp_path):
    energy = objective_fixture(tmp_path / "energy", "lio_er", "hetero_2x1p5_v1", True)
    baseline = objective_fixture(tmp_path / "baseline", "lio_er", "homogeneous_v1", profile="uniform_v1")
    assert not objective_summary.summarize_objectives([energy, baseline])[2]
    changed = deepcopy(baseline)
    changed[0]["seed"] += 1
    with pytest.raises(ValueError, match="matching training seed sets"):
        objective_summary.summarize_objectives([energy, changed], "hetero_2x1p5_v1")


def test_refine_reference_keeps_training_hardware_gate(tmp_path):
    energy = objective_fixture(tmp_path / "energy", "lio_er", "hetero_2x1p5_v1", True)
    baseline = objective_fixture(tmp_path / "refine", "refine_er", "homogeneous_v1")
    assert not objective_summary.summarize_objectives([energy, baseline], "hetero_2x1p5_v1")[2]
