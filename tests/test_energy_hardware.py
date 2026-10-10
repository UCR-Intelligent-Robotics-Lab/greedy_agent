"""Hardware propagation and immutable wave-1 contracts; fix wiring, not tolerances."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensys_runs import energy_profile_campaign as wave1
from sensys_runs import energy_hardware_campaign as wave2
from sensys_runs import energy_profile_runtime as runtime
from sensys_runs import energy_profile_runner as runner
from sensys_runs import energy_hardware_summary as hardware_summary
from sensys_runs import summarize_energy_profiles as summary
from sensys_runs.energy_profiles import (
    HARDWARE_PROFILES, HardwareProfile, EnergyProfileError, get_hardware_profile,
    action_cost, PROFILES,
)
from test_energy_profiles import source_objects, probe_agent, FeedRecorder
from test_energy_profile_eval import saved_fixture

BASE = "78971fe75c6314bf80264131a84048a81f0a3891"


def test_registry_immutable_and_registered():
    p = get_hardware_profile("hetero_2x1p5_v1")
    assert p.vector(4) == (1.0, 1.0, 1.5, 1.5)
    assert p.source_classification == "design choice; heterogeneous hardware profile lambda^i; not measured"
    assert get_hardware_profile("homogeneous_v1").vector(3) == (1.0,) * 3
    with pytest.raises(FrozenInstanceError): p.profile_id = "bad"
    with pytest.raises(TypeError): HARDWARE_PROFILES["bad"] = p
    with pytest.raises(EnergyProfileError): get_hardware_profile("unregistered")
    with pytest.raises(EnergyProfileError): p.vector(3)


@pytest.mark.parametrize("values", [(1, 1, 0, 1), (1, -1, 1, 1), (1, 1, float("nan"), 1),
                                    (float("inf"), 1, 1, 1), (True, 1, 1, 1), (1, 1)])
def test_invalid_vector(values):
    with pytest.raises(EnergyProfileError): HardwareProfile("probe", values, "test only").vector(4)


@pytest.mark.parametrize("method", ["refine_er", "lio_er"])
def test_cost_buffer_and_actor_penalty(monkeypatch, method):
    import numpy as np
    from lio.utils import configdict
    cfg = source_objects("lio/alg/config_room_REFiNE.py", {"get_config"}, {"configdict": configdict})["get_config"]()
    cfg.env.n_agents, cfg.env.min_at_lever = 4, 2
    cls = probe_agent()
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: SimpleNamespace(REFiNE=cls, LIO=cls))
    agents = [cls() for _ in range(4)]
    for i, a in enumerate(agents):
        a.agent_id, a.energy_param, a.include_cost_in_chain_rule, a.l_action = i, 1.0, False, 3
        a.list_copy_main_to_prime_ops = []
        a.can_give = False
        for key in ("obs", "action_taken", "r_ext", "ones", "epsilon", "r_from_others",
                    "energy_penalty", "fairness_coef", "policy_op_prime"):
            setattr(a, key, key)
    path = "lio/alg/train_REFiNE_er.py" if method == "refine_er" else "lio/alg/train_lio_er.py"
    ns = source_objects(path, {"run_episode", "Buffer"}, {"np": np})
    actor = source_objects("lio/env/room_agent.py", {"Actor"}, {"np": np})["Actor"]
    env_cls = source_objects("lio/env/room_symmetric.py", {"Env"}, {"np": np, "room_agent": SimpleNamespace(Actor=actor)})["Env"]
    def train(config):
        for a in agents:
            sequence = iter([1, 0 if a.agent_id in (0, 2) else 2])
            a.run_actor = lambda *args, sequence=sequence: next(sequence)
        buffers, done = ns["run_episode"](None, env_cls(config.env), agents, 0.0)
        assert done == 1
        feeds = []
        if method == "refine_er":
            for a, buf in zip(agents, buffers):
                recorder = FeedRecorder()
                a.update(recorder, buf, 0.0, buf.total_energy, 0.0)
                feeds.append(recorder.feed[a.energy_penalty])
            assert feeds == [b.total_energy for b in buffers]
        return ([b.energy_cost for b in buffers], [b.total_energy for b in buffers], feeds)
    # Exact equality, not approximate: the P1 calculation executes at default.
    for profile in PROFILES:
        with runtime.cost_profile_scope(profile, method): baseline = train(cfg)
        for hardware in (None, "homogeneous_v1"):
            if hardware:
                cfg.hardware_profile = get_hardware_profile(hardware).snapshot(4)
                cfg.hardware_profile_lambda = list(get_hardware_profile(hardware).vector(4))
            actual = runtime.wrap_training(train, profile, method)(cfg)
            assert json.dumps(actual) == json.dumps(baseline)
        cfg.pop("hardware_profile", None); cfg.pop("hardware_profile_lambda", None)
    cfg.hardware_profile = get_hardware_profile("hetero_2x1p5_v1").snapshot(4)
    cfg.hardware_profile_lambda = list(get_hardware_profile("hetero_2x1p5_v1").vector(4))
    costs, totals, feeds = runtime.wrap_training(train, "paper_ratio_v1", method)(cfg)
    assert costs[2][1] == 1.5 * costs[0][1]
    assert totals[2] == pytest.approx(1.5 * totals[0])
    if method == "refine_er": assert feeds[2] == totals[2]
    agents[2].energy_param = 1.5
    with pytest.raises(EnergyProfileError): runtime.wrap_training(train, "paper_ratio_v1", method)(cfg)


def test_wave1_frozen_and_wave2_isolated(monkeypatch, tmp_path):
    source = subprocess.check_output(["git", "show", BASE + ":sensys_runs/energy_profile_campaign.py"], cwd=ROOT, text=True)
    old = ModuleType("p3_frozen")
    old.__file__ = wave1.__file__
    exec(compile(source, old.__file__, "exec"), old.__dict__)
    configs = {m: wave1.load_config(m) for m in ("lio_er", "refine_er")}
    for module in (old, wave1):
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(module, "load_config", lambda m: deepcopy(configs[m]))
        monkeypatch.setattr(module, "source_identity", lambda: {"git_commit": BASE})
    for smoke in (False, True):
        expected = old.build_campaign("same", "local", 1, smoke=smoke)
        actual = wave1.build_campaign("same", "local", 1, smoke=smoke)
        assert json.dumps(actual["runs"], sort_keys=True) == json.dumps(expected["runs"], sort_keys=True)
        second = wave2.build_campaign("same", "local", 1, smoke=smoke)
        assert second["n_runs"] == (2 if smoke else 20)
        assert {r["seed"] for r in second["runs"]} == {r["seed"] for r in actual["runs"]}
        assert not {r["output_directory"] for r in second["runs"]} & {r["output_directory"] for r in actual["runs"]}
        for row in second["runs"]:
            cfg = row["effective_config"]
            assert cfg.lio.Fairness_multiplier == 0
            assert cfg.lio.energy_weight == configs["refine_er"].lio.energy_weight
            assert cfg.hardware_profile_lambda == [1, 1, 1.5, 1.5]
        assert [(r["method"], r["training_profile"]) for r in runner.shard_plan(second, 0, 1)[:2]] == list(wave2.WAVE2_ARMS)
    formal = wave1.build_campaign("same", "local", 1)
    smoke = wave1.build_campaign("same", "local", 1, smoke=True)
    assert runner.default_console_dir(formal).name == "same_console"
    assert runner.default_console_dir(smoke).name == "same_smoke_console"
    assert runner.default_console_dir(formal) != runner.default_console_dir(smoke)
    assert "default_console_dir" in (ROOT / "sensys_runs/energy_profile_launch.sh").read_text()


def test_support_share_varies_by_seed_and_weighted_cost():
    hardware = get_hardware_profile("hetero_2x1p5_v1").vector(4)
    values = []
    for seed, lever in [(1, [1, 1, 0, 0]), (2, [0, 0, 1, 1])]:
        rows = [dict(seed=seed, agent_id=i, eval_episode=1, n_lever=n, n_start=1-n, n_door=0, steps=1) for i, n in enumerate(lever)]
        scored = hardware_summary.score_hardware_rows(rows, PROFILES["paper_ratio_v1"].costs, hardware)
        assert scored["support_share_reference"] == 0.5
        assert scored["mean_agent_mission_cost"] == pytest.approx(scored["step_cost_component"] + scored["support_cost_component"])
        values.append(scored["support_share"])
        expected = [(1.4 if n else 0.8) * hardware[i] for i, n in enumerate(lever)]
        assert json.loads(scored["per_agent_mean_mission_cost"]) == expected
        assert scored["worst_agent_mean_mission_cost"] == max(expected)
    assert values == [1.0, 0.0]


def test_old_run_rescore_explicit_origin_and_refusal(tmp_path, monkeypatch, capsys):
    saved_fixture(tmp_path)
    meta, rows = summary.load_run(tmp_path)
    origin = hardware_summary.record_hardware_origin(meta, tmp_path)
    assert origin["profile_id"] == "homogeneous_v1" and BASE in origin["origin"]
    assert BASE in capsys.readouterr().out
    out = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["summary", "--runs", str(tmp_path), "--out", str(out),
                                     "--evaluation_hardware_profile", "hetero_2x1p5_v1"])
    assert summary.main() == 0
    assert (out / "hardware_seed_metrics.csv").is_file()
    assert json.loads((out / "inputs.json").read_text())["intervals"] == "not computed"
    (tmp_path / "provenance.json").write_text(json.dumps(dict(git_commit="unknown", git_dirty=False)))
    with pytest.raises(ValueError, match="pre-P4"): hardware_summary.record_hardware_origin(meta, tmp_path)


def test_recorded_hardware_must_match_both_config_and_provenance(tmp_path):
    saved_fixture(tmp_path)
    meta, _ = summary.load_run(tmp_path)
    snapshot = get_hardware_profile("hetero_2x1p5_v1").snapshot(4)
    meta["config"].update(hardware_profile=snapshot, hardware_profile_lambda=snapshot["hardware_profile_lambda"])
    prov = dict(hardware_profile=snapshot, hardware_profile_lambda=snapshot["hardware_profile_lambda"], git_commit="p4")
    (tmp_path / "provenance.json").write_text(json.dumps(prov))
    assert hardware_summary.record_hardware_origin(meta, tmp_path)["profile_id"] == "hetero_2x1p5_v1"
    prov["hardware_profile_lambda"] = [1]*4
    (tmp_path / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(ValueError, match="lambda"): hardware_summary.record_hardware_origin(meta, tmp_path)


@pytest.mark.parametrize("profile", ["legacy", "low_contrast_v1"])
def test_launcher_hardware_provenance_default(monkeypatch, tmp_path, profile):
    from test_energy_profiles import test_launcher_dispatch_and_provenance
    test_launcher_dispatch_and_provenance(monkeypatch, tmp_path, "refine_er", profile)
    record = json.loads((tmp_path / "lio/results/p1/probe/provenance.json").read_text())
    if profile == "legacy": assert "hardware_profile_lambda" not in record
    else: assert record["hardware_profile_lambda"] == [1.0, 1.0]


def test_hardware_flag_rejected_for_legacy_before_import(monkeypatch):
    from sensys_runs import run_single
    monkeypatch.setattr(sys, "argv", ["run_single", "--method", "lio_er", "--exp_name", "unused", "--dir_name", "unused",
                                     "--seed", "1", "--hardware_profile", "homogeneous_v1"])
    with pytest.raises(SystemExit) as exc: run_single.main()
    assert exc.value.code == 2


def test_heterogeneous_launcher_records_vector_and_scales(monkeypatch, tmp_path):
    from sensys_runs import run_single
    cfg = wave1.load_config("refine_er")
    class Agent:
        agent_id, energy_param = 2, 1.0
        def calculate_energy_cost(self, state, action): return -1
    captured = []
    def train(config):
        captured.append(config)
        assert Agent().calculate_energy_cost(None, 0) == 1.5 * action_cost("paper_ratio_v1", 0, 1.0)
    modules = {"lio.alg.config_room_REFiNE": SimpleNamespace(get_config=lambda: cfg),
               "lio.alg.train_REFiNE_er": SimpleNamespace(train=train),
               "lio.alg.REFiNE_er": SimpleNamespace(REFiNE=Agent)}
    original = runtime.importlib.import_module
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: modules[name] if name in modules else original(name))
    graph = SimpleNamespace(seed=None)
    def seed(value): graph.seed = value
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(__version__="stub", set_random_seed=seed,
                        compat=SimpleNamespace(v1=SimpleNamespace(set_random_seed=seed, get_default_graph=lambda: graph))))
    monkeypatch.setattr(run_single, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(run_single, "LIO_ALG", str(tmp_path))
    monkeypatch.setattr(run_single, "_git_field", lambda cmd: "fixture")
    monkeypatch.setattr(run_single, "_git_dirty", lambda: False)
    monkeypatch.setattr(run_single, "_set_thread_env", lambda n: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_single", "--method", "refine_er", "--exp_name", "p4", "--dir_name", "probe",
                                     "--seed", "1", "--n_agents", "4", "--cost_profile", "paper_ratio_v1",
                                     "--hardware_profile", "hetero_2x1p5_v1"])
    with pytest.raises(SystemExit) as exc: run_single.main()
    assert exc.value.code == 0
    record = json.loads((tmp_path / "lio/results/p4/probe/provenance.json").read_text())
    assert record["hardware_profile_lambda"] == captured[0].hardware_profile_lambda == [1, 1, 1.5, 1.5]
    assert record["hardware_profile"] == get_hardware_profile("hetero_2x1p5_v1").snapshot(4)
    assert Agent().calculate_energy_cost(None, 0) == -1
