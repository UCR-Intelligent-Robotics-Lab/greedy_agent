"""P1 checks against hand-computable costs and the current training source.

Source-isolated probes execute real buffer/episode/update bodies without TF;
the separately skipped integration check requires the existing TF environment.
Fix propagation or restoration failures rather than weakening the assertions.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensys_runs.energy_profiles import (ENERGY_PARAM, PROFILES, EnergyProfileError,
                                     action_cost, get_profile)
from sensys_runs import energy_profile_runtime as runtime
from sensys_runs import run_single


@pytest.mark.parametrize("profile,expected", [
    ("uniform_v1", (1, 1, 1)),
    ("low_contrast_v1", (138/121, 225/242, 225/242)),
    ("paper_ratio_v1", (1.4, 0.8, 0.8)),
])
def test_tables_and_provenance(profile, expected):
    """Check independent arithmetic and immutable, serializable provenance."""
    record = get_profile(profile)
    assert record.costs == pytest.approx(expected)
    assert sum(record.costs) == pytest.approx(3)
    assert all(action_cost(profile, a, ENERGY_PARAM) > 0 for a in range(3))
    snapshot = json.loads(json.dumps(record.snapshot()))
    assert snapshot["units"] == "normalized_modeled_cost"
    assert snapshot["actor_cost_form"] == "episode_mean"
    assert snapshot["energy_param"] == ENERGY_PARAM
    with pytest.raises(FrozenInstanceError):
        record.profile_id = "changed"
    with pytest.raises(TypeError):
        PROFILES[profile] = record


@pytest.mark.parametrize("action", [-1, 3, 1.0, True, "0", None])
def test_bad_actions(action):
    """Invalid actions must not silently index a table or truncate a float."""
    with pytest.raises(EnergyProfileError):
        action_cost("uniform_v1", action, ENERGY_PARAM)


@pytest.mark.parametrize("scale", [0, -1, 2, float("nan"), float("inf"), True])
def test_unregistered_scaling(scale):
    """A profile label must never hide an extra cost scale."""
    with pytest.raises(EnergyProfileError):
        action_cost("uniform_v1", 0, scale)


def test_invalid_profile_and_run_refusal(tmp_path):
    """Refused jobs must leave no files behind; legacy keeps old semantics."""
    fresh = tmp_path / "fresh"
    for profile, method in [("missing", "refine_er"), ("uniform_v1", "refine_eia_er"),
                            ("paper_ratio_v1", "lio_ipd")]:
        with pytest.raises(EnergyProfileError):
            runtime.validate_run(profile, method, str(fresh))
    assert not fresh.exists()
    with pytest.raises(EnergyProfileError):
        runtime.validate_run("uniform_v1", "refine_er", str(tmp_path))
    runtime.validate_run("legacy", "lio_ipd", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def source_objects(relative, names, namespace, class_name=None):
    """Load exact live bodies, excluding imports that require the TF stack."""
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def probe_agent():
    """Use original cost/update bodies, without pretending to build a TF graph."""
    np = pytest.importorskip("numpy")
    from lio.utils import util
    ns = source_objects("lio/alg/REFiNE_er.py",
                        {"calculate_energy_cost", "get_num_at_lever", "update"},
                        {"np": np, "util": util}, "REFiNE")
    return type("Probe", (), {k: ns[k] for k in
                              ("calculate_energy_cost", "get_num_at_lever", "update")})


def test_scope_purity_legacy_and_exception(monkeypatch):
    """Repeated unused calls and alternate one-hot observations must be inert."""
    np = pytest.importorskip("numpy")
    cls = probe_agent()
    agent = cls()
    agent.energy_param, agent.n_agents, agent.min_at_lever = ENERGY_PARAM, 4, 2
    original = cls.calculate_energy_cost
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: SimpleNamespace(REFiNE=cls))
    states = (np.tile([1, 0, 0], 4), np.tile([0, 1, 0], 4))
    before = [agent.calculate_energy_cost(s, a) for s in states for a in range(3)]
    with runtime.cost_profile_scope("legacy", "refine_er"):
        assert cls.calculate_energy_cost is original
        assert before == [agent.calculate_energy_cost(s, a) for s in states for a in range(3)]
    py_rng, np_rng = random.getstate(), np.random.get_state()
    fields = agent.__dict__.copy()
    with pytest.raises(RuntimeError, match="probe"):
        with runtime.cost_profile_scope("paper_ratio_v1", "refine_er"):
            for s in states:
                for _ in range(3):
                    assert agent.calculate_energy_cost(s, np.int64(0)) == 1.4
            raise RuntimeError("probe")
    assert cls.calculate_energy_cost is original
    assert agent.__dict__ == fields
    assert random.getstate() == py_rng
    current = np.random.get_state()
    assert current[0] == np_rng[0] and current[2:] == np_rng[2:]
    np.testing.assert_array_equal(current[1], np_rng[1])


class FeedRecorder:
    """Capture actor feeds, optionally forwarding to a real TensorFlow session."""
    def __init__(self, session=None):
        self.session, self.feed = session, None

    def run(self, ops, feed_dict=None):
        if feed_dict is not None:
            self.feed = feed_dict
        if self.session is not None:
            return self.session.run(ops, feed_dict=feed_dict)


def scripted_actors(agents):
    """Two-step ER(4,2) mission exercises start plus lever/door costs."""
    for agent in agents:
        sequence = iter([1, 0 if agent.agent_id < 2 else 2])

        def run_actor(state, sess, epsilon, prime, agent=agent, sequence=sequence):
            action = next(sequence)
            agent.calculate_energy_cost(state, action)  # Existing actor has an unused call.
            return action

        agent.run_actor = run_actor
        agent.can_give = False


def run_propagation(run_episode, agents, cfg, session, env_class):
    """Execute existing episode/buffer/update code and check two distinct totals."""
    recorder = FeedRecorder(session)
    totals = []
    for profile, expected in [("uniform_v1", 2.0), ("paper_ratio_v1", 2.2)]:
        scripted_actors(agents)
        with runtime.cost_profile_scope(profile, "refine_er"):
            buffers, done = run_episode(session, env_class(cfg.env), agents, 0.0)
            assert done == 1
            assert buffers[0].action == [1, 0]
            assert buffers[0].total_energy == pytest.approx(expected)
            agents[0].update(recorder, buffers[0], 0.0, buffers[0].total_energy, 0.0)
            assert recorder.feed[agents[0].energy_penalty] == pytest.approx(expected)
            totals.append(recorder.feed[agents[0].energy_penalty])
    assert totals[0] != totals[1]


def test_source_episode_buffer_actor_feed(monkeypatch):
    """Changing profiles must change actual buffer totals and update feeds."""
    np = pytest.importorskip("numpy")
    from lio.utils import configdict
    config_ns = source_objects("lio/alg/config_room_REFiNE.py", {"get_config"},
                               {"configdict": configdict})
    cfg = config_ns["get_config"]()
    cfg.env.n_agents, cfg.env.min_at_lever = 4, 2
    ns = source_objects("lio/alg/train_REFiNE_er.py", {"run_episode", "Buffer"}, {"np": np})
    cls = probe_agent()
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: SimpleNamespace(REFiNE=cls))
    agents = [cls() for _ in range(cfg.env.n_agents)]
    for i, agent in enumerate(agents):
        agent.agent_id, agent.energy_param = i, ENERGY_PARAM
        agent.include_cost_in_chain_rule, agent.l_action = False, 3
        agent.list_copy_main_to_prime_ops = []
        for key in ("obs", "action_taken", "r_ext", "ones", "epsilon", "r_from_others",
                    "energy_penalty", "fairness_coef", "policy_op_prime"):
            setattr(agent, key, key)
    actor_ns = source_objects("lio/env/room_agent.py", {"Actor"}, {"np": np})
    env_ns = source_objects("lio/env/room_symmetric.py", {"Env"},
                            {"np": np, "room_agent": SimpleNamespace(Actor=actor_ns["Actor"])})
    run_propagation(ns["run_episode"], agents, cfg, None, env_ns["Env"])


def test_cli_default_and_early_refusal(monkeypatch, tmp_path):
    """Both legacy spellings preserve dispatch; bad modes fail before TF imports."""
    args = ["run_single", "--method", "refine_er", "--exp_name", "p1", "--dir_name", "test", "--seed", "1"]
    monkeypatch.setattr(sys, "argv", args)
    assert run_single._parse_args().cost_profile == "legacy"
    monkeypatch.setattr(sys, "argv", args + ["--cost_profile", "legacy"])
    assert run_single._parse_args().cost_profile == "legacy"
    train = lambda cfg: cfg
    assert runtime.wrap_training(train, "legacy", "refine_er") is train
    monkeypatch.setattr(run_single, "REPO_ROOT", str(tmp_path))
    args[2] = "refine_eia_er"
    monkeypatch.setattr(sys, "argv", args + ["--cost_profile", "uniform_v1"])
    with pytest.raises(SystemExit) as error:
        run_single.main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("method", ["lio_er", "refine_er"])
@pytest.mark.parametrize("profile", ["legacy", "low_contrast_v1"])
def test_launcher_dispatch_and_provenance(monkeypatch, tmp_path, method, profile):
    """Exercise real launcher wiring with a stub trainer, not a simulated graph."""
    from lio.utils import configdict
    config_name, train_name = run_single.METHOD_MAP[method]
    ns = source_objects(f"lio/alg/{config_name}.py", {"get_config"}, {"configdict": configdict})
    cls = probe_agent()
    original = cls.calculate_energy_cost
    captured = []

    def train(cfg):
        captured.append(cfg)
        if profile == "legacy":
            assert "cost_profile" not in cfg
            assert cls.calculate_energy_cost is original
        else:
            assert cfg.cost_profile.profile_id == profile
            agent = cls()
            agent.energy_param = ENERGY_PARAM
            assert agent.calculate_energy_cost(None, 0) == pytest.approx(138/121)

    modules = {
        f"lio.alg.{config_name}": SimpleNamespace(get_config=ns["get_config"]),
        f"lio.alg.{train_name}": SimpleNamespace(train=train),
        runtime.AGENT_CLASSES[method][0]: SimpleNamespace(REFiNE=cls, LIO=cls),
    }
    real_import = importlib.import_module
    monkeypatch.setattr(runtime.importlib, "import_module",
                        lambda name: modules[name] if name in modules else real_import(name))
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(set_random_seed=lambda seed: None,
                                                                   __version__="test-double"))
    monkeypatch.setattr(run_single, "_set_thread_env", lambda threads: None)
    monkeypatch.setattr(run_single, "_git_field", lambda cmd: "test-double")
    monkeypatch.setattr(run_single, "_git_dirty", lambda: True)
    monkeypatch.setattr(run_single, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(run_single, "LIO_ALG", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_single", "--method", method,
                        "--exp_name", "p1", "--dir_name", "probe", "--seed", "1",
                        "--cost_profile", profile])
    with pytest.raises(SystemExit) as result:
        run_single.main()
    assert result.value.code == 0 and len(captured) == 1
    assert cls.calculate_energy_cost is original
    provenance_path = tmp_path / "lio/results/p1/probe/provenance.json"
    original_bytes = provenance_path.read_bytes()
    provenance = json.loads(original_bytes)
    if profile == "legacy":
        assert "cost_profile" not in provenance
    else:
        assert provenance["cost_profile"] == json.loads(json.dumps(captured[0].cost_profile))
        with pytest.raises(SystemExit) as result:
            run_single.main()
        assert result.value.code == 2 and len(captured) == 1
        assert provenance_path.read_bytes() == original_bytes


def test_tensorflow_propagation():
    """Required graph integration; a missing TF stack is a skip, never a pass."""
    tf = pytest.importorskip("tensorflow.compat.v1")
    from lio.alg.config_room_REFiNE import get_config
    from lio.alg.REFiNE_er import REFiNE
    from lio.alg.train_REFiNE_er import run_episode
    from lio.env.room_symmetric import Env
    tf.reset_default_graph()
    cfg = get_config()
    cfg.env.n_agents, cfg.env.min_at_lever = 4, 2
    env = Env(cfg.env)
    agents = [REFiNE(cfg.lio, env.l_obs, env.l_action, cfg.nn, f"agent_{i}",
                     cfg.env.r_multiplier, env.n_agents, i, ENERGY_PARAM)
              for i in range(env.n_agents)]
    for agent in agents:
        agent.receive_list_of_agents(agents)
        agent.create_policy_gradient_op()
        agent.create_update_op()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        run_propagation(run_episode, agents, cfg, session, Env)
    tf.reset_default_graph()
