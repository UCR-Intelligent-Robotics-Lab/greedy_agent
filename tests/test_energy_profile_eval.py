"""P2 integrity and metrics checks on hand-computable ER episodes.

Source-isolated tests run the unchanged evaluator/environment bodies. They are
not a substitute for the separately gated real TensorFlow integration check.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_energy_profiles import source_objects
from sensys_runs.energy_profiles import PROFILES, ENERGY_PARAM, action_cost
from sensys_runs.energy_profile_eval import (
    EPISODE_FIELDS, EPISODE_FILE, MANIFEST_FILE, EnergyEvaluationError,
    evaluate_final_policy, json_copy, observe_episodes, sha256,
)
from sensys_runs import summarize_energy_profiles as summary


def fixture_model(profile="paper_ratio_v1", method="refine_er"):
    """Use live ER definitions without package imports of TF, gym or LOLA."""
    np = pytest.importorskip("numpy")
    from lio.utils import configdict
    config_name = "config_room_REFiNE" if method == "refine_er" else "config_room_lio"
    cfg = source_objects(f"lio/alg/{config_name}.py", {"get_config"},
                         {"configdict": configdict})["get_config"]()
    cfg.env.n_agents, cfg.env.min_at_lever = 4, 2
    cfg.alg.n_test = cfg.env.n_agents  # Smallest count accepted by the unchanged evaluator.
    cfg.cost_profile = PROFILES[profile].snapshot()
    actor = source_objects("lio/env/room_agent.py", {"Actor"}, {"np": np})["Actor"]
    env_cls = source_objects("lio/env/room_symmetric.py", {"Env"},
                             {"np": np, "room_agent": SimpleNamespace(Actor=actor)})["Env"]
    evaluator = source_objects("lio/alg/evaluate.py", {"test_room_symmetric"},
                                {"np": np, "random": random})["test_room_symmetric"]
    env = env_cls(cfg.env)
    actions = [[0, 0, 2, 2]] + [[1]*4]*cfg.env.max_steps
    actions += [[2, 2, 0, 0]] + [[1]*4]*cfg.env.max_steps

    class Agent:
        def __init__(self, i):
            self.agent_id = i
            self.gamma = cfg.lio.gamma
            self.actions = iter(a[i] for a in actions)

        def run_actor(self, state, session, epsilon):
            return next(self.actions)

        def calculate_energy_cost(self, state, action):
            return action_cost(profile, action, ENERGY_PARAM)

        def give_reward(self, state, actions, session):
            return np.ones(env.n_agents) * 3  # Ensure env return excludes incentives.

    return cfg, env, [Agent(i) for i in range(env.n_agents)], evaluator


def saved_fixture(tmp_path, profile="paper_ratio_v1", method="refine_er"):
    """Create explicitly synthetic checkpoint bytes, never a learned checkpoint."""
    cfg, env, agents, evaluator = fixture_model(profile, method)
    prefix = tmp_path / cfg.main.model_name
    for suffix in (".index", ".meta", ".data-00000-of-00001"):
        Path(str(prefix) + suffix).write_bytes(b"synthetic checkpoint fixture")
    result = evaluate_final_policy(None, env, agents, cfg, prefix, method, evaluator)
    return cfg, env, prefix, result


@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("method", ["lio_er", "refine_er"])
def test_final_records_and_evaluator_transparency(tmp_path, profile, method):
    """Success, timeout, costs, incentives and original return tuple must agree."""
    np = pytest.importorskip("numpy")
    cfg, env, agents, evaluator = fixture_model(profile, method)
    random.seed(cfg.main.seed)
    baseline = evaluator(cfg.alg.n_test, env, None, agents, cfg.alg.name)
    baseline_rng = random.getstate()
    random.seed(cfg.main.seed)
    cfg, env, prefix, observed = saved_fixture(tmp_path, profile, method)
    assert random.getstate() == baseline_rng
    for expected, actual in zip(baseline, observed):
        np.testing.assert_array_equal(actual, expected)
    assert "step" not in vars(env)
    meta, rows = summary.load_run(tmp_path)
    assert len(rows) == cfg.alg.n_test * env.n_agents
    assert {r["terminal_code"] for r in rows} == {1, 2}
    assert all(r["steps"] == r["n_lever"] + r["n_start"] + r["n_door"] for r in rows)
    metrics = summary.score_rows(rows, PROFILES[profile].costs)
    assert metrics["success_rate"] == 0.5
    assert metrics["mean_mission_steps"] == (1 + cfg.env.max_steps) / 2
    # Two one-step successes: total env reward 18 each. Stationary failures: 0.
    assert metrics["mean_agent_env_return"] == 36 / (cfg.alg.n_test * env.n_agents)
    assert metrics["mean_agent_mission_cost"] == pytest.approx(sum(observed[11]) / len(rows))
    assert metrics["mean_agent_env_return"] == pytest.approx(sum(observed[1]) / len(rows))
    for agent in range(env.n_agents):
        online = observed[11][agent]
        offline = sum(sum(r[k]*c for k, c in zip(("n_lever", "n_start", "n_door"),
                                                  PROFILES[profile].costs))
                      for r in rows if r["agent_id"] == agent)
        assert offline == pytest.approx(online)
    assert meta["checkpoint_episode"] == cfg.alg.n_episodes
    assert len(meta["checkpoint_sha256"]) == 3
    before = (tmp_path / EPISODE_FILE).read_bytes()
    with pytest.raises(EnergyEvaluationError, match="already exists"):
        evaluate_final_policy(None, env, agents, cfg, prefix, method, evaluator)
    assert (tmp_path / EPISODE_FILE).read_bytes() == before


def test_observer_restoration_and_identity():
    """Both instance-bound and class-bound methods restore after failure."""
    _, env, _, _ = fixture_model()
    env.reset()
    original = env.step
    returned = []

    def instance_step(*args, **kwargs):
        result = original(*args, **kwargs)
        returned.append(result)
        return result

    env.step = instance_step
    with pytest.raises(RuntimeError, match="forced"):
        with observe_episodes(env) as records:
            result = env.step(actions=[0, 0, 2, 2], given_rewards=[[0]*3]*4)
            assert result is returned[-1] and len(records) == 4
            raise RuntimeError("forced")
    assert env.step is instance_step
    del env.step
    env.reset()
    with pytest.raises(EnergyEvaluationError, match="incomplete"):
        with observe_episodes(env):
            env.step([1]*4, [[0]*3]*4)
    assert "step" not in vars(env)


@pytest.mark.parametrize("failure", ["exception", "too_few_episodes", "too_small_n_test", "missing_checkpoint"])
def test_failed_evaluation_has_no_sidecars(tmp_path, failure):
    """Incomplete evaluation cannot be mistaken for complete scientific output."""
    cfg, env, agents, evaluator = fixture_model()
    prefix = tmp_path / cfg.main.model_name
    if failure != "missing_checkpoint":
        for suffix in (".index", ".meta", ".data-00000-of-00001"):
            Path(str(prefix) + suffix).write_bytes(b"synthetic checkpoint fixture")
    original = env.step

    def failing_evaluator(*args):
        env.reset()
        env.step([0, 0, 2, 2], [[0]*3]*4)
        if failure == "exception":
            raise RuntimeError("forced evaluator failure")

    if failure == "too_small_n_test":
        cfg.alg.n_test = env.n_agents - 1
    with pytest.raises((EnergyEvaluationError, RuntimeError)):
        evaluate_final_policy(None, env, agents, cfg, prefix, "refine_er", failing_evaluator)
    assert env.step == original and "step" not in vars(env)
    assert not (tmp_path / EPISODE_FILE).exists()
    assert not (tmp_path / MANIFEST_FILE).exists()


def test_averaging_and_worst_agent(tmp_path):
    """Duplicating complete episodes preserves means; rotating burden matters."""
    saved_fixture(tmp_path)
    _, rows = summary.load_run(tmp_path)
    costs = PROFILES["paper_ratio_v1"].costs
    original = summary.score_rows(rows, costs)
    duplicated = rows + [dict(r, eval_episode=r["eval_episode"]+4) for r in rows]
    doubled = summary.score_rows(duplicated, costs)
    for metric in summary.METRICS:
        assert doubled[metric] == pytest.approx(original[metric])
    # The two successful missions rotate lever roles. Each agent averages the
    # same cost; averaging the per-episode maximum would overstate the burden.
    assert original["worst_agent_mean_mission_cost"] == pytest.approx(original["mean_agent_mission_cost"])
    assert original["worst_agent_mean_mission_cost"] < (1.4 + 4.0 + 1.4 + 4.0) / 4
    assert original["mean_agent_mission_cost"] == pytest.approx(
        original["step_cost_component"] + original["support_cost_component"])


@pytest.mark.parametrize("defect", ["missing", "duplicate", "counts", "terminal", "reward", "checkpoint"])
def test_reader_rejects_corruption(tmp_path, defect):
    """A checksum update cannot disguise incomplete or invalid episode records."""
    _, _, prefix, _ = saved_fixture(tmp_path)
    if defect == "checkpoint":
        Path(str(prefix) + ".index").write_bytes(b"changed checkpoint")
    else:
        path = tmp_path / EPISODE_FILE
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if defect == "missing": rows.pop()
        if defect == "duplicate": rows[-1] = rows[0].copy()
        if defect == "counts": rows[0]["n_lever"] = "99"
        if defect == "terminal": rows[0]["terminal_code"] = "2"
        if defect == "reward": rows[0]["env_return"] = "nan"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest = tmp_path / MANIFEST_FILE
        meta = json.loads(manifest.read_text())
        meta["episodes_sha256"] = sha256(path)
        manifest.write_text(json.dumps(meta))
    with pytest.raises(EnergyEvaluationError):
        summary.load_run(tmp_path)


def test_summary_cli_and_paired_decomposition(tmp_path, monkeypatch):
    """Write seed-level metrics and signed paired deltas without new rollouts."""
    directories = []
    for method, profile in [("lio_er", "uniform_v1"), ("refine_er", "uniform_v1"),
                            ("refine_er", "paper_ratio_v1")]:
        directory = tmp_path / (method + profile)
        directory.mkdir()
        saved_fixture(directory, profile, method)
        directories.append(directory)
    runs = [summary.load_run(p) for p in directories]
    seeds, groups, pairs = summary.summarize_runs(runs)
    # Three comparisons (uniform-LIO, paper-LIO, paper-uniform), three tables.
    assert len(seeds) == 9 and len(groups) == 9 and len(pairs) == 9
    for row in pairs:
        assert row["delta_mean_agent_mission_cost"] == pytest.approx(
            row["delta_step_cost_component"] + row["delta_support_cost_component"])
    output = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", ["summary", "--runs", *map(str, directories), "--out", str(output)])
    assert summary.main() == 0
    assert {p.name for p in output.iterdir()} == {"seed_metrics.csv", "summary.csv", "paired_differences.csv", "inputs.json"}
    with pytest.raises(SystemExit) as error:
        summary.main()
    assert error.value.code == 2
    with pytest.raises(EnergyEvaluationError, match="Duplicate"):
        summary.summarize_runs(runs + runs[:1])
    changed = json_copy(runs)
    for row in changed[-1][1]:
        if row["eval_episode"] == 1:
            row["steps"] += 1
            row["n_start"] += 1
            if row["agent_id"] == 0:
                row["n_start"] -= 1
                row["n_lever"] += 1
    _, _, differences = summary.summarize_runs(changed)
    nonzero = [r for r in differences if r["training_profile"] == "paper_ratio_v1"]
    assert all(r["delta_mean_agent_mission_cost"] > 0 for r in nonzero)
    for row in nonzero:
        assert row["delta_mean_agent_mission_cost"] == pytest.approx(
            row["delta_step_cost_component"] + row["delta_support_cost_component"])
    unmatched = json_copy(runs)
    unmatched[-1][0]["seed"] += 1
    with pytest.raises(EnergyEvaluationError, match="matching training seed"):
        summary.summarize_runs(unmatched)


@pytest.mark.parametrize("method,filename", [("refine_er", "train_REFiNE_er.py"), ("lio_er", "train_lio_er.py")])
def test_hooks_follow_final_save_and_are_default_inert(method, filename):
    """Execute the actual final-save tail with spies; legacy never evaluates."""
    import ast
    tree = ast.parse((ROOT / "lio/alg" / filename).read_text(encoding="utf-8"))
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    assert isinstance(train.body[-1], ast.If)
    assert isinstance(train.body[-2], ast.Expr) and train.body[-2].value.func.attr == "save"
    from sensys_runs import energy_profile_eval
    from unittest.mock import patch
    import os
    for profile in ({}, {"cost_profile": {"profile_id": "legacy"}}, {"cost_profile": {"profile_id": "uniform_v1"}}):
        events = []
        namespace = dict(config=SimpleNamespace(**profile), sess=object(), env=object(), list_agents=[],
                         log_path="fixture", model_name="model.ckpt", os=os,
                         saver=SimpleNamespace(save=lambda *args: events.append("save")))
        with patch.object(energy_profile_eval, "evaluate_final_policy",
                          side_effect=lambda *args: events.append(args[-1])):
            exec(compile(ast.Module(body=train.body[-2:], type_ignores=[]), filename, "exec"), namespace)
        assert events == (["save", method] if profile.get("cost_profile", {}).get("profile_id") == "uniform_v1" else ["save"])


@pytest.mark.parametrize("method", ["lio_er", "refine_er"])
def test_tensorflow_final_evaluation(tmp_path, method):
    """Real graph and saver integration; missing TensorFlow is explicitly skipped."""
    tf = pytest.importorskip("tensorflow.compat.v1")
    import importlib
    from sensys_runs.energy_profile_runtime import AGENT_CLASSES, cost_profile_scope
    from lio.env.room_symmetric import Env
    config_module = "config_room_lio" if method == "lio_er" else "config_room_REFiNE"
    cfg = importlib.import_module("lio.alg." + config_module).get_config()
    cfg.env.n_agents, cfg.env.min_at_lever = 4, 2
    cfg.alg.n_test = cfg.env.n_agents
    cfg.cost_profile = PROFILES["paper_ratio_v1"].snapshot()
    module, name = AGENT_CLASSES[method]
    cls = getattr(importlib.import_module(module), name)
    with tf.Graph().as_default():
        env = Env(cfg.env)
        agents = [cls(cfg.lio, env.l_obs, env.l_action, cfg.nn, f"agent_{i}",
                      cfg.env.r_multiplier, env.n_agents, i, ENERGY_PARAM)
                  for i in range(env.n_agents)]
        saver = tf.train.Saver()
        with tf.Session() as session, cost_profile_scope("paper_ratio_v1", method):
            session.run(tf.global_variables_initializer())
            prefix = saver.save(session, str(tmp_path / cfg.main.model_name))
            before = session.run(tf.trainable_variables())
            result = evaluate_final_policy(session, env, agents, cfg, prefix, method)
            after = session.run(tf.trainable_variables())
        np = pytest.importorskip("numpy")
        for original, current in zip(before, after):
            np.testing.assert_array_equal(original, current)
        _, rows = summary.load_run(tmp_path)
        metrics = summary.score_rows(rows, PROFILES["paper_ratio_v1"].costs)
        assert metrics["mean_agent_mission_cost"] == pytest.approx(sum(result[11]) / len(rows))
