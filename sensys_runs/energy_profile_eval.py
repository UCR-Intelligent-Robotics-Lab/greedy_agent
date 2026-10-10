"""Observe final clean-ER evaluation without changing its behavior or RNG use.

Action counts are sufficient statistics for the P1 fixed cost tables. Completed
records include failures; a JSON manifest binds them to the saved checkpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Iterator

from sensys_runs.energy_profiles import PROFILES, get_profile

SCHEMA_VERSION = 1
EPISODE_FILE = "energy_eval_episodes.csv"
MANIFEST_FILE = "energy_eval_provenance.json"
EPISODE_FIELDS = (
    "schema_version", "run_id", "seed", "method", "training_profile",
    "checkpoint_episode", "eval_episode", "agent_id", "terminal_code", "steps",
    "n_lever", "n_start", "n_door", "env_return",
)
METHOD_ALGORITHMS = {"lio_er": "lio", "refine_er": "REFiNE"}


class EnergyEvaluationError(ValueError):
    """Invalid or incomplete evaluation data; never silently discard rows."""


def json_copy(value):
    """Normalize tuples and ConfigDict containers for stable provenance checks."""
    return json.loads(json.dumps(value, allow_nan=False))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(prefix: Path) -> dict[str, str]:
    """Pin the exact TF checkpoint prefix, including all data shards and graph."""
    files = [Path(str(prefix) + suffix) for suffix in (".index", ".meta")]
    shards = sorted(prefix.parent.glob(prefix.name + ".data-*-of-*"))
    parts = [re.fullmatch(re.escape(prefix.name) + r"\.data-(\d+)-of-(\d+)", p.name)
             for p in shards]
    if (not shards or not all(parts)
            or {int(m[2]) for m in parts} != {len(shards)}
            or {int(m[1]) for m in parts} != set(range(len(shards)))):
        raise EnergyEvaluationError(f"Missing or inconsistent checkpoint shards: {prefix}")
    files.extend(shards)
    if any(not p.is_file() for p in files):
        raise EnergyEvaluationError(f"Incomplete saved checkpoint: {prefix}")
    return {p.name: sha256(p) for p in files}


@contextmanager
def observe_episodes(env) -> Iterator[list[dict]]:
    """Forward env.step exactly; restore both method identity and lookup on exit."""
    original = env.step
    had_instance_step = "step" in vars(env)
    records = []
    counts = [[0, 0, 0] for _ in range(env.n_agents)]
    rewards = [0.0] * env.n_agents
    steps = 0
    episode = 1

    def step(*args, **kwargs):
        nonlocal steps, episode, counts, rewards
        actions = tuple(args[0] if args else kwargs["actions"])
        result = original(*args, **kwargs)
        _, env_rewards, done = result
        if len(actions) != env.n_agents or len(env_rewards) != env.n_agents:
            raise EnergyEvaluationError("Agent count changed during evaluation")
        if done not in (0, 1, 2):
            raise EnergyEvaluationError(f"Unknown terminal code: {done}")
        steps += 1
        for i, action in enumerate(actions):
            if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 3:
                raise EnergyEvaluationError(f"Invalid ER action: {action!r}")
            counts[i][action] += 1
            rewards[i] += float(env_rewards[i])
            if not math.isfinite(rewards[i]):
                raise EnergyEvaluationError("Non-finite environment return")
        if done:
            for i in range(env.n_agents):
                records.append(dict(eval_episode=episode, agent_id=i,
                                    terminal_code=int(done), steps=steps,
                                    n_lever=counts[i][0], n_start=counts[i][1],
                                    n_door=counts[i][2], env_return=rewards[i]))
            episode += 1
            steps = 0
            counts = [[0, 0, 0] for _ in range(env.n_agents)]
            rewards = [0.0] * env.n_agents
        return result

    env.step = step
    try:
        yield records
        if steps:
            raise EnergyEvaluationError("Evaluation ended with an incomplete episode")
    finally:
        if had_instance_step:
            env.step = original
        else:
            del env.step


def evaluate_final_policy(sess, env, agents, config, checkpoint_path,
                          method: str, evaluator=None):
    """Evaluate the just-saved policy; keep the existing evaluator return tuple."""
    if method not in METHOD_ALGORITHMS:
        raise EnergyEvaluationError("Final cost evaluation supports clean ER only")
    profile = get_profile(config.cost_profile["profile_id"])
    if json_copy(config.cost_profile) != json_copy(profile.snapshot()):
        raise EnergyEvaluationError("Training profile snapshot differs from the registry")
    n_test = config.alg.n_test
    # The unchanged evaluator indexes fairness_list once per agent at its end.
    if isinstance(n_test, bool) or not isinstance(n_test, Integral) or n_test < env.n_agents:
        raise EnergyEvaluationError("n_test must be an integer >= n_agents for the existing evaluator")
    if [a.agent_id for a in agents] != list(range(env.n_agents)):
        raise EnergyEvaluationError("Evaluator requires agents ordered by agent_id")
    prefix = Path(checkpoint_path).resolve()
    csv_path, manifest_path = prefix.parent / EPISODE_FILE, prefix.parent / MANIFEST_FILE
    if any(p.exists() or p.is_symlink() for p in (csv_path, manifest_path)):
        raise EnergyEvaluationError("Final evaluation output already exists")
    hashes = checkpoint_hashes(prefix)
    if evaluator is None:
        from lio.alg.evaluate import test_room_symmetric
        evaluator = test_room_symmetric
    with observe_episodes(env) as records:
        result = evaluator(n_test, env, sess, agents, METHOD_ALGORITHMS[method])
    if len(records) != n_test * env.n_agents:
        raise EnergyEvaluationError("Final evaluation episode count does not match n_test")
    common = dict(schema_version=SCHEMA_VERSION,
                  run_id=f"{config.main.exp_name}/{config.main.dir_name}",
                  seed=config.main.seed, method=method, training_profile=profile.profile_id,
                  checkpoint_episode=int(config.alg.n_episodes))
    manifest = dict(common, n_eval_episodes=int(n_test), n_agents=env.n_agents,
                    config=json_copy(config), training_profile_snapshot=profile.snapshot(),
                    evaluation_profiles={name: p.snapshot() for name, p in PROFILES.items()},
                    checkpoint_prefix=prefix.name, checkpoint_sha256=hashes)
    # Publish only after all episodes finish. A failed/partial write has no valid
    # matching manifest and is refused by the reader and by a subsequent run.
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(dict(common, **record) for record in records)
    manifest["episodes_sha256"] = sha256(csv_path)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    return result
