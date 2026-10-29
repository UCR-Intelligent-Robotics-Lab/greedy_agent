"""EIA-only adapter around the unchanged registered single-run launcher."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, redirect_stdout
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sensys_runs.energy_profiles import EnergyProfileError, action_cost, get_profile
from sensys_runs import run_single

METHODS = ("lio_eia_er", "refine_eia_er")
WEIGHTS = ((2.0, 0.2), (1.1, 0.9), (1.5, 1.5))
CLASS_MAP = {
    "lio_eia_er": (("lio.alg.lio_agent_er", "LIO"),
                   ("lio.alg.lio_eia_er", "ExploitativeLIO")),
    "refine_eia_er": (("lio.alg.REFiNE_er", "REFiNE"),
                      ("lio.alg.REFiNE_eia_er", "REFiNEExploitative")),
}
TRAINER_MAP = {"lio.alg.train_lio_eia_er": "lio.alg.lio_eia_er",
               "lio.alg.train_REFiNE_eia_er": "lio.alg.REFiNE_eia_er"}


class Proxy:
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


@contextmanager
def rebind(target, name, value):
    previous = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, previous)


def agent_classes(method):
    return tuple(getattr(importlib.import_module(module), name)
                 for module, name in CLASS_MAP[method])


def validate_arguments(args):
    if (args.method not in METHODS or args.cost_profile != "uniform_v1"
            or args.hardware_profile != "homogeneous_v1"
            or args.n_agents != 4 or args.min_at_lever != 2 or args.threads != 1
            or args.seed not in range(12341, 12351) or args.energy_in_reward
            or args.use_gpu or args.fairness_clip
            or (args.w_lever, args.w_door) not in WEIGHTS):
        raise EnergyProfileError("EIA requires the registered method, weights, uniform ER(4,2), homogeneous lambda and CPU/one thread")
    if args.method == "refine_eia_er" and (args.fairness_mult != 0.0 or args.energy_weight != 1.0):
        raise EnergyProfileError("REFiNE EIA requires explicit B=0 and beta=1")
    if args.method == "lio_eia_er" and (args.energy_weight is not None or args.fairness_mult is not None):
        raise EnergyProfileError("Plain LIO EIA does not take REFiNE objective parameters")
    for name in (args.exp_name, args.dir_name):
        if not name or name in (".", "..") or any(c in name for c in "/\\:"):
            raise EnergyProfileError("Experiment and run names must be safe directory components")


def validate_config(config, method):
    if (config.env.n_agents != 4 or config.env.min_at_lever != 2
            or config.env.max_steps != 5 or config.env.randomize
            or config.hardware_profile.profile_id != "homogeneous_v1"
            or config.hardware_profile_lambda != [1.0] * 4
            or config.cost_profile.profile_id != "uniform_v1"
            or config.alg.n_test != 100 or config.alg.n_eval < 4
            or config.alg.n_episodes <= 0 or config.alg.period <= 0
            or config.alg.n_episodes % config.alg.period):
        raise EnergyProfileError("EIA configuration differs from the registered task/evidence contract")
    if method == "refine_eia_er" and (config.lio.Fairness_multiplier != 0.0
                                       or config.lio.energy_weight != 1.0):
        raise EnergyProfileError("REFiNE EIA objective differs from B=0, beta=1")


@contextmanager
def uniform_cost_scope(method):
    """Patch both classes reached by the actual mixed team, then restore them."""
    def cost(agent, state, action):
        return action_cost("uniform_v1", action, agent.energy_param)
    with ExitStack() as stack:
        for cls in agent_classes(method):
            stack.enter_context(rebind(cls, "calculate_energy_cost", cost))
        yield


def probe_attack(method, config):
    """Probe actual give_reward bodies using their real constructors and fixed inputs."""
    import numpy as np
    classes = agent_classes(method)
    records = []
    class FixedSession:
        def run(self, operation, feed_dict):
            return np.ones((1, 4), dtype=np.float64)
    def fake_networks(agent):
        agent.obs, agent.action_others, agent.reward_function = object(), object(), object()
    for giver in range(4):
        cls = classes[1] if giver == 1 else classes[0]
        with rebind(cls, "create_networks", fake_networks), redirect_stdout(io.StringIO()):
            agent = cls(config.lio, 15, 3, config.nn, f"agent_{giver}", 1.0, 4, giver, 1.0)
        by_action = [agent.give_reward(np.zeros(15), [a] * 4, FixedSession()) for a in range(3)]
        for recipient in range(4):
            multipliers = [float(values[recipient]) for values in by_action]
            expected = [float(config.lio.eia_w_lever), 1.0, float(config.lio.eia_w_door)] if giver == 1 and recipient != giver else [1.0] * 3
            if multipliers != expected:
                raise EnergyProfileError(f"Effective EIA multipliers differ for {giver}->{recipient}: {multipliers}")
            records.append(dict(giver_id=giver, recipient_id=recipient,
                                multipliers=multipliers, delivered=recipient != giver))
    return dict(schema_version=1,
                configured_weights=dict(lever=float(config.lio.eia_w_lever), door=float(config.lio.eia_w_door)),
                action_order=["lever", "start", "door"], exploitative_giver_id=1,
                training_active=True, evaluation_active=True, applies_to="recipient_action",
                location="outside_differentiable_reward_graph", effective_multipliers=records,
                probe=dict(source="actual_give_reward", fixed_network_output=1.0,
                           r_multiplier=1.0, dtype="float64", self_payment="zeroed by evaluator"))


def source_hashes():
    paths = [Path(__file__), REPO / "lio/alg/REFiNE_eia_er.py",
             REPO / "lio/alg/lio_eia_er.py", REPO / "sensys_runs/run_single.py",
             REPO / "sensys_runs/energy_profiles.py", REPO / "sensys_runs/energy_profile_eval.py",
             REPO / "lio/alg/train_lio_eia_er.py", REPO / "lio/alg/train_REFiNE_eia_er.py"]
    return {str(p.relative_to(REPO)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def output_os(original, results_root):
    if results_root is None:
        return original
    default = (REPO / "lio/results").resolve()
    def join(*parts):
        joined = original.path.join(*parts)
        candidate = Path(original.path.abspath(joined))
        try:
            relative = candidate.relative_to(default)
        except ValueError:
            return joined
        return str(results_root / relative)
    return Proxy(original, path=Proxy(original.path, join=join))


def wrap_eia_training(train, profile, method, results_root=None):
    trainer = sys.modules[train.__module__]
    original_evaluate = trainer.evaluate
    def wrapped(config):
        validate_config(config, method)
        last = {}
        def periodic(n_eval, env, sess, agents, *args, **kwargs):
            result = original_evaluate.test_room_symmetric(n_eval, env, sess, agents, *args, **kwargs)
            last.update(env=env, sess=sess, agents=agents)
            return result
        with uniform_cost_scope(method), rebind(trainer, "evaluate", Proxy(original_evaluate, test_room_symmetric=periodic)), rebind(trainer, "os", output_os(trainer.os, results_root)):
            train(config)
            if not last:
                raise EnergyProfileError("No periodic evaluation captured before final checkpoint")
            from sensys_runs import energy_profile_eval as final
            # The original evaluator and observation recorder retain their behavior.
            root = results_root if results_root is not None else REPO / "lio/results"
            prefix = root / config.main.exp_name / config.main.dir_name / config.main.model_name
            algorithms = dict(final.METHOD_ALGORITHMS, **{method: config.alg.name})
            with rebind(final, "METHOD_ALGORITHMS", algorithms):
                final.evaluate_final_policy(last["sess"], last["env"], last["agents"], config, prefix, method)
            costs = [{"agent_id": agent.agent_id,
                      "costs": [agent.calculate_energy_cost([0.0] * 15, action) for action in range(3)]}
                     for agent in last["agents"]]
            if any(row["costs"] != [1.0] * 3 for row in costs):
                raise EnergyProfileError("Actual mixed-team uniform cost probe failed")
            metadata_path = prefix.parent / final.MANIFEST_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(eia_attack=config.eia_attack, effective_uniform_costs=costs)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            completion = dict(status="complete", method=method, seed=config.main.seed,
                              run_id=f"{config.main.exp_name}/{config.main.dir_name}",
                              final_evaluation_episodes=100, final_evaluation_rows=400,
                              final_evaluation_sha256=final.sha256(metadata_path))
            completion.update(schema_version=1, episodes_sha256=metadata["episodes_sha256"])
            with (prefix.parent / "eia_run_completion.json").open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(completion, stream, indent=2, sort_keys=True)
    return wrapped


@contextmanager
def adapter_scope(results_root=None):
    original_parse = run_single._parse_args
    original_import = run_single.importlib
    original_provenance = run_single._write_provenance
    original_aliases = {name: sys.modules.get(name) for name in ("lio_eia_er", "REFiNE_eia_er")}
    def parse():
        args = original_parse()
        validate_arguments(args)
        return args
    def validate(profile, method, path):
        if method not in METHODS or profile != "uniform_v1":
            raise EnergyProfileError("This adapter accepts only registered EIA uniform runs")
        get_profile(profile)
        if os.path.lexists(path):
            raise EnergyProfileError("Run output already exists: " + path)
    def import_module(name):
        if name in TRAINER_MAP:
            target = TRAINER_MAP[name]
            canonical = original_import.import_module(target)
            sys.modules[target.rsplit(".", 1)[-1]] = canonical
        return original_import.import_module(name)
    def provenance(path, args, config, tf_version, np_version):
        validate_config(config, args.method)
        config.eia_attack = probe_attack(args.method, config)
        original_provenance(path, args, config, tf_version, np_version)
        destination = Path(path) / "provenance.json"
        record = json.loads(destination.read_text(encoding="utf-8"))
        record.update(eia_attack=config.eia_attack, eia_source_sha256=source_hashes(),
                      eia_adapter="scoped_registered_eia_v1",
                      agent_cost_patch_targets=[x[0] + "." + x[1] for x in CLASS_MAP[args.method]])
        aliases = record.setdefault("agent_module_aliases", {})
        target = CLASS_MAP[args.method][1][0]
        aliases[target.rsplit(".", 1)[-1]] = target
        if sys.modules.get(target.rsplit(".", 1)[-1]) is not sys.modules.get(target):
            raise EnergyProfileError("EIA bare and canonical modules are not identical")
        destination.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    try:
        with ExitStack() as stack:
            for name, value in (("_parse_args", parse), ("validate_run", validate),
                                ("importlib", Proxy(original_import, import_module=import_module)),
                                ("_write_provenance", provenance),
                                ("wrap_training", lambda train, profile, method: wrap_eia_training(train, profile, method, results_root)),
                                ("os", output_os(run_single.os, results_root))):
                stack.enter_context(rebind(run_single, name, value))
            yield
    finally:
        for name, previous in original_aliases.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-root", type=Path)
    known, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    results_root = known.results_root
    if results_root is not None:
        if not results_root.is_absolute():
            parser.error("--results-root must be absolute")
        results_root = results_root.resolve()
        if results_root == REPO or REPO in results_root.parents:
            parser.error("Explicit --results-root must be outside the repository")
    previous_argv, previous_cwd = sys.argv, os.getcwd()
    try:
        sys.argv = [str(Path(__file__)), *remainder]
        with adapter_scope(results_root):
            run_single.main()
    except EnergyProfileError as exc:
        print("CONFIG_ERROR: " + str(exc), file=sys.stderr)
        return 2
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


if __name__ == "__main__":
    sys.exit(main())
