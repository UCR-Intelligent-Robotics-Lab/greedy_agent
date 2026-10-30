"""Energy-reward LIO+EIA using the unchanged A9 reward-buffer wrapper.

The backend method remains lio_eia_er. An explicit objective and display label
separate this arm from the completed plain LIO+EIA campaign in every config and
provenance record. No existing launch path imports this adapter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import copy
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sensys_runs import energy_reward_runtime as energy
from sensys_runs import run_single
from sensys_runs.energy_profiles import EnergyProfileError
from sensys_runs.iccps27_followup import eia_adapter as eia

METHOD = "lio_eia_er"
OBJECTIVE = "energy_reward"
ARM = "er_lio_eia_er"
DISPLAY_LABEL = "ER-LIO+EIA"
ADAPTER_ID = "scoped_registered_er_lio_eia_v1"


def configure(config):
    """Use the A9/A2 configuration function without changing its implementation."""
    energy.configure(config)
    config.followup_arm = ARM
    config.training_objective = OBJECTIVE
    config.display_label = DISPLAY_LABEL
    validate_energy_config(config)


def validate_energy_config(config):
    if (config.get("energy_in_reward") is not True
            or config.get("energy_reward_beta") != 1.0
            or energy.registered_beta() != 1.0
            or config.get("energy_reward_form") != energy.REWARD_FORM
            or config.lio.Fairness_multiplier != 0.0
            or config.get("followup_arm") != ARM
            or config.get("training_objective") != OBJECTIVE
            or config.get("display_label") != DISPLAY_LABEL):
        raise EnergyProfileError("ER-LIO+EIA requires the registered A9 beta=1 episode_mean objective")


def energy_provenance(config):
    validate_energy_config(config)
    return dict(schema_version=1, method=METHOD, arm=ARM,
                objective=OBJECTIVE, display_label=DISPLAY_LABEL,
                energy_in_reward=True, energy_reward_beta=config.energy_reward_beta,
                energy_reward_form=config.energy_reward_form,
                agent_ids=[0, 1, 2, 3], episode_kinds=["ordinary", "prime"],
                buffer_field="reward", energy_field="total_energy",
                penalty="beta * buffer.total_energy / len(buffer.reward)",
                total_energy_includes_lambda=True,
                consumers=["actor_update", "incentive_learner"],
                training_only=True, environment_reward_unchanged=True,
                implementation="sensys_runs.energy_reward_runtime.wrap_training")


def source_hashes():
    result = eia.source_hashes()
    paths = ("sensys_runs/iccps27_followup/er_eia_adapter.py",
             "sensys_runs/energy_reward_runtime.py",
             "sensys_runs/energy_profile_runtime.py",
             "sensys_runs/energy_profile_campaign.py",
             "lio/alg/lio_agent_er.py", "lio/alg/config_room_lio.py",
             "lio/alg/config_room_REFiNE.py")
    result.update({name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                   for name in paths})
    return result


def wrap_er_eia_training(train, profile, method, results_root=None):
    if method != METHOD:
        raise EnergyProfileError("ER-LIO+EIA accepts only the lio_eia_er backend")
    # Capture the actual trainer before either wrapper changes train.__module__.
    trainer = sys.modules[train.__module__]
    attacked = eia.wrap_eia_training(train, profile, method, results_root)
    charged = energy.wrap_training(attacked, trainer)

    @wraps(train)
    def wrapped(config):
        validate_energy_config(config)
        return charged(config)
    return wrapped


@contextmanager
def adapter_scope(results_root=None, launch_argv=None):
    """Compose the existing EIA adapter and restore every temporary binding."""
    with eia.adapter_scope(results_root):
        inherited_parse = run_single._parse_args
        inherited_provenance = run_single._write_provenance

        def parse():
            args = inherited_parse()
            if args.method != METHOD:
                raise EnergyProfileError("ER-LIO+EIA accepts only --method lio_eia_er")
            return args

        def provenance(path, args, config, tf_version, np_version):
            configure(config)
            recorded_args = copy(args)
            recorded_args.energy_in_reward = True
            inherited_provenance(path, recorded_args, config, tf_version, np_version)
            destination = Path(path)/"provenance.json"
            record = json.loads(destination.read_text(encoding="utf-8"))
            record.update(followup_arm=ARM, training_objective=OBJECTIVE,
                          display_label=DISPLAY_LABEL, er_eia_adapter=ADAPTER_ID,
                          energy_reward=energy_provenance(config),
                          er_eia_source_sha256=source_hashes())
            if launch_argv is not None:
                record["er_eia_launch_argv"] = list(launch_argv)
            destination.write_text(json.dumps(record, indent=2, sort_keys=True,
                                              allow_nan=False)+"\n",
                                   encoding="utf-8", newline="\n")

        with eia.rebind(run_single, "_parse_args", parse), \
             eia.rebind(run_single, "_write_provenance", provenance), \
             eia.rebind(run_single, "wrap_training",
                        lambda train, profile, method: wrap_er_eia_training(
                            train, profile, method, results_root)):
            yield


def main(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Registered ER-LIO+EIA adapter", add_help=False)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--energy_in_reward", action="store_true")
    known, remainder = parser.parse_known_args(supplied)
    if not known.energy_in_reward or supplied.count("--energy_in_reward") != 1:
        parser.error("exactly one explicit --energy_in_reward is required")
    results_root = known.results_root
    if results_root is not None:
        if not results_root.is_absolute():
            parser.error("--results-root must be absolute")
        results_root = results_root.resolve()
        if results_root == REPO or REPO in results_root.parents:
            parser.error("Explicit --results-root must be outside the repository")
    previous_argv, previous_cwd = sys.argv, os.getcwd()
    try:
        # The flag is consumed here, before the unchanged plain-EIA parser. The
        # only energy wrapper is installed by this scoped, explicitly opted-in path.
        sys.argv = [str(Path(__file__)), *remainder]
        launch_argv = [sys.executable, str(Path(__file__)), *supplied]
        with adapter_scope(results_root, launch_argv):
            run_single.main()
    except EnergyProfileError as exc:
        print("CONFIG_ERROR: " + str(exc), file=sys.stderr)
        return 2
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


if __name__ == "__main__":
    sys.exit(main())
