"""Training-only episode-mean energy in LIO's buffer reward.

P4's total_energy already contains lambda. Never change env.step or evaluation.
Both ordinary and prime training episodes use the temporarily rebound function.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import math

from sensys_runs.energy_profiles import EnergyProfileError

REWARD_FORM = "episode_mean"
REWARD_FIELDS = ("energy_in_reward", "energy_reward_beta", "energy_reward_form")


def registered_beta() -> float:
    """Call the existing REFiNE config factory without importing TensorFlow."""
    from sensys_runs.energy_profile_campaign import load_config
    beta = float(load_config("refine_er").lio.energy_weight)
    if not math.isfinite(beta) or beta < 0:
        raise EnergyProfileError("Registered energy weight must be finite and nonnegative")
    return beta


def configure(config) -> None:
    config.energy_in_reward = True
    config.energy_reward_beta = registered_beta()
    config.energy_reward_form = REWARD_FORM
    # Explicit study decision: this LIO baseline has no fairness shaping.
    config.lio.Fairness_multiplier = 0.0


@contextmanager
def energy_reward_scope(trainer, beta: float):
    """Restore the exact training function on normal exit and any exception."""
    original = trainer.run_episode
    if getattr(original, "_energy_reward_wrapper", False):
        raise EnergyProfileError("Nested energy-reward scopes would charge energy twice")

    @wraps(original)
    def run_episode(*args, **kwargs):
        result = original(*args, **kwargs)
        buffers, _ = result
        penalties = []
        for buf in buffers:
            if not buf.reward or not math.isfinite(buf.total_energy) or buf.total_energy < 0:
                raise EnergyProfileError("Energy shaping requires nonempty buffers and finite energy")
            penalties.append(beta * buf.total_energy / len(buf.reward))
        for buf, penalty in zip(buffers, penalties):
            buf.reward[:] = [reward - penalty for reward in buf.reward]
        return result

    run_episode._energy_reward_wrapper = True
    trainer.run_episode = run_episode
    try:
        yield
    finally:
        trainer.run_episode = original


def wrap_training(train, trainer):
    @wraps(train)
    def wrapped(config):
        if (config.energy_reward_beta != registered_beta()
                or config.energy_reward_form != REWARD_FORM):
            raise EnergyProfileError("Energy-reward config differs from registered design")
        with energy_reward_scope(trainer, config.energy_reward_beta):
            return train(config)
    return wrapped
