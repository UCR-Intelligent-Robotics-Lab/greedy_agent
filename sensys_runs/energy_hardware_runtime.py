"""Scoped registered hardware multipliers, without changing trainer call sites."""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from numbers import Integral

from sensys_runs.energy_profiles import EnergyProfileError, get_hardware_profile

ACTIVE_HARDWARE = ContextVar("energy_hardware", default=None)


def hardware_training(train):
    """Bind validated lambda only during the existing P1 cost scope."""
    @wraps(train)
    def wrapped(config):
        profile_id = config.get("hardware_profile", {}).get("profile_id", "homogeneous_v1")
        profile = get_hardware_profile(profile_id)
        vector = profile.vector(config.env.n_agents)
        if ("hardware_profile_lambda" in config
                and tuple(config.hardware_profile_lambda) != vector):
            raise EnergyProfileError("Config lambda differs from registered hardware profile")
        # Preserve the exact P1 computation for homogeneous defaults.
        token = ACTIVE_HARDWARE.set(None if profile_id == "homogeneous_v1" else vector)
        try:
            return train(config)
        finally:
            ACTIVE_HARDWARE.reset(token)
    return wrapped


def agent_lambda(agent) -> float:
    vector = ACTIVE_HARDWARE.get()
    index = agent.agent_id
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(vector):
        raise EnergyProfileError("Agent index outside registered hardware vector")
    return vector[index]
