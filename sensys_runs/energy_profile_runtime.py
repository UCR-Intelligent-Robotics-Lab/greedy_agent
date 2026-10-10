"""Scoped clean-ER cost adapter; legacy does not import or patch agent classes.

The original training and evaluation call sites remain authoritative. This
adapter changes only their cost lookup, with no accounting or RNG side effects.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import importlib
import os
from typing import Callable, Iterator

from sensys_runs.energy_profiles import EnergyProfileError, action_cost, get_profile


AGENT_CLASSES = {
    "lio_er": ("lio.alg.lio_agent_er", "LIO"),
    "refine_er": ("lio.alg.REFiNE_er", "REFiNE"),
}


def validate_run(profile: str, method: str, output_path: str) -> None:
    """Reject unsupported or stale new-profile runs before any output write."""
    if profile == "legacy":
        return
    get_profile(profile)
    if method not in AGENT_CLASSES:
        raise EnergyProfileError("Non-legacy costs support only clean lio_er and refine_er")
    if os.path.lexists(output_path):
        raise EnergyProfileError(f"Run output already exists: {output_path}")


@contextmanager
def cost_profile_scope(profile: str, method: str) -> Iterator[None]:
    """Patch only the selected class and always restore its original method."""
    if profile == "legacy":
        yield
        return
    get_profile(profile)
    if method not in AGENT_CLASSES:
        raise EnergyProfileError("Non-legacy costs support only clean lio_er and refine_er")
    module_name, class_name = AGENT_CLASSES[method]
    agent_class = getattr(importlib.import_module(module_name), class_name)
    original = agent_class.calculate_energy_cost

    def calculate_energy_cost(self, state, action):
        # Observations are deliberately irrelevant to the registered fixed table.
        return action_cost(profile, action, self.energy_param)

    agent_class.calculate_energy_cost = calculate_energy_cost
    try:
        yield
    finally:
        agent_class.calculate_energy_cost = original


def wrap_training(train: Callable, profile: str, method: str) -> Callable:
    """Keep the adapter active for training and its existing evaluations."""
    if profile == "legacy":
        return train

    @wraps(train)
    def wrapped(config):
        with cost_profile_scope(profile, method):
            return train(config)

    return wrapped
