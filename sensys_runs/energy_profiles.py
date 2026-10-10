"""Normalized ER action-cost scenarios from the minimal ICCPS 2027 plan.

Lives outside lio.alg because that package eagerly imports TensorFlow.
These are modeled costs, not Joules. Action order is lever, start, door;
the literature contrast uses an assumed ER action mapping.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from numbers import Integral, Real
from types import MappingProxyType


class EnergyProfileError(ValueError):
    """Invalid cost configuration; launchers report exit code 2."""


ENERGY_PARAM = 1.0
ACTION_LABELS = ("lever", "start", "door")


@dataclass(frozen=True)
class EnergyProfile:
    """Immutable provenance and exact source ratio for a fixed cost table."""

    profile_id: str
    source_ratio: Fraction
    source_classification: str
    source_location: str
    source_urls: tuple[str, ...] = ()
    version: int = 1
    units: str = "normalized_modeled_cost"
    normalization: str = "[r, 1, 1] / ((r + 2) / 3)"

    @property
    def costs(self) -> tuple[float, ...]:
        weights = (self.source_ratio, Fraction(1), Fraction(1))
        mean = sum(weights) / len(weights)
        return tuple(float(weight / mean) for weight in weights)

    def snapshot(self) -> dict:
        """Return a fresh JSON-compatible record for config and provenance."""
        record = asdict(self)
        record["source_ratio"] = {
            "numerator": self.source_ratio.numerator,
            "denominator": self.source_ratio.denominator,
            "value": float(self.source_ratio),
        }
        record.update(action_labels=ACTION_LABELS, normalized_vector=self.costs,
                      energy_param=ENERGY_PARAM, actor_cost_form="episode_mean")
        return record


PROFILES = MappingProxyType({
    p.profile_id: p for p in (
        EnergyProfile("uniform_v1", Fraction(1), "uniform-cost control",
                      "REFiNE_Minimal_Energy_Code_Plan.md, Section 4A"),
        EnergyProfile("low_contrast_v1", Fraction("9.2") / Fraction("7.5"),
                      "literature-informed contrast; assumed action mapping",
                      "Mokhtari et al., HONET 2024, Section V-A and Figure 4",
                      ("https://www.famaey.eu/papers/cnf-mokhtari2024a.pdf",)),
        EnergyProfile("paper_ratio_v1", Fraction("1.75"),
                      "manuscript scenario; physical calibration unverified",
                      "SenSys.tex, Section 4.1, Energy Profiling"),
    )
})
PROFILE_CHOICES = ("legacy", *PROFILES)


def get_profile(profile_id: str) -> EnergyProfile:
    """Resolve a registered fixed table; legacy has no substitute table."""
    if not isinstance(profile_id, str) or profile_id not in PROFILES:
        raise EnergyProfileError(f"Unknown fixed cost profile: {profile_id!r}")
    return PROFILES[profile_id]


def action_cost(profile: str, action: int, energy_param: float) -> float:
    """Pure lookup; reject unregistered scaling in this homogeneous campaign."""
    selected = get_profile(profile)
    if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < len(ACTION_LABELS):
        raise EnergyProfileError(f"Invalid ER action: {action!r}")
    if (isinstance(energy_param, bool) or not isinstance(energy_param, Real)
            or not math.isfinite(energy_param) or energy_param != ENERGY_PARAM):
        raise EnergyProfileError(f"This campaign requires energy_param={ENERGY_PARAM}")
    cost = selected.costs[action] * energy_param
    if not math.isfinite(cost) or cost <= 0:
        raise EnergyProfileError("Action costs must be finite and positive")
    return cost
