"""Separate, conservative sampling diagnostics for 100 final episodes.

Population reset bounds and sample-mean prediction envelopes are distinct.
No diagnostic changes a recorded observation, eligibility, or comparison.
"""
from __future__ import annotations

import math

N_EPISODES = 100
HORIZON = 5


def binomial_quantile(probability, n, q):
    """Small-n binomial inverse CDF using the finite PMF, with no random draws."""
    if (isinstance(n, bool) or not isinstance(n, int) or n < 1
            or not math.isfinite(q) or not 0 <= q <= 1
            or not 0 < probability < 1):
        raise ValueError("Invalid binomial quantile input")
    if q == 0:
        return 0
    if q == 1:
        return n
    masses = [math.comb(n, k) * q ** k * (1-q) ** (n-k)
              for k in range(n+1)]
    total = math.fsum(masses)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid finite binomial mass")
    for k in range(n+1):
        if math.fsum(masses[:k+1]) / total >= probability:
            return k
    return n


def _episode_count(value, name):
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    count = round(value)
    if abs(value-count) > 1e-8:
        raise ValueError(name + " is not a 100-episode count")
    return count


def diagnose(p1, observed_mean_steps, observed_one_step_fraction, n_episodes=100):
    """Return unadjusted evidence and a pointwise 95% prediction envelope."""
    if (isinstance(p1, bool) or not math.isfinite(p1) or not 0 <= p1 <= 1
            or n_episodes != N_EPISODES or isinstance(n_episodes, bool)):
        raise ValueError("Require valid reset P1 and exactly 100 episodes")
    if (isinstance(observed_mean_steps, bool) or not math.isfinite(observed_mean_steps)
            or not 1 <= observed_mean_steps <= HORIZON
            or isinstance(observed_one_step_fraction, bool)
            or not math.isfinite(observed_one_step_fraction)
            or not 0 <= observed_one_step_fraction <= 1):
        raise ValueError("Invalid observed duration or one-step fraction")
    total_steps = _episode_count(observed_mean_steps*n_episodes, "Total steps")
    one_step_count = _episode_count(observed_one_step_fraction*n_episodes, "One-step count")
    non_one_step = n_episodes-one_step_count
    if not n_episodes+non_one_step <= total_steps <= n_episodes+(HORIZON-1)*non_one_step:
        raise ValueError("Observed duration and one-step count are inconsistent")
    lower, upper = 2-p1, 1+(HORIZON-1)*(1-p1)
    k_low = binomial_quantile(.025, n_episodes, 1-p1)
    k_high = binomial_quantile(.975, n_episodes, 1-p1)
    envelope = [1+k_low/n_episodes, 1+(HORIZON-1)*k_high/n_episodes]
    side = ("below" if observed_mean_steps < lower else
            "above" if observed_mean_steps > upper else "inside")
    distance = (lower-observed_mean_steps if side == "below" else
                observed_mean_steps-upper if side == "above" else 0.0)
    mean_inside = envelope[0] <= observed_mean_steps <= envelope[1]
    count_inside = k_low <= non_one_step <= k_high
    return dict(
        P1=p1, observed_mean_steps=observed_mean_steps,
        observed_one_step_fraction=observed_one_step_fraction,
        n_episodes=n_episodes, horizon=HORIZON,
        population_E_T_bounds=[lower, upper], population_bound_side=side,
        distance_outside_population_bounds=distance,
        observed_non_one_step_count=non_one_step,
        binomial_count_prediction_interval_95=[k_low, k_high],
        sample_mean_prediction_envelope_95=envelope,
        mean_inside_prediction_envelope=mean_inside,
        count_inside_prediction_interval=count_inside,
        interpretation=("Compatible with 100-episode sampling under the conservative mean envelope"
                        if mean_inside else
                        "Not accounted for by the conservative pointwise mean envelope"),
        count_interpretation=("Observed non-one-step count is inside its central 95% interval"
                              if count_inside else
                              "Observed non-one-step count is outside its central 95% interval"),
        assumptions="Fixed deployed policy; independent reset episodes; integer T in [1,5]",
        qualification=("Pointwise prediction diagnostic, not a confidence interval for E[T], "
                       "not an equivalence test, and not a simultaneous guarantee."),
        observed_values_adjusted=False,
    )
