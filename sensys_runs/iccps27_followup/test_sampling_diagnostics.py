"""Sampling diagnostics distinguish finite samples from population bounds."""
import math

import pytest

from sensys_runs.iccps27_followup import sampling_diagnostics as sd


def test_deterministic_limits():
    assert sd.diagnose(1.0, 1.0, 1.0)["sample_mean_prediction_envelope_95"] == [1.0, 1.0]
    result = sd.diagnose(0.0, 3.0, 0.0)
    assert result["population_E_T_bounds"] == [2.0, 5.0]
    assert result["binomial_count_prediction_interval_95"] == [100, 100]
    assert result["sample_mean_prediction_envelope_95"] == [2.0, 5.0]


def test_mean_below_population_bound_can_be_sampling_compatible():
    result = sd.diagnose(.99, 1.0, 1.0)
    assert result["population_bound_side"] == "below"
    assert result["observed_mean_steps"] == 1.0
    assert result["binomial_count_prediction_interval_95"] == [0, 3]
    assert result["sample_mean_prediction_envelope_95"] == [1.0, 1.12]
    assert result["mean_inside_prediction_envelope"]
    assert result["count_inside_prediction_interval"]
    assert result["observed_values_adjusted"] is False


def test_mean_above_population_bound_can_be_sampling_compatible():
    result = sd.diagnose(.99, 1.08, .98)
    assert result["population_bound_side"] == "above"
    assert result["mean_inside_prediction_envelope"]
    assert result["count_inside_prediction_interval"]


def test_real_deviation_is_flagged_without_changing_the_value():
    result = sd.diagnose(.99, 1.4, .9)
    assert not result["mean_inside_prediction_envelope"]
    assert not result["count_inside_prediction_interval"]
    assert result["observed_mean_steps"] == 1.4
    assert result["observed_one_step_fraction"] == .9


def test_count_check_is_separate_from_conservative_mean_envelope():
    result = sd.diagnose(.99, 1.04, .96)
    assert result["mean_inside_prediction_envelope"]
    assert not result["count_inside_prediction_interval"]


@pytest.mark.parametrize("p1,mean,fraction,n", [
    (-.01, 1, 1, 100), (1.01, 1, 1, 100), (math.nan, 1, 1, 100),
    (.99, math.inf, 1, 100), (.99, 1, math.nan, 100),
    (.99, 1, 1, 99), (.99, 1.001, 1, 100), (.99, 1, .991, 100),
    (.99, 1, .99, 100), (.99, 1.05, .99, 100), (.99, 1, 1, True),
])
def test_invalid_evidence_rejected(p1, mean, fraction, n):
    with pytest.raises(ValueError):
        sd.diagnose(p1, mean, fraction, n)


def test_quantiles_against_independent_scipy_oracle():
    from scipy.stats import binom
    for p1 in [0, 1e-12, .01, .5, .95, .99, .999999999999, 1]:
        for probability in [.025, .975]:
            assert sd.binomial_quantile(probability, 100, 1-p1) == int(binom.ppf(probability, 100, 1-p1))


def test_envelope_coverage_inherits_binomial_count_coverage():
    for q in [.001, .01, .05, .2, .5, .8, .99]:
        low, high = [sd.binomial_quantile(p, 100, q) for p in [.025, .975]]
        coverage = math.fsum(math.comb(100, k)*q**k*(1-q)**(100-k) for k in range(low, high+1))
        assert coverage >= .95
        for k in range(low, high+1):
            assert 1+low/100 <= 1+k/100 <= 1+4*k/100 <= 1+4*high/100
