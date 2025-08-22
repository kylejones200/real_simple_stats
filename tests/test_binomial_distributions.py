import math

from real_simple_stats.binomial_distributions import (
    is_binomial_experiment,
    binomial_probability,
    binomial_mean,
    binomial_variance,
    binomial_std_dev,
    expected_value_single,
    expected_value_multiple,
    normal_approximation,
)


def test_is_binomial_experiment_basic():
    assert is_binomial_experiment(10, ["success", "fail"], 0.5)
    assert not is_binomial_experiment(0, ["s", "f"], 0.5)
    assert not is_binomial_experiment(10, ["s"], 0.5)
    assert not is_binomial_experiment(10, ["s", "f", "m"], 0.5)
    assert not is_binomial_experiment(10, ["s", "f"], -0.1)


def test_binomial_probability_and_moments():
    # Known value: n=10, k=3, p=0.5 => C(10,3)/2^10 = 120/1024
    p = binomial_probability(10, 3, 0.5)
    assert math.isclose(p, 120 / 1024)

    # Moments
    n, prob = 20, 0.3
    mu = binomial_mean(n, prob)
    var = binomial_variance(n, prob)
    std = binomial_std_dev(n, prob)
    assert math.isclose(mu, n * prob)
    assert math.isclose(var, n * prob * (1 - prob))
    assert math.isclose(std, math.sqrt(var))


def test_expected_value_helpers():
    assert expected_value_single(10, 0.25) == 2.5
    values = [1, 2, 3]
    probs = [0.2, 0.3, 0.5]
    assert math.isclose(
        expected_value_multiple(values, probs), 1 * 0.2 + 2 * 0.3 + 3 * 0.5
    )


def test_normal_approximation_reasonable_accuracy():
    # Compare normal approx P(X<=k) with true for Binomial(10,0.5)
    # True P(X<=5) = 0.623046875
    approx = normal_approximation(10, 0.5, 5, use_continuity=True)
    assert abs(approx - 0.623046875) < 0.03
