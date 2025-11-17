from real_simple_stats.binomial_distributions import (
    binomial_mean,
    binomial_probability,
    binomial_std_dev,
    binomial_variance,
    expected_value_multiple,
    expected_value_single,
    is_binomial_experiment,
    normal_approximation,
)

# Parameters
n, p = 10, 0.4
k = 3

# Check if this is a binomial setup
print("Is binomial:", is_binomial_experiment(n, ["pass", "fail"], p))

# Exact probability using binomial formula
print(f"P(X = {k}):", binomial_probability(n, k, p))

# Descriptive stats
print("Mean:", binomial_mean(n, p))
print("Variance:", binomial_variance(n, p))
print("Standard deviation:", binomial_std_dev(n, p))

# Expected values
print("Expected value (single outcome):", expected_value_single(85, 0.2))
print(
    "Expected value (multiple outcomes):",
    expected_value_multiple([10, 20, 30], [0.1, 0.5, 0.4]),
)

# Normal approximation with continuity correction
print(f"Normal approximation P(X ≤ {k}):", normal_approximation(n, p, k))
