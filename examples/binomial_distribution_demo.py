import logging

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
n, p = 10, 0.4
k = 3

# Check if this is a binomial setup
logger.info("Is binomial: %s", is_binomial_experiment(n, ["pass", "fail"], p))

# Exact probability using binomial formula
logger.info("P(X = %s): %s", k, binomial_probability(n, k, p))

# Descriptive stats
logger.info("Mean: %s", binomial_mean(n, p))
logger.info("Variance: %s", binomial_variance(n, p))
logger.info("Standard deviation: %s", binomial_std_dev(n, p))

# Expected values
logger.info("Expected value (single outcome): %s", expected_value_single(85, 0.2))
logger.info(
    "Expected value (multiple outcomes): %s",
    expected_value_multiple([10, 20, 30], [0.1, 0.5, 0.4]),
)

# Normal approximation with continuity correction
logger.info("Normal approximation P(X ≤ %s): %s", k, normal_approximation(n, p, k))
