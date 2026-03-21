"""Demo: Resampling - bootstrap and permutation tests."""

import logging

import numpy as np

from real_simple_stats.resampling import bootstrap, permutation_test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bootstrap: estimate CI for median
data = [23, 25, 28, 30, 32, 35, 38, 40, 42, 45]
result = bootstrap(data, np.median, n_iterations=1000, random_seed=42)
logger.info("Bootstrap for median:")
logger.info("  Sample median: %.2f", result["statistic"])
logger.info("  Bootstrap mean: %.2f", result["mean"])
logger.info("  95%% CI: %s", result["confidence_interval"])

# Permutation test: do two groups differ?
group1 = [28, 30, 32, 34, 36]
group2 = [24, 26, 28, 30, 32]
pt_result = permutation_test(
    group1,
    group2,
    statistic=lambda a, b: np.mean(a) - np.mean(b),
    n_permutations=1000,
    random_seed=42,
)
logger.info("\nPermutation test (difference in means):")
logger.info("  Observed difference: %.2f", pt_result["observed_statistic"])
logger.info("  P-value: %.4f", pt_result["p_value"])
logger.info("  Reject H0 at α=0.05: %s", pt_result["p_value"] < 0.05)
