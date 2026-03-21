"""Demo: Confidence intervals, sample size, and Central Limit Theorem."""

import logging

from real_simple_stats import descriptive_statistics as desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from real_simple_stats.sampling_and_intervals import (
    clt_probability_between,
    confidence_interval_known_std,
    confidence_interval_unknown_std,
    required_sample_size,
    slovins_formula,
)

# Sample data
data = [72, 68, 75, 71, 69, 74, 70, 73, 67, 72]
sample_mean = desc.mean(data)
sample_std = desc.sample_std_dev(data)
n = len(data)

# Confidence interval (unknown population std)
ci_low, ci_high = confidence_interval_unknown_std(sample_mean, sample_std, n, 0.95)
logger.info("95%% CI (t-dist): (%.2f, %.2f)", ci_low, ci_high)

# Confidence interval (known population std = 3)
ci_low, ci_high = confidence_interval_known_std(sample_mean, 3, n, 0.95)
logger.info("95%% CI (known σ=3): (%.2f, %.2f)", ci_low, ci_high)

# Required sample size for desired margin of error
width = 2  # total width of interval
std_dev = 3
required_n = required_sample_size(0.95, width, std_dev)
logger.info("Sample size for 95%% CI width≤%s: n ≥ %s", width, required_n)

# Slovin's formula: sample size for finite population
N = 1000
e = 0.05  # margin of error
n_slovin = slovins_formula(N, e)
logger.info("Slovin's formula (N=%s, e=%s): n ≥ %s", N, e, n_slovin)

# Central Limit Theorem
pop_mean, pop_std, n_samples = 80, 10, 100
prob = clt_probability_between(78, 82, pop_mean, pop_std, n_samples)
logger.info("P(78 < sample_mean < 82) for n=%s: %.4f", n_samples, prob)
