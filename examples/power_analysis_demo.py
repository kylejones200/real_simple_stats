"""Demo: Power analysis - sample size, power, and effect size."""

import logging

from real_simple_stats.power_analysis import (
    power_proportion_test,
    power_t_test,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# T-test power: how many subjects for 80% power to detect d=0.5?
result = power_t_test(delta=0.5, power=0.8, sd=1.0)
logger.info("T-test sample size for δ=0.5, power=0.8: n ≥ %s", result["n"])

# T-test: what power with n=30, delta=0.6?
result = power_t_test(n=30, delta=0.6)
logger.info("Power with n=30, δ=0.6: %.2f%%", result["power"] * 100)

# Proportion test: sample size for comparing p1=0.6 vs p0=0.5
result = power_proportion_test(p1=0.6, p0=0.5, power=0.8)
logger.info("\nProportion test sample size (p1=0.6 vs p0=0.5): n ≥ %s", result["n"])

# Proportion: what power with n=100?
result = power_proportion_test(n=100, p1=0.6, p0=0.5)
logger.info("Power with n=100: %.2f%%", result["power"] * 100)
