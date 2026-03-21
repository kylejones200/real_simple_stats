"""Demo: Hypothesis testing - critical values, p-values, decisions."""

import logging

from real_simple_stats.hypothesis_testing import (
    critical_value_t,
    critical_value_z,
    p_value_method,
    reject_null,
    state_alternate_hypothesis,
    state_null_hypothesis,
    t_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State hypotheses
logger.info("%s", state_null_hypothesis("μ = 100"))
logger.info("%s", state_alternate_hypothesis("μ ≠ 100"))

# One-sample t-test: sample vs population mean
sample_mean, pop_mean, sample_std, n = 105, 100, 12, 25
t_stat = t_score(sample_mean, pop_mean, sample_std, n)
logger.info("\nt-statistic: %.3f", t_stat)

# Critical values
alpha = 0.05
z_crit = critical_value_z(alpha, "two-tailed")
t_crit = critical_value_t(alpha, n - 1, "two-tailed")
logger.info("Critical z (α=0.05, two-tailed): ±%.2f", z_crit)
logger.info("Critical t (α=0.05, df=%s): ±%.2f", n - 1, t_crit)

# P-value and decision
p_val = p_value_method(t_stat, "two-tailed")
logger.info("P-value: %.4f", p_val)
logger.info("Reject H0 at α=0.05: %s", reject_null(p_val, alpha))
