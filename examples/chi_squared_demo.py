import logging

from real_simple_stats.chi_square_utils import (
    chi_square_statistic,
    critical_chi_square_value,
    reject_null_chi_square,
)

from real_simple_stats.plots import plot_observed_vs_expected

observed = [20, 30, 25]
expected = [25, 25, 25]

chi_stat = chi_square_statistic(observed, expected)
alpha = 0.05
df = len(observed) - 1
critical = critical_chi_square_value(alpha, df)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Chi-square: %.2f", chi_stat)
logger.info("Critical value: %.2f", critical)
logger.info("Reject null? %s", reject_null_chi_square(chi_stat, critical))

plot_observed_vs_expected(observed, expected)
