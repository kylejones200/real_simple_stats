"""Demo: Normal distributions - z-scores, areas, PDF, CDF."""

import logging

from real_simple_stats.normal_distributions import (
    area_between_z_scores,
    area_left_of_z,
    chebyshev_theorem,
    normal_cdf,
    normal_pdf,
    z_score,
    z_score_standard_error,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Z-score: how many standard deviations from the mean
x, mean, std_dev = 85, 75, 10
z = z_score(x, mean, std_dev)
logger.info("Z-score of %s (mean=%s, sd=%s): %.2f", x, mean, std_dev, z)

# Standard error z-score (for sample means)
sample_mean, pop_mean, pop_std, n = 82, 80, 10, 36
z_se = z_score_standard_error(sample_mean, pop_mean, pop_std, n)
logger.info("Z-score of sample mean: %.2f", z_se)

# Area under the curve
logger.info("P(Z < 1.96): %.4f", area_left_of_z(1.96))
logger.info("P(-1 < Z < 1): %.4f", area_between_z_scores(-1, 1))

# Chebyshev's theorem: at least 75% within 2 standard deviations
k = 2
min_proportion = chebyshev_theorem(k)
logger.info("Within %s std devs: at least %.0f%% of data", k, min_proportion * 100)

# PDF and CDF (standard normal)
logger.info("PDF at 0: %.4f", normal_pdf(0))
logger.info("CDF at 1.96: %.4f", normal_cdf(1.96))

# Non-standard normal: IQ scores (mean=100, sd=15)
iq = 115
prob_below = normal_cdf(iq, mean=100, std_dev=15)
logger.info("P(IQ < 115) where mean=100, sd=15: %.4f", prob_below)
