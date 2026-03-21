"""Demo: Poisson, geometric, and exponential distributions."""

import logging

from real_simple_stats.probability_distributions import (
    exponential_cdf,
    exponential_pdf,
    expected_value_poisson,
    geometric_cdf,
    geometric_pmf,
    poisson_cdf,
    poisson_pmf,
    variance_poisson,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Poisson: rare events (e.g., calls per hour)
lam = 3  # average rate
k = 2
logger.info("Poisson(λ=%s) P(X=%s): %.4f", lam, k, poisson_pmf(k, lam))
logger.info("Poisson(λ=%s) P(X≤%s): %.4f", lam, k, poisson_cdf(k, lam))
logger.info(
    "E[X] = %s, Var(X) = %s",
    expected_value_poisson(lam),
    variance_poisson(lam),
)

# Geometric: trials until first success
p = 0.2
k = 3
logger.info(
    "\nGeometric(p=%s) P(first success on trial %s): %.4f", p, k, geometric_pmf(k, p)
)
logger.info("Geometric(p=%s) P(success by trial %s): %.4f", p, k, geometric_cdf(k, p))

# Exponential: time until next event
lam = 2  # rate per unit time
x = 0.5
logger.info("\nExponential(λ=%s) PDF at x=%s: %.4f", lam, x, exponential_pdf(x, lam))
logger.info("Exponential(λ=%s) P(X≤%s): %.4f", lam, x, exponential_cdf(x, lam))
