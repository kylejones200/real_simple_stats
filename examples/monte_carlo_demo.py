"""Demo: Monte Carlo methods - integration and probability estimation."""

import logging

import numpy as np

from real_simple_stats.monte_carlo import (
    geometric_brownian_motion,
    monte_carlo_integration,
    monte_carlo_probability,
)

# Monte Carlo integration: estimate ∫x² dx from 0 to 1 (true = 1/3)
result = monte_carlo_integration(
    func=lambda x: x**2,
    lower_bounds=0,
    upper_bounds=1,
    n_samples=10000,
    random_seed=42,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("∫x² dx [0,1]: estimated=%.4f, true=0.3333", result["integral"])
logger.info("95%% CI: %s", result["confidence_interval"])

# Monte Carlo probability: P(x + y < 1) for x,y ~ Uniform(0,1)
result = monte_carlo_probability(
    condition=lambda s: (s[0] + s[1]) < 1,
    lower_bounds=[0, 0],
    upper_bounds=[1, 1],
    n_samples=10000,
    random_seed=42,
)
logger.info("\nP(x+y < 1), x,y~U(0,1): %.4f (true=0.5)", result["probability"])

# Geometric Brownian Motion: stock price simulation
gbm = geometric_brownian_motion(
    S0=100,
    mu=0.05,
    sigma=0.2,
    T=1.0,
    n_steps=252,
    n_simulations=1000,
    random_seed=42,
)
stats = gbm["statistics"]
logger.info("\nGBM (S0=100, μ=5%%, σ=20%%, T=1yr):")
logger.info("  Mean final price: $%.2f", stats["mean"])
logger.info("  Std dev: $%.2f", stats["std"])
logger.info(
    "  95%% CI: $%.2f - $%.2f",
    gbm["percentiles"][5],
    gbm["percentiles"][95],
)
