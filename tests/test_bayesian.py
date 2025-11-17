"""Tests for Bayesian statistics module."""

import numpy as np
import pytest

from real_simple_stats.bayesian_stats import (
    bayes_factor,
    beta_binomial_update,
    conjugate_prior_summary,
    credible_interval,
    empirical_bayes_estimate,
    gamma_poisson_update,
    highest_density_interval,
    normal_normal_update,
    posterior_predictive,
)


class TestBetaBinomialUpdate:
    def test_beta_binomial_update_uniform_prior(self):
        # Uniform prior: Beta(1, 1)
        post_alpha, post_beta = beta_binomial_update(1, 1, 7, 10)
        assert post_alpha == 8  # 1 + 7
        assert post_beta == 4  # 1 + (10-7)

    def test_beta_binomial_update_informative_prior(self):
        post_alpha, post_beta = beta_binomial_update(2, 2, 5, 10)
        assert post_alpha == 7  # 2 + 5
        assert post_beta == 7  # 2 + (10-5)

    def test_beta_binomial_update_no_successes(self):
        post_alpha, post_beta = beta_binomial_update(1, 1, 0, 10)
        assert post_alpha == 1
        assert post_beta == 11


class TestNormalNormalUpdate:
    def test_normal_normal_update_basic(self):
        post_mean, post_var = normal_normal_update(0, 1, [1, 2, 3], 1)
        assert isinstance(post_mean, float)
        assert isinstance(post_var, float)
        assert post_var > 0

    def test_normal_normal_update_single_observation(self):
        post_mean, post_var = normal_normal_update(0, 1, [2], 1)
        # Posterior mean should be between prior mean and data
        assert 0 < post_mean < 2


class TestGammaPoissonUpdate:
    def test_gamma_poisson_update_basic(self):
        post_shape, post_rate = gamma_poisson_update(1, 1, [2, 3, 4])
        assert isinstance(post_shape, float)
        assert isinstance(post_rate, float)
        assert post_shape > 0
        assert post_rate > 0

    def test_gamma_poisson_update_single_observation(self):
        post_shape, post_rate = gamma_poisson_update(2, 1, [5])
        assert post_shape > 2
        assert post_rate > 1


class TestCredibleInterval:
    def test_credible_interval_beta(self):
        lower, upper = credible_interval("beta", {"alpha": 8, "beta": 4}, 0.95)
        assert 0 <= lower < upper <= 1

    def test_credible_interval_normal(self):
        lower, upper = credible_interval("normal", {"mean": 0, "std": 1}, 0.95)
        assert lower < 0 < upper

    def test_credible_interval_gamma(self):
        lower, upper = credible_interval("gamma", {"shape": 2, "rate": 1}, 0.95)
        assert 0 <= lower < upper


class TestHighestDensityInterval:
    def test_highest_density_interval_normal(self):
        samples = list(np.random.normal(0, 1, 1000))
        lower, upper = highest_density_interval(samples, 0.95)
        assert lower < upper

    def test_highest_density_interval_contains_most_samples(self):
        samples = list(np.random.normal(0, 1, 1000))
        lower, upper = highest_density_interval(samples, 0.95)
        # Most samples should be in the interval
        in_interval = sum(1 for s in samples if lower <= s <= upper)
        assert in_interval / len(samples) >= 0.90  # At least 90%


class TestBayesFactor:
    def test_bayes_factor_equal_likelihoods(self):
        bf = bayes_factor(0.5, 0.5, 1.0)
        assert bf == pytest.approx(1.0)

    def test_bayes_factor_h1_more_likely(self):
        bf = bayes_factor(0.8, 0.2, 1.0)
        assert bf > 1.0

    def test_bayes_factor_h0_more_likely(self):
        bf = bayes_factor(0.2, 0.8, 1.0)
        assert bf < 1.0


class TestPosteriorPredictive:
    def test_posterior_predictive_beta(self):
        samples = posterior_predictive("beta", {"alpha": 8, "beta": 4}, 100)
        assert len(samples) == 100
        assert all(0 <= s <= 1 for s in samples)

    def test_posterior_predictive_normal(self):
        samples = posterior_predictive("normal", {"mean": 0, "std": 1}, 100)
        assert len(samples) == 100

    def test_posterior_predictive_gamma(self):
        samples = posterior_predictive("gamma", {"shape": 2, "rate": 1}, 100)
        assert len(samples) == 100
        assert all(s >= 0 for s in samples)


class TestEmpiricalBayesEstimate:
    def test_empirical_bayes_estimate_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = empirical_bayes_estimate(data)
        assert "mean" in result
        assert "variance" in result

    def test_empirical_bayes_estimate_values(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = empirical_bayes_estimate(data)
        assert result["mean"] == pytest.approx(3.0)
        assert result["variance"] > 0


class TestConjugatePriorSummary:
    def test_conjugate_prior_summary_binomial(self):
        result = conjugate_prior_summary("binomial")
        assert "likelihood" in result
        assert "prior" in result
        assert "posterior" in result

    def test_conjugate_prior_summary_normal(self):
        result = conjugate_prior_summary("normal")
        assert isinstance(result, dict)

    def test_conjugate_prior_summary_poisson(self):
        result = conjugate_prior_summary("poisson")
        assert isinstance(result, dict)
