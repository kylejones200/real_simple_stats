"""Bayesian statistics functions.

This module provides functions for Bayesian statistical analysis including
prior/posterior distributions and credible intervals.
"""

import numpy as np
from scipy import stats


def beta_binomial_update(
    prior_alpha: float, prior_beta: float, successes: int, trials: int
) -> tuple[float, float]:
    """Update Beta prior with binomial data to get posterior.

    Args:
        prior_alpha: Alpha parameter of Beta prior
        prior_beta: Beta parameter of Beta prior
        successes: Number of successes observed
        trials: Number of trials

    Returns:
        Tuple of (posterior_alpha, posterior_beta)

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> # Start with uniform prior Beta(1, 1)
        >>> post_a, post_b = beta_binomial_update(1, 1, 7, 10)
        >>> post_a, post_b
        (8.0, 4.0)
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive")
    if successes < 0 or trials < 0:
        raise ValueError("Successes and trials must be non-negative")
    if successes > trials:
        raise ValueError("Successes cannot exceed trials")

    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + (trials - successes)

    return float(posterior_alpha), float(posterior_beta)


def normal_normal_update(
    prior_mean: float,
    prior_variance: float,
    data: list[float],
    data_variance: float,
) -> tuple[float, float]:
    """Update Normal prior with Normal data to get posterior.

    Assumes known data variance (conjugate prior).

    Args:
        prior_mean: Mean of Normal prior
        prior_variance: Variance of Normal prior
        data: Observed data points
        data_variance: Known variance of data

    Returns:
        Tuple of (posterior_mean, posterior_variance)

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> data = [10.5, 11.2, 9.8, 10.1]
        >>> post_mean, post_var = normal_normal_update(10, 4, data, 1)
        >>> 9 < post_mean < 12
        True
    """
    if prior_variance <= 0 or data_variance <= 0:
        raise ValueError("Variances must be positive")
    if len(data) == 0:
        raise ValueError("Data cannot be empty")

    n = len(data)
    data_mean = np.mean(data)

    # Posterior parameters for conjugate Normal-Normal model
    posterior_variance = 1 / (1 / prior_variance + n / data_variance)
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance + n * data_mean / data_variance
    )

    return float(posterior_mean), float(posterior_variance)


def gamma_poisson_update(
    prior_shape: float, prior_rate: float, data: list[int]
) -> tuple[float, float]:
    """Update Gamma prior with Poisson data to get posterior.

    Args:
        prior_shape: Shape parameter of Gamma prior
        prior_rate: Rate parameter of Gamma prior
        data: Observed count data

    Returns:
        Tuple of (posterior_shape, posterior_rate)

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> data = [3, 5, 4, 6, 5]
        >>> post_shape, post_rate = gamma_poisson_update(1, 1, data)
        >>> post_shape > prior_shape
        True
    """
    if prior_shape <= 0 or prior_rate <= 0:
        raise ValueError("Prior parameters must be positive")
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    if any(x < 0 for x in data):
        raise ValueError("Poisson data must be non-negative")

    n = len(data)
    sum_data = sum(data)

    posterior_shape = prior_shape + sum_data
    posterior_rate = prior_rate + n

    return float(posterior_shape), float(posterior_rate)


def credible_interval(
    distribution: str, params: dict[str, float], credibility: float = 0.95
) -> tuple[float, float]:
    """Calculate credible interval for a posterior distribution.

    Args:
        distribution: Type of distribution ('beta', 'normal', 'gamma')
        params: Dictionary of distribution parameters
        credibility: Credibility level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        ValueError: If distribution type is unknown or parameters are invalid

    Examples:
        >>> # 95% credible interval for Beta(8, 4)
        >>> lower, upper = credible_interval('beta', {'alpha': 8, 'beta': 4})
        >>> 0 < lower < upper < 1
        True
    """
    if not 0 < credibility < 1:
        raise ValueError("Credibility must be between 0 and 1")

    alpha = (1 - credibility) / 2

    if distribution == "beta":
        if "alpha" not in params or "beta" not in params:
            raise ValueError("Beta distribution requires 'alpha' and 'beta' parameters")
        dist = stats.beta(params["alpha"], params["beta"])
    elif distribution == "normal":
        if "mean" not in params or "std" not in params:
            raise ValueError("Normal distribution requires 'mean' and 'std' parameters")
        dist = stats.norm(params["mean"], params["std"])
    elif distribution == "gamma":
        if "shape" not in params or "rate" not in params:
            raise ValueError(
                "Gamma distribution requires 'shape' and 'rate' parameters"
            )
        # scipy uses scale = 1/rate
        dist = stats.gamma(params["shape"], scale=1 / params["rate"])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    lower = dist.ppf(alpha)
    upper = dist.ppf(1 - alpha)

    return float(lower), float(upper)


def highest_density_interval(
    samples: list[float], credibility: float = 0.95
) -> tuple[float, float]:
    """Calculate highest density interval (HDI) from samples.

    The HDI is the shortest interval containing the specified probability mass.

    Args:
        samples: Samples from posterior distribution
        credibility: Credibility level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        ValueError: If samples are insufficient or credibility is invalid

    Examples:
        >>> samples = np.random.normal(0, 1, 1000).tolist()
        >>> lower, upper = highest_density_interval(samples)
        >>> lower < 0 < upper
        True
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples")
    if not 0 < credibility < 1:
        raise ValueError("Credibility must be between 0 and 1")

    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    n_included = int(np.ceil(credibility * n))

    if n_included >= n:
        return float(sorted_samples[0]), float(sorted_samples[-1])

    # Find shortest interval
    interval_widths = sorted_samples[n_included:] - sorted_samples[: n - n_included]
    min_idx = np.argmin(interval_widths)

    lower = sorted_samples[min_idx]
    upper = sorted_samples[min_idx + n_included]

    return float(lower), float(upper)


def bayes_factor(
    likelihood_h1: float, likelihood_h0: float, prior_odds: float = 1.0
) -> float:
    """Calculate Bayes factor for comparing two hypotheses.

    Args:
        likelihood_h1: Likelihood of data under hypothesis 1
        likelihood_h0: Likelihood of data under hypothesis 0
        prior_odds: Prior odds ratio H1/H0 (default: 1.0)

    Returns:
        Bayes factor (BF10)

    Raises:
        ValueError: If likelihoods are invalid

    Examples:
        >>> bf = bayes_factor(0.8, 0.2)
        >>> bf
        4.0
    """
    if likelihood_h0 <= 0:
        raise ValueError("Likelihood under H0 must be positive")
    if likelihood_h1 < 0:
        raise ValueError("Likelihood under H1 must be non-negative")

    bayes_factor_value = (likelihood_h1 / likelihood_h0) * prior_odds

    return float(bayes_factor_value)


def posterior_predictive(
    distribution: str, params: dict[str, float], n_samples: int = 1000
) -> list[float]:
    """Generate samples from posterior predictive distribution.

    Args:
        distribution: Type of distribution ('beta', 'beta_binomial', 'normal', 'gamma', 'gamma_poisson')
        params: Dictionary of posterior parameters
        n_samples: Number of samples to generate

    Returns:
        List of predictive samples

    Raises:
        ValueError: If distribution type is unknown or parameters are invalid

    Examples:
        >>> samples = posterior_predictive('beta_binomial', {'alpha': 8, 'beta': 4, 'n': 10})
        >>> len(samples)
        1000
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    if distribution == "beta":
        if "alpha" not in params or "beta" not in params:
            raise ValueError("Beta requires 'alpha' and 'beta' parameters")
        predictions = stats.beta.rvs(params["alpha"], params["beta"], size=n_samples)
        return predictions.tolist()

    elif distribution == "beta_binomial":
        if "alpha" not in params or "beta" not in params or "n" not in params:
            raise ValueError(
                "Beta-binomial requires 'alpha', 'beta', and 'n' parameters"
            )
        # Sample from Beta posterior, then from Binomial
        p_samples = stats.beta.rvs(params["alpha"], params["beta"], size=n_samples)
        predictions = stats.binom.rvs(params["n"], p_samples)
        return predictions.tolist()

    elif distribution == "normal":
        if "mean" not in params or "std" not in params:
            raise ValueError("Normal requires 'mean' and 'std' parameters")
        predictions = stats.norm.rvs(params["mean"], params["std"], size=n_samples)
        return predictions.tolist()

    elif distribution == "gamma":
        if "shape" not in params or "rate" not in params:
            raise ValueError("Gamma requires 'shape' and 'rate' parameters")
        predictions = stats.gamma.rvs(
            params["shape"], scale=1 / params["rate"], size=n_samples
        )
        return predictions.tolist()

    elif distribution == "gamma_poisson":
        if "shape" not in params or "rate" not in params:
            raise ValueError("Gamma-Poisson requires 'shape' and 'rate' parameters")
        # Sample from Gamma posterior, then from Poisson
        lambda_samples = stats.gamma.rvs(
            params["shape"], scale=1 / params["rate"], size=n_samples
        )
        predictions = stats.poisson.rvs(lambda_samples)
        return predictions.tolist()

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def empirical_bayes_estimate(data: list[float]) -> dict[str, float]:
    """Estimate prior parameters using empirical Bayes method.

    Assumes data comes from Normal distribution with unknown mean and variance.

    Args:
        data: Observed data

    Returns:
        Dictionary with estimated prior parameters

    Raises:
        ValueError: If data is insufficient

    Examples:
        >>> data = [10, 11, 9, 12, 10, 11]
        >>> params = empirical_bayes_estimate(data)
        >>> 'prior_mean' in params
        True
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")

    data_array = np.array(data)
    mean = np.mean(data_array)
    variance = np.var(data_array, ddof=1)

    return {
        "mean": float(mean),
        "variance": float(variance),
    }


def conjugate_prior_summary(family: str) -> dict[str, str]:
    """Get information about conjugate prior families.

    Args:
        family: Distribution family ('binomial', 'normal', 'poisson', 'exponential')

    Returns:
        Dictionary with prior information

    Raises:
        ValueError: If family is unknown

    Examples:
        >>> info = conjugate_prior_summary('binomial')
        >>> info['prior']
        'Beta'
    """
    conjugate_priors = {
        "binomial": {
            "likelihood": "Binomial",
            "prior": "Beta",
            "parameters": "alpha, beta",
            "posterior": "Beta(alpha + successes, beta + failures)",
            "interpretation": "Beta distribution is conjugate for binomial likelihood",
        },
        "normal": {
            "likelihood": "Normal",
            "prior": "Normal (known variance)",
            "parameters": "mean, variance",
            "posterior": "Normal with updated mean and variance",
            "interpretation": "Normal prior is conjugate for normal likelihood with known variance",
        },
        "poisson": {
            "likelihood": "Poisson",
            "prior": "Gamma",
            "parameters": "shape, rate",
            "posterior": "Gamma(shape + sum(data), rate + n)",
            "interpretation": "Gamma distribution is conjugate for Poisson likelihood",
        },
        "exponential": {
            "likelihood": "Exponential",
            "prior": "Gamma",
            "parameters": "shape, rate",
            "posterior": "Gamma(shape + n, rate + sum(data))",
            "interpretation": "Gamma distribution is conjugate for exponential likelihood",
        },
    }

    if family not in conjugate_priors:
        raise ValueError(f"Unknown family: {family}")

    return conjugate_priors[family]


__all__ = [
    "beta_binomial_update",
    "normal_normal_update",
    "gamma_poisson_update",
    "credible_interval",
    "highest_density_interval",
    "bayes_factor",
    "posterior_predictive",
    "empirical_bayes_estimate",
    "conjugate_prior_summary",
]
