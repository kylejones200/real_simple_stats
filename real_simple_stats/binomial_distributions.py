import math
from collections.abc import Sequence

from . import _rss

# --- BINOMIAL CORE FUNCTIONS ---


def is_binomial_experiment(
    trials: int, outcomes: Sequence[str], probability: float
) -> bool:
    """
    Checks if an experiment meets the binomial criteria:
    - fixed number of trials
    - each trial is independent
    - each trial has two possible outcomes
    - probability of success is constant
    """
    return (
        isinstance(trials, int)
        and trials > 0
        and len(outcomes) == 2
        and 0 <= probability <= 1
    )


def binomial_probability(n: int, k: int, p: float) -> float:
    """Computes probability of k successes in n binomial trials.

    Args:
        n: Number of trials (must be non-negative)
        k: Number of successes (must be between 0 and n)
        p: Probability of success on each trial (must be between 0 and 1)

    Returns:
        Probability of exactly k successes

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> binomial_probability(10, 3, 0.5)
        0.1171875
    """
    if n < 0:
        raise ValueError("Number of trials (n) must be non-negative")
    if k < 0 or k > n:
        raise ValueError(f"Number of successes (k) must be between 0 and {n}")
    if not 0 <= p <= 1:
        raise ValueError("Probability (p) must be between 0 and 1")

    comb = math.comb(n, k)
    return comb * (p**k) * ((1 - p) ** (n - k))


def binomial_mean(n: int, p: float) -> float:
    return n * p


def binomial_variance(n: int, p: float) -> float:
    return n * p * (1 - p)


def binomial_std_dev(n: int, p: float) -> float:
    return math.sqrt(binomial_variance(n, p))


def expected_value_single(value: float, probability: float) -> float:
    """Expected value of a single outcome."""
    return value * probability


def expected_value_multiple(
    values: Sequence[float], probabilities: Sequence[float]
) -> float:
    return sum(v * p for v, p in zip(values, probabilities))


# --- NORMAL APPROXIMATION AND CONTINUITY CORRECTION ---


def normal_approximation(
    n: int, p: float, k: int, use_continuity: bool = True
) -> float:
    """Uses normal approximation with continuity correction to estimate binomial P(X ≤ k)."""
    mu = binomial_mean(n, p)
    sigma = binomial_std_dev(n, p)
    z = (k + 0.5 - mu) / sigma if use_continuity else (k - mu) / sigma

    return _rss.norm_cdf(z)


def binomial_cdf(n: int, k: int, p: float) -> float:
    """P(X <= k): the chance of at most k successes in n independent trials.

    Args:
        n: Number of trials (non-negative)
        k: Success count to accumulate through
        p: Probability of success on a single trial, in [0, 1]

    Returns:
        Cumulative probability between 0 and 1

    Raises:
        ValueError: If n is negative or p is outside [0, 1]

    Example:
        >>> round(binomial_cdf(n=10, k=3, p=0.5), 6)
        0.171875
        >>> binomial_cdf(n=5, k=5, p=0.3)
        1.0
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    return _rss.binom_cdf(k, n, p)
