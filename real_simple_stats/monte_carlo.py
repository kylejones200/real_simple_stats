"""Monte Carlo simulation: price paths, integration, and probability estimation.

Path generation runs in Rust and in parallel; each simulation draws from its own
derived stream, so a given ``random_seed`` reproduces exactly regardless of core
count.

Two contract changes in 0.5.0, both consequences of dropping NumPy:

* ``paths`` and ``mean_path`` are lists (``paths`` a list of rows, one row per
  time step) rather than a 2-D ndarray.
* ``func`` and ``condition`` are called once per sample -- with a float in one
  dimension, or a tuple of floats in several -- instead of receiving a whole
  array. Scalar-style lambdas such as ``lambda x: x**2`` and
  ``lambda xy: xy[0]**2 + xy[1]**2 <= 1`` are unaffected; genuinely vectorised
  callables need rewriting as scalar ones.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from . import _rss
from ._rss import Rng

__all__ = [
    "geometric_brownian_motion",
    "monte_carlo_from_data",
    "monte_carlo_integration",
    "monte_carlo_probability",
]

_MAX_SEED = 2**63 - 1


def _seed_or_random(random_seed: int | None) -> int:
    if random_seed is not None:
        return int(random_seed) % _MAX_SEED
    return Rng().integers(0, _MAX_SEED, 1)[0]


def _as_bounds(value: float | Sequence[float]) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_simulations: int = 1000,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Simulate price paths under geometric Brownian motion.

    Args:
        S0: Starting value (must be positive).
        mu: Expected return per unit time.
        sigma: Volatility per unit time (non-negative).
        T: Total time horizon.
        n_steps: Number of time steps.
        n_simulations: Number of independent paths.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary with ``paths`` (a list of ``n_steps + 1`` rows, each holding
        every simulation's value at that step), ``times``, ``final_values``,
        ``mean_path``, ``percentiles`` and ``statistics``.

    Raises:
        ValueError: If any parameter is out of range.

    Example:
        >>> r = geometric_brownian_motion(100, 0.1, 0.2, 1.0, 12, 500, random_seed=1)
        >>> len(r["paths"]), len(r["paths"][0])
        (13, 500)
        >>> all(v > 0 for v in r["final_values"])
        True
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T <= 0:
        raise ValueError("T must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")

    seed = _seed_or_random(random_seed)
    flat = _rss.gbm_paths(
        float(S0), float(mu), float(sigma), float(T), n_steps, n_simulations, seed
    )

    rows = n_steps + 1
    paths = [flat[t * n_simulations : (t + 1) * n_simulations] for t in range(rows)]
    times = [T * t / n_steps for t in range(rows)]
    final_values = paths[-1]
    mean_path = [_rss.mean(row) for row in paths]

    q = _rss.quantiles(final_values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "paths": paths,
        "times": times,
        "final_values": final_values,
        "mean_path": mean_path,
        "percentiles": {5: q[0], 25: q[1], 50: q[2], 75: q[3], 95: q[4]},
        "statistics": {
            "mean": _rss.mean(final_values),
            "median": _rss.median(final_values),
            "std": _rss.std_dev(final_values, 0),
            "min": _rss.min_(final_values),
            "max": _rss.max_(final_values),
        },
    }


def monte_carlo_from_data(
    data: Sequence[float],
    n_steps: int,
    n_simulations: int = 1000,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Simulate forward using drift and volatility estimated from history.

    Args:
        data: Historical series, e.g. prices (at least 2 values, all positive).
        n_steps: Steps to project forward. Daily data is assumed, so the
            horizon is ``n_steps / 252`` years.
        n_simulations: Number of paths.
        random_seed: Seed for reproducibility.

    Returns:
        The :func:`geometric_brownian_motion` result plus a ``parameters``
        entry holding the estimated ``mu``, ``sigma``, ``drift`` and ``S0``.

    Raises:
        ValueError: If fewer than 2 values are given, or any value is not
            positive (log returns would be undefined).

    Example:
        >>> r = monte_carlo_from_data([100, 102, 101, 105, 103], 10, 100, random_seed=2)
        >>> "mu" in r["parameters"]
        True
    """
    values = [float(v) for v in data]
    if len(values) < 2:
        raise ValueError("Data must contain at least 2 values")
    if any(v <= 0 for v in values):
        raise ValueError("All data values must be positive to compute log returns")

    log_returns = [math.log(b / a) for a, b in zip(values, values[1:])]
    mu = _rss.mean(log_returns)
    sigma = _rss.std_dev(log_returns, 1) if len(log_returns) > 1 else 0.0
    drift = mu - 0.5 * sigma**2
    s0 = values[-1]

    result = geometric_brownian_motion(
        S0=s0,
        mu=drift,
        sigma=sigma,
        T=n_steps / 252,
        n_steps=n_steps,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    result["parameters"] = {"mu": mu, "sigma": sigma, "drift": drift, "S0": s0}
    return result


def _sample_box(
    lower: list[float], upper: list[float], n_samples: int, seed: int
) -> list:
    """Draw `n_samples` points from the box, as floats (1-D) or tuples (n-D)."""
    d = len(lower)
    flat = _rss.uniform_box(lower, upper, n_samples, seed)
    if d == 1:
        return flat
    return [tuple(flat[i * d : (i + 1) * d]) for i in range(n_samples)]


def monte_carlo_integration(
    func: Callable[..., float],
    lower_bounds: float | Sequence[float],
    upper_bounds: float | Sequence[float],
    n_samples: int = 10000,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Estimate a definite integral by averaging the integrand over the domain.

    Args:
        func: The integrand, called once per sample. It receives a float in one
            dimension, or a tuple of floats in several.
        lower_bounds: Lower limit per dimension.
        upper_bounds: Upper limit per dimension.
        n_samples: Number of samples.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary with ``integral``, ``std_error`` and a 95%
        ``confidence_interval``.

    Raises:
        ValueError: If the bounds disagree in length or ``n_samples`` < 1.

    Example:
        >>> r = monte_carlo_integration(lambda x: x**2, 0, 1, 20000, random_seed=7)
        >>> abs(r["integral"] - 1/3) < 0.01
        True
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    lower = _as_bounds(lower_bounds)
    upper = _as_bounds(upper_bounds)
    if len(lower) != len(upper):
        raise ValueError("lower_bounds and upper_bounds must have same length")

    samples = _sample_box(lower, upper, n_samples, _seed_or_random(random_seed))
    values = [float(func(s)) for s in samples]

    volume = 1.0
    for lo, hi in zip(lower, upper):
        volume *= hi - lo

    integral = volume * _rss.mean(values)
    std_error = (
        volume * _rss.std_dev(values, 1) / math.sqrt(n_samples) if n_samples > 1 else 0.0
    )
    return {
        "integral": integral,
        "std_error": std_error,
        "confidence_interval": (
            integral - 1.96 * std_error,
            integral + 1.96 * std_error,
        ),
    }


def monte_carlo_probability(
    condition: Callable[..., bool],
    lower_bounds: float | Sequence[float],
    upper_bounds: float | Sequence[float],
    n_samples: int = 10000,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Estimate the probability that a condition holds over a uniform domain.

    Args:
        condition: Predicate called once per sample, receiving a float in one
            dimension or a tuple of floats in several.
        lower_bounds: Lower limit per dimension.
        upper_bounds: Upper limit per dimension.
        n_samples: Number of samples.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary with ``probability``, ``std_error``, ``confidence_interval``,
        ``n_successes`` and ``n_samples``.

    Raises:
        ValueError: If the bounds disagree in length or ``n_samples`` < 1.

    Example:
        >>> r = monte_carlo_probability(
        ...     lambda xy: xy[0]**2 + xy[1]**2 <= 1, [0, 0], [1, 1], 20000, random_seed=4
        ... )
        >>> abs(r["probability"] * 4 - 3.14159) < 0.1
        True
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    lower = _as_bounds(lower_bounds)
    upper = _as_bounds(upper_bounds)
    if len(lower) != len(upper):
        raise ValueError("lower_bounds and upper_bounds must have same length")

    samples = _sample_box(lower, upper, n_samples, _seed_or_random(random_seed))
    n_successes = sum(1 for s in samples if condition(s))

    probability = n_successes / n_samples
    std_error = math.sqrt(probability * (1 - probability) / n_samples)
    return {
        "probability": probability,
        "std_error": std_error,
        "confidence_interval": (
            max(0.0, probability - 1.96 * std_error),
            min(1.0, probability + 1.96 * std_error),
        ),
        "n_successes": n_successes,
        "n_samples": int(n_samples),
    }
