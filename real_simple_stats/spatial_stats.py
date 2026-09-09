"""Spatial statistics — tools for analyzing geographically distributed data.

Spatial data breaks the standard independence assumption: nearby locations tend
to be more similar than distant ones.  These tools help you measure and model
that spatial structure.

- :func:`morans_i` — global spatial autocorrelation index.  Is the pattern
  more clustered, more dispersed, or random?

- :func:`compute_variogram` — experimental (empirical) variogram. Shows how
  dissimilarity between pairs of locations grows with distance.

- :func:`fit_variogram` — fits a parametric model (spherical, exponential, or
  Gaussian) to the experimental variogram using least-squares.

- :func:`variogram_spherical` / :func:`variogram_exponential` /
  :func:`variogram_gaussian` — the three standard variogram model functions,
  exposed directly for plotting or custom fitting.

All computation runs in the native Rust backend; there are no runtime
dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from . import _rss

__all__ = [
    "morans_i",
    "compute_variogram",
    "fit_variogram",
    "variogram_spherical",
    "variogram_exponential",
    "variogram_gaussian",
]


# ---------------------------------------------------------------------------
# Variogram model functions
# ---------------------------------------------------------------------------


def _elementwise(fn, h):
    """Apply `fn` to a scalar or to every element of a sequence."""
    if isinstance(h, (int, float)):
        return fn(float(h))
    return [fn(float(v)) for v in h]


def variogram_spherical(
    h: float | Sequence[float],
    nugget: float,
    sill: float,
    range_param: float,
) -> float | list[float]:
    """Spherical variogram model.

    Rises linearly near the origin, levels off at the *sill* beyond the
    *range*.  The most commonly used model in geostatistics.

    Args:
        h: Lag distances (non-negative).
        nugget: Discontinuity at h=0 (measurement error + micro-scale variation).
        sill: Variance at large distances (where spatial correlation vanishes).
        range_param: Distance beyond which spatial correlation is negligible.

    Returns:
        Semivariance: a float for a scalar lag, otherwise a list.
    """

    def one(hi: float) -> float:
        if hi >= range_param:
            return float(sill)
        hr = hi / range_param
        return nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr**3)

    return _elementwise(one, h)


def variogram_exponential(
    h: float | Sequence[float],
    nugget: float,
    sill: float,
    range_param: float,
) -> float | list[float]:
    """Exponential variogram model.

    Approaches the sill asymptotically — never fully flattens.  Good for
    data with strong near-origin structure.

    Args:
        h: Lag distances (non-negative).
        nugget: Nugget effect.
        sill: Asymptotic semivariance.
        range_param: Practical range parameter (effective range ≈ 3× range_param).

    Returns:
        Semivariance: a float for a scalar lag, otherwise a list.
    """
    return _elementwise(
        lambda hi: nugget + (sill - nugget) * (1.0 - math.exp(-hi / range_param)), h
    )


def variogram_gaussian(
    h: float | Sequence[float],
    nugget: float,
    sill: float,
    range_param: float,
) -> float | list[float]:
    """Gaussian variogram model.

    Very smooth near the origin — suitable for highly continuous phenomena.

    Args:
        h: Lag distances (non-negative).
        nugget: Nugget effect.
        sill: Asymptotic semivariance.
        range_param: Scale parameter controlling how fast the sill is reached.

    Returns:
        Semivariance: a float for a scalar lag, otherwise a list.
    """
    return _elementwise(
        lambda hi: nugget + (sill - nugget) * (1.0 - math.exp(-((hi / range_param) ** 2))), h
    )


_VARIOGRAM_MODELS = {
    "spherical": variogram_spherical,
    "exponential": variogram_exponential,
    "gaussian": variogram_gaussian,
}


# ---------------------------------------------------------------------------
# Core spatial statistics
# ---------------------------------------------------------------------------


def morans_i(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    distance_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute Moran's I — global spatial autocorrelation index.

    Moran's I measures whether similar values cluster together in space:

    - **I ≈ +1**: strong positive autocorrelation (clusters of similar values)
    - **I ≈  0**: spatial randomness
    - **I ≈ −1**: strong negative autocorrelation (checkerboard pattern)

    Under the null hypothesis of spatial randomness, E[I] = −1/(n−1) ≈ 0 for
    large n.  The z-score allows a quick significance test.

    Args:
        x: x-coordinates of each observation.
        y: y-coordinates of each observation.
        values: Attribute values at each location.
        distance_threshold: If given, only pairs within this distance are
            considered neighbours.  ``None`` uses all pairs (global weights).

    Returns:
        dict with keys:
            moran_i: The Moran's I statistic.
            expected_i: E[I] under spatial randomness = −1/(n−1).
            variance_i: Approximate variance under normality assumption.
            z_score: (I − E[I]) / sqrt(Var[I]).
            p_value: Two-sided p-value for the z-score.
            interpretation: Short plain-English description.
            n: Number of observations.

    Example:
        >>> import numpy as np
        >>> from real_simple_stats import Rng
        >>> rng = Rng(0)
        >>> x = rng.uniform(0, 100, 50)
        >>> y = rng.uniform(0, 100, 50)
        >>> v = 5 + 0.1 * x + rng.normal(0, 2, 50)  # correlated with location
        >>> r = morans_i(x, y, v, distance_threshold=30)
        >>> r["moran_i"] > 0  # expect positive autocorrelation
        True
    """
    x_ = [float(t) for t in x]
    y_ = [float(t) for t in y]
    v = [float(t) for t in values]
    n = len(v)

    # The O(n^2) pair loop and the Moran (1950) variance run in Rust.
    moran_I, E_I, var_I, z_score, p_value = _rss.morans_i(
        x_, y_, v, distance_threshold
    )

    if moran_I > 0.1:
        interp = "Positive spatial autocorrelation — similar values cluster together."
    elif moran_I < -0.1:
        interp = "Negative spatial autocorrelation — dissimilar values are neighbours."
    else:
        interp = "No strong spatial autocorrelation detected."

    return {
        "moran_i": float(moran_I),
        "expected_i": float(E_I),
        "variance_i": float(var_I),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "interpretation": interp,
        "n": n,
    }


def compute_variogram(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    n_lags: int = 15,
    max_lag: float | None = None,
) -> dict[str, Any]:
    """Compute the experimental (empirical) variogram.

    The variogram γ(h) is half the average squared difference between all
    pairs of locations separated by distance h.  It reveals spatial structure:
    a rising γ(h) that levels off means nearby points are more alike than
    distant ones.

    Args:
        x: x-coordinates.
        y: y-coordinates.
        values: Attribute values at each location.
        n_lags: Number of distance bins (default 15).
        max_lag: Maximum lag distance.  Defaults to half the maximum
            pairwise distance (the standard rule of thumb).

    Returns:
        dict with keys:
            lags: Bin centre distances.
            gamma: Semivariance at each lag.
            n_pairs: Number of data pairs contributing to each bin.
            max_lag: The max_lag used.
            total_variance: Overall variance of the data (the variogram sill
                for uncorrelated data).

    Example:
        >>> import numpy as np
        >>> from real_simple_stats import Rng
        >>> rng = Rng(1)
        >>> x, y = rng.uniform(0, 100, 80), rng.uniform(0, 100, 80)
        >>> v = [math.sin(xi / 20) + e for xi, e in zip(x, rng.normal(0, 0.3, 80))]
        >>> r = compute_variogram(x, y, v, n_lags=10)
        >>> len(r["lags"]) == 10
        True
    """
    x_ = [float(t) for t in x]
    y_ = [float(t) for t in y]
    v = [float(t) for t in values]

    lag_centers, gamma, n_pairs, max_lag_used, total_var = _rss.variogram(
        x_, y_, v, n_lags, max_lag
    )
    return {
        "lags": lag_centers,
        "gamma": gamma,
        "n_pairs": n_pairs,
        "max_lag": max_lag_used,
        "total_variance": total_var,
    }


def fit_variogram(
    lags: Sequence[float],
    gamma: Sequence[float],
    model: str = "spherical",
    n_pairs: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit a parametric variogram model to an experimental variogram.

    Uses weighted least squares (weights = number of pairs per bin) to fit
    one of three standard variogram models.

    Args:
        lags: Lag distances (bin centres from :func:`compute_variogram`).
        gamma: Experimental semivariance at each lag.
        model: One of ``"spherical"``, ``"exponential"``, or ``"gaussian"``
            (default ``"spherical"``).
        n_pairs: Optional array of pair counts per bin — used as weights.
            If ``None``, uses equal weights.

    Returns:
        dict with keys:
            model: Model name.
            nugget: Fitted nugget (discontinuity at zero lag).
            sill: Fitted sill (asymptotic variance).
            range_param: Fitted range parameter.
            rmse: Root-mean-square residual of the fit.
            model_fn: Callable ``gamma(h)`` using the fitted parameters.

    Raises:
        ValueError: If model name is unknown or fitting fails.

    Example:
        >>> import numpy as np
        >>> lags = [1 + i * 49 / 14 for i in range(15)]
        >>> gamma = variogram_spherical(lags, nugget=1, sill=10, range_param=30)
        >>> r = fit_variogram(lags, gamma, model="spherical")
        >>> abs(r["sill"] - 10) < 1
        True
    """
    model = model.lower()
    if model not in _VARIOGRAM_MODELS:
        raise ValueError(
            f"Unknown model {model!r}. Choose from: {', '.join(_VARIOGRAM_MODELS)}."
        )

    h = [float(v) for v in lags]
    g = [float(v) for v in gamma]

    # Keep only bins that actually contain pairs.
    keep = [i for i, gi in enumerate(g) if gi > 0]
    if len(keep) < 3:
        raise ValueError("Need at least 3 non-zero bins to fit a variogram model.")

    h_fit = [h[i] for i in keep]
    g_fit = [g[i] for i in keep]

    # Weight each bin by its pair count, as the SciPy version did via `sigma`.
    if n_pairs is not None:
        counts = [max(float(list(n_pairs)[i]), 1.0) for i in keep]
        weights = [c for c in counts]
    else:
        weights = [1.0] * len(h_fit)

    model_fn = _VARIOGRAM_MODELS[model]
    sill_guess = max(g_fit)
    range_guess = max(h_fit) / 3.0

    def residual(params: list[float]) -> list[float]:
        nugget_, sill_, range_ = params
        if range_ <= 0:
            return [1e6] * len(h_fit)
        pred = model_fn(h_fit, nugget_, sill_, range_)
        return [(p - o) * w for p, o, w in zip(pred, g_fit, weights)]

    params = _rss.curve_fit_lm(
        residual,
        [0.0, sill_guess, range_guess],
        [0.0, 0.0, 1e-6],
        [sill_guess, 2 * sill_guess, max(h_fit) * 2],
        len(h_fit),
        500,
    )

    nugget, sill, range_param = params
    fitted = model_fn(h_fit, nugget, sill, range_param)
    rmse = math.sqrt(
        sum((f - o) ** 2 for f, o in zip(fitted, g_fit)) / len(g_fit)
    )

    def fitted_fn(h_new: float) -> float:
        return float(model_fn(float(h_new), nugget, sill, range_param))

    return {
        "model": model,
        "nugget": float(nugget),
        "sill": float(sill),
        "range_param": float(range_param),
        "rmse": rmse,
        "model_fn": fitted_fn,
    }
