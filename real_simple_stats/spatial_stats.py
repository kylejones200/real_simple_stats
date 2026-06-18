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

All functions use only numpy and scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix as _dist_matrix

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


def variogram_spherical(
    h: np.ndarray,
    nugget: float,
    sill: float,
    range_param: float,
) -> np.ndarray:
    """Spherical variogram model.

    Rises linearly near the origin, levels off at the *sill* beyond the
    *range*.  The most commonly used model in geostatistics.

    Args:
        h: Lag distances (non-negative).
        nugget: Discontinuity at h=0 (measurement error + micro-scale variation).
        sill: Variance at large distances (where spatial correlation vanishes).
        range_param: Distance beyond which spatial correlation is negligible.

    Returns:
        Array of semivariance values.
    """
    h = np.asarray(h, dtype=float)
    gamma = np.full_like(h, float(sill))
    mask = h < range_param
    hr = h[mask] / range_param
    gamma[mask] = nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr**3)
    return gamma


def variogram_exponential(
    h: np.ndarray,
    nugget: float,
    sill: float,
    range_param: float,
) -> np.ndarray:
    """Exponential variogram model.

    Approaches the sill asymptotically — never fully flattens.  Good for
    data with strong near-origin structure.

    Args:
        h: Lag distances (non-negative).
        nugget: Nugget effect.
        sill: Asymptotic semivariance.
        range_param: Practical range parameter (effective range ≈ 3× range_param).

    Returns:
        Array of semivariance values.
    """
    h = np.asarray(h, dtype=float)
    return nugget + (sill - nugget) * (1.0 - np.exp(-h / range_param))


def variogram_gaussian(
    h: np.ndarray,
    nugget: float,
    sill: float,
    range_param: float,
) -> np.ndarray:
    """Gaussian variogram model.

    Very smooth near the origin — suitable for highly continuous phenomena.

    Args:
        h: Lag distances (non-negative).
        nugget: Nugget effect.
        sill: Asymptotic semivariance.
        range_param: Scale parameter controlling how fast the sill is reached.

    Returns:
        Array of semivariance values.
    """
    h = np.asarray(h, dtype=float)
    return nugget + (sill - nugget) * (1.0 - np.exp(-((h / range_param) ** 2)))


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
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(0, 100, 50)
        >>> y = rng.uniform(0, 100, 50)
        >>> v = 5 + 0.1 * x + rng.normal(0, 2, 50)  # correlated with location
        >>> r = morans_i(x, y, v, distance_threshold=30)
        >>> r["moran_i"] > 0  # expect positive autocorrelation
        True
    """
    x_ = np.asarray(x, dtype=float)
    y_ = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(v)
    if not (len(x_) == len(y_) == n):
        raise ValueError("x, y, and values must have the same length.")
    if n < 3:
        raise ValueError("Need at least 3 observations.")

    coords = np.column_stack([x_, y_])
    D = _dist_matrix(coords, coords)

    if distance_threshold is not None:
        W = ((D > 0) & (D <= distance_threshold)).astype(float)
    else:
        W = (D > 0).astype(float)

    W_sum = W.sum()
    if W_sum == 0:
        raise ValueError(
            "Spatial weights matrix is all zeros — no neighbours found. "
            "Try increasing distance_threshold."
        )

    z = v - v.mean()
    numerator = float(np.sum(W * np.outer(z, z)))
    denominator = float(np.sum(z**2))

    if denominator == 0:
        raise ValueError("All values are identical; Moran's I is undefined.")

    I = (n / W_sum) * (numerator / denominator)
    E_I = -1.0 / (n - 1)

    # Variance under normality assumption (Moran 1950)
    S1 = 0.5 * float(np.sum((W + W.T) ** 2))
    S2 = float(np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2))
    n2 = n * n
    m2 = float(np.sum(z**2)) / n
    m4 = float(np.sum(z**4)) / n
    b2 = m4 / (m2**2) if m2 > 0 else 0.0

    A = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * W_sum**2)
    B = b2 * ((n**2 - n) * S1 - 2 * n * S2 + 6 * W_sum**2)
    C = (n - 1) * (n - 2) * (n - 3) * W_sum**2
    var_I = max((A - B) / C - E_I**2, 1e-12)

    z_score = (I - E_I) / math.sqrt(var_I)
    from scipy.stats import norm as _norm
    p_value = float(2 * _norm.sf(abs(z_score)))

    if I > 0.1:
        interp = "Positive spatial autocorrelation — similar values cluster together."
    elif I < -0.1:
        interp = "Negative spatial autocorrelation — dissimilar values are neighbours."
    else:
        interp = "No strong spatial autocorrelation detected."

    return {
        "moran_i": float(I),
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
        >>> rng = np.random.default_rng(1)
        >>> x, y = rng.uniform(0, 100, 80), rng.uniform(0, 100, 80)
        >>> v = np.sin(x / 20) + rng.normal(0, 0.3, 80)
        >>> r = compute_variogram(x, y, v, n_lags=10)
        >>> len(r["lags"]) == 10
        True
    """
    x_ = np.asarray(x, dtype=float)
    y_ = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(v)
    if not (len(x_) == len(y_) == n):
        raise ValueError("x, y, and values must have the same length.")
    if n < 4:
        raise ValueError("Need at least 4 observations.")
    if n_lags < 2:
        raise ValueError("n_lags must be at least 2.")

    coords = np.column_stack([x_, y_])
    D = _dist_matrix(coords, coords)

    # Upper triangle only (unique pairs)
    idx = np.triu_indices(n, k=1)
    distances = D[idx]
    sq_diffs = (v[idx[0]] - v[idx[1]]) ** 2

    if max_lag is None:
        max_lag = float(distances.max()) / 2.0

    bins = np.linspace(0.0, max_lag, n_lags + 1)
    lag_centers = (bins[:-1] + bins[1:]) / 2.0

    gamma = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)

    for i in range(n_lags):
        mask = (distances >= bins[i]) & (distances < bins[i + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            gamma[i] = 0.5 * float(sq_diffs[mask].mean())
            n_pairs[i] = cnt

    return {
        "lags": lag_centers,
        "gamma": gamma,
        "n_pairs": n_pairs,
        "max_lag": float(max_lag),
        "total_variance": float(np.var(v, ddof=1)),
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
        >>> lags = np.linspace(1, 50, 15)
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

    h = np.asarray(lags, dtype=float)
    g = np.asarray(gamma, dtype=float)

    # Use only bins that have pairs (non-zero bins)
    valid = g > 0
    if valid.sum() < 3:
        raise ValueError("Need at least 3 non-zero bins to fit a variogram model.")

    h_fit = h[valid]
    g_fit = g[valid]
    sigma = None
    if n_pairs is not None:
        counts = np.asarray(n_pairs, dtype=float)[valid]
        sigma = 1.0 / np.maximum(counts, 1.0)

    model_fn = _VARIOGRAM_MODELS[model]
    sill_guess = float(g_fit.max())
    range_guess = float(h_fit.max()) / 3.0

    try:
        params, _ = curve_fit(
            model_fn,
            h_fit,
            g_fit,
            p0=[0.0, sill_guess, range_guess],
            bounds=([0, 0, 1e-6], [sill_guess, 2 * sill_guess, h_fit.max() * 2]),
            sigma=sigma,
            maxfev=5000,
        )
    except RuntimeError as e:
        raise ValueError(f"Variogram fitting failed: {e}") from e

    nugget, sill, range_param = params
    residuals = g_fit - model_fn(h_fit, *params)
    rmse = float(np.sqrt((residuals**2).mean()))

    def fitted_fn(h_new: float) -> float:
        return float(model_fn(np.asarray([h_new]), nugget, sill, range_param)[0])

    return {
        "model": model,
        "nugget": float(nugget),
        "sill": float(sill),
        "range_param": float(range_param),
        "rmse": rmse,
        "model_fn": fitted_fn,
    }
