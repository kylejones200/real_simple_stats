"""Causal inference — quasi-experimental designs for treatment effect estimation.

When a randomized experiment isn't possible, these methods let you reason about
cause and effect from observational or panel data:

- :func:`difference_in_differences` — pre/post × treatment/control panel design
- :func:`regression_discontinuity` — local polynomial estimation at a cutoff
- :func:`synthetic_control` — weighted counterfactual from donor control units
- :func:`panel_fixed_effects` — within-entity OLS with entity effects absorbed

All functions take numpy-compatible sequences and return plain ``dict`` results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as norm_dist
from scipy.stats import t as t_dist

__all__ = [
    "difference_in_differences",
    "regression_discontinuity",
    "synthetic_control",
    "panel_fixed_effects",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ols(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS via lstsq. Returns (beta, se, sigma)."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = max(n - k, 1)
    sigma2 = float(resid @ resid) / df
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se, math.sqrt(sigma2)


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def difference_in_differences(
    outcome: Sequence[float],
    post: Sequence[int],
    treated: Sequence[int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate a treatment effect via difference-in-differences (DiD).

    Compares how much the treated group changed relative to the control group
    across the pre/post period boundary.  Relies on the *parallel trends*
    assumption: absent treatment, both groups would have trended the same way.

    Model: outcome = β₀ + β₁·post + β₂·treated + β₃·(post×treated) + ε

    The coefficient β₃ is the DiD estimator.

    Args:
        outcome: Outcome variable, length n.
        post: Binary — 1 for post-treatment period, 0 for pre. Length n.
        treated: Binary — 1 for treatment group, 0 for control. Length n.
        alpha: Significance level for the confidence interval (default 0.05).

    Returns:
        dict with keys: did_estimate, se, t_stat, p_value, ci, reject_null,
        coefficients, n, df_residual, r_squared.

    Example:
        >>> outcome = [100, 102, 110, 114,  103, 101, 104, 103]
        >>> post    = [  0,   0,   1,   1,    0,   0,   1,   1]
        >>> treated = [  1,   1,   1,   1,    0,   0,   0,   0]
        >>> r = difference_in_differences(outcome, post, treated)
        >>> round(r["did_estimate"], 1)
        5.0
    """
    y = np.asarray(outcome, dtype=float)
    post_ = np.asarray(post, dtype=float)
    treated_ = np.asarray(treated, dtype=float)
    n = len(y)
    if not (len(post_) == len(treated_) == n):
        raise ValueError("outcome, post, and treated must have the same length.")
    if n < 4:
        raise ValueError("Need at least 4 observations for DiD.")

    did_ = post_ * treated_
    X = np.column_stack([np.ones(n), post_, treated_, did_])
    beta, se, _ = _ols(X, y)

    df_resid = n - 4
    did_est = float(beta[3])
    did_se = float(se[3])
    t_stat = did_est / did_se if did_se > 0 else float("nan")
    p_value = float(2 * t_dist.sf(abs(t_stat), df_resid))
    t_crit = float(t_dist.ppf(1 - alpha / 2, df_resid))
    ci = (did_est - t_crit * did_se, did_est + t_crit * did_se)

    return {
        "did_estimate": did_est,
        "se": did_se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci": ci,
        "reject_null": p_value < alpha,
        "coefficients": {
            "intercept": float(beta[0]),
            "post": float(beta[1]),
            "treated": float(beta[2]),
            "did": float(beta[3]),
        },
        "n": n,
        "df_residual": df_resid,
        "r_squared": _r_squared(y, X @ beta),
    }


def regression_discontinuity(
    outcome: Sequence[float],
    running_var: Sequence[float],
    cutoff: float,
    degree: int = 1,
    bandwidth: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate a treatment effect via regression discontinuity design (RDD).

    Units at or above the cutoff are treated; units below are controls.  A local
    polynomial is fit on each side of the threshold and the *jump* in predicted
    outcomes at the cutoff is the causal estimate.

    Model (degree p):
        outcome = β₀ + β₁x + … + βₚxᵖ + τT + γ₁Tx + … + γₚTxᵖ + ε

    where x = running_var − cutoff and T = 1{running_var ≥ cutoff}.
    The coefficient τ is the RDD estimate.

    Args:
        outcome: Outcome variable, length n.
        running_var: Variable that determines treatment assignment.
        cutoff: Threshold — units ≥ cutoff are treated.
        degree: Polynomial degree (default 1 = local linear).
        bandwidth: If given, restrict to observations within ±bandwidth of the
            cutoff.  ``None`` uses all data.
        alpha: Significance level (default 0.05).

    Returns:
        dict with keys: effect, se, t_stat, p_value, ci, reject_null,
        n_used, n_total, cutoff, degree, bandwidth.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.uniform(-2, 2, 500)
        >>> y = 1.0 + 0.5 * x + 3.0 * (x >= 0) + rng.normal(0, 0.5, 500)
        >>> r = regression_discontinuity(y, x, cutoff=0.0)
        >>> 2.0 < r["effect"] < 4.0
        True
    """
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(running_var, dtype=float)
    n_all = len(y)
    if len(x) != n_all:
        raise ValueError("outcome and running_var must have the same length.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    x_c = x - cutoff
    T = (x >= cutoff).astype(float)

    if bandwidth is not None:
        mask = np.abs(x_c) <= bandwidth
        y, x_c, T = y[mask], x_c[mask], T[mask]

    n = len(y)
    k_cols = 2 * (degree + 1)
    if n < k_cols + 1:
        raise ValueError(
            f"Too few observations ({n}) for degree-{degree} RDD; "
            f"need at least {k_cols + 1}."
        )

    # [1, x, x², …, xᵖ, T, Tx, Tx², …, Txᵖ]
    cols = [np.ones(n)]
    for d in range(1, degree + 1):
        cols.append(x_c**d)
    cols.append(T)
    for d in range(1, degree + 1):
        cols.append(T * x_c**d)
    X = np.column_stack(cols)

    beta, se, _ = _ols(X, y)

    T_idx = degree + 1
    effect = float(beta[T_idx])
    effect_se = float(se[T_idx])
    df_resid = n - X.shape[1]
    t_stat = effect / effect_se if effect_se > 0 else float("nan")
    p_value = float(2 * t_dist.sf(abs(t_stat), df_resid))
    t_crit = float(t_dist.ppf(1 - alpha / 2, df_resid))
    ci = (effect - t_crit * effect_se, effect + t_crit * effect_se)

    return {
        "effect": effect,
        "se": effect_se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci": ci,
        "reject_null": p_value < alpha,
        "n_used": n,
        "n_total": n_all,
        "cutoff": cutoff,
        "degree": degree,
        "bandwidth": bandwidth,
    }


def synthetic_control(
    y_treated: Sequence[float],
    Y_controls: Sequence[Sequence[float]],
    n_pre: int,
) -> dict[str, Any]:
    """Build a synthetic control counterfactual for a single treated unit.

    Finds non-negative weights (summing to 1) over control units such that the
    weighted average of control outcomes best matches the treated unit's
    pre-treatment trajectory.  The post-treatment *gap* (treated − synthetic)
    is the estimated treatment effect.

    Args:
        y_treated: Outcome series for the treated unit over all T periods,
            shape (T,).
        Y_controls: Outcome matrix for control (donor) units, shape (T, n_controls).
            Rows are time periods, columns are individual controls.
        n_pre: Number of pre-treatment periods.  Weights are fit on periods
            0 … n_pre−1.  Treatment starts at period n_pre.

    Returns:
        dict with keys:
            weights: Array of donor weights, shape (n_controls,).
            synthetic: Synthetic control series over all T periods.
            gap: Treated − synthetic for every period.
            ate_post: Average gap in the post-treatment period.
            pre_fit_rmse: In-sample RMSE for the pre-treatment fit (lower = better).

    Raises:
        ValueError: If shapes are inconsistent or n_pre is out of range.
        RuntimeError: If weight optimization fails to converge.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> Y = rng.normal(size=(20, 5))
        >>> y = Y[:, 0] + np.concatenate([np.zeros(10), np.full(10, 2.0)])
        >>> r = synthetic_control(y, Y[:, 1:], n_pre=10)
        >>> r["ate_post"] > 1.0
        True
    """
    y = np.asarray(y_treated, dtype=float)
    Y = np.asarray(Y_controls, dtype=float)
    T = len(y)
    if Y.ndim == 1:
        Y = Y.reshape(T, 1)
    if Y.shape[0] != T:
        raise ValueError(
            f"y_treated has length {T} but Y_controls has {Y.shape[0]} rows. "
            "Both must have T rows (one per time period)."
        )
    if not 1 <= n_pre < T:
        raise ValueError(f"n_pre must be between 1 and T−1; got {n_pre} with T={T}.")

    n_controls = Y.shape[1]
    y_pre = y[:n_pre]
    Y_pre = Y[:n_pre]

    w0 = np.full(n_controls, 1.0 / n_controls)

    def obj(w: np.ndarray) -> float:
        return float(((y_pre - Y_pre @ w) ** 2).sum())

    result = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_controls,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(
            f"Synthetic control optimization failed: {result.message}"
        )

    w = np.maximum(result.x, 0.0)
    w = w / w.sum()

    synthetic = Y @ w
    gap = y - synthetic

    return {
        "weights": w,
        "synthetic": synthetic,
        "gap": gap,
        "ate_post": float(gap[n_pre:].mean()),
        "pre_fit_rmse": float(np.sqrt(((y_pre - synthetic[:n_pre]) ** 2).mean())),
    }


def panel_fixed_effects(
    outcome: Sequence[float],
    predictors: Sequence[Sequence[float]] | Sequence[float],
    entity: Sequence[Any],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fixed-effects OLS regression via within-group demeaning.

    Removes time-invariant entity-specific levels by subtracting each entity's
    mean from the outcome and all predictors before running OLS.  This is
    equivalent to including entity dummy variables but avoids the memory cost of
    a large dummy matrix.

    Use this when entities differ in stable, unobserved ways (e.g. a store has a
    permanently high baseline) and you want coefficients that reflect
    within-entity variation only.

    Args:
        outcome: Outcome variable, length n.
        predictors: Predictor matrix shape (n, k) or length-n 1D array.
        entity: Entity identifier for each observation, length n.
        alpha: Significance level (default 0.05).

    Returns:
        dict with keys: coefficients, se, t_stats, p_values, ci, n,
        n_entities, df_residual, sigma.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> entity = np.repeat([0, 1, 2], 20)
        >>> x = rng.normal(size=60)
        >>> y = 2.0 * x + np.repeat([0, 5, -3], 20) + rng.normal(size=60)
        >>> r = panel_fixed_effects(y, x, entity)
        >>> abs(r["coefficients"][0] - 2.0) < 0.5
        True
    """
    y = np.asarray(outcome, dtype=float)
    X = np.asarray(predictors, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    entities = np.asarray(entity)
    n, k = X.shape

    if not (len(y) == n == len(entities)):
        raise ValueError(
            "outcome, predictors, and entity must all have the same length."
        )
    if n < k + 2:
        raise ValueError(f"Need more observations than predictors; got n={n}, k={k}.")

    unique_entities = np.unique(entities)
    n_entities = len(unique_entities)

    y_dm = y.copy()
    X_dm = X.copy()
    for e in unique_entities:
        mask = entities == e
        y_dm[mask] -= y[mask].mean()
        X_dm[mask] -= X[mask].mean(axis=0)

    beta, se, sigma = _ols(X_dm, y_dm)

    df_resid = max(n - k - n_entities, 1)
    # Recompute se with the correct df (the helper uses X.shape for df)
    resid = y_dm - X_dm @ beta
    sigma2_corrected = float(resid @ resid) / df_resid
    try:
        cov = sigma2_corrected * np.linalg.inv(X_dm.T @ X_dm)
    except np.linalg.LinAlgError:
        cov = sigma2_corrected * np.linalg.pinv(X_dm.T @ X_dm)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    sigma = math.sqrt(sigma2_corrected)

    t_stats = np.where(se > 0, beta / se, np.nan)
    p_values = 2.0 * t_dist.sf(np.abs(t_stats), df_resid)
    t_crit = float(t_dist.ppf(1 - alpha / 2, df_resid))
    cis = [
        (float(b - t_crit * s), float(b + t_crit * s))
        for b, s in zip(beta.tolist(), se.tolist())
    ]

    return {
        "coefficients": beta.tolist(),
        "se": se.tolist(),
        "t_stats": t_stats.tolist(),
        "p_values": p_values.tolist(),
        "ci": cis,
        "n": n,
        "n_entities": n_entities,
        "df_residual": df_resid,
        "sigma": sigma,
    }
