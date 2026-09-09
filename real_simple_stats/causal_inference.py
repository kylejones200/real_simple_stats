"""Causal inference — quasi-experimental designs for treatment effect estimation.

When a randomized experiment isn't possible, these methods let you reason about
cause and effect from observational or panel data:

- :func:`difference_in_differences` — pre/post × treatment/control panel design
- :func:`regression_discontinuity` — local polynomial estimation at a cutoff
- :func:`synthetic_control` — weighted counterfactual from donor control units
- :func:`panel_fixed_effects` — within-entity OLS with entity effects absorbed

All functions take any numeric sequences (lists, tuples, or NumPy arrays if you
have them) and return plain ``dict`` results. The library itself has no runtime
dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from . import _rss

__all__ = [
    "difference_in_differences",
    "regression_discontinuity",
    "synthetic_control",
    "panel_fixed_effects",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ols(X: list[list[float]], y: list[float]) -> tuple[list[float], list[float], float]:
    """OLS on a design matrix given as a list of rows.

    Returns ``(coefficients, standard_errors, residual_sigma)``. The native
    backend falls back to the pseudo-inverse for a rank-deficient design, so
    this does not raise where the NumPy version caught LinAlgError.
    """
    n = len(X)
    k = len(X[0]) if n else 0
    flat = [v for row in X for v in row]
    coef, se, _t, _p, _resid, _r2, _ar2, _df, sigma2, _cov = _rss.ols(n, k, flat, y)
    return coef, se, math.sqrt(sigma2)


def _predict(X: list[list[float]], beta: list[float]) -> list[float]:
    return [sum(v * b for v, b in zip(row, beta)) for row in X]


def _r_squared(y: list[float], y_hat: list[float]) -> float:
    mean_y = _rss.mean(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, y_hat))
    ss_tot = sum((a - mean_y) ** 2 for a in y)
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
    y = [float(v) for v in outcome]
    post_ = [float(v) for v in post]
    treated_ = [float(v) for v in treated]
    n = len(y)
    if not (len(post_) == len(treated_) == n):
        raise ValueError("outcome, post, and treated must have the same length.")
    if n < 4:
        raise ValueError("Need at least 4 observations for DiD.")

    X = [[1.0, p, t, p * t] for p, t in zip(post_, treated_)]
    beta, se, _ = _ols(X, y)

    df_resid = n - 4
    did_est = float(beta[3])
    did_se = float(se[3])
    t_stat = did_est / did_se if did_se > 0 else float("nan")
    p_value = 2 * _rss.t_sf(abs(t_stat), df_resid)
    t_crit = _rss.t_ppf(1 - alpha / 2, df_resid)
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
        "r_squared": _r_squared(y, _predict(X, beta)),
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
        >>> from real_simple_stats import Rng
        >>> rng = Rng(0)
        >>> x = rng.uniform(-2, 2, 500)
        >>> noise = rng.normal(0, 0.5, 500)
        >>> y = [1.0 + 0.5 * xi + 3.0 * (xi >= 0) + e for xi, e in zip(x, noise)]
        >>> r = regression_discontinuity(y, x, cutoff=0.0)
        >>> 2.0 < r["effect"] < 4.0
        True
    """
    y = [float(v) for v in outcome]
    x = [float(v) for v in running_var]
    n_all = len(y)
    if len(x) != n_all:
        raise ValueError("outcome and running_var must have the same length.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    x_c = [xi - cutoff for xi in x]
    T = [1.0 if xi >= cutoff else 0.0 for xi in x]

    if bandwidth is not None:
        keep = [i for i, xc in enumerate(x_c) if abs(xc) <= bandwidth]
        y = [y[i] for i in keep]
        x_c = [x_c[i] for i in keep]
        T = [T[i] for i in keep]

    n = len(y)
    k_cols = 2 * (degree + 1)
    if n < k_cols + 1:
        raise ValueError(
            f"Too few observations ({n}) for degree-{degree} RDD; "
            f"need at least {k_cols + 1}."
        )

    # [1, x, x^2, ..., x^p, T, Tx, Tx^2, ..., Tx^p]
    X = []
    for i in range(n):
        row = [1.0]
        row.extend(x_c[i] ** d for d in range(1, degree + 1))
        row.append(T[i])
        row.extend(T[i] * x_c[i] ** d for d in range(1, degree + 1))
        X.append(row)

    beta, se, _ = _ols(X, y)

    T_idx = degree + 1
    effect = float(beta[T_idx])
    effect_se = float(se[T_idx])
    df_resid = n - k_cols
    t_stat = effect / effect_se if effect_se > 0 else float("nan")
    p_value = 2 * _rss.t_sf(abs(t_stat), df_resid)
    t_crit = _rss.t_ppf(1 - alpha / 2, df_resid)
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
        >>> from real_simple_stats import Rng
        >>> rng = Rng(42)
        >>> Y = [rng.normal(0, 1, 5) for _ in range(20)]
        >>> y = [Y[t][0] + (0.0 if t < 10 else 2.0) for t in range(20)]
        >>> controls = [row[1:] for row in Y]
        >>> r = synthetic_control(y, controls, n_pre=10)
        >>> r["ate_post"] > 1.0
        True
    """
    y = [float(v) for v in y_treated]
    T = len(y)

    rows = [
        [float(r)] if isinstance(r, (int, float)) else [float(v) for v in r]
        for r in Y_controls
    ]
    if len(rows) != T:
        raise ValueError(
            f"y_treated has length {T} but Y_controls has {len(rows)} rows. "
            "Both must have T rows (one per time period)."
        )
    if not 1 <= n_pre < T:
        raise ValueError(f"n_pre must be between 1 and T-1; got {n_pre} with T={T}.")

    n_controls = len(rows[0])
    # Column-major view of the pre-period, which is what the solver consumes.
    y_pre = y[:n_pre]
    columns_pre = [[rows[t][j] for t in range(n_pre)] for j in range(n_controls)]

    # Weights on the simplex (non-negative, summing to 1). The Rust solver is
    # projected gradient with an exact simplex projection, so every iterate is
    # feasible by construction and there is no optimiser failure to handle.
    w = _rss.simplex_least_squares(y_pre, columns_pre)

    synthetic = [sum(rows[t][j] * w[j] for j in range(n_controls)) for t in range(T)]
    gap = [a - b for a, b in zip(y, synthetic)]

    return {
        "weights": w,
        "synthetic": synthetic,
        "gap": gap,
        "ate_post": _rss.mean(gap[n_pre:]),
        "pre_fit_rmse": math.sqrt(
            sum((a - b) ** 2 for a, b in zip(y_pre, synthetic[:n_pre])) / n_pre
        ),
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
        >>> from real_simple_stats import Rng
        >>> rng = Rng(0)
        >>> entity = [0] * 20 + [1] * 20 + [2] * 20
        >>> x = rng.normal(0, 1, 60)
        >>> fixed = [0, 5, -3]
        >>> noise = rng.normal(0, 1, 60)
        >>> y = [2.0 * xi + fixed[e] + n for xi, e, n in zip(x, entity, noise)]
        >>> r = panel_fixed_effects(y, x, entity)
        >>> abs(r["coefficients"][0] - 2.0) < 0.5
        True
    """
    y = [float(v) for v in outcome]
    X = [
        [float(r)] if isinstance(r, (int, float)) else [float(v) for v in r]
        for r in predictors
    ]
    entities = list(entity)
    n = len(X)
    k = len(X[0]) if n else 0

    if not (len(y) == n == len(entities)):
        raise ValueError(
            "outcome, predictors, and entity must all have the same length."
        )
    if n < k + 2:
        raise ValueError(f"Need more observations than predictors; got n={n}, k={k}.")

    # Preserve first-appearance order so results do not depend on sortability
    # of the entity labels.
    unique_entities: list = []
    for e in entities:
        if e not in unique_entities:
            unique_entities.append(e)
    n_entities = len(unique_entities)

    # Within transformation: subtract each entity's own mean.
    y_dm = list(y)
    X_dm = [list(row) for row in X]
    for e in unique_entities:
        idx = [i for i, ent in enumerate(entities) if ent == e]
        y_bar = _rss.mean([y[i] for i in idx])
        x_bar = [_rss.mean([X[i][j] for i in idx]) for j in range(k)]
        for i in idx:
            y_dm[i] -= y_bar
            for j in range(k):
                X_dm[i][j] -= x_bar[j]

    beta, _se_naive, _sigma = _ols(X_dm, y_dm)

    # Standard errors must use the fixed-effects degrees of freedom, which
    # charge one parameter per entity on top of the k slopes.
    df_resid = max(n - k - n_entities, 1)
    resid = [a - b for a, b in zip(y_dm, _predict(X_dm, beta))]
    sigma2_corrected = sum(r * r for r in resid) / df_resid

    flat_dm = [v for row in X_dm for v in row]
    xtx = _rss.mat_matmul(
        k, n, [X_dm[i][j] for j in range(k) for i in range(n)], n, k, flat_dm
    )[2]
    try:
        xtx_inv = _rss.mat_inv(k, k, xtx)
    except ValueError:
        _r, _c, xtx_inv = _rss.mat_pinv(k, k, xtx)

    se = [math.sqrt(max(sigma2_corrected * xtx_inv[j * k + j], 0.0)) for j in range(k)]
    sigma = math.sqrt(sigma2_corrected)

    t_stats = [b / s if s > 0 else math.nan for b, s in zip(beta, se)]
    p_values = [
        2.0 * _rss.t_sf(abs(t), df_resid) if t == t else math.nan for t in t_stats
    ]
    t_crit = _rss.t_ppf(1 - alpha / 2, df_resid)
    cis = [(b - t_crit * s, b + t_crit * s) for b, s in zip(beta, se)]

    return {
        "coefficients": beta,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "ci": cis,
        "n": n,
        "n_entities": n_entities,
        "df_residual": df_resid,
        "sigma": sigma,
    }
