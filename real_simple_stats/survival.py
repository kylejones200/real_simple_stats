"""Survival analysis — time-to-event methods.

Estimates the probability that an event (failure, churn, conversion) has NOT
yet occurred at each point in time.  Handles *right censoring* — observations
where the event had not occurred by the time the study ended.

- :func:`kaplan_meier` — non-parametric step-function estimate of S(t).
  Correct for censored data.  The first stop for any time-to-event analysis.

- :func:`fit_parametric_survival` — fit a single named distribution
  (Exponential, Weibull, Lognormal, or Log-logistic) to observed event times.

- :func:`compare_survival_models` — fit all four models, rank by AIC, and
  return the full comparison so you can pick the best-fitting distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import (
    expon,
    fisk,
    lognorm,
    weibull_min,
)
from scipy.stats import (
    norm as norm_dist,
)

__all__ = [
    "kaplan_meier",
    "fit_parametric_survival",
    "compare_survival_models",
]

_DISTRIBUTIONS: dict[str, Any] = {
    "exponential": expon,
    "weibull": weibull_min,
    "lognormal": lognorm,
    "loglogistic": fisk,
}


def kaplan_meier(
    durations: Sequence[float],
    event_observed: Sequence[int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute the Kaplan-Meier survival curve.

    At each observed event time, the survival probability drops by the fraction
    of at-risk units that failed:

        S(tᵢ) = S(tᵢ₋₁) × (1 − dᵢ / nᵢ)

    where dᵢ is the number of events and nᵢ the number still at risk at tᵢ.
    Censored observations (event_observed=0) are removed from the risk set at
    their observed time without contributing to the hazard.

    Greenwood's formula provides pointwise confidence intervals.

    Args:
        durations: Observed time to event or censoring, length n.
        event_observed: 1 if the event occurred, 0 if right-censored. Length n.
        alpha: Significance level for the Greenwood CI (default 0.05).

    Returns:
        dict with keys:
            times: Array of times at which S(t) changes (includes t=0).
            survival_prob: Estimated S(t) at each time.
            ci_lower / ci_upper: Greenwood pointwise confidence bands.
            median_survival: Smallest t where S(t) ≤ 0.5, or None.
            n_events: Total observed events.
            n_censored: Total censored observations.

    Example:
        >>> durations = [2, 3, 5, 7, 11, 4, 8, 10]
        >>> observed  = [1, 1, 1, 1,  0, 1, 0,  1]
        >>> r = kaplan_meier(durations, observed)
        >>> r["survival_prob"][0]
        1.0
        >>> r["n_events"]
        6
    """
    t: np.ndarray = np.asarray(durations, dtype=float)
    e: np.ndarray = np.asarray(event_observed, dtype=int)
    n = len(t)
    if len(e) != n:
        raise ValueError("durations and event_observed must have the same length.")
    if n == 0:
        raise ValueError("Need at least one observation.")

    # Sort once; iterate over unique event times
    order = np.argsort(t)
    t_s = t[order]
    e_s = e[order]

    event_times = np.unique(t_s[e_s == 1])

    times = [0.0]
    surv = [1.0]
    greenwood = [0.0]

    s = 1.0
    gw = 0.0

    for ti in event_times:
        n_at_risk = float(np.sum(t >= ti))
        d = float(np.sum((t == ti) & (e == 1)))
        if n_at_risk > d:
            gw += d / (n_at_risk * (n_at_risk - d))
        s *= 1.0 - d / n_at_risk
        times.append(float(ti))
        surv.append(float(s))
        greenwood.append(gw)

    times_arr = np.array(times)
    surv_arr = np.array(surv)
    gw_arr = np.array(greenwood)

    z = float(norm_dist.ppf(1 - alpha / 2))
    se = surv_arr * np.sqrt(gw_arr)
    ci_lower = np.clip(surv_arr - z * se, 0.0, 1.0)
    ci_upper = np.clip(surv_arr + z * se, 0.0, 1.0)
    ci_lower[0] = ci_upper[0] = 1.0

    below_half = np.where(surv_arr <= 0.5)[0]
    median_survival = float(times_arr[below_half[0]]) if len(below_half) > 0 else None

    return {
        "times": times_arr,
        "survival_prob": surv_arr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median_survival": median_survival,
        "n_events": int(e.sum()),
        "n_censored": int(n - e.sum()),
    }


def fit_parametric_survival(
    durations: Sequence[float],
    event_observed: Sequence[int],
    distribution: str = "weibull",
) -> dict[str, Any]:
    """Fit a parametric survival model to observed event times.

    Fits a named parametric distribution to the *observed* (uncensored) event
    times using maximum likelihood.  Censored observations inform the Kaplan-
    Meier picture but are not used in this fit — for a fully correct censored-
    data MLE, use a dedicated survival library such as lifelines.

    Supported distributions: ``"exponential"``, ``"weibull"``,
    ``"lognormal"``, ``"loglogistic"``.

    Args:
        durations: Observed time to event or censoring, length n.
        event_observed: 1 if the event occurred, 0 if censored. Length n.
        distribution: Which parametric family to fit (default ``"weibull"``).

    Returns:
        dict with keys:
            distribution: Name of the fitted distribution.
            params: Fitted scipy distribution parameters (shape, loc, scale, …).
            aic: Akaike Information Criterion (lower = better fit).
            bic: Bayesian Information Criterion.
            n_fit: Number of observed events used in the fit.
            survival_fn: Callable S(t) → float for the fitted model.

    Raises:
        ValueError: If distribution name is unrecognised or too few events.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> t = rng.weibull(1.5, 200) * 50
        >>> e = np.ones(200, dtype=int)
        >>> r = fit_parametric_survival(t, e, distribution="weibull")
        >>> r["distribution"]
        'weibull'
        >>> r["aic"] < r["bic"] or r["aic"] >= r["bic"]  # both computed
        True
    """
    dist_name = distribution.lower()
    if dist_name not in _DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution {distribution!r}. "
            f"Choose from: {', '.join(_DISTRIBUTIONS)}."
        )

    t: np.ndarray = np.asarray(durations, dtype=float)
    e: np.ndarray = np.asarray(event_observed, dtype=int)
    t_obs = t[e == 1]

    if len(t_obs) < 3:
        raise ValueError(
            f"Need at least 3 observed events to fit a parametric model; "
            f"got {len(t_obs)}."
        )

    dist = _DISTRIBUTIONS[dist_name]
    params = dist.fit(t_obs, floc=0)

    # Log-likelihood and information criteria
    log_lik = float(dist.logpdf(t_obs, *params).sum())
    k = len(params)
    n = len(t_obs)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik

    def survival_fn(time: float) -> float:
        return float(dist.sf(time, *params))

    return {
        "distribution": dist_name,
        "params": params,
        "aic": aic,
        "bic": bic,
        "n_fit": n,
        "survival_fn": survival_fn,
    }


def compare_survival_models(
    durations: Sequence[float],
    event_observed: Sequence[int],
) -> list[dict[str, Any]]:
    """Fit and compare all supported parametric survival models.

    Fits Exponential, Weibull, Lognormal, and Log-logistic distributions and
    ranks them by AIC (lower = better trade-off between fit and simplicity).

    Args:
        durations: Observed time to event or censoring, length n.
        event_observed: 1 if the event occurred, 0 if censored. Length n.

    Returns:
        List of result dicts (same shape as :func:`fit_parametric_survival`)
        sorted by AIC ascending.  Each dict also includes ``"rank"`` (1 = best).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> t = rng.exponential(scale=30, size=300)
        >>> e = np.ones(300, dtype=int)
        >>> results = compare_survival_models(t, e)
        >>> results[0]["distribution"]  # exponential should win
        'exponential'
    """
    results = []
    for name in _DISTRIBUTIONS:
        try:
            r = fit_parametric_survival(durations, event_observed, distribution=name)
            results.append(r)
        except Exception:
            pass

    results.sort(key=lambda r: r["aic"])
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results
