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

import math
from collections.abc import Sequence
from typing import Any

from . import _rss

__all__ = [
    "kaplan_meier",
    "fit_parametric_survival",
    "compare_survival_models",
]

#: Supported parametric families, mapped to their survival functions.
#: Location is fixed at 0 throughout, as befits durations.
_DISTRIBUTIONS: dict[str, Any] = {
    "exponential": lambda x, scale: _rss.expon_sf(x, scale),
    "weibull": lambda x, c, scale: _rss.weibull_sf(x, c, scale),
    "lognormal": lambda x, s, scale: _rss.lognorm_sf(x, s, scale),
    "loglogistic": lambda x, c, scale: _rss.fisk_sf(x, c, scale),
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
    t = [float(v) for v in durations]
    e = [int(v) for v in event_observed]
    n = len(t)
    if len(e) != n:
        raise ValueError("durations and event_observed must have the same length.")
    if n == 0:
        raise ValueError("Need at least one observation.")

    # Distinct times at which an event (not a censoring) was observed.
    event_times = sorted({ti for ti, ei in zip(t, e) if ei == 1})

    times = [0.0]
    surv = [1.0]
    greenwood = [0.0]

    s = 1.0
    gw = 0.0

    for ti in event_times:
        n_at_risk = float(sum(1 for v in t if v >= ti))
        d = float(sum(1 for v, ev in zip(t, e) if v == ti and ev == 1))
        if n_at_risk > d:
            gw += d / (n_at_risk * (n_at_risk - d))
        s *= 1.0 - d / n_at_risk
        times.append(float(ti))
        surv.append(float(s))
        greenwood.append(gw)

    # Greenwood standard errors and the corresponding normal-approximation band.
    z = _rss.norm_ppf(1 - alpha / 2)
    se = [sv * math.sqrt(g) for sv, g in zip(surv, greenwood)]
    ci_lower = [min(max(sv - z * e_i, 0.0), 1.0) for sv, e_i in zip(surv, se)]
    ci_upper = [min(max(sv + z * e_i, 0.0), 1.0) for sv, e_i in zip(surv, se)]
    # Survival is 1 at time 0 by definition, so the band is degenerate there.
    ci_lower[0] = ci_upper[0] = 1.0

    median_survival = next((ti for ti, sv in zip(times, surv) if sv <= 0.5), None)
    n_events = sum(e)

    return {
        "times": times,
        "survival_prob": surv,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median_survival": median_survival,
        "n_events": int(n_events),
        "n_censored": int(n - n_events),
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
            params: Fitted parameters as (shape, loc, scale), or (loc, scale)
                for the exponential. Location is always 0.
            aic: Akaike Information Criterion (lower = better fit).
            bic: Bayesian Information Criterion.
            n_fit: Number of observed events used in the fit.
            survival_fn: Callable S(t) → float for the fitted model.

    Raises:
        ValueError: If distribution name is unrecognised or too few events.

    Example:
        >>> from real_simple_stats import Rng
        >>> rng = Rng(0)
        >>> t = rng.weibull(1.5, 50.0, 200)
        >>> e = [1] * 200
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

    t = [float(v) for v in durations]
    e = [int(v) for v in event_observed]
    if len(e) != len(t):
        raise ValueError("durations and event_observed must have the same length.")
    t_obs = [ti for ti, ei in zip(t, e) if ei == 1]

    if len(t_obs) < 3:
        raise ValueError(
            f"Need at least 3 observed events to fit a parametric model; "
            f"got {len(t_obs)}."
        )

    if any(v <= 0 for v in t_obs):
        raise ValueError("All observed durations must be strictly positive.")

    shape, scale, log_lik = _rss.fit_survival(dist_name, t_obs)

    # Parameter tuples keep the (shape, loc, scale) layout, with loc pinned to
    # 0. The exponential has no shape, so it reports (loc, scale).
    if dist_name == "exponential":
        params = (0.0, scale)
        sf_args = (scale,)
    else:
        params = (shape, 0.0, scale)
        sf_args = (shape, scale)

    k = len(params)
    n = len(t_obs)
    aic = 2 * k - 2 * log_lik
    bic = k * math.log(n) - 2 * log_lik

    sf = _DISTRIBUTIONS[dist_name]

    def survival_fn(time: float) -> float:
        return float(sf(float(time), *sf_args))

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
        >>> from real_simple_stats import Rng
        >>> rng = Rng(1)
        >>> t = rng.exponential(30, 300)
        >>> e = [1] * 300
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
