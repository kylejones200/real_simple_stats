# Survival Analysis Guide

Survival analysis answers questions about *time until an event*: How long until a customer churns? When does a machine fail? How quickly does a patient recover? The defining feature of this data type is **right censoring** — you often don't observe the event for every subject.

---

## What counts as a "survival" problem?

The name comes from medical research, but the methods apply anywhere you're measuring time to an event:

| Domain | Event | Censored when |
|---|---|---|
| SaaS / e-commerce | Customer churns (cancels) | Customer still active at end of study |
| Manufacturing | Machine fails | Machine still running at end of study |
| Lending | Borrower defaults | Loan fully repaid, still active, or study ends |
| HR | Employee resigns | Employee still at company at end of study |
| Clinical | Patient relapses | Patient withdrew, lost to follow-up, or study ended |

---

## Right censoring: the central concept

A **right-censored** observation is one where you know the subject survived *at least* until their last observation time, but you don't know when (or if) they eventually experienced the event.

```
Timeline for 5 customers, observed for 12 months:

Customer 1: ──────────────●  (churned at month 8)
Customer 2: ─────────────────────────────────>   (still active at month 12, censored)
Customer 3: ──────●               (churned at month 5)
Customer 4: ─────────────────>    (dropped out at month 9 — reason unknown, censored)
Customer 5: ──────────────────────●  (churned at month 11)
                                       ^month 12
```

**Why you can't just ignore censored observations**: If you drop customers 2 and 4 from the analysis and compute the average churn time from the remaining three, you get (8 + 5 + 11) / 3 = 8 months. But customers 2 and 4 survived longer than your censored times — you're systematically underestimating survival. This is **survivor bias**.

```python
import real_simple_stats as rss

# 0 = censored, 1 = event observed
durations      = [8, 12, 5, 9, 11]
event_observed = [1,  0, 1, 0,  1]   # customers 2 and 4 are censored

r = rss.kaplan_meier(durations, event_observed)
print(f"Median survival: {r['median_survival']}")
print(f"Events observed: {r['n_events']} / {r['n_events'] + r['n_censored']}")
```

---

## The Kaplan-Meier estimator

KM builds a survival curve step by step. At each observed event time:

1. Count how many subjects are still at risk (haven't had the event yet and haven't been censored)
2. Record how many experienced the event at this time
3. Multiply the previous survival probability by (1 − events / at-risk)

Censored subjects leave the risk set at their censoring time without contributing to the hazard — this is the key correction.

```python
result = rss.kaplan_meier_explained(durations, event_observed)
print(result)
# Prints: question, how KM works, interpretation of THIS result,
#         censoring assumption caveat, next steps

result.plot()
# Step-function curve with Greenwood confidence band
```

### Reading the KM curve

- **The steps**: each drop represents one or more events
- **The ticks** (if shown): censored observations are marked with a small tick on the curve
- **The CI band**: Greenwood's formula — pointwise, not simultaneous
- **Flat right tail**: no more events observed after this point; S(t) is unknown beyond here, not zero

### When KM gives `median_survival = None`

If the survival curve never drops to 0.5, the median is not reached within the observation window. This is common when your follow-up period is short relative to the typical event time.

---

## Parametric survival models

KM is non-parametric — it makes no assumptions about the shape of the hazard. This is a strength for description, but a limitation for extrapolation. If you need to estimate survival probabilities beyond your observation window, fit a parametric model.

### The four distributions

| Distribution | Hazard | Best for |
|---|---|---|
| **Exponential** | Constant (memory-less) | Events that happen at a fixed rate regardless of age — rare in practice |
| **Weibull** | Increasing or decreasing monotonically | Most failure and churn processes; the most flexible of the four |
| **Lognormal** | Increases then decreases (log-normal hazard) | Processes with an initial period before the event becomes likely |
| **Log-logistic** | Also increases then decreases | Similar to lognormal but heavier tails |

```python
# Fit a specific distribution
r = rss.fit_parametric_survival(durations, event_observed, distribution="weibull")
print(f"Parameters: {r['params']}")
print(f"AIC: {r['aic']:.1f}")
print(f"P(survive > 30 days): {r['survival_fn'](30):.3f}")

# Compare all four — returns AIC-ranked list
ranked = rss.compare_survival_models(durations, event_observed)
for m in ranked:
    print(f"  {m['distribution']:12s}  AIC={m['aic']:.1f}  (rank {m['rank']})")
```

### Choosing by AIC

Lower AIC = better model. The differences matter more than the absolute values:

- ΔAIC < 2: essentially equivalent; prefer the simpler model (Exponential if it qualifies)
- ΔAIC 2–10: moderate evidence for the lower-AIC model
- ΔAIC > 10: strong evidence; use the lower-AIC model

Even the best-ranked parametric model can be a poor fit overall. Always overlay the fitted curve on the KM estimate and check visually.

---

## Full worked example: customer churn

```python
import real_simple_stats as rss
import numpy as np

rng = np.random.default_rng(0)
n = 200

# Simulate: churn follows a Weibull distribution
# Shape > 1 → hazard increases over time (churn accelerates)
from scipy.stats import weibull_min
true_shape, true_scale = 1.5, 15.0
durations = weibull_min.rvs(true_shape, scale=true_scale, size=n, random_state=rng)

# ~30% censored (customers still active)
censoring_time = rng.uniform(5, 30, n)
event_observed = (durations <= censoring_time).astype(int)
durations = np.minimum(durations, censoring_time)

# Step 1: describe the curve
km_result = rss.kaplan_meier_explained(durations, event_observed)
print(km_result)
km_result.plot()

# Step 2: find the best parametric model
ranked = rss.compare_survival_models(durations, event_observed)
best = ranked[0]
print(f"\nBest model: {best['distribution']}  (AIC={best['aic']:.1f})")

# Step 3: forecast churn probability at business-relevant horizons
for t in [7, 14, 30, 60, 90]:
    p_survive = best["survival_fn"](t)
    p_churn = 1 - p_survive
    print(f"  P(churn within {t:2d} days): {p_churn:.1%}")
```

---

## Common mistakes

**Ignoring censoring**: Averaging over only the customers who churned, ignoring those still active. Always use KM — never use `np.mean(durations[event_observed == 1])`.

**Informative censoring**: KM assumes censored subjects are no more or less likely to experience the event than those who remain. If sicker patients drop out of a clinical trial, the KM curve is optimistic. There is no statistical fix for this — it requires study design changes.

**Extrapolating past the data without a parametric model**: The KM curve is flat after the last event, but that doesn't mean survival stays high forever. Use `fit_parametric_survival` if you need to project beyond the observation window.

**Comparing groups without a log-rank test**: KM describes one group at a time. To test whether two groups differ (e.g. treated vs. control), you need a log-rank test (not yet in `rss` — use `lifelines` or `scipy` for this).

---

## See also

- [WHICH_TEST.md](WHICH_TEST.md) — when to use survival analysis vs. other methods
- [MATHEMATICAL_FORMULAS.md](MATHEMATICAL_FORMULAS.md) — KM formula, Greenwood CI, AIC
- [FAQ.md](FAQ.md) — censoring, KM vs. parametric, AIC interpretation
