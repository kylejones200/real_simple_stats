# Causal Inference Guide

The four quasi-experimental methods in `real_simple_stats` all answer the same fundamental question — did X *cause* Y? — but they require different data structures and rest on different identifying assumptions. This guide explains when to use each one, what can go wrong, and how to check your work.

---

## The fundamental problem of causal inference

You can never observe the same unit in two states at the same time. You can see what happened to the customers who received the discount; you cannot see what *would have happened* to those same customers if they hadn't received it. The counterfactual is fundamentally unobservable.

All four methods below solve this problem in different ways — by constructing a credible counterfactual from the data you do have.

---

## Method 1: Difference-in-Differences (DiD)

**The counterfactual**: the change the control group experienced over the same period.

**When to use it**: You have measurements from *before* and *after* an event, on both a treated group and a control group that wasn't treated. Classic examples: a minimum wage law in one state but not another, a product feature rolled out to half your user base.

### The estimator

DiD runs OLS with an interaction term:

```
outcome = β₀ + β₁·Post + β₂·Treated + β₃·(Post × Treated) + ε
```

**β₃ is the DiD estimate** — the extra change in the treated group, net of the time trend the control group experienced.

```python
import real_simple_stats as rss
import numpy as np

rng = np.random.default_rng(42)
n = 60

# Simulate: treatment adds 8 units, control group rises by 2 due to time trend
ctrl_pre  = rng.normal(100, 5, n)
ctrl_post = rng.normal(102, 5, n)   # +2 time trend
trt_pre   = rng.normal(100, 5, n)
trt_post  = rng.normal(110, 5, n)   # +2 time trend + 8 treatment = +10

outcome = np.concatenate([ctrl_pre, ctrl_post, trt_pre, trt_post])
post    = np.array([0]*n + [1]*n + [0]*n + [1]*n)
treated = np.array([0]*(2*n) + [1]*(2*n))

r = rss.difference_in_differences(outcome, post, treated)
print(f"DiD estimate: {r['did_estimate']:.2f}")   # ≈ 8.0
print(f"p-value: {r['p_value']:.4f}")
print(f"95% CI: ({r['ci'][0]:.2f}, {r['ci'][1]:.2f})")

# Self-explaining version with misconception guard
result = rss.difference_in_differences_explained(outcome, post, treated)
print(result)
result.plot()   # 2×2 DiD diagram with counterfactual line
```

### The key assumption: parallel trends

The parallel trends assumption says: *absent treatment, the treated and control groups would have changed at the same rate.* If the treated group was already trending faster *before* treatment began, DiD overestimates the effect.

**How to partially test it**: compute DiD using only pre-treatment periods. If you find a spurious "effect" in the pre-period, the assumption is suspect.

```python
# Test with pre-treatment data only (you need at least two pre periods)
pre_only_outcome = ...  # two pre-period waves
pre_only_post    = ...  # 0/1 for "later pre period"
pre_only_treated = ...
r_placebo = rss.difference_in_differences(pre_only_outcome, pre_only_post, pre_only_treated)
print(f"Placebo DiD: {r_placebo['did_estimate']:.2f}")  # should be near 0
```

### Common mistakes

- **Using only the treated group**: comparing treated-post to treated-pre without a control group conflates the treatment effect with the time trend.
- **Contaminated control group**: if control-group units were indirectly affected by the treatment (spillover effects, general equilibrium), DiD overestimates.
- **Composition changes**: if the mix of people in each group changes across periods (new customers entering, churned customers leaving), the comparison is confounded.

---

## Method 2: Regression Discontinuity (RDD)

**The counterfactual**: units just on the other side of the cutoff.

**When to use it**: Treatment was assigned by comparing a continuous score to a threshold — scholarship above 65 points, price cap for incomes below €30,000, parole for sentences under 12 months. Units just above and just below the threshold are approximately exchangeable, differing only in which side of the line they fall on.

```python
# Students above score 65 get tutoring; below 65 don't
# Simulate: tutoring raises final exam by 5 points
rng = np.random.default_rng(0)
n = 200
running_var = rng.uniform(40, 90, n)         # entrance score
treated_mask = running_var >= 65
outcome = (
    50 + 0.5 * running_var                   # linear baseline
    + 5 * treated_mask.astype(float)         # treatment effect
    + rng.normal(0, 3, n)
)

r = rss.regression_discontinuity(outcome, running_var, cutoff=65)
print(f"Treatment effect at cutoff: {r['effect']:.2f}")   # ≈ 5.0
print(f"p-value: {r['p_value']:.4f}")
```

### What RDD estimates

RDD identifies the **Local Average Treatment Effect (LATE)** — the effect *at the cutoff*, among units near the threshold. It does not identify the ATE for the whole population. If the effect varies across score levels (effect heterogeneity), the LATE at the cutoff may not generalise.

### Key checks

**Density test (McCrary test)**: If agents can manipulate their score to get just above or just below the cutoff, the identifying assumption fails. Check whether the density of the running variable shows a discontinuity at the cutoff — a suspicious pile-up just above (or below) the threshold is a red flag.

**Bandwidth sensitivity**: Results should be qualitatively stable across reasonable bandwidths around the cutoff. The library's `bandwidth` parameter controls the window; leaving it as `None` uses all data, which can introduce bias far from the cutoff.

---

## Method 3: Synthetic Control

**The counterfactual**: a weighted average of control units, constructed to match the treated unit's pre-treatment history.

**When to use it**: You have one (or a few) treated unit(s) and many potential control "donors," with a long pre-treatment history. Classic example: one state passes a law; you build a "synthetic state" from a weighted combination of the other states.

```python
rng = np.random.default_rng(1)
T = 40          # total time periods
n_pre = 25      # pre-treatment periods
n_donors = 8

# Treated unit: follows a trend, then treatment adds 4 units post-period 25
y_treated = np.concatenate([
    rng.normal(100 + np.arange(n_pre) * 0.5, 1),
    rng.normal(100 + n_pre * 0.5 + 4 + np.arange(T - n_pre) * 0.5, 1),
])

# Donor units: similar trends, no treatment
Y_controls = rng.normal(
    (100 + np.arange(T)[:, None] * 0.5) + rng.normal(0, 3, (1, n_donors)),
    1, (T, n_donors)
)

r = rss.synthetic_control(y_treated, Y_controls, n_pre=n_pre)
print("Donor weights:", r["weights"].round(3))
print(f"Pre-fit MSE: {r['pre_fit_mse']:.4f}")  # lower = better counterfactual
print(f"Average post-treatment effect: {r['effect'].mean():.2f}")  # ≈ 4.0
```

### Interpretation

- `weights`: each donor's contribution to the synthetic counterfactual. Weights are non-negative and sum to 1.
- `pre_fit_mse`: how closely the synthetic control matched the treated unit before treatment. High MSE = unreliable counterfactual.
- `effect`: post-treatment gap (treated − synthetic) at each time step.

**Inference by permutation**: run the same synthetic control procedure for each control unit as if it were treated, then compare the treated unit's effect to the distribution of placebo effects. This library returns the point estimate; the permutation loop is left to the user.

---

## Method 4: Panel Fixed Effects

**The counterfactual**: each entity's own mean, used to absorb time-invariant confounders.

**When to use it**: You have repeated measurements on the same entities (people, firms, countries) over multiple time periods. Fixed effects absorb all entity-level characteristics that don't change over time — eliminating a large class of confounders without explicitly modelling them.

```python
rng = np.random.default_rng(2)
n_entities = 30
n_time = 4

entity  = np.repeat(np.arange(n_entities), n_time)
time    = np.tile(np.arange(n_time), n_entities)
treat   = (entity % 2 == 0).astype(float)   # half the entities treated
post    = (time >= 2).astype(float)
treated = treat * post

# True effect: treatment raises outcome by 3
outcome = (
    5.0 * entity / n_entities               # entity-level fixed effect
    + 2.0 * post                             # time trend
    + 3.0 * treated                          # treatment effect
    + rng.normal(0, 1, n_entities * n_time)
)

r = rss.panel_fixed_effects(outcome, treated[:, None], entity)
print(f"Coefficient: {r['coefficients'][0]:.2f}")   # ≈ 3.0
print(f"R²: {r['r_squared']:.3f}")
```

### What fixed effects do (and don't) control for

Fixed effects absorb **time-invariant** unobserved confounders — things that differ across entities but are constant over time for each entity. They do **not** control for:

- Time-varying confounders (a change inside an entity that both affects outcome and correlates with treatment)
- The treatment itself if it is time-invariant (you can't estimate the effect of something that never changes within an entity)

---

## Choosing between the four methods

| Data structure | Best method |
|---|---|
| Pre/post + control group | Difference-in-Differences |
| Assignment by numerical cutoff | Regression Discontinuity |
| One treated unit, many donors, long pre-treatment history | Synthetic Control |
| Repeated measurements per entity, treatment varies over time | Panel Fixed Effects |

When multiple methods apply, use more than one and check whether they agree. Agreement builds confidence; disagreement is a diagnostic signal.

---

## See also

- [WHICH_TEST.md](WHICH_TEST.md) — causal branch of the decision tree
- [MATHEMATICAL_FORMULAS.md](MATHEMATICAL_FORMULAS.md) — DiD formula
- [FAQ.md](FAQ.md) — parallel trends, synthetic control interpretation, panel FE
