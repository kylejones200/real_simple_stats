# What Can Statistics Do?

Statistics helps you answer five kinds of questions. Start with your question, find the right category below, and follow the path to the function you need.

---

## The five objectives

### 1. Describe — "What does my data look like?"

Summarise the shape, centre, spread, and unusual values of a dataset before drawing any conclusions.

**Functions**

```python
import real_simple_stats as rss

data = [23, 25, 28, 30, 32, 35, 38, 40]

rss.mean(data)                      # 31.4
rss.median(data)                    # 31.0
rss.sample_std_dev(data)            # 5.9
rss.five_number_summary(data)       # min/Q1/median/Q3/max
rss.iqr(data)                       # interquartile range
rss.skewness(data)                  # symmetry measure
rss.kurtosis(data)                  # tail heaviness
rss.detect_outliers_iqr(data)       # list of outlier values
rss.frequency_table(data)           # value → count

# Time series summaries
rss.rolling_statistics(data, window=3)   # rolling mean/std/min/max
rss.linear_trend(data)                   # (slope, intercept, r²)
```

**When to use**: always first — before any test or model.

---

### 2. Compare — "Are these groups different?"

Detect differences between groups, time periods, or reference values. Tells you whether a gap is real or likely to be sampling noise.

**Frequentist tests**

```python
# Two means
rss.one_sample_t_test(data, mu=30)             # vs. fixed reference
rss.two_sample_t_test(group_a, group_b)        # independent groups
rss.paired_t_test(before, after)               # same units measured twice

# Three or more means
r = rss.one_way_anova(g1, g2, g3)
# r["f_stat"], r["p_value"], r["eta_squared"]

# Categorical variables
r = rss.chi_square_independence([[40, 5], [5, 40]])
# r["chi2"], r["p_value"], r["cramers_v"]

# Non-parametric alternatives
rss.mann_whitney_u(group_a, group_b)
rss.wilcoxon_signed_rank(before, after)
```

**Self-explaining variants** — see the narrative alongside the numbers:

```python
result = rss.one_sample_t_test_explained(data, mu=30)
result.p_value          # the number
print(result)           # the explanation + misconception guards
result.plot()           # p-value as shaded tail area

result = rss.one_way_anova_explained(g1, g2, g3)
result.plot()           # box plots of each group
```

**Effect sizes** — significance alone is not enough:

```python
rss.cohens_d(group_a, group_b)        # standardised mean difference
rss.eta_squared(groups)               # fraction of variance explained
rss.cramers_v(contingency_table)      # association strength for chi-square
```

---

### 3. Explain causally — "Did X cause Y?"

Observational data doesn't directly answer causal questions — you need a design that controls for confounders. Use these when a randomised experiment isn't possible.

**Difference-in-differences** — pre/post data with a control group

```python
# "Did the policy change the outcome, net of the time trend?"
r = rss.difference_in_differences(outcome, post, treated)
r["did_estimate"]   # causal effect (β₃ in the interaction model)
r["p_value"]
r["ci"]             # 95% confidence interval

result = rss.difference_in_differences_explained(outcome, post, treated)
result.plot()       # 2×2 DiD diagram with counterfactual line
print(result)       # parallel trends caveat, key assumptions
```

**Regression discontinuity** — units assigned by a threshold

```python
# "Did crossing the 65-point cutoff cause a change?"
r = rss.regression_discontinuity(outcome, running_var, cutoff=65)
r["effect"]         # LATE at the cutoff
r["p_value"]
```

**Synthetic control** — no comparable control group exists

```python
# "What would California's GDP have been without the law?"
r = rss.synthetic_control(y_treated, Y_controls, n_pre=20)
r["weights"]        # donor weights (≥ 0, sum to 1)
r["effect"]         # post-treatment effect
```

**Panel fixed effects** — repeated measurements per entity

```python
# "Controlling for country-level factors, what's the wage effect?"
r = rss.panel_fixed_effects(outcome, predictors, entity)
r["coefficients"]
r["r_squared"]
```

---

### 4. Predict — "What will happen next?"

Use historical patterns to forecast future values. Returns numbers you can plug into decisions.

```python
# Simple linear extrapolation
slope, intercept = rss.linear_regression(x, y)
r2 = rss.r_squared(x, y)

# Multiple predictors
r = rss.multiple_regression(X, y)
r["coefficients"]
r["r_squared"]

# Time series forecasting
rss.exponential_smoothing(data, alpha=0.3)
r = rss.double_exponential_smoothing(data, alpha=0.8, beta=0.2)
r["smoothed"]   # in-sample fits
r["trend"]      # trend component — extrapolate with l_T + h·b_T

# Forecast accuracy (scale-independent)
rss.mean_absolute_scaled_error(actual, forecast)
# < 1.0 means your model beats a naïve lag-1 baseline
```

**Time-to-event prediction** (churn, failure, default):

```python
# When will the customer churn?
ranked = rss.compare_survival_models(durations, observed)
best = ranked[0]                 # lowest AIC
best["distribution"]             # e.g. "weibull"
best["survival_fn"](t=30)        # P(survive past day 30)
```

---

### 5. Discover structure — "What patterns exist in my data?"

Find associations, clusters, and structure that isn't obvious from looking at the raw numbers.

**Association rules** (market basket, cross-sell, co-occurrence):

```python
matrix, items = rss.encode_transactions(transactions)
itemsets = rss.frequent_itemsets(matrix, items, min_support=0.3)
rules = rss.association_rules(itemsets, min_confidence=0.6, min_lift=1.0)
for r in rules:
    print(f"{r['antecedent']} → {r['consequent']}  lift={r['lift']:.2f}")
```

**Spatial autocorrelation** (are similar values clustered?):

```python
r = rss.morans_i(x, y, values, distance_threshold=20)
r["moran_i"]        # +1 = clustered, 0 = random, -1 = dispersed
r["p_value"]

result = rss.morans_i_explained(x, y, values)
result.plot()       # spatial scatter coloured by value
```

**Change points** (when did the mean shift?):

```python
r = rss.detect_change_points(data, n_breaks=2)
r["change_points"]   # [index_1, index_2]
r["segment_means"]   # mean of each segment

result = rss.detect_change_points_explained(data, n_breaks=2)
result.plot()        # series with break lines
```

**Multivariate structure**:

```python
rss.pca(X, n_components=2)            # dimensionality reduction
rss.mahalanobis_distance(X)           # multivariate outlier detection
rss.pearson_correlation(x, y)
rss.spearman_correlation(x, y)        # monotone (non-linear) relationships
```

---

## Quick-reference table

| Question | Category | Key functions |
|---|---|---|
| What does my data look like? | Describe | `mean`, `five_number_summary`, `rolling_statistics` |
| Is group A different from B? | Compare | `two_sample_t_test`, `one_way_anova_explained` |
| Are these variables associated? | Compare | `chi_square_independence_explained`, `pearson_correlation` |
| Did the policy cause a change? | Explain causally | `difference_in_differences_explained` |
| Who was assigned by a cutoff? | Explain causally | `regression_discontinuity` |
| What will sales be next quarter? | Predict | `double_exponential_smoothing`, `multiple_regression` |
| When will the customer churn? | Predict | `kaplan_meier_explained`, `compare_survival_models` |
| Do values cluster in space? | Discover | `morans_i_explained`, `compute_variogram` |
| Which products co-occur? | Discover | `frequent_itemsets`, `association_rules` |
| When did the mean shift? | Discover | `detect_change_points_explained` |

---

## Choosing a method: the one-paragraph version

**Start with your data type.**

- *Continuous outcome, one or two groups* → t-test family. Use `one_sample_t_test_explained` or `two_sample_t_test`. Always pair with an effect size (`cohens_d`).
- *Continuous outcome, three+ groups* → `one_way_anova_explained`. Significant? Follow up with pairwise post-hoc tests.
- *Categorical outcome* → `chi_square_independence_explained`. Report Cramér's V, not just p.
- *Time-to-event outcome (churn, failure)* → `kaplan_meier_explained` first, then `compare_survival_models` to fit and extrapolate.
- *"Did X cause Y?" question* → Pick a quasi-experimental design: DiD if you have a before/after + control group; RDD if assignment was by a threshold; synthetic control if there's no comparable control; panel FE if you have repeated observations per entity.
- *Spatial data* → `morans_i_explained` to test clustering, then `compute_variogram` + `fit_variogram` to model how autocorrelation decays with distance.
- *Transaction data* → `encode_transactions` → `frequent_itemsets` → `association_rules`.
- *Time series with unknown breaks* → `detect_change_points_explained`.

**When in doubt, see [docs/WHICH_TEST.md](WHICH_TEST.md) for the full decision tree.**

---

## Further reading

- [WHICH_TEST.md](WHICH_TEST.md) — full decision tree with one-paragraph rationale per leaf
- [CAUSAL_INFERENCE_GUIDE.md](CAUSAL_INFERENCE_GUIDE.md) — when to use DiD vs. RDD vs. synthetic control
- [SURVIVAL_ANALYSIS_GUIDE.md](SURVIVAL_ANALYSIS_GUIDE.md) — censoring, KM vs. parametric, model selection
- [SPATIAL_STATS_GUIDE.md](SPATIAL_STATS_GUIDE.md) — Moran's I, variogram concepts, model families
- [SELF_EXPLAINING_RESULTS.md](SELF_EXPLAINING_RESULTS.md) — the ExplainedResult pattern
- [FAQ.md](FAQ.md) — common questions by module
