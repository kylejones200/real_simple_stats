# Real Simple Stats — Quick Reference

**Version 0.4.1** · `pip install real-simple-stats` · Python 3.12+

```python
import real_simple_stats as rss
import numpy as np
```

---

## Self-explaining results

The library's key feature: every `_explained` function returns an `ExplainedResult` that works as both a data object and a teacher.

```python
result = rss.one_sample_t_test_explained(data, mu=5.0)
result.p_value          # use it as a number
result.ci               # (lower, upper) confidence interval
result.reject_null      # bool
result.plot()           # signature visualization
print(result)           # full narrative with misconception guards

# All seven explained functions:
rss.one_sample_t_test_explained(data, mu=5.0)
rss.one_way_anova_explained(g1, g2, g3)
rss.chi_square_independence_explained([[40, 5], [5, 40]])
rss.difference_in_differences_explained(outcome, post, treated)
rss.kaplan_meier_explained(durations, event_observed)
rss.morans_i_explained(x, y, values, distance_threshold=20)
rss.detect_change_points_explained(data, n_breaks=2)
```

---

## Descriptive statistics

```python
data = [12, 15, 18, 20, 22, 25, 28, 30]

rss.mean(data)                     # 21.25
rss.median(data)                   # 21.0
rss.mode(data)                     # most frequent value
rss.sample_std_dev(data)           # 6.41
rss.variance(data)                 # sample variance
rss.five_number_summary(data)      # min / Q1 / median / Q3 / max
rss.iqr(data)                      # Q3 - Q1
rss.coefficient_of_variation(data) # std / mean × 100
rss.skewness(data)
rss.kurtosis(data)
rss.detect_outliers_iqr(data)      # list of values outside 1.5×IQR fence
rss.frequency_table(data)          # {value: count}
```

---

## Probability

```python
rss.simple_probability(favorable=3, total=10)         # 0.3
rss.joint_probability(0.4, 0.3)                       # 0.12
rss.conditional_probability(0.12, 0.3)                # 0.4
rss.bayes_theorem(prior=0.01, sensitivity=0.95, specificity=0.90)

rss.combinations(n=10, k=3)                           # 120
rss.permutations(n=10, k=3)                           # 720
```

---

## Distributions

```python
# Normal
rss.normal_pdf(x=1.0, mean=0, std_dev=1)   # 0.2420
rss.normal_cdf(x=1.96, mean=0, std_dev=1)  # 0.9750
rss.z_score(value=75, mean=70, std_dev=10) # 0.5

# Binomial
rss.binomial_probability(n=10, k=3, p=0.5) # P(X=3)
rss.binomial_cdf(n=10, k=3, p=0.5)         # P(X≤3)
rss.binomial_mean(n=10, p=0.5)             # 5.0

# Poisson
rss.poisson_probability(k=3, lam=2.5)

# Geometric / Exponential
rss.geometric_probability(k=4, p=0.3)
rss.exponential_probability(x=2.0, lam=0.5)
```

---

## Hypothesis testing

```python
# One-sample t-test
t, p = rss.one_sample_t_test(data, mu=30)
result = rss.one_sample_t_test_explained(data, mu=30)

# Two-sample / paired
rss.two_sample_t_test(g1, g2, equal_var=True)
rss.paired_t_test(before, after)

# One-way ANOVA
r = rss.one_way_anova(g1, g2, g3)
# r["f_stat"], r["p_value"], r["eta_squared"], r["reject_null"]
result = rss.one_way_anova_explained(g1, g2, g3)

# Chi-square independence
r = rss.chi_square_independence([[40, 5], [5, 40]])
# r["chi2"], r["p_value"], r["cramers_v"], r["reject_null"]
result = rss.chi_square_independence_explained([[40, 5], [5, 40]])

# Z-test and proportions
rss.z_test(data, mu=100, sigma=15)
rss.one_proportion_z_test(p_hat=0.6, n=50, p0=0.5)

# Non-parametric
rss.mann_whitney_u(g1, g2)
rss.wilcoxon_signed_rank(before, after)

# Critical values
rss.critical_value_z(alpha=0.05)           # 1.96 (two-sided)
rss.critical_value_t(alpha=0.05, df=24)
rss.p_value_method(test_statistic=2.1, test_type="two-tailed")
rss.reject_null(p_value=0.03, alpha=0.05)  # True
```

---

## Regression & correlation

```python
# Simple
slope, intercept = rss.linear_regression(x, y)
r2 = rss.r_squared(x, y)
rss.pearson_correlation(x, y)
rss.spearman_correlation(x, y)

# Multiple
r = rss.multiple_regression(X, y)
r["coefficients"], r["r_squared"], r["p_values"]

# Predictions
y_hat = rss.predict(slope, intercept, x_new=5.0)
residuals = rss.calculate_residuals(y, y_hat)

# Assumptions check
rss.check_regression_assumptions(x, y, verbose=True)
```

---

## Causal inference

```python
# Difference-in-Differences
r = rss.difference_in_differences(outcome, post, treated)
r["did_estimate"]   # β₃ from the interaction model
r["p_value"], r["ci"]
result = rss.difference_in_differences_explained(outcome, post, treated)
result.plot()       # 2×2 DiD diagram

# Regression Discontinuity
r = rss.regression_discontinuity(outcome, running_var, cutoff=65)
r["effect"], r["p_value"]

# Synthetic Control
r = rss.synthetic_control(y_treated, Y_controls, n_pre=20)
r["weights"]        # donor weights (≥ 0, sum to 1)
r["effect"]         # post-treatment effect

# Panel Fixed Effects
r = rss.panel_fixed_effects(outcome, predictors, entity)
r["coefficients"], r["r_squared"]
```

---

## Survival analysis

```python
durations      = [2, 3, 5, 7, 11, 4, 8, 10, 6, 14]
event_observed = [1, 1, 1, 1,  0, 1, 0,  1, 1,  0]  # 0 = censored

# Kaplan-Meier
r = rss.kaplan_meier(durations, event_observed)
r["median_survival"]          # time where S(t) = 0.5
r["survival_prob"]            # array
r["ci_lower"], r["ci_upper"]  # Greenwood confidence bands

result = rss.kaplan_meier_explained(durations, event_observed)
result.median_survival
result.plot()   # step-function curve with CI

# Parametric fit
r = rss.fit_parametric_survival(durations, event_observed, distribution="weibull")
r["params"], r["aic"], r["survival_fn"]  # survival_fn(t) → float

# Compare all distributions (AIC-ranked)
ranked = rss.compare_survival_models(durations, event_observed)
ranked[0]["distribution"]        # best-fitting model
ranked[0]["survival_fn"](t=30)   # P(survive to day 30)
```

---

## Market basket analysis

```python
transactions = [
    ["bread", "milk", "eggs"],
    ["bread", "butter"],
    ["milk", "diapers", "beer"],
]

# Step 1: encode
matrix, items = rss.encode_transactions(transactions)

# Step 2: frequent itemsets
itemsets = rss.frequent_itemsets(matrix, items, min_support=0.3)
# [{"itemset": {"milk"}, "support": 0.67}, ...]

# Step 3: rules
rules = rss.association_rules(itemsets, min_confidence=0.5, min_lift=1.0)
for r in rules:
    print(f"{r['antecedent']} → {r['consequent']}  lift={r['lift']:.2f}")
```

---

## Spatial statistics

```python
x = np.random.uniform(0, 100, 80)
y = np.random.uniform(0, 100, 80)
values = np.sin(x / 20) + np.random.normal(0, 0.3, 80)

# Moran's I
r = rss.morans_i(x, y, values, distance_threshold=20)
r["moran_i"]          # +1=clustered, 0=random, -1=dispersed
r["z_score"], r["p_value"]
result = rss.morans_i_explained(x, y, values, distance_threshold=20)
result.plot()         # spatial scatter coloured by value

# Experimental variogram
vario = rss.compute_variogram(x, y, values, n_lags=15)
vario["lags"], vario["gamma"], vario["n_pairs"]

# Fit variogram model
fit = rss.fit_variogram(vario["lags"], vario["gamma"], model="spherical")
fit["nugget"], fit["sill"], fit["range_param"], fit["rmse"]
fit["model_fn"](15.0)   # γ at h=15

# Three model families
rss.variogram_spherical(lags, nugget=0.5, sill=8.0, range_param=20.0)
rss.variogram_exponential(lags, nugget=0.5, sill=8.0, range_param=20.0)
rss.variogram_gaussian(lags, nugget=0.5, sill=8.0, range_param=20.0)
```

---

## Time series

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Smoothing
rss.moving_average(data, window_size=3)                         # simple MA
rss.moving_average(data, window_size=3, method="exponential")   # EMA
rss.exponential_smoothing(data, alpha=0.3)                      # SES list

# Holt's double exponential smoothing (level + trend)
r = rss.double_exponential_smoothing(data, alpha=0.8, beta=0.2)
r["smoothed"], r["level"], r["trend"]

# Rolling statistics (min_periods=1)
r = rss.rolling_statistics(data, window=3)
r["mean"], r["std"], r["minimum"], r["maximum"], r["expanding_mean"]

# Autocorrelation
rss.autocorrelation(data, max_lag=5)
rss.partial_autocorrelation(data, max_lag=5)

# Trend and structure
rss.linear_trend(data)               # (slope, intercept, r²)
rss.detrend(data, method="linear")
rss.seasonal_decompose(data * 4, period=10)  # (trend, seasonal, residual)
rss.difference(data, lag=1, order=1)

# Change points (binary segmentation)
r = rss.detect_change_points(data, n_breaks=1, min_size=5)
r["change_points"]     # [index where mean shifts]
r["segment_means"]     # mean of each segment
r["rss_reduction"]     # total variance reduction
result = rss.detect_change_points_explained(data)
result.plot()          # series + break lines

# Forecast accuracy
rss.mean_absolute_scaled_error(actual, forecast)  # < 1.0 beats naïve
```

---

## Bayesian statistics

```python
# Beta-Binomial conjugate (A/B tests)
post_a, post_b = rss.beta_binomial_update(
    prior_alpha=1, prior_beta=1, successes=7, trials=10
)

# Normal-Normal conjugate
post_mean, post_var = rss.normal_normal_update(
    prior_mean=10, prior_variance=4, data=data, data_variance=1
)

# Credible intervals
lower, upper = rss.credible_interval(
    "beta", {"alpha": 8, "beta": 4}, credibility=0.95
)

# Highest Density Interval
samples = np.random.normal(0, 1, 1000).tolist()
lo, hi = rss.highest_density_interval(samples)

# Bayes Factor  (> 10 = strong evidence for H1)
bf = rss.bayes_factor(likelihood_h1=0.8, likelihood_h0=0.2)
```

---

## Resampling & bootstrapping

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Bootstrap CI
r = rss.bootstrap(data, np.mean, n_iterations=1000, confidence_level=0.95)
r["statistic"], r["confidence_interval"]

# Bootstrap hypothesis test
r = rss.bootstrap_hypothesis_test(
    g1, g2,
    test_statistic=lambda x, y: np.mean(x) - np.mean(y),
    n_iterations=1000,
)

# Permutation test
r = rss.permutation_test(
    g1, g2,
    test_statistic=lambda x, y: np.mean(x) - np.mean(y),
    n_permutations=1000, alternative="two-sided",
)

# Jackknife
r = rss.jackknife(data, np.mean)
r["bias"], r["std_error"]

# Cross-validation
r = rss.cross_validate(X, y, model_fn, k_folds=5)
r["mean_score"], r["fold_scores"]
```

---

## Power analysis

```python
# How many subjects do I need?
rss.power_t_test(delta=0.5, power=0.8, sig_level=0.05)["n"]

# What power do I have?
rss.power_t_test(n=50, delta=0.5, sig_level=0.05)["power"]

# What effect can I detect?
rss.power_t_test(n=50, power=0.8, sig_level=0.05)["delta"]

# Other tests
rss.power_proportion_test(p1=0.6, p2=0.5, power=0.8)
rss.power_anova(n_groups=3, effect_size=0.25, power=0.8)
rss.power_correlation(r=0.3, power=0.8)

# Cross-test summary
rss.sample_size_summary(effect_size=0.5, power=0.8)
```

---

## Effect sizes

```python
# Continuous outcomes
rss.cohens_d(g1, g2)           # standardised mean difference
rss.hedges_g(g1, g2)           # bias-corrected d
rss.glass_delta(g1, g2)        # d using control group SD
rss.interpret_effect_size(d, "d")   # "small" / "medium" / "large"

# ANOVA
rss.eta_squared(groups)
rss.omega_squared(groups)

# Categorical
rss.cramers_v(contingency_table)
rss.phi_coefficient(table_2x2)

# Binary outcomes
rss.odds_ratio(table_2x2)
rss.relative_risk(table_2x2)

# Proportions
rss.cohens_h(p1=0.7, p2=0.5)
```

---

## Utilities

```python
# Assumption checking
rss.check_t_test_assumptions(data, verbose=True)
rss.check_regression_assumptions(x, y, verbose=True)

# Glossary
rss.lookup("p-value")   # definition + context
rss.GLOSSARY            # full dict of 200+ terms

# Verbose step-by-step mode
from real_simple_stats.verbose_stats import t_test_verbose, regression_verbose
t_test_verbose(data, mu_null=100, verbose=True)
regression_verbose(x, y, verbose=True)

# Multivariate
rss.pca(X, n_components=2)
rss.mahalanobis_distance(X)
rss.factor_analysis(X, n_factors=2)

# CLI
# rss-calc describe --data 12 15 18 20 22 25
# rss-calc prob --type binomial --n 10 --k 3 --p 0.5
# rss-calc test --type t-test --data 23 25 28 30 32 --mu 25
```

---

## Effect size interpretation

| Measure | Negligible | Small | Medium | Large |
|---|---|---|---|---|
| Cohen's d | < 0.2 | 0.2–0.5 | 0.5–0.8 | > 0.8 |
| Pearson r | < 0.1 | 0.1–0.3 | 0.3–0.5 | > 0.5 |
| η² (eta-squared) | < 0.01 | 0.01–0.06 | 0.06–0.14 | > 0.14 |
| Cramér's V | < 0.1 | 0.1–0.3 | 0.3–0.5 | > 0.5 |
| MASE | > 1 (worse than naïve) | — | — | 0 (perfect) |

| Bayes Factor | Evidence |
|---|---|
| < 1 | For H₀ |
| 1–3 | Anecdotal for H₁ |
| 3–10 | Moderate for H₁ |
| 10–30 | Strong for H₁ |
| > 30 | Very strong for H₁ |

---

**Full docs**: see `docs/` · **Decision guide**: `docs/WHICH_TEST.md` · **API**: `help(rss.<function>)`
