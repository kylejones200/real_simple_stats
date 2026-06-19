# Real Simple Stats

[![PyPI version](https://badge.fury.io/py/real-simple-stats.svg)](https://badge.fury.io/py/real-simple-stats)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/kylejones200/real_simple_stats/workflows/CI/badge.svg)](https://github.com/kylejones200/real_simple_stats/actions)

**A statistics library that teaches while it computes.**

Most libraries hand you a number and walk away. Real Simple Stats hands you the number *and* explains what it means, what assumptions it rests on, what it does not license you to claim, and what to do next.

---

## The difference

```python
import real_simple_stats as rss

result = rss.one_sample_t_test_explained([5.2, 5.4, 5.1, 5.5, 5.3], mu=5.0)

result.p_value    # 0.0421  → use it as data, just like any other library
result.plot()     # → the p-value drawn as a shaded tail area

print(result)     # → the full narrative below
```

```
=== One-Sample t-Test ===

QUESTION
  Is the true population mean different enough from 5.0 that we shouldn't
  chalk the gap up to random sampling?

RESULT
  n                     5
  sample mean           5.3000
  t statistic           3.0732
  p_value               0.0374
  ci                    (5.0284, 5.5716)
  effect_size           1.3416
  decision              Reject H₀

WHAT THE TEST IS DOING
  The t statistic is a signal-to-noise ratio. The signal is how far the
  sample mean (5.3) sits from the hypothesized mean (5.0); the noise is the
  standard error (0.0976). Here t = 3.073, so the gap is about 3.07 standard
  errors wide.

ASSUMPTIONS
  Small sample (n=5), but the data look roughly normal, so the t-test is
  reasonable.

INTERPRETATION
  At α = 0.05, the result is statistically significant (p = 0.0374). We
  reject H₀. The sample mean is 5.3, and a 95% confidence interval for the
  true mean runs from 5.028 to 5.572. The effect size (Cohen's d = 1.342)
  is large.

WHAT THIS DOES *NOT* MEAN
  • p = 0.0374 is NOT the probability that H₀ is true. It is the probability
    of data this extreme assuming H₀ is true.
  • Statistical significance is not practical importance.
  • α = 0.05 is a convention, not a bright line in nature.

NEXT STEPS
  → Report the effect size and confidence interval alongside p.
  → Call result.plot() to see the p-value as a shaded tail area.
```

Every explained function works this way. The numbers are still accessible as plain attributes. The narrative is the bonus.

---

## Install

```bash
pip install real-simple-stats
```

**Requirements**: Python 3.12+, NumPy, SciPy, Matplotlib. No pandas, statsmodels, or scikit-learn required.

---

## Contents

- [Self-explaining results](#self-explaining-results)
- [Descriptive statistics](#descriptive-statistics)
- [Probability & distributions](#probability--distributions)
- [Hypothesis testing](#hypothesis-testing)
- [Regression & correlation](#regression--correlation)
- [Causal inference](#causal-inference)
- [Survival analysis](#survival-analysis)
- [Market basket analysis](#market-basket-analysis)
- [Spatial statistics](#spatial-statistics)
- [Time series](#time-series)
- [Bayesian statistics](#bayesian-statistics)
- [Resampling & bootstrapping](#resampling--bootstrapping)
- [Power analysis](#power-analysis)
- [Effect sizes](#effect-sizes)
- [Utilities](#utilities)

---

## Self-explaining results

Seven tests have an `_explained` variant that returns an `ExplainedResult`. All seven work identically: use them as data objects or let them teach.

| Function | Test | What `result.plot()` shows |
|---|---|---|
| `one_sample_t_test_explained` | One-sample t-test | p-value as shaded tail area |
| `one_way_anova_explained` | One-way ANOVA | Box plots of each group |
| `chi_square_independence_explained` | Chi-square test of independence | Observed vs. expected counts |
| `difference_in_differences_explained` | Difference-in-differences | 2×2 DiD diagram with counterfactual |
| `kaplan_meier_explained` | Kaplan-Meier survival curve | Step-function curve with Greenwood CI |
| `morans_i_explained` | Moran's I spatial autocorrelation | Spatial scatter coloured by value |
| `detect_change_points_explained` | Binary segmentation change points | Series with break lines + segment means |

Every `ExplainedResult` carries:
- **question** — the real-world question the test answers
- **values** — headline numbers, also accessible as attributes (`result.p_value`, `result.f_stat`, …)
- **intuition** — how the machinery works in plain language, with your specific values baked in
- **assumptions** — what must hold and whether it's a concern for your data
- **interpretation** — what this specific result means
- **caveats** — the misconception guard (what this result does NOT license you to say)
- **next steps** — concrete follow-up actions

```python
import real_simple_stats as rss
import numpy as np

rng = np.random.default_rng(0)
g1 = rng.normal(0, 1, 40)
g2 = rng.normal(2, 1, 40)
g3 = rng.normal(4, 1, 40)

result = rss.one_way_anova_explained(g1, g2, g3)

result.f_stat        # 154.7
result.eta_squared   # 0.80
result.reject_null   # True
result.plot()        # box plots of the three groups
print(result)        # full narrative with misconception guards
```

---

## Descriptive statistics

```python
import real_simple_stats as rss

data = [12, 15, 18, 20, 22, 25, 28, 30]

rss.mean(data)                  # 21.25
rss.median(data)                # 21.0
rss.sample_std_dev(data)        # 6.41
rss.five_number_summary(data)   # {'min': 12, 'Q1': 16.5, 'median': 21.0, 'Q3': 26.5, 'max': 30}
rss.iqr(data)                   # 10.0
rss.coefficient_of_variation(data)  # 30.2 (percent)
rss.skewness(data)
rss.kurtosis(data)
rss.detect_outliers_iqr(data)   # returns list of outlier values
rss.frequency_table(data)       # value → count mapping
```

---

## Probability & distributions

```python
# Basic probability
rss.simple_probability(favorable=3, total=10)      # 0.3
rss.joint_probability(0.4, 0.3)                    # 0.12
rss.conditional_probability(0.12, 0.3)             # 0.4
rss.bayes_theorem(prior=0.01, sensitivity=0.95, specificity=0.90)

# Combinatorics
rss.combinations(n=10, k=3)      # 120
rss.permutations(n=10, k=3)      # 720

# Binomial
rss.binomial_probability(n=10, k=3, p=0.5)    # 0.1172
rss.binomial_mean(n=10, p=0.5)                # 5.0
rss.binomial_cdf(n=10, k=3, p=0.5)           # cumulative through k=3

# Normal
rss.normal_pdf(x=1.0, mean=0, std_dev=1)     # 0.2420
rss.normal_cdf(x=1.96, mean=0, std_dev=1)    # 0.9750
rss.z_score(value=75, mean=70, std_dev=10)   # 0.5

# Poisson, geometric, exponential
rss.poisson_probability(k=3, lam=2.5)
rss.geometric_probability(k=4, p=0.3)
rss.exponential_probability(x=2.0, lam=0.5)
```

---

## Hypothesis testing

```python
# One-sample t-test
rss.one_sample_t_test(data, mu=5.0)
# returns (t_statistic, p_value)

# Self-explaining variant
result = rss.one_sample_t_test_explained(data, mu=5.0)
result.p_value; result.ci; print(result)

# Two-sample and paired t-tests
rss.two_sample_t_test(group1, group2, equal_var=True)
rss.paired_t_test(before, after)

# One-way ANOVA
r = rss.one_way_anova(g1, g2, g3)
# r["f_stat"], r["p_value"], r["eta_squared"], r["reject_null"]

result = rss.one_way_anova_explained(g1, g2, g3)
# Adds: intuition, misconception guard, box plot

# Chi-square independence
r = rss.chi_square_independence([[40, 5], [5, 40]])
# r["chi2"], r["p_value"], r["cramers_v"], r["reject_null"]

result = rss.chi_square_independence_explained([[40, 5], [5, 40]])
# Adds: Cramér's V narrative, observed vs expected bar chart

# Z-test and proportion tests
rss.z_test(data, mu=100, sigma=15)
rss.one_proportion_z_test(p_hat=0.6, n=50, p0=0.5)

# Non-parametric
rss.mann_whitney_u(group1, group2)
rss.wilcoxon_signed_rank(before, after)
```

---

## Regression & correlation

```python
# Simple linear regression
slope, intercept = rss.linear_regression(x, y)
r2 = rss.r_squared(x, y)
rss.pearson_correlation(x, y)      # correlation coefficient
rss.spearman_correlation(x, y)     # rank correlation

# Multiple regression
result = rss.multiple_regression(X, y)
result["coefficients"]
result["r_squared"]
result["p_values"]

# Predictions and residuals
y_hat = rss.predict(slope, intercept, x_new)
residuals = rss.calculate_residuals(y, y_hat)

# Diagnostics
rss.check_regression_assumptions(x, y, verbose=True)
```

---

## Causal inference

Quasi-experimental designs for situations where randomized experiments aren't possible.

```python
# Difference-in-differences
# Isolates treatment effect by comparing treated vs. control group changes
r = rss.difference_in_differences(outcome, post, treated)
r["did_estimate"]   # causal effect estimate
r["p_value"]
r["ci"]             # 95% confidence interval

result = rss.difference_in_differences_explained(outcome, post, treated)
result.plot()       # 2×2 DiD diagram with counterfactual line
print(result)       # parallel trends caveat, interpretation, next steps

# Regression discontinuity
# Compares units just above/below a cutoff threshold
r = rss.regression_discontinuity(outcome, running_var, cutoff=65)
r["effect"]         # local average treatment effect at the cutoff
r["p_value"]

# Synthetic control
# Builds a weighted counterfactual from donor control units
r = rss.synthetic_control(y_treated, Y_controls, n_pre=20)
r["weights"]        # donor weights (sum to 1, non-negative)
r["effect"]         # post-treatment treatment effect

# Panel fixed effects
# Within-entity demeaning to absorb entity-level confounders
r = rss.panel_fixed_effects(outcome, predictors, entity)
r["coefficients"]
r["r_squared"]
```

---

## Survival analysis

Time-to-event analysis for churn, failure, clinical trials, and any right-censored data.

```python
durations = [2, 3, 5, 7, 11, 4, 8, 10, 6, 14]
observed  = [1, 1, 1, 1,  0, 1, 0,  1, 1,  0]  # 0 = censored

# Kaplan-Meier (non-parametric)
r = rss.kaplan_meier(durations, observed)
r["median_survival"]   # time at which S(t) = 0.5
r["survival_prob"]     # array of S(t) values
r["ci_lower"], r["ci_upper"]   # Greenwood confidence bands

result = rss.kaplan_meier_explained(durations, observed)
result.median_survival
result.plot()      # step-function curve with confidence band
print(result)      # censoring caveat, interpretation, follow-up suggestions

# Parametric models (MLE fit)
r = rss.fit_parametric_survival(durations, observed, distribution="weibull")
r["params"]           # fitted shape and scale parameters
r["survival_fn"]      # callable: S(t) → float
r["aic"]              # model selection criterion

# Compare all four distributions — returns AIC-ranked list
ranked = rss.compare_survival_models(durations, observed)
ranked[0]["distribution"]   # best-fitting model
ranked[0]["aic"]
```

---

## Market basket analysis

Association rule mining for transaction data — discover which products co-occur.

```python
transactions = [
    ["bread", "milk", "eggs"],
    ["bread", "butter"],
    ["milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "beer"],
    ["milk", "eggs"],
]

# Step 1: encode transactions into a binary matrix
matrix, items = rss.encode_transactions(transactions)

# Step 2: find frequent itemsets
itemsets = rss.frequent_itemsets(matrix, items, min_support=0.3)
# [{"itemset": {"milk"}, "support": 0.8}, {"itemset": {"bread", "milk"}, ...}, ...]

# Step 3: generate association rules
rules = rss.association_rules(itemsets, min_confidence=0.6, min_lift=1.0)
for rule in rules:
    print(f"{rule['antecedent']} → {rule['consequent']}")
    print(f"  support={rule['support']:.2f}, confidence={rule['confidence']:.2f}, lift={rule['lift']:.2f}")
```

---

## Spatial statistics

Measure and model spatial autocorrelation — do similar values cluster together in space?

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 100, 80)
y = rng.uniform(0, 100, 80)
values = np.sin(x / 20) + rng.normal(0, 0.3, 80)   # spatially structured

# Moran's I — global spatial autocorrelation
r = rss.morans_i(x, y, values, distance_threshold=20)
r["moran_i"]        # +1 = clustered, 0 = random, -1 = dispersed
r["z_score"]
r["p_value"]
r["interpretation"]

result = rss.morans_i_explained(x, y, values, distance_threshold=20)
result.plot()       # spatial scatter coloured by value
print(result)       # local vs global caveat, weight matrix sensitivity warning

# Experimental variogram
# Quantifies how spatial autocorrelation decays with distance
vario = rss.compute_variogram(x, y, values, n_lags=15)
vario["lags"]      # lag distances
vario["gamma"]     # semivariance at each lag
vario["n_pairs"]   # number of point pairs per bin

# Fit a variogram model
fit = rss.fit_variogram(vario["lags"], vario["gamma"], model="spherical")
fit["nugget"]       # variance at zero distance
fit["sill"]         # total variance
fit["range_param"]  # distance at which autocorrelation effectively vanishes
fit["model_fn"]     # callable: h → γ(h)
fit["rmse"]         # fit quality

# Three model families
rss.variogram_spherical(lags, nugget, sill, range_param)
rss.variogram_exponential(lags, nugget, sill, range_param)
rss.variogram_gaussian(lags, nugget, sill, range_param)
```

---

## Time series

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Classic smoothing
rss.moving_average(data, window_size=3)                    # simple MA
rss.moving_average(data, window_size=3, method="weighted") # WMA

# Exponential smoothing — no seasonal component
rss.exponential_smoothing(data, alpha=0.3)   # list of smoothed values

# Holt's double exponential smoothing — handles linear trends
r = rss.double_exponential_smoothing(data, alpha=0.8, beta=0.2)
r["smoothed"]   # in-sample fitted values
r["level"]      # level component at each step
r["trend"]      # trend component at each step

# Rolling statistics
r = rss.rolling_statistics(data, window=3)
r["mean"]           # rolling mean
r["std"]            # rolling std (min_periods=1 behaviour)
r["minimum"]
r["maximum"]
r["expanding_mean"] # cumulative mean

# Autocorrelation and partial autocorrelation
rss.autocorrelation(data, max_lag=5)
rss.partial_autocorrelation(data, max_lag=5)

# Trend and decomposition
rss.linear_trend(data)                         # (slope, intercept, r²)
rss.detrend(data, method="linear")
rss.seasonal_decompose(data * 4, period=10)    # (trend, seasonal, residual)
rss.difference(data, lag=1, order=1)

# Change point detection (binary segmentation)
r = rss.detect_change_points(data, n_breaks=1, min_size=5)
r["change_points"]   # list of 0-based indices where the mean shifts
r["segment_means"]   # mean of each segment
r["rss_reduction"]   # total variance reduction (higher = stronger evidence)

result = rss.detect_change_points_explained([0]*20 + [5]*20, n_breaks=1)
result.plot()        # time series with break lines and segment means

# Forecast accuracy
rss.mean_absolute_scaled_error(actual, forecast)
# < 1.0 means your model beats a naïve lag-1 forecast
```

---

## Bayesian statistics

```python
# Beta-Binomial conjugate update (e.g. A/B testing)
post_alpha, post_beta = rss.beta_binomial_update(
    prior_alpha=1, prior_beta=1, successes=7, trials=10
)

# Normal-Normal conjugate update
post_mean, post_var = rss.normal_normal_update(
    prior_mean=10, prior_variance=4, data=data, data_variance=1
)

# Credible intervals and HDI
lower, upper = rss.credible_interval("beta", {"alpha": 8, "beta": 4}, credibility=0.95)
hdi_lo, hdi_hi = rss.highest_density_interval(samples)

# Bayes Factor
bf = rss.bayes_factor(likelihood_h1=0.8, likelihood_h0=0.2)
# > 10 → strong evidence for H1; < 1 → evidence for H0
```

---

## Resampling & bootstrapping

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Bootstrap confidence interval
result = rss.bootstrap(data, np.mean, n_iterations=1000, confidence_level=0.95)
result["statistic"]           # bootstrap estimate
result["confidence_interval"] # (lower, upper)

# Bootstrap hypothesis test
result = rss.bootstrap_hypothesis_test(
    group1, group2,
    test_statistic=lambda x, y: np.mean(x) - np.mean(y),
    n_iterations=1000
)

# Permutation test
result = rss.permutation_test(
    group1, group2,
    test_statistic=lambda x, y: np.mean(x) - np.mean(y),
    n_permutations=1000, alternative="two-sided"
)

# Jackknife
result = rss.jackknife(data, np.mean)
result["bias"]
result["std_error"]

# Cross-validation
result = rss.cross_validate(X, y, model_fn, k_folds=5)
result["mean_score"]
result["fold_scores"]
```

---

## Power analysis

```python
# How many subjects do I need?
result = rss.power_t_test(delta=0.5, power=0.8, sig_level=0.05)
result["n"]       # required per group

# What power do I have with my current n?
result = rss.power_t_test(n=50, delta=0.5, sig_level=0.05)
result["power"]

# What is the smallest effect I can detect?
result = rss.power_t_test(n=50, power=0.8, sig_level=0.05)
result["delta"]

# Proportion tests, ANOVA, correlation
rss.power_proportion_test(p1=0.6, p2=0.5, power=0.8)
rss.power_anova(n_groups=3, effect_size=0.25, power=0.8)
rss.power_correlation(r=0.3, power=0.8)

# Summary across test types at once
rss.sample_size_summary(effect_size=0.5, power=0.8)
```

---

## Effect sizes

```python
# Cohen's d — standardised mean difference
rss.cohens_d(group1, group2)
rss.interpret_effect_size(d, "d")   # "small" / "medium" / "large"

# Variants
rss.hedges_g(group1, group2)        # bias-corrected
rss.glass_delta(group1, group2)     # when control SD is the reference

# ANOVA
rss.eta_squared(groups)
rss.omega_squared(groups)           # less biased than η²

# Categorical
rss.cramers_v(contingency_table)    # 0 → no association, 1 → perfect
rss.phi_coefficient(table_2x2)

# Binary outcomes
rss.odds_ratio(table_2x2)
rss.relative_risk(table_2x2)

# Proportions
rss.cohens_h(p1=0.7, p2=0.5)
```

---

## Utilities

### Assumption checking

Always check before running a test.

```python
rss.check_t_test_assumptions(data, verbose=True)
rss.check_regression_assumptions(x, y, verbose=True)
# Returns dict of checks; verbose=True prints a readable report
```

### Glossary

```python
rss.lookup("p-value")       # definition + context
rss.lookup("Type I error")
rss.GLOSSARY                # dict of all 200+ terms
```

### Verbose step-by-step mode

See the arithmetic as it runs — useful for learning and for auditing.

```python
from real_simple_stats.verbose_stats import t_test_verbose, regression_verbose

t_test_verbose(data, mu_null=100, verbose=True)
# Prints: subtract mean, divide by SE, find critical value, compare, decide

regression_verbose(x, y, verbose=True)
# Prints: compute Σxy, Σx², solve for slope and intercept, calculate R²
```

### Command-line interface

```bash
# Descriptive stats
rss-calc describe --data 12 15 18 20 22 25

# Probability
rss-calc prob --type binomial --n 10 --k 3 --p 0.5
rss-calc prob --type normal --x 1.96 --mean 0 --std 1

# Hypothesis test
rss-calc test --type t-test --data 23 25 28 30 32 --mu 25

# Full list
rss-calc --help
```

### Multivariate

```python
rss.multiple_regression(X, y)
rss.pca(X, n_components=2)
result["components"]
result["explained_variance_ratio"]
rss.mahalanobis_distance(X)
```

---

## Interpretation cheat sheet

| Measure | Negligible | Small | Medium | Large |
|---|---|---|---|---|
| Cohen's d | < 0.2 | 0.2–0.5 | 0.5–0.8 | > 0.8 |
| Pearson r | < 0.1 | 0.1–0.3 | 0.3–0.5 | > 0.5 |
| η² (eta-squared) | < 0.01 | 0.01–0.06 | 0.06–0.14 | > 0.14 |
| Cramér's V | < 0.1 | 0.1–0.3 | 0.3–0.5 | > 0.5 |
| MASE | > 1.0 (worse than naïve) | — | — | 0 (perfect) |

| Bayes Factor | Evidence |
|---|---|
| < 1 | Evidence for H₀ |
| 1–3 | Anecdotal for H₁ |
| 3–10 | Moderate for H₁ |
| 10–30 | Strong for H₁ |
| > 30 | Very strong for H₁ |

---

## Choosing the right test

```
Is your outcome continuous?
├── One group → one-sample t-test
├── Two independent groups → two-sample t-test
├── Two paired measurements → paired t-test
├── Three or more groups → one-way ANOVA
└── Did a treatment cause the change?
    ├── Pre/post with a control group → difference-in-differences
    ├── Cutoff-based assignment → regression discontinuity
    └── Panel data → panel fixed effects

Is your outcome a count / category?
├── Goodness-of-fit → chi-square goodness-of-fit
└── Are two variables related? → chi-square independence test

Is your outcome a time-to-event?
├── Describe the survival curve → Kaplan-Meier
└── Find the best-fitting distribution → compare_survival_models()

Is your data spatial?
├── Do values cluster in space? → Moran's I
└── How does similarity decay with distance? → compute_variogram + fit_variogram

Is your data a time series?
├── Smooth or forecast → exponential_smoothing / double_exponential_smoothing
└── Did the mean shift? → detect_change_points
```

---

## Development

```bash
git clone https://github.com/kylejones200/real_simple_stats.git
cd real_simple_stats
pip install -e ".[dev]"
pytest                          # 763 tests
pytest --cov=real_simple_stats  # with coverage
```

---

## Examples

The `examples/` directory has standalone scripts for every module:

```
examples/
├── explained_t_test_demo.py        # ExplainedResult full demo
├── causal_inference_demo.py        # DiD, RDD, synthetic control, panel FE
├── survival_demo.py                # Kaplan-Meier + Weibull fit
├── market_basket_demo.py           # Apriori / association rules
├── recipes/                        # end-to-end analysis workflows
│   ├── compare_two_groups.py
│   ├── hypothesis_testing_workflow.py
│   ├── power_analysis_planning.py
│   └── regression_analysis.py
└── data/                           # sample datasets
```

---

## Contributing

1. Fork, create a branch, make changes
2. Add tests — new functions need coverage
3. `pytest` must pass before opening a PR
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide

---

## License

MIT — see [LICENSE](LICENSE).

---

**PyPI**: [real-simple-stats](https://pypi.org/project/real-simple-stats/) · **Docs**: [real-simple-stats.readthedocs.io](https://real-simple-stats.readthedocs.io/) · **Source**: [github.com/kylejones200/real_simple_stats](https://github.com/kylejones200/real_simple_stats)
