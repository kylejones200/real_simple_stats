# Real Simple Stats - Quick Reference Guide v0.3.0

## 🚀 Quick Start

```python
import real_simple_stats as rss
import numpy as np
```

---

## 📊 Time Series Analysis

```python
# Moving averages
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sma = rss.moving_average(data, window_size=3, method='simple')
ema = rss.moving_average(data, window_size=3, method='exponential')

# Autocorrelation
acf = rss.autocorrelation(data, max_lag=5)
pacf = rss.partial_autocorrelation(data, max_lag=5)

# Trend analysis
slope, intercept, r2 = rss.linear_trend(data)
detrended = rss.detrend(data, method='linear')

# Seasonal decomposition
trend, seasonal, residual = rss.seasonal_decompose(data * 3, period=10)

# Differencing
diff_data = rss.difference(data, lag=1, order=1)
```

---

## 🔢 Multivariate Analysis

```python
# Multiple regression
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [2, 4, 5, 4, 5]
result = rss.multiple_regression(X, y)
# Access: result['coefficients'], result['r_squared'], result['p_value']

# Principal Component Analysis
pca_result = rss.pca(X, n_components=2)
# Access: pca_result['components'], pca_result['explained_variance_ratio']

# Factor Analysis
fa_result = rss.factor_analysis(X, n_factors=2)
# Access: fa_result['loadings'], fa_result['communalities']

# Mahalanobis Distance
distances = rss.mahalanobis_distance(X)
```

---

## 🎲 Bayesian Statistics

```python
# Beta-Binomial conjugate prior
post_alpha, post_beta = rss.beta_binomial_update(
    prior_alpha=1, prior_beta=1, successes=7, trials=10
)

# Normal-Normal conjugate prior
data = [10.5, 11.2, 9.8, 10.1]
post_mean, post_var = rss.normal_normal_update(
    prior_mean=10, prior_variance=4, data=data, data_variance=1
)

# Credible intervals
lower, upper = rss.credible_interval(
    'beta', {'alpha': 8, 'beta': 4}, credibility=0.95
)

# Highest Density Interval (from samples)
samples = np.random.normal(0, 1, 1000).tolist()
hdi_lower, hdi_upper = rss.highest_density_interval(samples)

# Bayes Factor
bf = rss.bayes_factor(likelihood_h1=0.8, likelihood_h0=0.2)

# Posterior predictive
predictions = rss.posterior_predictive(
    'beta_binomial', {'alpha': 8, 'beta': 4, 'n': 10}, n_samples=1000
)
```

---

## 🔄 Resampling Methods

```python
# Bootstrap
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
boot_result = rss.bootstrap(data, np.mean, n_iterations=1000, confidence_level=0.95)
# Access: boot_result['statistic'], boot_result['confidence_interval']

# Bootstrap hypothesis test
group1 = [1, 2, 3, 4, 5]
group2 = [3, 4, 5, 6, 7]
boot_test = rss.bootstrap_hypothesis_test(
    group1, group2, lambda x, y: np.mean(x) - np.mean(y), n_iterations=1000
)

# Permutation test
perm_result = rss.permutation_test(
    group1, group2, lambda x, y: np.mean(x) - np.mean(y),
    n_permutations=1000, alternative='two-sided'
)

# Jackknife
jack_result = rss.jackknife(data, np.mean)
# Access: jack_result['bias'], jack_result['std_error']

# Cross-validation
X = [[i] for i in range(20)]
y = [2*i + 1 for i in range(20)]
def simple_model(X_train, y_train, X_test):
    return [np.mean(y_train)] * len(X_test)

cv_result = rss.cross_validate(X, y, simple_model, k_folds=5)
# Access: cv_result['mean_score'], cv_result['fold_scores']

# Stratified split
X = [[i] for i in range(100)]
y = [0] * 50 + [1] * 50
X_train, X_test, y_train, y_test = rss.stratified_split(X, y, test_size=0.2)
```

---

## 📏 Effect Sizes

```python
# Cohen's d (for t-tests)
group1 = [1, 2, 3, 4, 5]
group2 = [3, 4, 5, 6, 7]
d = rss.cohens_d(group1, group2, pooled=True)
interpretation = rss.interpret_effect_size(d, 'd')

# Hedges' g (bias-corrected)
g = rss.hedges_g(group1, group2)

# Glass's delta
delta = rss.glass_delta(group1, group2)

# Eta-squared (for ANOVA)
groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
eta2 = rss.eta_squared(groups)
partial_eta2 = rss.partial_eta_squared(groups)
omega2 = rss.omega_squared(groups)

# Cramér's V (for chi-square)
table = [[10, 20], [30, 40]]
v = rss.cramers_v(table)

# Phi coefficient (2x2 tables)
phi = rss.phi_coefficient(table)

# Odds ratio
or_value, ci = rss.odds_ratio(table)

# Relative risk
rr, ci = rss.relative_risk(table)

# Cohen's h (for proportions)
h = rss.cohens_h(p1=0.7, p2=0.5)
```

---

## ⚡ Power Analysis

```python
# T-test power analysis
# Calculate sample size
result = rss.power_t_test(delta=0.5, power=0.8, sig_level=0.05)
n_required = result['n']

# Calculate power
result = rss.power_t_test(n=50, delta=0.5, sig_level=0.05)
power = result['power']

# Calculate detectable effect
result = rss.power_t_test(n=50, power=0.8, sig_level=0.05)
min_effect = result['delta']

# Proportion test power analysis
result = rss.power_proportion_test(p1=0.6, p2=0.5, power=0.8)

# ANOVA power analysis
result = rss.power_anova(n_groups=3, effect_size=0.25, power=0.8)

# Correlation power analysis
result = rss.power_correlation(r=0.3, power=0.8)

# Minimum detectable effect
mde = rss.minimum_detectable_effect(n=50, power=0.8, test_type='t-test')

# Sample size summary
summary = rss.sample_size_summary(effect_size=0.5, power=0.8)
```

---

## 📖 Interpretation Guidelines

### Cohen's d
- **< 0.2**: Negligible
- **0.2 - 0.5**: Small
- **0.5 - 0.8**: Medium
- **> 0.8**: Large

### Correlation (r)
- **< 0.1**: Negligible
- **0.1 - 0.3**: Small
- **0.3 - 0.5**: Medium
- **> 0.5**: Large

### Eta-squared
- **< 0.01**: Negligible
- **0.01 - 0.06**: Small
- **0.06 - 0.14**: Medium
- **> 0.14**: Large

### Cramér's V
- **< 0.1**: Negligible
- **0.1 - 0.3**: Small
- **0.3 - 0.5**: Medium
- **> 0.5**: Large

### Bayes Factor
- **< 1**: Evidence for H0
- **1 - 3**: Anecdotal evidence for H1
- **3 - 10**: Moderate evidence for H1
- **10 - 30**: Strong evidence for H1
- **> 30**: Very strong evidence for H1

---

## 🎯 Common Workflows

### Study Design
```python
# 1. Determine effect size from pilot or literature
effect_size = 0.5

# 2. Calculate required sample size
result = rss.power_t_test(delta=effect_size, power=0.8, sig_level=0.05)
print(f"Need {result['n']} participants per group")

# 3. Get summary for multiple test types
summary = rss.sample_size_summary(effect_size, power=0.8)
```

### Data Analysis
```python
# 1. Perform statistical test
t_stat, p_value = rss.one_sample_t_test(data, mu=5)

# 2. Calculate effect size
d = rss.cohens_d(group1, group2)
interpretation = rss.interpret_effect_size(d, 'd')

# 3. Bootstrap confidence interval
boot_result = rss.bootstrap(data, np.mean, n_iterations=1000)
ci = boot_result['confidence_interval']

print(f"Effect size: {d:.3f} ({interpretation})")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### Time Series Analysis
```python
# 1. Check for trend
slope, intercept, r2 = rss.linear_trend(data)

# 2. Detrend if necessary
if r2 > 0.5:
    detrended = rss.detrend(data, method='linear')
else:
    detrended = data

# 3. Check autocorrelation
acf = rss.autocorrelation(detrended, max_lag=10)

# 4. Apply moving average for smoothing
smoothed = rss.moving_average(detrended, window_size=5)
```

---

## 💡 Tips & Best Practices

1. **Always check assumptions** before applying statistical tests
2. **Report effect sizes** alongside p-values for practical significance
3. **Use bootstrap** when sample size is small or distribution is unknown
4. **Cross-validate** models to avoid overfitting
5. **Calculate power** before collecting data to ensure adequate sample size
6. **Use Bayesian methods** when you have prior information
7. **Check autocorrelation** in time series before applying standard tests
8. **Interpret results** in context - statistical significance ≠ practical importance

---

## 🔗 See Also

- **Full Documentation**: [ADVANCED_FEATURES_SUMMARY.md](ADVANCED_FEATURES_SUMMARY.md)
- **API Reference**: Use `help(rss.function_name)` for detailed documentation
- **Examples**: Check docstrings for working code examples
- **Tests**: See `tests/` directory for more usage examples

---

**Version**: 0.3.0
**Last Updated**: 2025
**License**: MIT
