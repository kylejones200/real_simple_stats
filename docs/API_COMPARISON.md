# API Comparison Table - Quick Function Lookup

## Overview
This guide helps you quickly find the Real Simple Stats function you need, with comparisons to similar functions in other popular libraries.

---

## 📊 Descriptive Statistics

| Task | Real Simple Stats | NumPy | SciPy | Pandas | statsmodels |
|------|-------------------|-------|-------|--------|-------------|
| **Mean** | `mean(data)` | `np.mean(data)` | - | `df.mean()` | - |
| **Median** | `median(data)` | `np.median(data)` | - | `df.median()` | - |
| **Mode** | `mode(data)` | - | `stats.mode(data)` | `df.mode()` | - |
| **Std Dev** | `sample_std_dev(data)` | `np.std(data, ddof=1)` | - | `df.std()` | - |
| **Variance** | `sample_variance(data)` | `np.var(data, ddof=1)` | - | `df.var()` | - |
| **Range** | `data_range(data)` | `np.ptp(data)` | - | `df.max() - df.min()` | - |
| **IQR** | `interquartile_range(data)` | `np.percentile(data, 75) - np.percentile(data, 25)` | - | `df.quantile(0.75) - df.quantile(0.25)` | - |
| **5-Number Summary** | `five_number_summary(data)` | - | - | `df.describe()` | - |
| **CV** | `coefficient_of_variation(data)` | - | `stats.variation(data)` | - | - |

**Example:**
```python
import real_simple_stats as rss

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(rss.mean(data))                    # 5.5
print(rss.five_number_summary(data))     # {'min': 1, 'q1': 3.25, ...}
```

---

## 📈 Probability Distributions

### Normal Distribution

| Task | Real Simple Stats | SciPy | NumPy |
|------|-------------------|-------|-------|
| **PDF** | `normal_pdf(x, mu, sigma)` | `stats.norm.pdf(x, mu, sigma)` | - |
| **CDF** | `normal_cdf(x, mu, sigma)` | `stats.norm.cdf(x, mu, sigma)` | - |
| **Inverse CDF** | `normal_ppf(p, mu, sigma)` | `stats.norm.ppf(p, mu, sigma)` | - |
| **Z-score** | `z_score(x, mu, sigma)` | `(x - mu) / sigma` | - |
| **Random samples** | - | `stats.norm.rvs(mu, sigma, size=n)` | `np.random.normal(mu, sigma, n)` |

### Binomial Distribution

| Task | Real Simple Stats | SciPy | NumPy |
|------|-------------------|-------|-------|
| **PMF** | `binomial_probability(n, k, p)` | `stats.binom.pmf(k, n, p)` | - |
| **CDF** | `binomial_cdf(k, n, p)` | `stats.binom.cdf(k, n, p)` | - |
| **Mean** | `binomial_mean(n, p)` | `n * p` | - |
| **Variance** | `binomial_variance(n, p)` | `n * p * (1-p)` | - |

### Other Distributions

| Distribution | Real Simple Stats | SciPy Equivalent |
|--------------|-------------------|------------------|
| **Poisson** | `poisson_pmf(k, lam)` | `stats.poisson.pmf(k, lam)` |
| **Geometric** | `geometric_pmf(k, p)` | `stats.geom.pmf(k, p)` |
| **Exponential** | `exponential_pdf(x, lam)` | `stats.expon.pdf(x, scale=1/lam)` |
| **Negative Binomial** | `negative_binomial_pmf(k, r, p)` | `stats.nbinom.pmf(k, r, p)` |

**Example:**
```python
import real_simple_stats as rss

# Normal distribution
prob = rss.normal_cdf(1.96, 0, 1)  # P(Z ≤ 1.96) ≈ 0.975

# Binomial distribution
prob = rss.binomial_probability(10, 7, 0.5)  # P(X=7) when n=10, p=0.5
```

---

## 🧪 Hypothesis Testing

| Test | Real Simple Stats | SciPy | statsmodels |
|------|-------------------|-------|-------------|
| **One-sample t-test** | `one_sample_t_test(data, mu0)` | `stats.ttest_1samp(data, mu0)` | `sm.stats.ttest_ind(data, mu0)` |
| **Two-sample t-test** | `two_sample_t_test(data1, data2)` | `stats.ttest_ind(data1, data2)` | `sm.stats.ttest_ind(data1, data2)` |
| **Paired t-test** | `paired_t_test(data1, data2)` | `stats.ttest_rel(data1, data2)` | - |
| **Z-test** | `one_sample_z_test(data, mu0, sigma)` | `sm.stats.ztest(data, value=mu0)` | `sm.stats.ztest(data, value=mu0)` |
| **Chi-square test** | `chi_square_statistic(obs, exp)` | `stats.chisquare(obs, exp)` | - |
| **ANOVA** | `one_way_anova(groups)` | `stats.f_oneway(*groups)` | `sm.stats.anova_lm()` |

**Example:**
```python
import real_simple_stats as rss

# One-sample t-test
data = [23, 25, 28, 30, 32]
t_stat, p_value = rss.one_sample_t_test(data, mu0=30)
print(f"t = {t_stat:.3f}, p = {p_value:.3f}")

# Two-sample t-test
group1 = [1, 2, 3, 4, 5]
group2 = [3, 4, 5, 6, 7]
t_stat, p_value = rss.two_sample_t_test(group1, group2)
```

---

## 📉 Regression & Correlation

| Task | Real Simple Stats | SciPy | scikit-learn | statsmodels |
|------|-------------------|-------|--------------|-------------|
| **Pearson correlation** | `pearson_correlation(x, y)` | `stats.pearsonr(x, y)` | - | `sm.stats.correlation()` |
| **Simple linear regression** | `linear_regression(x, y)` | `stats.linregress(x, y)` | `LinearRegression()` | `sm.OLS(y, x)` |
| **R-squared** | `coefficient_of_determination(x, y)` | `linregress(x, y).rvalue**2` | `model.score(X, y)` | `results.rsquared` |
| **Multiple regression** | `multiple_regression(X, y)` | - | `LinearRegression()` | `sm.OLS(y, X)` |
| **Prediction** | `regression_equation(x, slope, intercept)` | `slope * x + intercept` | `model.predict(X)` | `results.predict(X)` |

**Example:**
```python
import real_simple_stats as rss

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Correlation
r = rss.pearson_correlation(x, y)
print(f"Correlation: {r:.3f}")

# Regression
slope, intercept, r_value, p_value, std_err = rss.linear_regression(x, y)
print(f"y = {slope:.2f}x + {intercept:.2f}")

# Prediction
y_pred = rss.regression_equation(6, slope, intercept)
```

---

## 🔄 Time Series Analysis

| Task | Real Simple Stats | pandas | statsmodels |
|------|-------------------|--------|-------------|
| **Moving average** | `moving_average(data, window, 'simple')` | `df.rolling(window).mean()` | - |
| **Exponential MA** | `moving_average(data, window, 'exponential')` | `df.ewm(span=window).mean()` | - |
| **Autocorrelation** | `autocorrelation(data, max_lag)` | `pd.Series(data).autocorr(lag)` | `sm.tsa.acf(data)` |
| **Partial ACF** | `partial_autocorrelation(data, max_lag)` | - | `sm.tsa.pacf(data)` |
| **Linear trend** | `linear_trend(data)` | - | `sm.tsa.deterministic.DeterministicTerm()` |
| **Detrend** | `detrend(data, 'linear')` | - | `sm.tsa.detrend(data)` |
| **Seasonal decompose** | `seasonal_decompose(data, period)` | - | `sm.tsa.seasonal_decompose()` |
| **Differencing** | `difference(data, lag, order)` | `df.diff(lag)` | - |

**Example:**
```python
import real_simple_stats as rss

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Moving average
ma = rss.moving_average(data, window_size=3, method='simple')

# Autocorrelation
acf = rss.autocorrelation(data, max_lag=5)

# Trend analysis
slope, intercept, r2 = rss.linear_trend(data)
```

---

## 🎲 Resampling Methods

| Method | Real Simple Stats | scikit-learn | scipy |
|--------|-------------------|--------------|-------|
| **Bootstrap** | `bootstrap(data, statistic, n_iterations)` | - | - |
| **Bootstrap CI** | Returns `confidence_interval` | - | - |
| **Permutation test** | `permutation_test(data1, data2, statistic)` | `permutation_test()` | - |
| **Jackknife** | `jackknife(data, statistic)` | - | - |
| **Cross-validation** | `cross_validate(X, y, model_fn, k_folds)` | `cross_val_score()` | - |
| **Stratified split** | `stratified_split(X, y, test_size)` | `train_test_split(stratify=y)` | - |

**Example:**
```python
import real_simple_stats as rss
import numpy as np

data = [1, 2, 3, 4, 5]

# Bootstrap confidence interval
result = rss.bootstrap(data, np.mean, n_iterations=1000)
print(f"95% CI: {result['confidence_interval']}")

# Permutation test
group1 = [1, 2, 3, 4, 5]
group2 = [3, 4, 5, 6, 7]
result = rss.permutation_test(group1, group2, 
                               lambda d1, d2: np.mean(d1) - np.mean(d2))
print(f"p-value: {result['p_value']:.3f}")
```

---

## 📊 Effect Sizes

| Effect Size | Real Simple Stats | Other Libraries |
|-------------|-------------------|-----------------|
| **Cohen's d** | `cohens_d(group1, group2)` | `pg.compute_effsize()` (pingouin) |
| **Hedges' g** | `hedges_g(group1, group2)` | `pg.compute_effsize(eftype='hedges')` |
| **Glass's Δ** | `glass_delta(group1, group2)` | - |
| **Eta-squared** | `eta_squared(groups)` | `pg.anova()['np2']` |
| **Partial η²** | `partial_eta_squared(groups)` | - |
| **Omega-squared** | `omega_squared(groups)` | - |
| **Cramér's V** | `cramers_v(contingency_table)` | `scipy.stats.contingency.association()` |
| **Phi coefficient** | `phi_coefficient(table)` | - |
| **Odds ratio** | `odds_ratio(table)` | `statsmodels.stats.contingency_tables` |
| **Relative risk** | `relative_risk(table)` | - |
| **Cohen's h** | `cohens_h(p1, p2)` | - |
| **Interpretation** | `interpret_effect_size(es, measure)` | - |

**Example:**
```python
import real_simple_stats as rss

group1 = [1, 2, 3, 4, 5]
group2 = [3, 4, 5, 6, 7]

# Cohen's d
d = rss.cohens_d(group1, group2)
interpretation = rss.interpret_effect_size(d, 'd')
print(f"Cohen's d = {d:.3f} ({interpretation})")

# Cramér's V for categorical data
table = [[10, 20], [30, 40]]
v = rss.cramers_v(table)
print(f"Cramér's V = {v:.3f}")
```

---

## 🔬 Power Analysis

| Analysis | Real Simple Stats | statsmodels | G*Power |
|----------|-------------------|-------------|---------|
| **t-test power** | `power_t_test(delta, power, sig_level)` | `sm.stats.TTestPower()` | Manual |
| **Proportion test** | `power_proportion_test(p1, p2, power)` | `sm.stats.proportion_effectsize()` | Manual |
| **ANOVA power** | `power_anova(effect_size, k, power)` | `sm.stats.FTestAnovaPower()` | Manual |
| **Correlation power** | `power_correlation(r, power, sig_level)` | - | Manual |
| **Min detectable effect** | `minimum_detectable_effect(n, power)` | - | Manual |
| **Sample size summary** | `sample_size_summary(test_type, params)` | - | - |

**Example:**
```python
import real_simple_stats as rss

# Calculate required sample size for t-test
result = rss.power_t_test(delta=0.5, power=0.8, sig_level=0.05)
print(f"Required n per group: {result['n']}")

# Calculate power for given sample size
result = rss.power_t_test(delta=0.5, n=64, sig_level=0.05)
print(f"Statistical power: {result['power']:.3f}")
```

---

## 🎯 Bayesian Statistics

| Method | Real Simple Stats | PyMC | Stan |
|--------|-------------------|------|------|
| **Beta-Binomial update** | `beta_binomial_update(α, β, k, n)` | Manual | Manual |
| **Normal-Normal update** | `normal_normal_update(μ₀, σ₀², data, σ²)` | Manual | Manual |
| **Gamma-Poisson update** | `gamma_poisson_update(α, β, data)` | Manual | Manual |
| **Credible interval** | `credible_interval(dist, params, cred)` | `pm.hdi()` | Manual |
| **HDI** | `highest_density_interval(samples, cred)` | `pm.hdi()` | Manual |
| **Bayes factor** | `bayes_factor(L_H1, L_H0, prior_odds)` | Manual | Manual |
| **Posterior predictive** | `posterior_predictive(dist, params, n)` | `pm.sample_posterior_predictive()` | Manual |

**Example:**
```python
import real_simple_stats as rss

# Update Beta prior with binomial data
prior_alpha, prior_beta = 1, 1  # Uniform prior
successes, trials = 7, 10

post_alpha, post_beta = rss.beta_binomial_update(prior_alpha, prior_beta, 
                                                   successes, trials)

# Calculate credible interval
lower, upper = rss.credible_interval('beta', 
                                      {'alpha': post_alpha, 'beta': post_beta}, 
                                      0.95)
print(f"95% Credible Interval: [{lower:.3f}, {upper:.3f}]")
```

---

## 📐 Multivariate Analysis

| Method | Real Simple Stats | scikit-learn | statsmodels |
|--------|-------------------|--------------|-------------|
| **Multiple regression** | `multiple_regression(X, y)` | `LinearRegression()` | `sm.OLS()` |
| **PCA** | `pca(X, n_components)` | `PCA()` | - |
| **Factor analysis** | `factor_analysis(X, n_factors)` | `FactorAnalysis()` | - |
| **Canonical correlation** | `canonical_correlation(X, Y)` | `CCA()` | - |
| **Mahalanobis distance** | `mahalanobis_distance(X, point)` | - | - |

**Example:**
```python
import real_simple_stats as rss

X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [2, 4, 5, 4, 5]

# Multiple regression
result = rss.multiple_regression(X, y)
print(f"R² = {result['r_squared']:.3f}")
print(f"Coefficients: {result['coefficients']}")

# PCA
result = rss.pca(X, n_components=2)
print(f"Explained variance: {result['explained_variance']}")
```

---

## 🔍 Quick Lookup by Use Case

### "I want to..."

**...compare two groups**
- `two_sample_t_test(group1, group2)` - Test for mean difference
- `cohens_d(group1, group2)` - Calculate effect size
- `permutation_test(group1, group2, statistic)` - Non-parametric test

**...analyze relationships**
- `pearson_correlation(x, y)` - Linear correlation
- `linear_regression(x, y)` - Fit regression line
- `coefficient_of_determination(x, y)` - R² value

**...work with time series**
- `moving_average(data, window)` - Smooth data
- `autocorrelation(data, max_lag)` - Find patterns
- `seasonal_decompose(data, period)` - Decompose components

**...calculate confidence**
- `confidence_interval_known_std(mean, std, n, conf)` - Known σ
- `confidence_interval_unknown_std(mean, std, n, conf)` - Unknown σ
- `bootstrap(data, statistic, n_iterations)` - Bootstrap CI

**...plan a study**
- `power_t_test(delta, power, sig_level)` - Sample size for t-test
- `required_sample_size(confidence, width, std)` - CI-based planning
- `slovins_formula(N, e)` - Survey sample size

**...do Bayesian analysis**
- `beta_binomial_update(α, β, k, n)` - Update beliefs
- `credible_interval(dist, params, cred)` - Bayesian CI
- `bayes_factor(L_H1, L_H0)` - Compare hypotheses

---

## 📚 Function Categories

### By Statistical Domain

**Descriptive Statistics**: `mean`, `median`, `mode`, `sample_std_dev`, `sample_variance`, `five_number_summary`, `interquartile_range`, `coefficient_of_variation`

**Probability**: `normal_pdf`, `normal_cdf`, `binomial_probability`, `poisson_pmf`, `geometric_pmf`, `exponential_pdf`

**Inference**: `one_sample_t_test`, `two_sample_t_test`, `paired_t_test`, `one_sample_z_test`, `chi_square_statistic`, `one_way_anova`

**Regression**: `linear_regression`, `multiple_regression`, `pearson_correlation`, `coefficient_of_determination`

**Time Series**: `moving_average`, `autocorrelation`, `seasonal_decompose`, `detrend`, `difference`

**Resampling**: `bootstrap`, `permutation_test`, `jackknife`, `cross_validate`

**Effect Sizes**: `cohens_d`, `eta_squared`, `cramers_v`, `odds_ratio`

**Power Analysis**: `power_t_test`, `power_anova`, `minimum_detectable_effect`

**Bayesian**: `beta_binomial_update`, `credible_interval`, `bayes_factor`

**Multivariate**: `pca`, `factor_analysis`, `canonical_correlation`

---

## 🎓 Learning Path

**Beginner → Intermediate → Advanced**

1. **Start here**: `mean`, `median`, `std_dev`, `normal_cdf`
2. **Then learn**: `t_test`, `linear_regression`, `confidence_interval`
3. **Next**: `bootstrap`, `effect_sizes`, `power_analysis`
4. **Advanced**: `time_series`, `bayesian_stats`, `multivariate`

---

## 💡 Tips

- **All functions return simple Python types** (floats, lists, dicts) - no custom objects
- **Type hints included** for better IDE support
- **Comprehensive docstrings** with examples in every function
- **Consistent naming** - functions do what their names say
- **Educational focus** - designed for learning and teaching

---

**See also:**
- [Mathematical Formulas](MATHEMATICAL_FORMULAS.md) - LaTeX notation for all functions
- [FAQ](FAQ.md) - Common questions
- [Migration Guide](MIGRATION_GUIDE.md) - Switching from other libraries
