# Migration Guide - Switching to Real Simple Stats

Complete guide for migrating from other statistical libraries to Real Simple Stats.

---

## 📚 Overview

This guide helps you transition from:
- **R** - Statistical programming language
- **SciPy** - Python scientific computing
- **statsmodels** - Python statistical models
- **SPSS** - Commercial statistical software
- **Excel** - Spreadsheet analysis

---

## 🔄 From R to Real Simple Stats

### Philosophy Differences

| Aspect | R | Real Simple Stats |
|--------|---|-------------------|
| **Syntax** | `function(data, param=value)` | `function(data, param=value)` |
| **Data structures** | data.frames, vectors | Lists, NumPy arrays |
| **Output** | Complex objects | Simple dicts/tuples |
| **Installation** | `install.packages()` | `pip install` |

---

### Common Function Translations

#### Descriptive Statistics

| R | Real Simple Stats |
|---|-------------------|
| `mean(x)` | `rss.mean(x)` |
| `median(x)` | `rss.median(x)` |
| `sd(x)` | `rss.sample_std_dev(x)` |
| `var(x)` | `rss.sample_variance(x)` |
| `quantile(x, c(0.25, 0.75))` | `rss.five_number_summary(x)` |
| `IQR(x)` | `rss.interquartile_range(x)` |
| `summary(x)` | `rss.five_number_summary(x)` |

**Example Migration:**
```r
# R code
data <- c(1, 2, 3, 4, 5)
mean_val <- mean(data)
sd_val <- sd(data)
```

```python
# Python equivalent
import real_simple_stats as rss

data = [1, 2, 3, 4, 5]
mean_val = rss.mean(data)
sd_val = rss.sample_std_dev(data)
```

---

#### Hypothesis Tests

| R | Real Simple Stats |
|---|-------------------|
| `t.test(x, mu=0)` | `rss.one_sample_t_test(x, mu0=0)` |
| `t.test(x, y)` | `rss.two_sample_t_test(x, y)` |
| `t.test(x, y, paired=TRUE)` | `rss.paired_t_test(x, y)` |
| `chisq.test(obs, p=exp)` | `rss.chi_square_statistic(obs, exp)` |
| `aov(y ~ group)` | `rss.one_way_anova(groups)` |

**Example Migration:**
```r
# R code
group1 <- c(23, 25, 28, 30, 32)
group2 <- c(28, 30, 35, 38, 40)
result <- t.test(group1, group2)
print(result$p.value)
```

```python
# Python equivalent
import real_simple_stats as rss

group1 = [23, 25, 28, 30, 32]
group2 = [28, 30, 35, 38, 40]
t_stat, p_value = rss.two_sample_t_test(group1, group2)
print(p_value)
```

---

#### Regression

| R | Real Simple Stats |
|---|-------------------|
| `cor(x, y)` | `rss.pearson_correlation(x, y)` |
| `lm(y ~ x)` | `rss.linear_regression(x, y)` |
| `summary(lm(y ~ x))$r.squared` | `rss.coefficient_of_determination(x, y)` |
| `predict(model, newdata)` | `rss.regression_equation(x_new, slope, intercept)` |

**Example Migration:**
```r
# R code
x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 5, 4, 5)
model <- lm(y ~ x)
summary(model)
```

```python
# Python equivalent
import real_simple_stats as rss

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
slope, intercept, r_value, p_value, std_err = rss.linear_regression(x, y)
r_squared = r_value ** 2

print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print(f"R²: {r_squared:.3f}")
print(f"p-value: {p_value:.4f}")
```

---

#### Distributions

| R | Real Simple Stats |
|---|-------------------|
| `dnorm(x, mean, sd)` | `rss.normal_pdf(x, mu, sigma)` |
| `pnorm(x, mean, sd)` | `rss.normal_cdf(x, mu, sigma)` |
| `qnorm(p, mean, sd)` | `rss.normal_ppf(p, mu, sigma)` |
| `dbinom(k, n, p)` | `rss.binomial_probability(n, k, p)` |
| `pbinom(k, n, p)` | `rss.binomial_cdf(k, n, p)` |
| `dpois(k, lambda)` | `rss.poisson_pmf(k, lam)` |

---

### Key Differences

1. **Return Values:**
   ```r
   # R returns complex object
   result <- t.test(x, y)
   result$statistic
   result$p.value
   result$conf.int
   ```
   
   ```python
   # Python returns tuple
   t_stat, p_value = rss.two_sample_t_test(x, y)
   # Simpler, but less information
   ```

2. **Data Frames:**
   ```r
   # R uses data frames natively
   df <- data.frame(x=c(1,2,3), y=c(4,5,6))
   cor(df$x, df$y)
   ```
   
   ```python
   # Python uses lists or pandas
   import pandas as pd
   df = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})
   rss.pearson_correlation(df['x'].tolist(), df['y'].tolist())
   ```

3. **Missing Values:**
   ```r
   # R handles NA automatically
   mean(c(1, 2, NA, 4), na.rm=TRUE)
   ```
   
   ```python
   # Python requires manual handling
   data = [1, 2, None, 4]
   clean_data = [x for x in data if x is not None]
   rss.mean(clean_data)
   ```

---

## 🐍 From SciPy to Real Simple Stats

### When to Use Each

| Use Case | SciPy | Real Simple Stats |
|----------|-------|-------------------|
| **Learning statistics** | ❌ Complex | ✅ Simple |
| **Teaching** | ❌ Too technical | ✅ Educational |
| **Quick analysis** | ❌ Verbose | ✅ Concise |
| **Advanced features** | ✅ Comprehensive | ❌ Basic |
| **Performance critical** | ✅ Optimized | ⚠️ Good enough |

---

### Function Translations

#### Descriptive Statistics

| SciPy/NumPy | Real Simple Stats |
|-------------|-------------------|
| `np.mean(x)` | `rss.mean(x)` |
| `np.median(x)` | `rss.median(x)` |
| `np.std(x, ddof=1)` | `rss.sample_std_dev(x)` |
| `np.var(x, ddof=1)` | `rss.sample_variance(x)` |
| `stats.mode(x)` | `rss.mode(x)` |

---

#### Hypothesis Tests

| SciPy | Real Simple Stats |
|-------|-------------------|
| `stats.ttest_1samp(x, popmean)` | `rss.one_sample_t_test(x, mu0)` |
| `stats.ttest_ind(x, y)` | `rss.two_sample_t_test(x, y)` |
| `stats.ttest_rel(x, y)` | `rss.paired_t_test(x, y)` |
| `stats.chisquare(obs, exp)` | `rss.chi_square_statistic(obs, exp)` |
| `stats.f_oneway(*groups)` | `rss.one_way_anova(groups)` |

**Example Migration:**
```python
# SciPy code
from scipy import stats
import numpy as np

data = [23, 25, 28, 30, 32]
t_stat, p_value = stats.ttest_1samp(data, 30)
```

```python
# Real Simple Stats equivalent
import real_simple_stats as rss

data = [23, 25, 28, 30, 32]
t_stat, p_value = rss.one_sample_t_test(data, mu0=30)
```

---

#### Distributions

| SciPy | Real Simple Stats |
|-------|-------------------|
| `stats.norm.pdf(x, loc, scale)` | `rss.normal_pdf(x, mu, sigma)` |
| `stats.norm.cdf(x, loc, scale)` | `rss.normal_cdf(x, mu, sigma)` |
| `stats.norm.ppf(p, loc, scale)` | `rss.normal_ppf(p, mu, sigma)` |
| `stats.binom.pmf(k, n, p)` | `rss.binomial_probability(n, k, p)` |
| `stats.poisson.pmf(k, mu)` | `rss.poisson_pmf(k, lam)` |

---

#### Regression

| SciPy | Real Simple Stats |
|-------|-------------------|
| `stats.pearsonr(x, y)` | `rss.pearson_correlation(x, y)` |
| `stats.linregress(x, y)` | `rss.linear_regression(x, y)` |

**Example Migration:**
```python
# SciPy code
from scipy import stats

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
```

```python
# Real Simple Stats equivalent (identical!)
import real_simple_stats as rss

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
slope, intercept, r_value, p_value, std_err = rss.linear_regression(x, y)
```

---

### Key Advantages of Real Simple Stats

1. **Simpler imports:**
   ```python
   # SciPy
   from scipy import stats
   from scipy.stats import norm, binom
   import numpy as np
   
   # Real Simple Stats
   import real_simple_stats as rss
   ```

2. **Clearer function names:**
   ```python
   # SciPy
   stats.ttest_ind(group1, group2)
   
   # Real Simple Stats (more descriptive)
   rss.two_sample_t_test(group1, group2)
   ```

3. **Educational focus:**
   ```python
   # Real Simple Stats has better docstrings
   help(rss.two_sample_t_test)
   # Includes: explanation, formula, interpretation
   ```

---

## 📊 From statsmodels to Real Simple Stats

### Function Translations

| statsmodels | Real Simple Stats |
|-------------|-------------------|
| `sm.OLS(y, X).fit()` | `rss.multiple_regression(X, y)` |
| `sm.stats.ztest(x, value=mu0)` | `rss.one_sample_z_test(x, mu0, sigma)` |
| `sm.stats.ttest_ind(x, y)` | `rss.two_sample_t_test(x, y)` |
| `sm.stats.anova_lm()` | `rss.one_way_anova(groups)` |

**Example Migration:**
```python
# statsmodels code
import statsmodels.api as sm
import numpy as np

X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [2, 4, 5, 4, 5]
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
```

```python
# Real Simple Stats equivalent
import real_simple_stats as rss

X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [2, 4, 5, 4, 5]
result = rss.multiple_regression(X, y, include_intercept=True)

print(f"R² = {result['r_squared']:.3f}")
print(f"Coefficients: {result['coefficients']}")
print(f"Intercept: {result['intercept']}")
```

---

### When to Use Each

**Use statsmodels when:**
- Need detailed regression diagnostics
- Require time series models (ARIMA, VAR)
- Need generalized linear models (GLM)
- Want comprehensive statistical tests

**Use Real Simple Stats when:**
- Learning statistics
- Quick exploratory analysis
- Teaching or presentations
- Simple regression/correlation

---

## 💼 From SPSS to Real Simple Stats

### Menu-Driven to Code-Based

| SPSS Menu | Real Simple Stats Code |
|-----------|------------------------|
| Analyze → Descriptive Statistics → Descriptives | `rss.five_number_summary(data)` |
| Analyze → Compare Means → One-Sample T Test | `rss.one_sample_t_test(data, mu0)` |
| Analyze → Compare Means → Independent-Samples T Test | `rss.two_sample_t_test(group1, group2)` |
| Analyze → Compare Means → Paired-Samples T Test | `rss.paired_t_test(before, after)` |
| Analyze → Correlate → Bivariate | `rss.pearson_correlation(x, y)` |
| Analyze → Regression → Linear | `rss.linear_regression(x, y)` |

---

### Common SPSS Tasks

#### Task 1: Descriptive Statistics

**SPSS:**
```
DESCRIPTIVES VARIABLES=score
  /STATISTICS=MEAN STDDEV MIN MAX.
```

**Real Simple Stats:**
```python
import real_simple_stats as rss

score = [85, 90, 78, 92, 88]
print(f"Mean: {rss.mean(score)}")
print(f"Std Dev: {rss.sample_std_dev(score)}")
print(f"Min: {min(score)}")
print(f"Max: {max(score)}")
```

---

#### Task 2: Independent t-test

**SPSS:**
```
T-TEST GROUPS=group(1 2)
  /VARIABLES=score.
```

**Real Simple Stats:**
```python
import real_simple_stats as rss

group1 = [85, 90, 78, 92, 88]
group2 = [75, 80, 72, 82, 78]

t_stat, p_value = rss.two_sample_t_test(group1, group2)
d = rss.cohens_d(group1, group2)

print(f"t = {t_stat:.3f}, p = {p_value:.3f}")
print(f"Cohen's d = {d:.3f}")
```

---

#### Task 3: Correlation

**SPSS:**
```
CORRELATIONS
  /VARIABLES=height weight.
```

**Real Simple Stats:**
```python
import real_simple_stats as rss

height = [65, 70, 68, 72, 66]
weight = [150, 180, 165, 190, 155]

r = rss.pearson_correlation(height, weight)
print(f"r = {r:.3f}")
```

---

#### Task 4: Linear Regression

**SPSS:**
```
REGRESSION
  /DEPENDENT score
  /METHOD=ENTER hours_studied.
```

**Real Simple Stats:**
```python
import real_simple_stats as rss

hours_studied = [1, 2, 3, 4, 5]
score = [55, 65, 70, 80, 85]

slope, intercept, r_value, p_value, std_err = rss.linear_regression(
    hours_studied, score
)

print(f"Equation: score = {slope:.2f} * hours + {intercept:.2f}")
print(f"R² = {r_value**2:.3f}")
print(f"p = {p_value:.4f}")
```

---

### Advantages of Real Simple Stats over SPSS

1. **Free and open-source** (SPSS is expensive)
2. **Reproducible** (code vs. clicking)
3. **Automatable** (scripts vs. manual)
4. **Portable** (runs anywhere Python runs)
5. **Integrates with Python ecosystem** (pandas, matplotlib, etc.)

---

## 📊 From Excel to Real Simple Stats

### Common Excel Functions

| Excel | Real Simple Stats |
|-------|-------------------|
| `=AVERAGE(A1:A10)` | `rss.mean(data)` |
| `=MEDIAN(A1:A10)` | `rss.median(data)` |
| `=STDEV.S(A1:A10)` | `rss.sample_std_dev(data)` |
| `=CORREL(A1:A10, B1:B10)` | `rss.pearson_correlation(x, y)` |
| `=T.TEST(A1:A10, B1:B10, 2, 2)` | `rss.two_sample_t_test(x, y)` |
| `=SLOPE(Y1:Y10, X1:X10)` | `rss.linear_regression(x, y)[0]` |
| `=INTERCEPT(Y1:Y10, X1:X10)` | `rss.linear_regression(x, y)[1]` |

---

### Example Migration: Data Analysis

**Excel Workflow:**
1. Enter data in columns A and B
2. Click Data → Data Analysis → t-Test
3. Select ranges
4. Click OK
5. View output

**Real Simple Stats Workflow:**
```python
import real_simple_stats as rss
import pandas as pd

# Read Excel file
df = pd.read_excel('data.xlsx')

# Perform t-test
t_stat, p_value = rss.two_sample_t_test(
    df['Group1'].tolist(),
    df['Group2'].tolist()
)

# Calculate effect size
d = rss.cohens_d(
    df['Group1'].tolist(),
    df['Group2'].tolist()
)

# Report results
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {d:.3f}")
```

---

### Advantages over Excel

1. **Reproducibility**: Code can be re-run
2. **Scalability**: Handle large datasets
3. **Automation**: Process multiple files
4. **Version control**: Track changes with Git
5. **Advanced statistics**: More methods available

---

## 🔄 Complete Migration Example

### Scenario: Comparing Two Groups

**R Code:**
```r
# Load data
group1 <- c(23, 25, 28, 30, 32)
group2 <- c(28, 30, 35, 38, 40)

# Descriptive statistics
mean1 <- mean(group1)
mean2 <- mean(group2)
sd1 <- sd(group1)
sd2 <- sd(group2)

# t-test
result <- t.test(group1, group2)

# Effect size (requires package)
library(effsize)
d <- cohen.d(group1, group2)

# Report
cat(sprintf("Group 1: M=%.2f, SD=%.2f\n", mean1, sd1))
cat(sprintf("Group 2: M=%.2f, SD=%.2f\n", mean2, sd2))
cat(sprintf("t(%.0f)=%.2f, p=%.3f\n", 
            result$parameter, result$statistic, result$p.value))
cat(sprintf("Cohen's d=%.2f\n", d$estimate))
```

**Real Simple Stats Code:**
```python
import real_simple_stats as rss

# Load data
group1 = [23, 25, 28, 30, 32]
group2 = [28, 30, 35, 38, 40]

# Descriptive statistics
mean1 = rss.mean(group1)
mean2 = rss.mean(group2)
sd1 = rss.sample_std_dev(group1)
sd2 = rss.sample_std_dev(group2)

# t-test
t_stat, p_value = rss.two_sample_t_test(group1, group2)

# Effect size
d = rss.cohens_d(group1, group2)
interpretation = rss.interpret_effect_size(d, 'd')

# Report
print(f"Group 1: M={mean1:.2f}, SD={sd1:.2f}")
print(f"Group 2: M={mean2:.2f}, SD={sd2:.2f}")
print(f"t({len(group1)+len(group2)-2})={t_stat:.2f}, p={p_value:.3f}")
print(f"Cohen's d={d:.2f} ({interpretation})")
```

---

## 📋 Migration Checklist

### Before Migration

- [ ] Identify which functions you use most
- [ ] Check if Real Simple Stats supports them
- [ ] Review [API Comparison](API_COMPARISON.md)
- [ ] Test with sample data

### During Migration

- [ ] Install Real Simple Stats: `pip install real-simple-stats`
- [ ] Convert data structures (data.frames → lists/arrays)
- [ ] Translate function calls
- [ ] Verify results match original
- [ ] Update documentation/comments

### After Migration

- [ ] Run tests to ensure correctness
- [ ] Update analysis scripts
- [ ] Train team members
- [ ] Document any limitations

---

## 🎯 Quick Reference Card

### Most Common Translations

```python
# Descriptive Statistics
mean(x)                    → rss.mean(x)
sd(x) / np.std(x, ddof=1)  → rss.sample_std_dev(x)
median(x)                  → rss.median(x)

# Hypothesis Tests
t.test(x, y)               → rss.two_sample_t_test(x, y)
cor.test(x, y)             → rss.pearson_correlation(x, y)
chisq.test(obs, exp)       → rss.chi_square_statistic(obs, exp)

# Regression
lm(y ~ x)                  → rss.linear_regression(x, y)
predict(model, newdata)    → rss.regression_equation(x, slope, intercept)

# Distributions
pnorm(x, mean, sd)         → rss.normal_cdf(x, mu, sigma)
qnorm(p, mean, sd)         → rss.normal_ppf(p, mu, sigma)
```

---

## 💡 Tips for Successful Migration

1. **Start small**: Migrate one analysis at a time
2. **Verify results**: Compare outputs with original software
3. **Use version control**: Track changes with Git
4. **Document differences**: Note any discrepancies
5. **Leverage Python ecosystem**: Combine with pandas, matplotlib
6. **Ask for help**: Use [GitHub issues](https://github.com/kylejones200/real_simple_stats/issues)

---

## 🔗 Additional Resources

- **API Comparison**: [Detailed function mapping](API_COMPARISON.md)
- **Examples**: [Interactive tutorials](INTERACTIVE_EXAMPLES.md)
- **FAQ**: [Common questions](FAQ.md)
- **Troubleshooting**: [Error solutions](TROUBLESHOOTING.md)

---

**Need help migrating?** [Open an issue](https://github.com/kylejones200/real_simple_stats/issues) with your use case!

**Last Updated**: 2025  
**Version**: 0.3.0
