# Frequently Asked Questions (FAQ)

Common questions about Real Simple Stats, answered.

---

## 📦 Installation & Setup

### Q: How do I install Real Simple Stats?

**A:** Use pip:
```bash
pip install real-simple-stats
```

For the latest development version:
```bash
pip install git+https://github.com/kylejones200/real_simple_stats.git
```

---

### Q: What are the system requirements?

**A:**
- **Python**: 3.7 or higher
- **Dependencies**: NumPy, SciPy (automatically installed)
- **Optional**: matplotlib (for plotting), pandas (for data handling)

---

### Q: Can I use this in Google Colab or Jupyter?

**A:** Yes! Install in the first cell:
```python
!pip install real-simple-stats
import real_simple_stats as rss
```

---

### Q: Do I need to install matplotlib separately?

**A:** No, matplotlib is included as a dependency. However, if you only want the statistical functions without plotting, you can skip it.

---

## 🎯 General Usage

### Q: How do I import the package?

**A:** Standard import:
```python
import real_simple_stats as rss

# Use functions
mean = rss.mean([1, 2, 3, 4, 5])
```

Or import specific functions:
```python
from real_simple_stats import mean, median, std_dev

mean([1, 2, 3])
```

---

### Q: What data types does the package accept?

**A:** Most functions accept:
- Python lists: `[1, 2, 3, 4, 5]`
- NumPy arrays: `np.array([1, 2, 3, 4, 5])`
- Tuples: `(1, 2, 3, 4, 5)`

For multivariate functions, use lists of lists or 2D NumPy arrays.

---

### Q: Do functions modify my original data?

**A:** No! All functions return new values without modifying your input data.

```python
data = [1, 2, 3, 4, 5]
result = rss.mean(data)
# data is unchanged
```

---

### Q: What's the difference between sample and population functions?

**A:**
- **Sample functions** (e.g., `sample_std_dev`): Use $n-1$ in denominator (Bessel's correction)
- **Population functions** (e.g., `population_std_dev`): Use $n$ in denominator

**Rule of thumb**: Use sample functions for real-world data (most common).

```python
# Sample standard deviation (n-1)
rss.sample_std_dev([1, 2, 3, 4, 5])

# Population standard deviation (n)
rss.population_std_dev([1, 2, 3, 4, 5])
```

---

## 📊 Statistical Tests

### Q: When should I use a t-test vs. z-test?

**A:**
- **t-test**: Unknown population standard deviation (most common)
- **z-test**: Known population standard deviation (rare in practice)

```python
# Unknown σ (use t-test)
t_stat, p_value = rss.one_sample_t_test(data, mu0=100)

# Known σ (use z-test)
z_stat, p_value = rss.one_sample_z_test(data, mu0=100, sigma=15)
```

---

### Q: How do I interpret p-values?

**A:**
- **p < 0.05**: Statistically significant (reject null hypothesis)
- **p ≥ 0.05**: Not statistically significant (fail to reject null hypothesis)

**Important**: p-value is NOT the probability that the null hypothesis is true!

```python
t_stat, p_value = rss.two_sample_t_test(group1, group2)

if p_value < 0.05:
    print("Significant difference between groups")
else:
    print("No significant difference")
```

---

### Q: What's the difference between one-tailed and two-tailed tests?

**A:**
- **Two-tailed** (default): Tests if means are different (either direction)
- **One-tailed**: Tests if one mean is specifically greater or less

Most Real Simple Stats functions use two-tailed tests by default.

---

### Q: Should I use paired or independent t-test?

**A:**
- **Paired t-test**: Same subjects measured twice (before/after, matched pairs)
- **Independent t-test**: Different subjects in each group

```python
# Paired (same subjects)
before = [120, 130, 125, 135, 140]
after = [115, 125, 120, 130, 135]
t_stat, p_value = rss.paired_t_test(before, after)

# Independent (different subjects)
group1 = [120, 130, 125, 135, 140]
group2 = [115, 125, 120, 130, 135]
t_stat, p_value = rss.two_sample_t_test(group1, group2)
```

---

### Q: What sample size do I need?

**A:** Use power analysis:
```python
# For t-test with medium effect size (d=0.5), 80% power
result = rss.power_t_test(delta=0.5, power=0.8, sig_level=0.05)
print(f"Need {result['n']} participants per group")
```

---

## 📈 Regression & Correlation

### Q: What's the difference between correlation and regression?

**A:**
- **Correlation** (`pearson_correlation`): Measures strength of linear relationship (-1 to 1)
- **Regression** (`linear_regression`): Predicts one variable from another

```python
# Correlation
r = rss.pearson_correlation(x, y)  # Just a number

# Regression
slope, intercept, r_value, p_value, std_err = rss.linear_regression(x, y)
# Can make predictions: y = slope*x + intercept
```

---

### Q: How do I interpret R²?

**A:** R² (coefficient of determination) = proportion of variance explained

- **R² = 0.00**: No predictive power
- **R² = 0.25**: Weak relationship
- **R² = 0.50**: Moderate relationship
- **R² = 0.75**: Strong relationship
- **R² = 1.00**: Perfect prediction

```python
slope, intercept, r_value, p_value, std_err = rss.linear_regression(x, y)
r_squared = r_value ** 2
print(f"Model explains {r_squared*100:.1f}% of variance")
```

---

### Q: Can I do multiple regression?

**A:** Yes! Use `multiple_regression`:
```python
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]  # Multiple predictors
y = [2, 4, 5, 4, 5]

result = rss.multiple_regression(X, y)
print(f"R² = {result['r_squared']:.3f}")
print(f"Coefficients: {result['coefficients']}")
```

---

## 🎲 Probability & Distributions

### Q: How do I calculate probabilities for normal distribution?

**A:**
```python
# P(X ≤ x)
prob = rss.normal_cdf(x=100, mu=100, sigma=15)

# P(X > x) = 1 - P(X ≤ x)
prob = 1 - rss.normal_cdf(x=100, mu=100, sigma=15)

# P(a < X < b)
prob = rss.normal_cdf(b, mu, sigma) - rss.normal_cdf(a, mu, sigma)
```

---

### Q: What's the difference between PDF and CDF?

**A:**
- **PDF** (Probability Density Function): Height of distribution curve
- **CDF** (Cumulative Distribution Function): Area under curve up to x

For probabilities, use **CDF**:
```python
# Probability that X ≤ 1.96 for standard normal
prob = rss.normal_cdf(1.96, 0, 1)  # ≈ 0.975
```

---

### Q: How do I find critical values?

**A:**
```python
# For normal distribution (z-score)
z_critical = rss.normal_ppf(0.975, 0, 1)  # 1.96 for 95% CI

# For chi-square
chi_critical = rss.critical_chi_square_value(alpha=0.05, df=5)
```

---

## 🔄 Advanced Topics

### Q: What's the difference between bootstrap and permutation test?

**A:**
- **Bootstrap**: Estimates uncertainty (confidence intervals)
- **Permutation test**: Tests hypotheses (p-values)

```python
# Bootstrap for CI
result = rss.bootstrap(data, np.mean, n_iterations=1000)
print(f"95% CI: {result['confidence_interval']}")

# Permutation test for hypothesis
result = rss.permutation_test(group1, group2,
                               lambda d1, d2: np.mean(d1) - np.mean(d2))
print(f"p-value: {result['p_value']}")
```

---

### Q: When should I use Bayesian vs. frequentist methods?

**A:**
- **Frequentist** (t-tests, p-values): Traditional, widely accepted
- **Bayesian**: Incorporates prior knowledge, gives probability of hypotheses

Use Bayesian when:
- You have prior information
- You want probability statements about parameters
- You need to update beliefs with new data

```python
# Bayesian update
post_alpha, post_beta = rss.beta_binomial_update(
    prior_alpha=1, prior_beta=1,  # Uniform prior
    successes=7, trials=10
)

# Credible interval (Bayesian CI)
lower, upper = rss.credible_interval('beta',
                                      {'alpha': post_alpha, 'beta': post_beta})
```

---

### Q: What's PCA and when should I use it?

**A:** PCA (Principal Component Analysis) reduces dimensions while preserving variance.

**Use when:**
- You have many correlated variables
- You want to visualize high-dimensional data
- You need to reduce multicollinearity

```python
result = rss.pca(X, n_components=2)
print(f"Explained variance: {result['explained_variance']}")
```

---

## 🎯 Effect Sizes

### Q: Why do I need effect sizes?

**A:** P-values tell you if an effect exists; effect sizes tell you how large it is.

**Example:**
```python
# Significant but small effect
t_stat, p_value = rss.two_sample_t_test(group1, group2)
d = rss.cohens_d(group1, group2)

print(f"p-value: {p_value:.4f}")  # p < 0.05 (significant)
print(f"Cohen's d: {d:.3f}")      # d = 0.15 (tiny effect)
```

**Interpretation**: Statistically significant but practically meaningless.

---

### Q: Which effect size should I use?

**A:**
- **Cohen's d**: Comparing two means
- **Eta-squared**: ANOVA (multiple groups)
- **Cramér's V**: Categorical data (chi-square)
- **R²**: Regression

```python
# Two groups
d = rss.cohens_d(group1, group2)

# Multiple groups (ANOVA)
eta_sq = rss.eta_squared([group1, group2, group3])

# Categorical
v = rss.cramers_v([[10, 20], [30, 40]])
```

---

### Q: How do I interpret Cohen's d?

**A:**
- **Small**: d ≈ 0.2
- **Medium**: d ≈ 0.5
- **Large**: d ≈ 0.8

```python
d = rss.cohens_d(group1, group2)
interpretation = rss.interpret_effect_size(d, 'd')
print(f"Cohen's d = {d:.3f} ({interpretation})")
```

---

## 🔧 Technical Questions

### Q: Are the functions vectorized?

**A:** Yes, most functions use NumPy internally for efficient computation.

---

### Q: Can I use this with pandas DataFrames?

**A:** Yes! Convert columns to lists or arrays:
```python
import pandas as pd
import real_simple_stats as rss

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Method 1: Convert to list
mean_A = rss.mean(df['A'].tolist())

# Method 2: Use values (NumPy array)
mean_A = rss.mean(df['A'].values)

# Regression
slope, intercept, *_ = rss.linear_regression(df['A'].values, df['B'].values)
```

---

### Q: How accurate are the calculations?

**A:** Real Simple Stats uses SciPy and NumPy for numerical computations, which are industry-standard and highly accurate. Results match those from R, SPSS, and other statistical software.

---

### Q: Can I use this for production/research?

**A:** Yes! The package is:
- ✅ Well-tested (86% code coverage)
- ✅ Based on established statistical methods
- ✅ Uses reliable numerical libraries (SciPy, NumPy)
- ✅ Documented with references

However, always validate results for critical applications.

---

### Q: Is this package maintained?

**A:** Yes! Check the [GitHub repository](https://github.com/kylejones200/real_simple_stats) for:
- Latest updates
- Issue tracking
- Contribution guidelines

---

## 🎓 Educational Questions

### Q: Can I use this for teaching?

**A:** Absolutely! Real Simple Stats is designed for education:
- Clear function names
- Comprehensive docstrings
- Step-by-step examples
- Educational focus over performance

---

### Q: Is there a textbook or course that uses this?

**A:** While not tied to a specific textbook, Real Simple Stats aligns with standard introductory statistics curricula. See [INTERACTIVE_EXAMPLES.md](INTERACTIVE_EXAMPLES.md) for tutorials.

---

### Q: How does this compare to R or SPSS?

**A:**
- **Simpler**: Easier to learn than R
- **More accessible**: Free and open-source (unlike SPSS)
- **Python-based**: Integrates with data science ecosystem
- **Educational**: Designed for learning, not just analysis

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed comparisons.

---

## 🐛 Troubleshooting

### Q: I get "ModuleNotFoundError: No module named 'real_simple_stats'"

**A:** Install the package:
```bash
pip install real-simple-stats
```

Make sure you're using the correct package name (with hyphens).

---

### Q: Functions return unexpected results

**A:** Check:
1. **Data format**: Are you passing lists/arrays?
2. **Sample vs. population**: Using correct function?
3. **Parameter order**: Check docstring with `help(rss.function_name)`

```python
# Check documentation
help(rss.two_sample_t_test)
```

---

### Q: I get "ValueError: Input arrays must have the same length"

**A:** For paired tests and correlation, ensure both arrays have the same length:
```python
# Wrong
x = [1, 2, 3]
y = [4, 5]  # Different length!

# Correct
x = [1, 2, 3]
y = [4, 5, 6]  # Same length
```

---

### Q: Plots don't show up

**A:**
```python
import matplotlib.pyplot as plt
import real_simple_stats as rss

rss.plot_normal_histogram(data)
plt.show()  # Add this!
```

---

### Q: I get warnings about "divide by zero"

**A:** This can happen with:
- Empty datasets
- Zero variance (all values the same)
- Zero expected frequencies (chi-square)

Check your data:
```python
data = [5, 5, 5, 5, 5]
std = rss.sample_std_dev(data)  # Will be 0
```

---

## 💡 Best Practices

### Q: What's the recommended workflow?

**A:**
1. **Explore data**: Use descriptive statistics
2. **Visualize**: Create plots
3. **Test hypotheses**: Run appropriate tests
4. **Calculate effect sizes**: Assess practical significance
5. **Report results**: Include all relevant statistics

```python
import real_simple_stats as rss

# 1. Descriptive statistics
print(rss.five_number_summary(data))

# 2. Visualize
rss.plot_box_plot(data)

# 3. Test
t_stat, p_value = rss.one_sample_t_test(data, mu0=100)

# 4. Effect size
d = rss.cohens_d(data, [100]*len(data))

# 5. Report
print(f"t({len(data)-1}) = {t_stat:.2f}, p = {p_value:.3f}, d = {d:.2f}")
```

---

### Q: How should I report results?

**A:** Include:
- Test statistic and degrees of freedom
- P-value
- Effect size
- Confidence interval (when appropriate)

**Example**:
```
"A two-sample t-test revealed a significant difference between groups,
t(18) = 2.45, p = .025, d = 0.73, 95% CI [0.5, 3.2]."
```

---

### Q: Should I correct for multiple comparisons?

**A:** Yes, if you're running multiple tests on the same dataset. Common methods:
- Bonferroni correction: Divide α by number of tests
- False Discovery Rate (FDR)

```python
# 3 tests, use α = 0.05/3 = 0.0167
alpha_corrected = 0.05 / 3
```

---

## 🔗 Additional Resources

### Q: Where can I learn more?

**A:**
- **Documentation**: [ReadTheDocs](https://real-simple-stats.readthedocs.io/)
- **Examples**: [Interactive Tutorials](INTERACTIVE_EXAMPLES.md)
- **API Reference**: [Function Comparison](API_COMPARISON.md)
- **Math Details**: [Mathematical Formulas](MATHEMATICAL_FORMULAS.md)

---

### Q: How do I report bugs or request features?

**A:**
1. Check [existing issues](https://github.com/kylejones200/real_simple_stats/issues)
2. Create a new issue with:
   - Description of problem/feature
   - Example code (if applicable)
   - Expected vs. actual behavior

---

### Q: Can I contribute?

**A:** Yes! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📞 Still Have Questions?

- **GitHub Issues**: [Ask a question](https://github.com/kylejones200/real_simple_stats/issues)
- **Documentation**: [Full docs](https://real-simple-stats.readthedocs.io/)
- **Examples**: [Interactive tutorials](INTERACTIVE_EXAMPLES.md)

---

**Last Updated**: 2025
**Version**: 0.3.0
