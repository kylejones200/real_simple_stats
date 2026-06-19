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

## General Usage

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

## Statistical Tests

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

## Regression & Correlation

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

## Probability & Distributions

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

## Effect Sizes

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

**A:** Yes! You have several options:

**Option 1: Use pandas compatibility module (recommended)**
```python
import pandas as pd
from real_simple_stats.pandas_compat import mean, median, standard_deviation

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
mean_A = mean(df['A'])  # Works directly with Series!
```

**Option 2: Convert to list or array**
```python
import pandas as pd
import real_simple_stats as rss

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Method 1: Convert to list
mean_A = rss.mean(df['A'].tolist())

# Method 2: Use values (NumPy array)
mean_A = rss.mean(df['A'].values)
```

---

### Q: How accurate are the calculations?

**A:** Real Simple Stats uses SciPy and NumPy for numerical computations, which are industry-standard and highly accurate. Results match those from R, SPSS, and other statistical software.

---

### Q: Can I use this for production/research?

**A:** Yes! The package is:
- Well-tested (86% code coverage)
- Based on established statistical methods
- Uses reliable numerical libraries (SciPy, NumPy)
- Documented with references

However, always validate results for critical applications.

---

### Q: Is this package maintained?

**A:** Yes! Check the [GitHub repository](https://github.com/kylejones200/real_simple_stats) for:
- Latest updates
- Issue tracking
- Contribution guidelines

---

## Educational Questions

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

## Common Pitfalls

### Q: I'm getting different results than expected. What's wrong?

**A:** Common issues:

1. **Using population functions instead of sample functions**
   ```python
   # Wrong for sample data
   std = rss.standard_deviation(data)  # Population formula (divides by n)
   
   # Correct for samples
   std = rss.sample_std_dev(data)  # Sample formula (divides by n-1)
   ```

2. **Forgetting to handle missing values**
   ```python
   # Wrong - includes None/NaN
   data = [1, 2, None, 4, 5]
   mean = rss.mean(data)  # Error!
   
   # Correct - filter missing values
   clean_data = [x for x in data if x is not None]
   mean = rss.mean(clean_data)
   ```

3. **Using wrong test for your data**
   - Paired data → use `paired_t_test()`, not `two_sample_t_test()`
   - Small samples → use t-test, not z-test
   - Non-normal data → consider non-parametric tests

4. **Confusing one-tailed vs two-tailed tests**
   ```python
   # Two-tailed (default): H₁: μ ≠ μ₀
   t_stat, p_value = rss.one_sample_t_test(data, mu=5)
   
   # One-tailed: H₁: μ < μ₀ or μ > μ₀
   # Check the alternative parameter in function docs
   ```

---

### Q: Why is my p-value different from what I calculated by hand?

**A:** Common reasons:

1. **Rounding differences**: Hand calculations often round intermediate steps
2. **Using wrong degrees of freedom**: Check df = n - 1 for one-sample t-test
3. **Two-tailed vs one-tailed**: Make sure you're comparing the right p-value
4. **Using z-table instead of t-table**: For small samples, use t-distribution

The package uses exact calculations, so trust the computed p-value over hand calculations.

---

### Q: My confidence interval seems wrong. What did I do?

**A:** Check these:

1. **Wrong critical value**: Make sure you're using the right distribution (t vs z)
2. **Wrong standard error**: Sample vs population standard error
3. **One-tailed vs two-tailed**: CI should be two-tailed (default)
4. **Wrong degrees of freedom**: Especially for two-sample tests

```python
# Correct: Two-sample t-test CI
t_stat, p_value = rss.two_sample_t_test(group1, group2)
# Calculate CI using pooled standard error and correct df
```

---

### Q: I'm getting a ValueError. What does it mean?

**A:** Common ValueErrors and fixes:

1. **"Cannot calculate standard deviation of empty list"**
   - Fix: Check that your data list isn't empty
   - Check for None values that got filtered out

2. **"Standard deviation must be positive"**
   - Fix: Check that std_dev parameter > 0
   - For normal distributions, variance must be positive

3. **"Probability must be between 0 and 1"**
   - Fix: Check that probability values are in valid range
   - Don't use percentages (0.05 not 5%)

4. **"Number of successes (k) must be between 0 and n"**
   - Fix: For binomial, k must satisfy 0 ≤ k ≤ n
   - Check your k and n values

---

### Q: Why does my statistical test give different results than Excel/SPSS/R?

**A:** Possible reasons:

1. **Different default settings**: 
   - Excel might use population formulas by default
   - R uses sample formulas (n-1) by default
   - Check which formula you're using

2. **Different algorithms**: 
   - Some software uses approximations
   - Real Simple Stats uses SciPy (exact methods)

3. **Data handling differences**:
   - Missing value handling
   - Rounding precision
   - Ties in rank-based tests

4. **Test parameters**:
   - One-tailed vs two-tailed
   - Different significance levels
   - Different assumptions (equal variance, etc.)

**Solution**: Always check:
- Which formula is being used (sample vs population)
- Test parameters (one-tailed vs two-tailed)
- Data preprocessing (missing values, outliers)

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

## Best Practices

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

## Additional Resources

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

---

## Causal Inference

### Q: When should I use difference-in-differences vs. regression discontinuity?

**A:** They answer the same broad question ("did X cause Y?") but require different data structures.

| Method | You need | Assumption |
|---|---|---|
| DiD | Pre/post measurements + a control group that didn't receive the treatment | Parallel trends: absent treatment, both groups would have changed at the same rate |
| RDD | A numerical score used to assign treatment (e.g. test cutoff, income threshold) | Local continuity: potential outcomes are smooth through the cutoff |
| Synthetic control | Multiple control units, long pre-treatment history, no single comparable control | Donor units span the same covariate space as the treated unit |
| Panel FE | Repeated observations per entity over time | Treatment is uncorrelated with unobserved time-invariant entity characteristics |

Use DiD when you have a natural "before vs. after" event and a control group (e.g. minimum wage law in one state, not another). Use RDD when assignment was determined by a cutoff score — e.g., students above 65 get tutoring, below 65 don't.

```python
import real_simple_stats as rss

# DiD
r = rss.difference_in_differences(outcome, post, treated)
result = rss.difference_in_differences_explained(outcome, post, treated)
print(result)   # prints parallel trends caveat and next steps

# RDD
r = rss.regression_discontinuity(outcome, running_var, cutoff=65)
r["effect"]     # local average treatment effect at the cutoff
```

---

### Q: What is the parallel trends assumption, and how do I check it?

**A:** The parallel trends assumption says: *absent treatment, the treated and control groups would have changed at the same rate.* It's the core identifying assumption of DiD — without it, the DiD estimate is biased.

You cannot test it in the post-treatment period (that's the period you're trying to study). But you can partially validate it with **pre-treatment trend tests**:

1. Plot the outcome for both groups in the pre-treatment period. If the lines run roughly parallel, the assumption is more credible.
2. Run a DiD regression using only the pre-treatment periods to check whether there's a spurious "effect" before the intervention — there shouldn't be.
3. Be especially cautious when treated and control groups differ on observable baseline characteristics.

The `difference_in_differences_explained` function prints a specific caveat about this and suggests next steps.

---

### Q: How do I interpret a synthetic control result?

**A:** The synthetic control creates a weighted average of control units that closely matches the treated unit during the pre-treatment period. After the intervention, the gap between the treated unit and its synthetic counterfactual is the estimated treatment effect.

```python
r = rss.synthetic_control(y_treated, Y_controls, n_pre=30)
r["weights"]         # float array, one per donor unit; sums to ~1
r["effect"]          # post-treatment gap at each time step
r["pre_fit_mse"]     # how well the synthetic control matched pre-treatment
```

Key checks:
- `pre_fit_mse` should be small — if the synthetic control doesn't fit pre-treatment well, the post-treatment comparison is unreliable.
- Inspect `weights` — if one donor has weight ≈ 1 and all others ≈ 0, you've essentially just used that one unit as a control; consider whether that's appropriate.
- Inference is done via permutation (run the same analysis on each control unit as if it were treated). This library returns the point estimate; formal permutation p-values require the user to iterate.

---

## Survival Analysis

### Q: What is right-censoring, and why does it matter?

**A:** A censored observation is one where the event had not yet occurred when the study ended (or the subject dropped out). For example, in a customer churn study, a customer who is still active at the end of your observation window is censored — you know they survived *at least* that long, but not when (or if) they'll eventually churn.

If you ignore censored observations and only analyse customers who did churn, you introduce survivor bias — your median churn time will be far too short. Kaplan-Meier and parametric survival models handle censoring correctly by removing censored observations from the risk set at their last known time without contributing to the hazard estimate.

```python
durations      = [2, 3, 5, 7, 11, 4, 8, 10, 6, 14]
event_observed = [1, 1, 1, 1,  0, 1, 0,  1, 1,  0]  # 0 = censored

r = rss.kaplan_meier(durations, event_observed)
r["median_survival"]   # time where S(t) = 0.5
r["n_censored"]        # 3 in this example
```

---

### Q: When should I use Kaplan-Meier vs. a parametric survival model?

**A:**

| | Kaplan-Meier | Parametric (Weibull, etc.) |
|---|---|---|
| **Assumptions** | None — fully non-parametric | Distributional form (shape of hazard) |
| **Extrapolation** | Not reliable beyond last event | Can extrapolate into the future |
| **Interpretation** | Empirical S(t) at each observed time | Smooth curve with interpretable parameters |
| **Best for** | First look, small samples, comparing groups | Forecasting future survival, fitting a hazard model |

Always start with Kaplan-Meier. Then use `compare_survival_models` to find the best-fitting parametric family (Exponential, Weibull, Lognormal, Log-logistic) if you need to extrapolate beyond your observation window.

```python
# Step 1: non-parametric description
result = rss.kaplan_meier_explained(durations, event_observed)
print(result)

# Step 2: find best parametric fit
ranked = rss.compare_survival_models(durations, event_observed)
ranked[0]["distribution"]   # "weibull" or "lognormal" etc.
ranked[0]["aic"]            # lower is better
ranked[0]["survival_fn"](t=90)  # probability of surviving to day 90
```

---

### Q: What does AIC mean in `compare_survival_models`?

**A:** AIC (Akaike Information Criterion) balances goodness of fit against model complexity. Lower AIC = better model. Use it to choose among the four distributions — Exponential (1 parameter), Weibull (2), Lognormal (2), Log-logistic (2).

AIC differences matter more than the absolute value:
- ΔAIC < 2: models are roughly equivalent; prefer the simpler one (Exponential)
- ΔAIC 2–10: modest evidence for the better model
- ΔAIC > 10: strong evidence; use the lower-AIC model

```python
ranked = rss.compare_survival_models(durations, event_observed)
for m in ranked:
    print(f"{m['distribution']:12s}  AIC={m['aic']:.1f}  rank={m['rank']}")
```

---

## Market Basket Analysis

### Q: What do support, confidence, and lift measure?

**A:** These three metrics describe different aspects of an association rule `A → B`:

| Metric | Formula | Meaning | Good value |
|---|---|---|---|
| **Support** | P(A ∩ B) | How often A and B appear together | Depends on data volume; typically > 0.01 |
| **Confidence** | P(B \| A) | When A is bought, how often is B also bought? | > 0.5 for actionable rules |
| **Lift** | P(B \| A) / P(B) | Does A increase B's purchase probability vs. baseline? | > 1.0 (lift < 1 = A suppresses B) |

Lift is the most useful for actionability — it tells you whether A and B co-occur *more than chance* would predict. A lift of 3.2 means customers who buy A are 3.2× more likely to also buy B compared to a random customer.

```python
matrix, items = rss.encode_transactions(transactions)
itemsets = rss.frequent_itemsets(matrix, items, min_support=0.05)
rules = rss.association_rules(itemsets, min_confidence=0.4, min_lift=1.5)

for r in sorted(rules, key=lambda x: x["lift"], reverse=True)[:5]:
    print(f"{r['antecedent']} → {r['consequent']}")
    print(f"  support={r['support']:.3f}  confidence={r['confidence']:.3f}  lift={r['lift']:.2f}")
```

---

### Q: How many transactions do I need for reliable association rules?

**A:** As a rough guide:
- `min_support = 0.01` means an itemset must appear in at least 1% of transactions
- For that to represent ≥ 10 transactions (the practical minimum for a reliable count), you need at least 1,000 transactions
- With sparse data (many unique items, few items per basket), prefer higher `min_support` thresholds

If you have fewer than ~500 transactions, consider raising `min_support` to 0.05–0.10 to avoid spurious rules driven by a handful of co-occurrences. Rules with very high confidence but low support (e.g. `{A, B} → C`, support = 0.002, confidence = 0.95) usually reflect a handful of transactions and shouldn't drive business decisions without validation.

---

### Q: What's the difference between `frequent_itemsets` and `association_rules`?

**A:** They're two steps in the Apriori pipeline:

1. `frequent_itemsets` finds sets of items that appear together frequently (above `min_support`). Output: `[{"itemset": {"milk", "bread"}, "support": 0.4}, ...]`
2. `association_rules` takes those itemsets and derives directional rules (`A → B`) filtered by `min_confidence` and `min_lift`. Output: `[{"antecedent": {"milk"}, "consequent": {"bread"}, "confidence": 0.8, "lift": 2.1}, ...]`

You always run `frequent_itemsets` first. `association_rules` needs the itemset list as input.

---

## Spatial Statistics

### Q: How do I choose a distance threshold for Moran's I?

**A:** The distance threshold defines who is a "neighbour" — it directly controls the result. There is no universally correct choice; it depends on the spatial process you're trying to measure.

Practical guidance:
- **Domain knowledge first**: if you're studying disease spread and the transmission radius is ~2 km, use that.
- **Vary and compare**: compute Moran's I at several thresholds (e.g. 5th, 10th, 25th percentile of all pairwise distances). If clustering only appears at one specific threshold, it may be an artefact.
- **Rule of thumb**: each observation should have at least 2–4 neighbours on average at your chosen threshold.

```python
import numpy as np
from scipy.spatial.distance import cdist

coords = np.column_stack([x, y])
dists = cdist(coords, coords)

# Check: how many neighbours at distance d?
d = 20
avg_neighbors = (dists < d).sum(axis=1).mean() - 1
print(f"Average neighbours at d={d}: {avg_neighbors:.1f}")

r = rss.morans_i(x, y, values, distance_threshold=d)
```

Using `distance_threshold=None` builds a fully global weight matrix (all pairs) — often appropriate for small datasets or when spatial structure is expected at all scales.

---

### Q: What is a variogram, and when do I need one?

**A:** Moran's I gives a single number summarising spatial autocorrelation across the entire study area. A variogram goes further: it shows *how* spatial autocorrelation changes with distance — specifically, when it effectively disappears.

The three key parameters of a fitted variogram model are:
- **Nugget** (γ at h=0): measurement error or variation at distances smaller than your minimum sampling distance.
- **Sill** (γ as h→∞): total variance in the data; the variogram plateaus here.
- **Range**: the distance at which the variogram reaches the sill — beyond this, points are spatially uncorrelated.

```python
# Step 1: compute the experimental variogram
vario = rss.compute_variogram(x, y, values, n_lags=15)

# Step 2: fit a model to the experimental variogram
fit = rss.fit_variogram(vario["lags"], vario["gamma"], model="spherical")
print(f"Range: {fit['range_param']:.1f}  Sill: {fit['sill']:.3f}  Nugget: {fit['nugget']:.3f}")
print(f"RMSE: {fit['rmse']:.4f}")

# Step 3: predict semivariance at a new distance
fit["model_fn"](15.0)   # γ at h=15
```

Use a variogram when you need to understand the scale of spatial structure, compare multiple fields, or prepare for kriging interpolation.

---

### Q: Which variogram model should I use — spherical, exponential, or Gaussian?

**A:** All three are valid; choose by fit quality (RMSE) and by the shape of your experimental variogram:

| Model | Behaviour near origin | Practical implication |
|---|---|---|
| **Spherical** | Linear near h=0 | Abrupt transition from correlated to uncorrelated; common in geology |
| **Exponential** | Linear near h=0 (slower approach to sill) | Never fully reaches sill; correlated at all distances |
| **Gaussian** | Parabolic near h=0 (smooth) | Very smooth spatial variation; often soil or atmospheric data |

```python
for model in ("spherical", "exponential", "gaussian"):
    fit = rss.fit_variogram(vario["lags"], vario["gamma"], model=model)
    print(f"{model:12s}  RMSE={fit['rmse']:.4f}")
```

Start with spherical (it's the most commonly appropriate). Switch to exponential if the variogram doesn't fully plateau, or Gaussian if the data is very smooth.

---

## Still Have Questions?

- **GitHub Issues**: [Ask a question](https://github.com/kylejones200/real_simple_stats/issues)
- **Documentation**: [Full docs](https://real-simple-stats.readthedocs.io/)
- **Examples**: [Interactive tutorials](INTERACTIVE_EXAMPLES.md)

---

**Last Updated**: 2025
**Version**: 0.3.0
