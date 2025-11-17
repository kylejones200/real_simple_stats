# What Can Statistics Do?

Statistics helps you accomplish five main objectives. Understanding these categories helps you choose the right statistical method for your question.

## The Five Objectives

### 1. **Describe** 📊
Characterize populations and samples using descriptive statistics, visualizations, and summary measures.

**What it answers**: "What does my data look like?"

**Real Simple Stats Functions**:
- `mean()`, `median()`, `mode()` - Central tendency
- `standard_deviation()`, `variance()` - Variability
- `five_number_summary()` - Complete distribution summary
- `coefficient_of_variation()` - Relative variability
- `plots.histogram()`, `plots.box_plot()` - Visualizations

**Example Applications**:
- Opinion surveys: "What's the average satisfaction rating?"
- Demographic surveys: "What's the age distribution?"
- Quality control: "What's the typical defect rate?"

**Example**:
```python
from real_simple_stats import descriptive_statistics as desc

data = [23, 25, 28, 30, 32, 35, 38, 40]
print(f"Mean: {desc.mean(data)}")
print(f"Summary: {desc.five_number_summary(data)}")
```

---

### 2. **Compare or Test** 🔬
Detect differences between statistical populations or reference values using hypothesis tests.

**What it answers**: "Are these groups different?" or "Is this different from expected?"

**Real Simple Stats Functions**:
- `t_score()` - T-test statistics
- `critical_value_t()`, `critical_value_z()` - Critical values
- `cohens_d()`, `hedges_g()` - Effect sizes
- `one_way_anova()` - Multiple group comparisons
- `chi_square_statistic()` - Categorical comparisons

**Example Applications**:
- A/B tests: "Does the new design increase conversions?"
- Marketing effectiveness: "Which ad campaign performed better?"
- Quality control: "Is this batch different from the standard?"

**Example**:
```python
from real_simple_stats import effect_sizes as es

group_a = [78, 82, 85, 79, 83]
group_b = [72, 75, 78, 74, 76]
effect = es.cohens_d(group_a, group_b)
print(f"Effect size: {effect:.3f}")
```

---

### 3. **Identify or Classify** 🏷️
Classify and identify entities or groups using descriptive statistics, intervals, and multivariate techniques.

**What it answers**: "What category does this belong to?" or "What patterns exist?"

**Real Simple Stats Functions**:
- `pca()` - Principal Component Analysis (dimensionality reduction)
- `factor_analysis()` - Factor analysis
- Descriptive statistics for classification
- Confidence intervals for identification

**Example Applications**:
- Customer churn: "Which customers are likely to leave?"
- Employee retention: "What characteristics predict retention?"
- Medical diagnosis: "Does this patient have condition X?"

**Note**: Real Simple Stats focuses on basic classification tools. Advanced machine learning classification (random forests, neural networks) is beyond the library's scope.

---

### 4. **Predict** 🔮
Forecast future values using regression, time series, and predictive models.

**What it answers**: "What will happen next?" or "What value should I expect?"

**Real Simple Stats Functions**:
- `linear_regression()` - Simple linear regression
- `multiple_regression()` - Multiple regression
- `moving_average()` - Time series smoothing
- `autocorrelation()` - Time series patterns
- `linear_trend()` - Trend analysis

**Example Applications**:
- Credit scoring: "What's this person's credit score?"
- House pricing: "What should this house cost?"
- Sales forecasting: "What will next quarter's sales be?"

**Example**:
```python
from real_simple_stats import linear_regression_utils as lr

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
slope, intercept, r_squared = lr.linear_regression(x, y)
print(f"Prediction equation: y = {slope:.2f}x + {intercept:.2f}")
```

---

### 5. **Explain** 🔍
Understand relationships and causes using regression, ANOVA, and explanatory models.

**What it answers**: "Why did this happen?" or "What explains this relationship?"

**Real Simple Stats Functions**:
- `linear_regression()` - Relationship between variables
- `pearson_correlation()` - Linear relationships
- `one_way_anova()` - Explaining variance
- `multiple_regression()` - Multiple explanatory variables
- `pca()` - Understanding underlying factors

**Example Applications**:
- Research: "What factors explain student performance?"
- Root cause analysis: "Why did sales drop?"
- Policy analysis: "What explains unemployment rates?"

**Example**:
```python
from real_simple_stats import linear_regression_utils as lr

# Study hours (x) vs Test scores (y)
hours = [5, 10, 15, 20, 25]
scores = [60, 70, 75, 85, 90]
slope, intercept, r_squared = lr.linear_regression(hours, scores)
print(f"R² = {r_squared:.3f} - {r_squared*100:.1f}% of variance explained")
```

---

## Quick Reference Table

| Objective | Question | Key Functions | Example Use Case |
|----------|----------|---------------|------------------|
| **Describe** | "What does my data look like?" | `mean()`, `five_number_summary()`, `plots.histogram()` | Survey summaries, quality control |
| **Compare/Test** | "Are these groups different?" | `cohens_d()`, `critical_value_t()`, `anova()` | A/B tests, treatment effects |
| **Identify/Classify** | "What category is this?" | `pca()`, `factor_analysis()` | Customer segmentation, diagnosis |
| **Predict** | "What will happen next?" | `linear_regression()`, `moving_average()` | Sales forecasting, credit scores |
| **Explain** | "Why did this happen?" | `linear_regression()`, `correlation()`, `anova()` | Research, root cause analysis |

---

## Choosing the Right Method

**Start with your question**, then choose the appropriate category:

1. **"What's the average?"** → **Describe** (use `mean()`, `median()`)
2. **"Is group A different from group B?"** → **Compare/Test** (use `cohens_d()`, t-tests)
3. **"What category does this belong to?"** → **Identify/Classify** (use `pca()`, classification)
4. **"What will sales be next quarter?"** → **Predict** (use `linear_regression()`, `moving_average()`)
5. **"What explains the relationship?"** → **Explain** (use `regression()`, `correlation()`)

---

## Real-World Workflow Example

**Scenario**: Analyzing customer satisfaction survey

1. **Describe**: Calculate mean satisfaction, create histogram
   ```python
   mean_satisfaction = desc.mean(satisfaction_scores)
   desc.plots.histogram(satisfaction_scores)
   ```

2. **Compare**: Compare satisfaction between regions
   ```python
   effect = es.cohens_d(region_a_scores, region_b_scores)
   ```

3. **Explain**: Understand what drives satisfaction
   ```python
   slope, intercept, r2 = lr.linear_regression(service_quality, satisfaction)
   ```

4. **Predict**: Forecast future satisfaction
   ```python
   predicted = slope * future_service_quality + intercept
   ```

---

## Limitations

Real Simple Stats focuses on **basic to intermediate statistics**. For advanced needs:

- **Advanced econometrics** (DiD, IV, RDD) → Use `statsmodels` or `causal inference` libraries
- **Machine learning classification** → Use `scikit-learn`
- **Advanced time series** (ARIMA, VAR) → Use `statsmodels` or `pmdarima`
- **Spatial statistics** → Use specialized spatial libraries

Real Simple Stats provides the **foundation** - the statistical thinking and basic methods that apply across all domains.

---

## Further Reading

- [Quick Start Guide](../README.md#-quick-start)
- [Statistical Recipes](../examples/recipes/)
- [API Reference](../docs/source/api/)
- [FAQ](../docs/FAQ.md)

