# Interactive Examples - Try Real Simple Stats Online

Run Real Simple Stats directly in your browser without any installation!

---

## 🚀 Quick Start Options

### Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/interactive_tutorial.ipynb)



### Option 3: Local Jupyter
```bash
pip install real-simple-stats jupyter
jupyter notebook
```

---

## 📚 Interactive Notebooks

### 1. Getting Started Tutorial
**File:** `examples/01_getting_started.ipynb`

**Topics Covered:**
- Installation and setup
- Basic descriptive statistics
- Creating your first analysis
- Interpreting results

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/01_getting_started.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/01_getting_started.ipynb)

---

### 2. Hypothesis Testing Workshop
**File:** `examples/02_hypothesis_testing.ipynb`

**Topics Covered:**
- t-tests (one-sample, two-sample, paired)
- Chi-square tests
- ANOVA
- Interpreting p-values
- Effect sizes

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/02_hypothesis_testing.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/02_hypothesis_testing.ipynb)

---

### 3. Regression Analysis
**File:** `examples/03_regression_analysis.ipynb`

**Topics Covered:**
- Simple linear regression
- Multiple regression
- Model diagnostics
- Predictions
- Visualization

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/03_regression_analysis.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/03_regression_analysis.ipynb)

---

### 4. Time Series Analysis
**File:** `examples/04_time_series.ipynb`

**Topics Covered:**
- Moving averages
- Autocorrelation
- Trend analysis
- Seasonal decomposition
- Forecasting basics

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/04_time_series.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/04_time_series.ipynb)

---

### 5. Bayesian Statistics
**File:** `examples/05_bayesian_stats.ipynb`

**Topics Covered:**
- Prior and posterior distributions
- Conjugate updates
- Credible intervals
- Bayes factors
- Practical applications

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/05_bayesian_stats.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/05_bayesian_stats.ipynb)

---

### 6. Resampling Methods
**File:** `examples/06_resampling.ipynb`

**Topics Covered:**
- Bootstrap confidence intervals
- Permutation tests
- Jackknife estimation
- Cross-validation
- When to use each method

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/06_resampling.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/06_resampling.ipynb)

---

### 7. Power Analysis & Study Design
**File:** `examples/07_power_analysis.ipynb`

**Topics Covered:**
- Sample size calculations
- Power analysis for different tests
- Effect size planning
- Study design optimization

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/07_power_analysis.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/07_power_analysis.ipynb)

---

### 8. Real-World Case Studies
**File:** `examples/08_case_studies.ipynb`

**Topics Covered:**
- A/B testing analysis
- Survey data analysis
- Clinical trial simulation
- Quality control
- Social science research

**Launch:**
- [Google Colab](https://colab.research.google.com/github/kylejones200/real_simple_stats/blob/main/examples/08_case_studies.ipynb)
- [Binder](https://mybinder.org/v2/gh/kylejones200/real_simple_stats/main?filepath=examples/08_case_studies.ipynb)

---

## 🎯 Quick Examples

### Example 1: Basic Statistics
```python
# Run this in Colab or Binder!
!pip install real-simple-stats

import real_simple_stats as rss

# Your data
data = [23, 25, 28, 30, 32, 35, 38, 40, 42, 45]

# Calculate statistics
print(f"Mean: {rss.mean(data):.2f}")
print(f"Median: {rss.median(data)}")
print(f"Std Dev: {rss.sample_std_dev(data):.2f}")
print(f"5-Number Summary: {rss.five_number_summary(data)}")
```

---

### Example 2: Hypothesis Testing
```python
import real_simple_stats as rss

# Two groups to compare
control = [23, 25, 28, 30, 32]
treatment = [28, 30, 35, 38, 40]

# Perform t-test
t_stat, p_value = rss.two_sample_t_test(control, treatment)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# Calculate effect size
d = rss.cohens_d(control, treatment)
interpretation = rss.interpret_effect_size(d, 'd')
print(f"Cohen's d: {d:.3f} ({interpretation})")
```

---

### Example 3: Regression Analysis
```python
import real_simple_stats as rss

# Data
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
test_scores = [55, 60, 65, 70, 75, 80, 85, 90]

# Fit regression
slope, intercept, r_value, p_value, std_err = rss.linear_regression(
    hours_studied, test_scores
)

print(f"Equation: y = {slope:.2f}x + {intercept:.2f}")
print(f"R² = {r_value**2:.3f}")
print(f"p-value = {p_value:.4f}")

# Make prediction
predicted_score = rss.regression_equation(10, slope, intercept)
print(f"Predicted score for 10 hours: {predicted_score:.1f}")
```

---

### Example 4: Bootstrap Confidence Interval
```python
import real_simple_stats as rss
import numpy as np

# Sample data
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Bootstrap CI for the mean
result = rss.bootstrap(data, np.mean, n_iterations=1000, confidence_level=0.95)

print(f"Sample mean: {result['statistic']:.2f}")
print(f"95% CI: {result['confidence_interval']}")
print(f"Standard error: {np.std(result['bootstrap_distribution']):.2f}")
```

---

### Example 5: Power Analysis
```python
import real_simple_stats as rss

# Calculate required sample size
result = rss.power_t_test(
    delta=0.5,      # Effect size (Cohen's d)
    power=0.8,      # Desired power
    sig_level=0.05  # Significance level
)

print(f"Required sample size per group: {result['n']}")
print(f"Total participants needed: {result['n'] * 2}")
```

---

## 🎓 Educational Modules

### Module 1: Understanding p-values
**Interactive Visualization**

```python
import real_simple_stats as rss
import matplotlib.pyplot as plt
import numpy as np

# Simulate null distribution
null_samples = [rss.mean(np.random.normal(0, 1, 30)) for _ in range(1000)]

# Your observed statistic
observed = 0.5

# Calculate p-value
p_value = sum(abs(x) >= abs(observed) for x in null_samples) / len(null_samples)

# Visualize
plt.hist(null_samples, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(observed, color='red', linestyle='--', label=f'Observed ({observed})')
plt.axvline(-observed, color='red', linestyle='--')
plt.title(f'Null Distribution (p-value = {p_value:.3f})')
plt.xlabel('Test Statistic')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

---

### Module 2: Effect Size Interpretation
**Interactive Calculator**

```python
import real_simple_stats as rss

def effect_size_calculator(group1, group2):
    """Interactive effect size calculator"""

    # Calculate multiple effect sizes
    d = rss.cohens_d(group1, group2)
    g = rss.hedges_g(group1, group2)

    # Interpret
    d_interp = rss.interpret_effect_size(d, 'd')

    print("=" * 50)
    print("EFFECT SIZE ANALYSIS")
    print("=" * 50)
    print(f"Cohen's d:  {d:.3f} ({d_interp})")
    print(f"Hedges' g:  {g:.3f}")
    print(f"\nGroup 1: M = {rss.mean(group1):.2f}, SD = {rss.sample_std_dev(group1):.2f}")
    print(f"Group 2: M = {rss.mean(group2):.2f}, SD = {rss.sample_std_dev(group2):.2f}")
    print("=" * 50)

# Try it!
control = [20, 22, 24, 26, 28]
treatment = [25, 28, 30, 32, 35]
effect_size_calculator(control, treatment)
```

---

### Module 3: Confidence Interval Simulator
**Visualize Coverage**

```python
import real_simple_stats as rss
import matplotlib.pyplot as plt
import numpy as np

def simulate_confidence_intervals(true_mean=100, true_std=15, n=30,
                                   n_simulations=100, confidence=0.95):
    """Simulate confidence intervals to show coverage"""

    covers_true_mean = 0

    plt.figure(figsize=(12, 8))

    for i in range(n_simulations):
        # Generate sample
        sample = np.random.normal(true_mean, true_std, n)

        # Calculate CI
        sample_mean = rss.mean(sample)
        sample_std = rss.sample_std_dev(sample)
        lower, upper = rss.confidence_interval_unknown_std(
            sample_mean, sample_std, n, confidence
        )

        # Check if CI covers true mean
        covers = lower <= true_mean <= upper
        covers_true_mean += covers

        # Plot
        color = 'green' if covers else 'red'
        plt.plot([lower, upper], [i, i], color=color, alpha=0.5)
        plt.plot(sample_mean, i, 'o', color=color, markersize=3)

    # Add true mean line
    plt.axvline(true_mean, color='blue', linestyle='--', linewidth=2,
                label=f'True Mean ({true_mean})')

    coverage = covers_true_mean / n_simulations
    plt.title(f'{confidence*100}% Confidence Intervals\n'
              f'Coverage: {coverage*100:.1f}% (Expected: {confidence*100}%)')
    plt.xlabel('Value')
    plt.ylabel('Simulation')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Coverage rate: {coverage*100:.1f}%")
    print(f"Expected: {confidence*100}%")

# Run simulation
simulate_confidence_intervals()
```

---

## 🔬 Advanced Interactive Examples

### Bayesian Updating Visualization
```python
import real_simple_stats as rss
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def visualize_bayesian_update(prior_alpha=1, prior_beta=1,
                               successes=7, trials=10):
    """Visualize Bayesian updating"""

    # Update
    post_alpha, post_beta = rss.beta_binomial_update(
        prior_alpha, prior_beta, successes, trials
    )

    # Plot
    x = np.linspace(0, 1, 1000)
    prior = stats.beta.pdf(x, prior_alpha, prior_beta)
    posterior = stats.beta.pdf(x, post_alpha, post_beta)

    plt.figure(figsize=(10, 6))
    plt.plot(x, prior, label='Prior', linewidth=2)
    plt.plot(x, posterior, label='Posterior', linewidth=2)
    plt.axvline(successes/trials, color='red', linestyle='--',
                label=f'MLE ({successes}/{trials})')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title(f'Bayesian Update: Beta({prior_alpha}, {prior_beta}) → '
              f'Beta({post_alpha}, {post_beta})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Try it!
visualize_bayesian_update(prior_alpha=2, prior_beta=2, successes=7, trials=10)
```

---

## 📱 Mobile-Friendly Options

### Google Colab on Mobile
1. Open [Google Colab](https://colab.research.google.com/)
2. Sign in with Google account
3. File → Open notebook → GitHub
4. Enter: `kylejones200/real_simple_stats`
5. Select any example notebook

### Jupyter Lite (No Installation)
Try Jupyter in your browser: [JupyterLite](https://jupyter.org/try)

---

## 🎮 Interactive Widgets

### Widget Example: Statistical Test Selector
```python
from ipywidgets import interact, widgets
import real_simple_stats as rss

@interact(
    test_type=widgets.Dropdown(
        options=['One-Sample t-test', 'Two-Sample t-test', 'Paired t-test'],
        description='Test:'
    ),
    alpha=widgets.FloatSlider(min=0.01, max=0.10, step=0.01, value=0.05,
                               description='α:')
)
def run_test(test_type, alpha):
    """Interactive test runner"""

    if test_type == 'One-Sample t-test':
        data = [23, 25, 28, 30, 32]
        mu0 = 30
        t_stat, p_value = rss.one_sample_t_test(data, mu0)
        print(f"One-Sample t-test (H₀: μ = {mu0})")

    elif test_type == 'Two-Sample t-test':
        group1 = [23, 25, 28, 30, 32]
        group2 = [28, 30, 35, 38, 40]
        t_stat, p_value = rss.two_sample_t_test(group1, group2)
        print(f"Two-Sample t-test")

    elif test_type == 'Paired t-test':
        before = [23, 25, 28, 30, 32]
        after = [25, 27, 30, 33, 35]
        t_stat, p_value = rss.paired_t_test(before, after)
        print(f"Paired t-test")

    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significance level: {alpha}")

    if p_value < alpha:
        print(f"✓ Reject H₀ (p < {alpha})")
    else:
        print(f"✗ Fail to reject H₀ (p ≥ {alpha})")
```

---

## 🌐 Web-Based Demos

### Streamlit Apps
Coming soon: Interactive web apps for common analyses

### Observable Notebooks
Coming soon: JavaScript-based interactive visualizations

---

## 📦 Setup Instructions

### For Google Colab
```python
# First cell - install package
!pip install real-simple-stats

# Second cell - import and use
import real_simple_stats as rss
# Your code here
```

### For Binder
No setup needed! Everything is pre-configured.

### For Local Jupyter
```bash
# Install
pip install real-simple-stats jupyter matplotlib

# Launch
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

---

## 🎯 Learning Paths

### Path 1: Complete Beginner
1. Getting Started Tutorial
2. Hypothesis Testing Workshop
3. Regression Analysis
4. Case Studies

### Path 2: Intermediate User
1. Time Series Analysis
2. Resampling Methods
3. Power Analysis
4. Advanced Case Studies

### Path 3: Advanced Topics
1. Bayesian Statistics
2. Multivariate Analysis
3. Custom Workflows
4. Research Applications

---

## 💡 Tips for Interactive Learning

**Best Practices:**
1. **Run all cells** from top to bottom first
2. **Modify parameters** to see how results change
3. **Try your own data** by replacing example datasets
4. **Save your work** (File → Save a copy in Drive for Colab)
5. **Share notebooks** with collaborators

**Common Issues:**
- **Package not found**: Run `!pip install real-simple-stats` first
- **Kernel died**: Restart kernel and run all cells again
- **Import errors**: Make sure all imports are in the first cell

---

## 🔗 Additional Resources

- **Documentation**: [ReadTheDocs](https://real-simple-stats.readthedocs.io/)
- **GitHub**: [Source Code](https://github.com/kylejones200/real_simple_stats)
- **PyPI**: [Package Page](https://pypi.org/project/real-simple-stats/)
- **Issues**: [Report Problems](https://github.com/kylejones200/real_simple_stats/issues)

---

## 🤝 Contributing Examples

Want to add your own interactive example?

1. Fork the repository
2. Create a new notebook in `examples/`
3. Add Colab/Binder badges
4. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

**Happy Learning! 🎓**

Try the examples above and explore statistical concepts interactively!
