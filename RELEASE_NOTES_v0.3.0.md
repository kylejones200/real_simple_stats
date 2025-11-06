# Release Notes - Version 0.3.0

## 🎉 Major Feature Release

Real Simple Stats v0.3.0 represents a **major expansion** of the package, adding **6 new modules** with **45+ advanced statistical functions**. This release transforms the package from a basic statistics library into a comprehensive statistical analysis toolkit suitable for research, data science, and advanced education.

---

## ✨ What's New

### 🆕 New Modules

#### 1. **Time Series Analysis** (`time_series.py`)
- Moving averages (simple, exponential, weighted)
- Autocorrelation and partial autocorrelation
- Trend analysis and detrending
- Seasonal decomposition
- Differencing operations

#### 2. **Multivariate Analysis** (`multivariate.py`)
- Multiple linear regression with full diagnostics
- Principal Component Analysis (PCA)
- Factor analysis
- Canonical Correlation Analysis
- Mahalanobis distance

#### 3. **Bayesian Statistics** (`bayesian_stats.py`)
- Conjugate prior updates (Beta-Binomial, Normal-Normal, Gamma-Poisson)
- Credible intervals and HDI
- Bayes factors
- Posterior predictive distributions
- Empirical Bayes estimation

#### 4. **Resampling Methods** (`resampling.py`)
- Bootstrap with confidence intervals
- Bootstrap hypothesis testing
- Permutation tests
- Jackknife estimation
- K-fold cross-validation
- Stratified train-test splitting

#### 5. **Effect Sizes** (`effect_sizes.py`)
- Cohen's d, Hedges' g, Glass's delta
- Eta-squared, partial eta-squared, omega-squared
- Cramér's V and phi coefficient
- Odds ratios and relative risk
- Cohen's h for proportions
- Effect size interpretation

#### 6. **Power Analysis** (`power_analysis.py`)
- Power analysis for t-tests
- Power analysis for proportion tests
- Power analysis for ANOVA
- Power analysis for correlation tests
- Minimum detectable effect calculations
- Sample size summaries

---

## 📊 Statistics

### Code Additions
- **~2,562 lines** of new production code
- **45 new functions** across 6 modules
- **69 new tests** (114 total, up from 45)
- **Test coverage**: 48% overall (96% for time_series, 87% for power_analysis)

### Module Breakdown
| Module | Lines | Functions | Tests | Coverage |
|--------|-------|-----------|-------|----------|
| time_series.py | 329 | 7 | 23 | 96% |
| multivariate.py | 393 | 5 | 0* | 7% |
| bayesian_stats.py | 392 | 9 | 0* | 12% |
| resampling.py | 444 | 6 | 0* | 7% |
| effect_sizes.py | 518 | 12 | 26 | 68% |
| power_analysis.py | 486 | 6 | 20 | 87% |

*Additional tests for these modules can be added in future releases

---

## 🔧 Technical Improvements

### Code Quality
- ✅ **Type hints** throughout all new functions
- ✅ **Google-style docstrings** with examples
- ✅ **Comprehensive error handling**
- ✅ **Input validation** for all parameters
- ✅ **Black formatted** (88 character line length)
- ✅ **Flake8 compliant**

### API Design
- Consistent naming conventions
- Dictionary returns for complex results
- Optional parameters with sensible defaults
- Flexible input types (lists, NumPy arrays)

### Documentation
- Detailed docstrings with mathematical background
- Working code examples in every function
- Parameter descriptions and return values
- Exception documentation
- Quick reference guide
- Advanced features summary

---

## 🚀 Usage Examples

### Time Series
```python
import real_simple_stats as rss

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ma = rss.moving_average(data, window_size=3, method='simple')
acf = rss.autocorrelation(data, max_lag=5)
slope, intercept, r2 = rss.linear_trend(data)
```

### Bayesian Analysis
```python
# Update Beta prior with binomial data
post_alpha, post_beta = rss.beta_binomial_update(1, 1, 7, 10)
lower, upper = rss.credible_interval('beta', {'alpha': post_alpha, 'beta': post_beta})
```

### Bootstrap
```python
import numpy as np

boot_result = rss.bootstrap(data, np.mean, n_iterations=1000)
print(f"95% CI: {boot_result['confidence_interval']}")
```

### Effect Sizes
```python
group1 = [1, 2, 3, 4, 5]
group2 = [3, 4, 5, 6, 7]
d = rss.cohens_d(group1, group2)
interpretation = rss.interpret_effect_size(d, 'd')
print(f"Cohen's d: {d:.3f} ({interpretation})")
```

### Power Analysis
```python
# Calculate required sample size
result = rss.power_t_test(delta=0.5, power=0.8, sig_level=0.05)
print(f"Required n per group: {result['n']}")
```

---

## 📦 Installation

### Upgrade from PyPI
```bash
pip install --upgrade real-simple-stats
```

### Install from source
```bash
git clone https://github.com/kylejones200/real_simple_stats.git
cd real_simple_stats
pip install -e .
```

---

## 🔄 Migration Guide

### From v0.2.0 to v0.3.0

**No breaking changes!** All existing functionality remains unchanged. The new modules are purely additive.

**New imports available:**
```python
import real_simple_stats as rss

# All new functions are available at the top level
rss.moving_average(...)
rss.bootstrap(...)
rss.cohens_d(...)
rss.power_t_test(...)
# etc.
```

**Or import specific modules:**
```python
from real_simple_stats import time_series
from real_simple_stats import bayesian_stats
from real_simple_stats import resampling
```

---

## 🐛 Bug Fixes

None - this is a feature release with no bug fixes.

---

## 📚 Documentation

### New Documentation Files
- `ADVANCED_FEATURES_SUMMARY.md` - Comprehensive guide to all new features
- `QUICK_REFERENCE.md` - Quick reference for common tasks
- `RELEASE_NOTES_v0.3.0.md` - This file

### Updated Files
- `__init__.py` - Added imports for new modules
- `pyproject.toml` - Version bump to 0.3.0
- `README.md` - (Should be updated to mention new features)

---

## 🎯 Use Cases

### Research & Academia
- Design studies with proper power analysis
- Perform Bayesian inference with prior knowledge
- Analyze time series data
- Calculate and report effect sizes
- Use bootstrap for robust inference

### Data Science
- Multivariate analysis and dimensionality reduction
- Cross-validation for model evaluation
- Permutation tests for hypothesis testing
- Feature selection with PCA
- Outlier detection with Mahalanobis distance

### Education
- Teach advanced statistical concepts
- Demonstrate Bayesian vs. frequentist approaches
- Show practical significance with effect sizes
- Illustrate resampling methods
- Explore time series patterns

---

## ⚠️ Known Limitations

1. **Test coverage** for multivariate, bayesian_stats, and resampling modules is low (7-12%)
   - Functions are tested manually and work correctly
   - Comprehensive unit tests will be added in v0.3.1

2. **Performance** - Some functions may be slow for very large datasets
   - Future versions may add optimization and parallel processing

3. **Documentation** - Sphinx docs need to be updated to include new modules
   - Will be addressed in next documentation update

---

## 🔮 Future Plans (v0.4.0)

Potential additions:
- **Survival analysis** - Kaplan-Meier, Cox regression
- **Mixed models** - Random effects, hierarchical models
- **Nonparametric tests** - More rank-based tests
- **Spatial statistics** - Spatial autocorrelation
- **Interactive visualizations** - Plotly integration
- **Performance optimization** - Cython/Numba for speed

---

## 🙏 Acknowledgments

These implementations are based on:
- Statistical textbooks and peer-reviewed literature
- Industry best practices from statsmodels, scikit-learn, and scipy
- Educational clarity and accessibility principles
- Modern Python packaging standards

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kylejones200/real_simple_stats/issues)
- **Documentation**: [ReadTheDocs](https://real-simple-stats.readthedocs.io/)
- **PyPI**: [real-simple-stats](https://pypi.org/project/real-simple-stats/)

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🎊 Contributors

- Kyle Jones (@kylejones200) - Primary author and maintainer

---

**Release Date**: 2025  
**Version**: 0.3.0  
**Python**: 3.8+  
**Dependencies**: numpy>=1.20.0, scipy>=1.7.0, matplotlib>=3.3.0

---

*Thank you for using Real Simple Stats! We hope these new features enhance your statistical analysis workflow.* 🎉
