# Real Simple Stats Handbook — Python Edition

[![PyPI version](https://badge.fury.io/py/real-simple-stats.svg)](https://badge.fury.io/py/real-simple-stats)
[![Python versions](https://img.shields.io/pypi/pyversions/real-simple-stats.svg)](https://pypi.org/project/real-simple-stats/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/kylejones200/real_simple_stats/workflows/Continuous%20Integration/badge.svg)](https://github.com/kylejones200/real_simple_stats/actions)
[![Documentation](https://github.com/kylejones200/real_simple_stats/workflows/Documentation/badge.svg)](https://kylejones200.github.io/real_simple_stats/)
[![PyPI Publish](https://github.com/kylejones200/real_simple_stats/workflows/Publish%20to%20PyPI/badge.svg)](https://github.com/kylejones200/real_simple_stats/actions)

A comprehensive, educational Python statistics library covering basic through advanced statistical concepts. Perfect for students, educators, and anyone learning statistics!

## 🚀 Quick Start

```bash
pip install real-simple-stats
```

```python
import real_simple_stats as rss

# Basic descriptive statistics
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Mean: {rss.mean(data)}")
print(f"Median: {rss.median(data)}")
print(f"Standard Deviation: {rss.sample_std_dev(data)}")

# Probability calculations
print(f"Binomial probability: {rss.binomial_probability(n=10, k=3, p=0.5)}")
print(f"Normal distribution: {rss.normal_probability(x=1.96, mean=0, std_dev=1)}")

# Hypothesis testing
t_stat, p_value = rss.one_sample_t_test(data, mu=5)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

## 📚 Features

### Core Statistics
- **Descriptive Statistics**: mean, median, mode, variance, standard deviation, five-number summary
- **Data Validation**: discrete/continuous detection, outlier identification
- **Frequency Analysis**: frequency tables, cumulative distributions

### Probability & Distributions
- **Basic Probability**: simple, joint, conditional, Bayes' theorem
- **Combinatorics**: combinations, permutations, fundamental counting
- **Distributions**: binomial, normal, Poisson, geometric, exponential
- **Distribution Properties**: PDF, CDF, mean, variance calculations

### Statistical Inference
- **Hypothesis Testing**: z-tests, t-tests, F-tests, chi-square tests
- **Confidence Intervals**: for means, proportions, differences
- **P-values and Critical Values**: comprehensive lookup tables
- **Effect Sizes**: Cohen's d, correlation coefficients

### Advanced Analysis
- **Linear Regression**: slope, intercept, R², residual analysis
- **ANOVA**: one-way and two-way analysis of variance
- **Non-parametric Tests**: Mann-Whitney U, Wilcoxon signed-rank
- **Sampling Theory**: Central Limit Theorem demonstrations

### Utilities
- **Interactive Glossary**: 200+ statistical terms with definitions
- **Data Visualization**: built-in plotting functions
- **Command Line Interface**: quick calculations from terminal
- **Educational Examples**: step-by-step worked problems

## 📖 Documentation

Comprehensive documentation is available at [real-simple-stats.readthedocs.io](https://real-simple-stats.readthedocs.io/)

### Quick Examples

#### Descriptive Statistics
```python
from real_simple_stats import descriptive_statistics as desc

data = [12, 15, 18, 20, 22, 25, 28, 30]
summary = desc.five_number_summary(data)
print(summary)  # {'min': 12, 'Q1': 16.5, 'median': 21, 'Q3': 26.5, 'max': 30}
```

#### Probability Distributions
```python
from real_simple_stats import normal_distributions as norm

# Calculate probability that X < 1.96 for standard normal
prob = norm.normal_cdf(1.96, mean=0, std_dev=1)
print(f"P(X < 1.96) = {prob:.4f}")  # 0.9750
```

#### Hypothesis Testing
```python
from real_simple_stats import hypothesis_testing as ht

# One-sample t-test
sample = [23, 25, 28, 30, 32, 35, 38, 40]
t_stat, p_val = ht.one_sample_t_test(sample, mu=30)
print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
```

## 🛠️ Installation

### From PyPI (Recommended)
```bash
pip install real-simple-stats
```

### From Source
```bash
git clone https://github.com/kylejones200/real_simple_stats.git
cd real_simple_stats
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/kylejones200/real_simple_stats.git
cd real_simple_stats
pip install -e ".[dev]"
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=real_simple_stats --cov-report=html
```

## 🎯 Use Cases

- **Education**: Perfect for statistics courses and self-study
- **Research**: Quick statistical calculations and hypothesis testing
- **Data Analysis**: Exploratory data analysis and validation
- **Reference**: Comprehensive statistical function library

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by educational statistics textbooks and courses
- Built with modern Python packaging standards
- Designed for clarity and educational value

## 📊 Project Stats

- **Language**: Python 3.8+
- **Dependencies**: NumPy, SciPy, Matplotlib
- **Test Coverage**: >95%
- **Documentation**: Sphinx + ReadTheDocs

---

**Made with ❤️ for statistics education**
