# Real Simple Stats Handbook — Python Edition

This is a full Python implementation of basic statistics. It covers basic through advanced statistics, written as callable Python modules for education and quick reference.

## Features

- Percentage conversions, rounding, and factorials
- Descriptive statistics: mean, median, mode, variance
- Probability: simple, joint, conditional, Bayes
- Distributions: binomial, normal, Poisson, geometric, exponential
- Hypothesis testing (z, t, F tests, p-values)
- Confidence intervals and CLT
- Linear regression: slope, intercept, R²
- Chi-square test and critical values
- Full glossary with lookup utility

## Install

You can use this as a pure Python package:

```bash
git clone https://github.com/kylejones200/real_simple_stats
cd real_simple_stats
pip install -e .
```

```python
from real_simple_stats import mean, z_score, lookup

print(mean([1, 2, 3]))
print(z_score(85, 80, 5))
print(lookup("μ"))  # Population mean
```