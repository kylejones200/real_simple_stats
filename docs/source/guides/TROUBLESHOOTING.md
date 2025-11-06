# Troubleshooting Guide

Solutions to common errors and issues when using Real Simple Stats.

---

## 🚨 Installation Issues

### Error: "ModuleNotFoundError: No module named 'real_simple_stats'"

**Symptoms:**
```python
import real_simple_stats as rss
# ModuleNotFoundError: No module named 'real_simple_stats'
```

**Solutions:**

1. **Install the package:**
   ```bash
   pip install real-simple-stats
   ```

2. **Check installation:**
   ```bash
   pip list | grep real-simple-stats
   ```

3. **Verify Python environment:**
   ```bash
   which python
   which pip
   ```

4. **For Jupyter/Colab:**
   ```python
   !pip install real-simple-stats
   import real_simple_stats as rss
   ```

---

### Error: "pip: command not found"

**Solution:**
```bash
# Try pip3 instead
pip3 install real-simple-stats

# Or use python -m pip
python -m pip install real-simple-stats
```

---

### Error: "Permission denied" during installation

**Solution:**
```bash
# Install for current user only
pip install --user real-simple-stats

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install real-simple-stats
```

---

### Error: Package version conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Solutions:**

1. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

2. **Use virtual environment:**
   ```bash
   python -m venv clean_env
   source clean_env/bin/activate
   pip install real-simple-stats
   ```

3. **Check dependencies:**
   ```bash
   pip show real-simple-stats
   ```

---

## 🐍 Import Errors

### Error: "ImportError: cannot import name 'function_name'"

**Symptoms:**
```python
from real_simple_stats import nonexistent_function
# ImportError: cannot import name 'nonexistent_function'
```

**Solutions:**

1. **Check function name:**
   ```python
   import real_simple_stats as rss
   print(dir(rss))  # List all available functions
   ```

2. **Use correct import:**
   ```python
   # Correct
   from real_simple_stats import mean, median

   # Or
   import real_simple_stats as rss
   rss.mean([1, 2, 3])
   ```

3. **Check version:**
   ```python
   import real_simple_stats
   print(real_simple_stats.__version__)
   ```

---

### Error: "AttributeError: module 'real_simple_stats' has no attribute 'X'"

**Cause:** Function doesn't exist or typo in name.

**Solution:**
```python
# Check available functions
import real_simple_stats as rss
help(rss)

# Common typos:
# Wrong: rss.standard_deviation()
# Right: rss.sample_std_dev()

# Wrong: rss.ttest()
# Right: rss.two_sample_t_test()
```

---

## 📊 Data Input Errors

### Error: "TypeError: 'int' object is not iterable"

**Symptoms:**
```python
rss.mean(5)
# TypeError: 'int' object is not iterable
```

**Cause:** Passing single value instead of list/array.

**Solution:**
```python
# Wrong
rss.mean(5)

# Correct
rss.mean([5])
rss.mean([1, 2, 3, 4, 5])
```

---

### Error: "ValueError: Input arrays must have the same length"

**Symptoms:**
```python
x = [1, 2, 3]
y = [4, 5]
rss.pearson_correlation(x, y)
# ValueError: Input arrays must have the same length
```

**Cause:** Mismatched array lengths for paired operations.

**Solution:**
```python
# Check lengths
print(f"x length: {len(x)}, y length: {len(y)}")

# Ensure same length
x = [1, 2, 3]
y = [4, 5, 6]  # Same length as x
rss.pearson_correlation(x, y)
```

---

### Error: "ValueError: Data must contain at least one element"

**Symptoms:**
```python
rss.mean([])
# ValueError: Data must contain at least one element
```

**Cause:** Empty dataset.

**Solution:**
```python
# Check if data is empty
data = []
if len(data) > 0:
    mean = rss.mean(data)
else:
    print("No data to analyze")

# Or use try-except
try:
    mean = rss.mean(data)
except ValueError as e:
    print(f"Error: {e}")
```

---

### Error: "TypeError: unsupported operand type(s)"

**Symptoms:**
```python
data = ['1', '2', '3']
rss.mean(data)
# TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

**Cause:** Non-numeric data (strings, None, etc.).

**Solution:**
```python
# Convert strings to numbers
data = ['1', '2', '3']
data_numeric = [float(x) for x in data]
rss.mean(data_numeric)

# Handle missing values
data = [1, 2, None, 4, 5]
data_clean = [x for x in data if x is not None]
rss.mean(data_clean)

# With pandas
import pandas as pd
df = pd.DataFrame({'values': [1, 2, None, 4, 5]})
clean_data = df['values'].dropna().tolist()
rss.mean(clean_data)
```

---

## 🔢 Numerical Errors

### Warning: "RuntimeWarning: invalid value encountered in divide"

**Symptoms:**
```python
data = [5, 5, 5, 5, 5]
rss.sample_std_dev(data)
# RuntimeWarning: invalid value encountered in divide
```

**Cause:** Division by zero (e.g., zero variance).

**Solution:**
```python
# Check for constant data
data = [5, 5, 5, 5, 5]
if len(set(data)) == 1:
    print("All values are the same (zero variance)")
else:
    std = rss.sample_std_dev(data)

# Or handle the warning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    std = rss.sample_std_dev(data)
```

---

### Error: "ZeroDivisionError: division by zero"

**Symptoms:**
```python
rss.coefficient_of_variation([0, 0, 0])
# ZeroDivisionError: division by zero
```

**Cause:** Mean is zero (CV = std/mean).

**Solution:**
```python
data = [0, 0, 0]
mean_val = rss.mean(data)

if mean_val == 0:
    print("Cannot calculate CV when mean is zero")
else:
    cv = rss.coefficient_of_variation(data)
```

---

### Error: "ValueError: math domain error"

**Symptoms:**
```python
rss.normal_pdf(-1, 0, -1)
# ValueError: math domain error
```

**Cause:** Invalid parameters (e.g., negative standard deviation).

**Solution:**
```python
# Check parameters
mu = 0
sigma = 1  # Must be positive!

if sigma <= 0:
    raise ValueError("Standard deviation must be positive")

result = rss.normal_pdf(x, mu, sigma)
```

---

### Warning: "RuntimeWarning: overflow encountered"

**Cause:** Very large numbers in calculations.

**Solution:**
```python
# Use log-scale for large factorials
import math
log_factorial = math.lgamma(n + 1)

# Or limit input ranges
if n > 170:
    print("Value too large for factorial")
```

---

## 📈 Statistical Test Errors

### Error: "ValueError: Degrees of freedom must be positive"

**Symptoms:**
```python
rss.one_sample_t_test([1], mu0=0)
# ValueError: Degrees of freedom must be positive
```

**Cause:** Sample size too small (n=1 gives df=0).

**Solution:**
```python
data = [1, 2, 3]  # Need at least 2 observations
if len(data) < 2:
    print("Need at least 2 observations for t-test")
else:
    t_stat, p_value = rss.one_sample_t_test(data, mu0=0)
```

---

### Error: "ValueError: Observed and expected lists must be the same length"

**Symptoms:**
```python
observed = [10, 20, 30]
expected = [15, 25]
rss.chi_square_statistic(observed, expected)
# ValueError: Observed and expected lists must be the same length
```

**Solution:**
```python
# Ensure same length
observed = [10, 20, 30]
expected = [15, 25, 35]  # Same length

# Or check first
if len(observed) != len(expected):
    raise ValueError("Lengths must match")
```

---

### Issue: P-value is NaN or inf

**Cause:** Numerical instability or invalid test conditions.

**Solutions:**

1. **Check data validity:**
   ```python
   import numpy as np

   # Check for NaN or inf
   if any(np.isnan(data)) or any(np.isinf(data)):
       print("Data contains NaN or inf values")
   ```

2. **Check variance:**
   ```python
   # Zero variance causes issues
   if rss.sample_variance(data) == 0:
       print("Zero variance - all values are identical")
   ```

3. **Check sample size:**
   ```python
   if len(data) < 3:
       print("Sample size too small for reliable inference")
   ```

---

## 🎨 Plotting Errors

### Error: "No module named 'matplotlib'"

**Solution:**
```bash
pip install matplotlib
```

---

### Issue: Plots don't show

**Symptoms:**
```python
rss.plot_normal_histogram(data)
# Nothing appears
```

**Solutions:**

1. **Add plt.show():**
   ```python
   import matplotlib.pyplot as plt
   import real_simple_stats as rss

   rss.plot_normal_histogram(data)
   plt.show()  # Add this!
   ```

2. **For Jupyter:**
   ```python
   %matplotlib inline
   import real_simple_stats as rss

   rss.plot_normal_histogram(data)
   ```

3. **Check backend:**
   ```python
   import matplotlib
   print(matplotlib.get_backend())

   # Change if needed
   matplotlib.use('TkAgg')  # or 'Qt5Agg', 'MacOSX'
   ```

---

### Error: "UserWarning: No artists with labels found"

**Cause:** Legend called but no labels defined.

**Solution:**
```python
# This is just a warning, can be ignored
# Or suppress it:
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

---

## 🔄 Advanced Function Errors

### Error: "LinAlgError: Singular matrix"

**Symptoms:**
```python
X = [[1, 2], [2, 4], [3, 6]]  # Perfectly correlated
y = [1, 2, 3]
rss.multiple_regression(X, y)
# LinAlgError: Singular matrix
```

**Cause:** Perfect multicollinearity in predictors.

**Solution:**
```python
# Check correlation between predictors
import numpy as np
X_array = np.array(X)
corr_matrix = np.corrcoef(X_array.T)
print(corr_matrix)

# Remove perfectly correlated variables
# Or use regularization (not in this package)
```

---

### Error: "ValueError: n_components must be <= min(n_samples, n_features)"

**Symptoms:**
```python
X = [[1, 2], [3, 4]]  # 2 samples, 2 features
rss.pca(X, n_components=3)
# ValueError: n_components must be <= 2
```

**Solution:**
```python
n_samples, n_features = len(X), len(X[0])
max_components = min(n_samples, n_features)

n_components = min(desired_components, max_components)
result = rss.pca(X, n_components=n_components)
```

---

### Issue: Bootstrap/Permutation tests are slow

**Cause:** Too many iterations.

**Solutions:**

1. **Reduce iterations for testing:**
   ```python
   # Fast (for testing)
   result = rss.bootstrap(data, np.mean, n_iterations=100)

   # Accurate (for final analysis)
   result = rss.bootstrap(data, np.mean, n_iterations=10000)
   ```

2. **Use progress indicator:**
   ```python
   from tqdm import tqdm

   # Custom implementation with progress bar
   results = []
   for i in tqdm(range(n_iterations)):
       # Your bootstrap code
       pass
   ```

---

## 🎯 Result Interpretation Issues

### Issue: "Unexpected p-value"

**Checklist:**
1. ✅ Using correct test (one-sample vs. two-sample)?
2. ✅ Data in correct format?
3. ✅ Assumptions met (normality, equal variance)?
4. ✅ Using two-tailed vs. one-tailed correctly?

**Debug:**
```python
# Check descriptive statistics
print(f"Group 1: mean={rss.mean(group1)}, std={rss.sample_std_dev(group1)}")
print(f"Group 2: mean={rss.mean(group2)}, std={rss.sample_std_dev(group2)}")

# Visualize
rss.plot_box_plot(group1)
rss.plot_box_plot(group2)

# Check assumptions
# (normality tests not in this package - use scipy.stats.shapiro)
```

---

### Issue: "Effect size doesn't match p-value"

**This is normal!** P-value depends on sample size; effect size doesn't.

**Example:**
```python
# Large sample, small effect
group1 = [100] * 1000
group2 = [100.1] * 1000
t_stat, p_value = rss.two_sample_t_test(group1, group2)
d = rss.cohens_d(group1, group2)

print(f"p-value: {p_value:.4f}")  # Very small (significant)
print(f"Cohen's d: {d:.3f}")      # Very small (trivial effect)
```

**Lesson:** Always report both!

---

## 🔧 Performance Issues

### Issue: Functions are slow

**Solutions:**

1. **Use NumPy arrays:**
   ```python
   import numpy as np

   # Slower
   data_list = list(range(10000))

   # Faster
   data_array = np.array(data_list)
   ```

2. **Reduce bootstrap/permutation iterations:**
   ```python
   # Faster
   result = rss.bootstrap(data, np.mean, n_iterations=1000)

   # Slower but more accurate
   result = rss.bootstrap(data, np.mean, n_iterations=10000)
   ```

3. **Profile your code:**
   ```python
   import time

   start = time.time()
   result = rss.some_function(data)
   print(f"Time: {time.time() - start:.2f}s")
   ```

---

## 🐛 Debugging Strategies

### General Debugging Workflow

1. **Check data:**
   ```python
   print(f"Data type: {type(data)}")
   print(f"Data length: {len(data)}")
   print(f"First few values: {data[:5]}")
   print(f"Data range: {min(data)} to {max(data)}")
   ```

2. **Check for missing values:**
   ```python
   import numpy as np
   if any(x is None for x in data):
       print("Contains None values")
   if any(np.isnan(x) for x in data):
       print("Contains NaN values")
   ```

3. **Verify function signature:**
   ```python
   help(rss.function_name)
   ```

4. **Test with simple data:**
   ```python
   # Use known values
   simple_data = [1, 2, 3, 4, 5]
   result = rss.mean(simple_data)  # Should be 3.0
   ```

5. **Enable detailed errors:**
   ```python
   import traceback

   try:
       result = rss.some_function(data)
   except Exception as e:
       traceback.print_exc()
   ```

---

## 📞 Getting Help

### Before asking for help:

1. ✅ Read error message carefully
2. ✅ Check this troubleshooting guide
3. ✅ Review [FAQ](FAQ.md)
4. ✅ Check [API documentation](API_COMPARISON.md)
5. ✅ Search [existing issues](https://github.com/kylejones200/real_simple_stats/issues)

### When reporting issues:

Include:
- **Error message** (full traceback)
- **Code to reproduce** (minimal example)
- **Expected behavior**
- **Actual behavior**
- **Environment** (Python version, OS, package version)

**Template:**
```python
import real_simple_stats as rss

# Minimal reproducible example
data = [1, 2, 3, 4, 5]
result = rss.some_function(data)

# Error:
# [paste full error message]

# Expected: [describe expected result]
# Actual: [describe actual result]

# Environment:
# Python 3.9.0
# real-simple-stats 0.3.0
# macOS 12.0
```

---

## 🔗 Additional Resources

- **FAQ**: [Common questions](FAQ.md)
- **API Reference**: [Function lookup](API_COMPARISON.md)
- **Examples**: [Interactive tutorials](INTERACTIVE_EXAMPLES.md)
- **GitHub Issues**: [Report bugs](https://github.com/kylejones200/real_simple_stats/issues)

---

## 💡 Prevention Tips

### Best Practices to Avoid Errors

1. **Validate input data:**
   ```python
   def validate_data(data):
       if not data:
           raise ValueError("Data is empty")
       if not all(isinstance(x, (int, float)) for x in data):
           raise TypeError("Data must be numeric")
       return True
   ```

2. **Use type hints:**
   ```python
   from typing import List

   def my_analysis(data: List[float]) -> float:
       return rss.mean(data)
   ```

3. **Handle exceptions gracefully:**
   ```python
   try:
       result = rss.two_sample_t_test(group1, group2)
   except ValueError as e:
       print(f"Invalid input: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")
   ```

4. **Document your assumptions:**
   ```python
   # Assumes:
   # - Data is normally distributed
   # - Equal variances
   # - Independent samples
   t_stat, p_value = rss.two_sample_t_test(group1, group2)
   ```

---

**Last Updated**: 2025
**Version**: 0.3.0

**Still stuck?** [Open an issue](https://github.com/kylejones200/real_simple_stats/issues) on GitHub!
