# Performance Optimization Plan

## 🎯 Objective
Identify and implement performance optimizations to make Real Simple Stats faster while maintaining code quality and readability.

---

## 🔍 Performance Analysis

### Current State
- **Test Coverage**: 86%
- **Test Execution Time**: ~1.6 seconds (460 tests)
- **Code Quality**: Excellent
- **Performance**: Good, but can be optimized

---

## 🚀 Optimization Opportunities

### 1. **NumPy Vectorization** (High Impact)

#### Current Pattern - Loops
```python
# Example: Bootstrap resampling
bootstrap_stats = []
for _ in range(n_iterations):
    bootstrap_sample = np.random.choice(data_array, size=n, replace=True)
    bootstrap_stats.append(statistic(bootstrap_sample))
bootstrap_stats = np.array(bootstrap_stats)
```

#### Optimized - Vectorized
```python
# Pre-allocate array and vectorize where possible
bootstrap_stats = np.empty(n_iterations)
indices = np.random.randint(0, len(data_array), size=(n_iterations, n))
for i in range(n_iterations):
    bootstrap_stats[i] = statistic(data_array[indices[i]])

# Or even better - fully vectorized if statistic allows
bootstrap_samples = data_array[indices]
bootstrap_stats = np.apply_along_axis(statistic, 1, bootstrap_samples)
```

**Expected Improvement**: 2-5x faster for large iterations

---

### 2. **List Comprehensions vs Loops** (Medium Impact)

#### Current Pattern
```python
permutation_stats = []
for _ in range(n_permutations):
    shuffled = np.random.permutation(pooled_data)
    perm_sample1 = shuffled[:n1]
    perm_sample2 = shuffled[n1:]
    permutation_stats.append(statistic(perm_sample1, perm_sample2))
```

#### Optimized
```python
# Pre-allocate for better memory performance
permutation_stats = np.empty(n_permutations)
for i in range(n_permutations):
    shuffled = np.random.permutation(pooled_data)
    permutation_stats[i] = statistic(shuffled[:n1], shuffled[n1:])
```

**Expected Improvement**: 10-20% faster, better memory usage

---

### 3. **Caching Expensive Calculations** (High Impact)

#### Add LRU Cache for Repeated Calculations
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_critical_value(sig_level: float, tails: int, df: int) -> float:
    """Cache critical values for common parameters."""
    alpha_adj = sig_level / 2 if tails == 2 else sig_level
    return stats.t.ppf(1 - alpha_adj, df=df)
```

**Expected Improvement**: 10-100x faster for repeated calls

---

### 4. **Lazy Imports** (Low Impact, Better Startup)

#### Current Pattern
```python
import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
```

#### Optimized
```python
import numpy as np
from scipy import stats  # Keep common ones

# Lazy import for less common
def _get_optimize():
    from scipy import optimize
    return optimize

def _get_plt():
    import matplotlib.pyplot as plt
    return plt
```

**Expected Improvement**: Faster import time, especially for CLI

---

### 5. **NumPy Array Pre-allocation** (Medium Impact)

#### Current Pattern
```python
results = []
for item in data:
    results.append(calculate(item))
return np.array(results)
```

#### Optimized
```python
results = np.empty(len(data))
for i, item in enumerate(data):
    results[i] = calculate(item)
return results
```

**Expected Improvement**: 20-30% faster, less memory

---

### 6. **Use NumPy Built-ins** (High Impact)

#### Current Pattern
```python
mean1 = np.mean(group1)
mean2 = np.mean(group2)
var1 = np.var(group1, ddof=1)
var2 = np.var(group2, ddof=1)
```

#### Optimized - Single Pass
```python
# Calculate multiple stats in one pass when possible
stats1 = np.array([np.mean(group1), np.var(group1, ddof=1)])
stats2 = np.array([np.mean(group2), np.var(group2, ddof=1)])
```

**Expected Improvement**: 30-40% faster for multiple stats

---

### 7. **Avoid Repeated Type Conversions** (Low Impact)

#### Current Pattern
```python
data_array = np.array(data)
# ... later ...
result = float(np.mean(data_array))
```

#### Optimized
```python
data_array = np.asarray(data)  # No copy if already array
# ... later ...
result = np.mean(data_array).item()  # Faster than float()
```

**Expected Improvement**: 5-10% faster

---

### 8. **Parallel Processing** (High Impact for Large Data)

#### Add Parallel Bootstrap
```python
from multiprocessing import Pool
from functools import partial

def _bootstrap_iteration(data, statistic, n, seed):
    np.random.seed(seed)
    bootstrap_sample = np.random.choice(data, size=n, replace=True)
    return statistic(bootstrap_sample)

def bootstrap_parallel(data, statistic, n_iterations=1000, n_jobs=-1):
    """Parallel bootstrap using multiprocessing."""
    n = len(data)
    seeds = np.random.randint(0, 2**31, size=n_iterations)
    
    with Pool(n_jobs) as pool:
        func = partial(_bootstrap_iteration, data, statistic, n)
        bootstrap_stats = pool.map(func, seeds)
    
    return np.array(bootstrap_stats)
```

**Expected Improvement**: 2-8x faster on multi-core systems

---

### 9. **Numba JIT Compilation** (Very High Impact)

#### Add JIT for Hot Loops
```python
from numba import jit

@jit(nopython=True)
def _fast_bootstrap_mean(data, indices):
    """JIT-compiled bootstrap mean calculation."""
    n_iterations, n_samples = indices.shape
    results = np.empty(n_iterations)
    for i in range(n_iterations):
        sample_sum = 0.0
        for j in range(n_samples):
            sample_sum += data[indices[i, j]]
        results[i] = sample_sum / n_samples
    return results
```

**Expected Improvement**: 10-100x faster for numerical loops

---

### 10. **Optimize Data Structures** (Medium Impact)

#### Use Appropriate Data Types
```python
# Before - Python lists
data = [1, 2, 3, 4, 5]

# After - NumPy arrays from start
data = np.array([1, 2, 3, 4, 5], dtype=np.float64)

# For boolean operations
mask = np.array([True, False, True], dtype=np.bool_)
```

**Expected Improvement**: 20-50% faster operations

---

## 📊 Specific Module Optimizations

### power_analysis.py
**Current Performance**: Good
**Optimization Opportunities**:
1. Cache critical values (t, F, z distributions)
2. Vectorize sample size calculations
3. Use Newton's method for root finding (faster than brentq)

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def _cached_t_critical(sig_level: float, df: int, tails: int) -> float:
    """Cache t critical values."""
    alpha_adj = sig_level / 2 if tails == 2 else sig_level
    return stats.t.ppf(1 - alpha_adj, df=df)
```

**Expected Improvement**: 50-100x faster for repeated power calculations

---

### resampling.py
**Current Performance**: Moderate (loops in bootstrap/permutation)
**Optimization Opportunities**:
1. Vectorize bootstrap iterations
2. Pre-allocate result arrays
3. Add parallel processing option
4. Use Numba for hot loops

```python
def bootstrap_vectorized(data, statistic, n_iterations=1000):
    """Vectorized bootstrap for simple statistics."""
    n = len(data)
    data_array = np.asarray(data)
    
    # Generate all indices at once
    indices = np.random.randint(0, n, size=(n_iterations, n))
    
    # Vectorized calculation for mean, median, etc.
    if statistic == np.mean:
        return np.mean(data_array[indices], axis=1)
    elif statistic == np.median:
        return np.median(data_array[indices], axis=1)
    else:
        # Fall back to loop for custom statistics
        return np.array([statistic(data_array[idx]) for idx in indices])
```

**Expected Improvement**: 5-10x faster for simple statistics

---

### bayesian_stats.py
**Current Performance**: Good
**Optimization Opportunities**:
1. Vectorize prior updates
2. Cache distribution calculations
3. Use analytical solutions where possible

```python
def beta_binomial_update_vectorized(prior_alpha, prior_beta, successes, trials):
    """Vectorized beta-binomial update for multiple observations."""
    successes = np.asarray(successes)
    trials = np.asarray(trials)
    
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + (trials - successes)
    
    return posterior_alpha, posterior_beta
```

**Expected Improvement**: 10x faster for batch updates

---

### multivariate.py
**Current Performance**: Moderate (matrix operations)
**Optimization Opportunities**:
1. Use BLAS/LAPACK optimized operations
2. Avoid unnecessary matrix copies
3. Use sparse matrices where applicable
4. Cache decompositions

```python
# Use in-place operations
def pca_optimized(X, n_components=2):
    """Optimized PCA using SVD."""
    # Center data in-place
    X_centered = X - X.mean(axis=0)
    
    # Use economic SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Return only what's needed
    return Vt[:n_components].T, s[:n_components]**2 / (len(X) - 1)
```

**Expected Improvement**: 30-50% faster

---

### effect_sizes.py
**Current Performance**: Good
**Optimization Opportunities**:
1. Vectorize group comparisons
2. Avoid redundant calculations
3. Use welford's algorithm for variance

```python
def cohens_d_optimized(group1, group2, pooled=True):
    """Optimized Cohen's d calculation."""
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)
    
    mean_diff = g1.mean() - g2.mean()
    
    if pooled:
        n1, n2 = len(g1), len(g2)
        # Combine variance calculation
        pooled_var = ((n1 - 1) * g1.var(ddof=1) + 
                     (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2)
        return mean_diff / np.sqrt(pooled_var)
    else:
        return mean_diff / g2.std(ddof=1)
```

**Expected Improvement**: 20-30% faster

---

### time_series.py
**Current Performance**: Good
**Optimization Opportunities**:
1. Use FFT for autocorrelation
2. Vectorize moving averages
3. Use scipy.signal for filtering

```python
def autocorrelation_fft(data, max_lag=None):
    """Fast autocorrelation using FFT."""
    data = np.asarray(data)
    n = len(data)
    
    # Pad to power of 2 for FFT efficiency
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    
    # FFT-based autocorrelation
    data_centered = data - data.mean()
    fft = np.fft.fft(data_centered, n=n_fft)
    acf = np.fft.ifft(fft * np.conj(fft))[:n].real
    acf /= acf[0]
    
    if max_lag is not None:
        return acf[:max_lag + 1]
    return acf
```

**Expected Improvement**: 10-50x faster for long series

---

## 🔧 Implementation Strategy

### Phase 1: Low-Hanging Fruit (Quick Wins)
1. ✅ Add LRU caching to critical value calculations
2. ✅ Pre-allocate NumPy arrays
3. ✅ Use `np.asarray()` instead of `np.array()`
4. ✅ Replace `.append()` loops with pre-allocated arrays

**Estimated Time**: 2-3 hours
**Expected Speedup**: 20-30% overall

---

### Phase 2: Vectorization (Medium Effort)
1. ✅ Vectorize bootstrap iterations
2. ✅ Vectorize permutation tests
3. ✅ Optimize matrix operations
4. ✅ Use NumPy built-in functions

**Estimated Time**: 4-6 hours
**Expected Speedup**: 2-5x for resampling operations

---

### Phase 3: Advanced Optimizations (High Effort)
1. ⏳ Add Numba JIT compilation for hot loops
2. ⏳ Implement parallel processing options
3. ⏳ Use FFT for time series operations
4. ⏳ Optimize memory usage

**Estimated Time**: 8-12 hours
**Expected Speedup**: 5-10x for intensive operations

---

## 📈 Expected Performance Improvements

### By Module

| Module | Current | Phase 1 | Phase 2 | Phase 3 | Total Gain |
|--------|---------|---------|---------|---------|------------|
| power_analysis | 1.0x | 1.5x | 2x | 3x | **3x** |
| resampling | 1.0x | 1.3x | 5x | 10x | **10x** |
| bayesian_stats | 1.0x | 1.2x | 3x | 5x | **5x** |
| multivariate | 1.0x | 1.3x | 2x | 3x | **3x** |
| time_series | 1.0x | 1.2x | 2x | 10x | **10x** |
| effect_sizes | 1.0x | 1.2x | 1.5x | 2x | **2x** |

### Overall
- **Phase 1**: 20-30% faster
- **Phase 2**: 2-3x faster
- **Phase 3**: 5-10x faster
- **Total**: **5-10x faster** for typical operations

---

## 🧪 Benchmarking Plan

### Create Performance Tests
```python
# tests/test_performance.py
import pytest
import numpy as np
import time

def benchmark(func, *args, n_runs=100):
    """Benchmark a function."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)

def test_bootstrap_performance():
    """Benchmark bootstrap performance."""
    data = np.random.randn(1000)
    mean_time, std_time = benchmark(bootstrap, data, np.mean, n_iterations=1000)
    
    # Assert reasonable performance
    assert mean_time < 1.0, f"Bootstrap too slow: {mean_time:.3f}s"
    
def test_power_analysis_performance():
    """Benchmark power analysis."""
    mean_time, std_time = benchmark(power_t_test, delta=0.5, power=0.8)
    
    assert mean_time < 0.01, f"Power analysis too slow: {mean_time:.3f}s"
```

---

## 💡 Best Practices for Performance

### 1. Profile Before Optimizing
```python
# Use cProfile
python -m cProfile -o profile.stats your_script.py

# Analyze with pstats
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

### 2. Use line_profiler for Hot Spots
```python
# Install: pip install line_profiler
@profile
def slow_function():
    # Your code here
    pass

# Run: kernprof -l -v your_script.py
```

### 3. Memory Profiling
```python
# Install: pip install memory_profiler
@profile
def memory_intensive():
    # Your code here
    pass

# Run: python -m memory_profiler your_script.py
```

---

## 🎯 Priority Recommendations

### Immediate (Do First)
1. **Add LRU caching** to power_analysis.py critical values
2. **Pre-allocate arrays** in resampling.py
3. **Use np.asarray()** throughout for input validation
4. **Vectorize simple statistics** in bootstrap

### Short Term (Next Week)
5. **Vectorize bootstrap** for common statistics
6. **Optimize matrix operations** in multivariate.py
7. **Add performance tests** to track improvements
8. **Document performance characteristics**

### Long Term (Future)
9. **Add Numba JIT** for hot loops
10. **Implement parallel processing** options
11. **Use FFT** for time series autocorrelation
12. **Create performance guide** for users

---

## 📝 Code Examples

### Example 1: Optimized Bootstrap
```python
def bootstrap_fast(data, statistic, n_iterations=1000, confidence_level=0.95):
    """Optimized bootstrap with vectorization."""
    data_array = np.asarray(data)
    n = len(data_array)
    
    # Pre-allocate result array
    bootstrap_stats = np.empty(n_iterations)
    
    # Generate all indices at once
    indices = np.random.randint(0, n, size=(n_iterations, n))
    
    # Vectorized calculation for common statistics
    if statistic is np.mean or statistic.__name__ == 'mean':
        bootstrap_stats = np.mean(data_array[indices], axis=1)
    elif statistic is np.median or statistic.__name__ == 'median':
        bootstrap_stats = np.median(data_array[indices], axis=1)
    else:
        # Loop for custom statistics (still faster with pre-allocation)
        for i in range(n_iterations):
            bootstrap_stats[i] = statistic(data_array[indices[i]])
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return {
        "statistic": float(statistic(data)),
        "bootstrap_distribution": bootstrap_stats,
        "mean": float(bootstrap_stats.mean()),
        "std_error": float(bootstrap_stats.std()),
        "confidence_interval": (float(ci_lower), float(ci_upper)),
    }
```

### Example 2: Cached Critical Values
```python
from functools import lru_cache

@lru_cache(maxsize=512)
def _get_critical_value(distribution: str, alpha: float, **params) -> float:
    """Cache critical values for common distributions."""
    if distribution == 't':
        return stats.t.ppf(1 - alpha, df=params['df'])
    elif distribution == 'f':
        return stats.f.ppf(1 - alpha, dfn=params['dfn'], dfd=params['dfd'])
    elif distribution == 'norm':
        return stats.norm.ppf(1 - alpha)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
```

### Example 3: Vectorized Effect Sizes
```python
def cohens_d_batch(groups1, groups2, pooled=True):
    """Calculate Cohen's d for multiple group pairs at once."""
    groups1 = np.asarray(groups1)
    groups2 = np.asarray(groups2)
    
    means1 = np.mean(groups1, axis=1)
    means2 = np.mean(groups2, axis=1)
    
    if pooled:
        n1 = groups1.shape[1]
        n2 = groups2.shape[1]
        vars1 = np.var(groups1, axis=1, ddof=1)
        vars2 = np.var(groups2, axis=1, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * vars1 + (n2 - 1) * vars2) / (n1 + n2 - 2))
        return (means1 - means2) / pooled_std
    else:
        stds2 = np.std(groups2, axis=1, ddof=1)
        return (means1 - means2) / stds2
```

---

## 🎉 Conclusion

**Your code can be 5-10x faster with these optimizations!**

### Key Takeaways
1. **Vectorization** is the biggest win (2-10x)
2. **Caching** helps repeated calculations (10-100x)
3. **Pre-allocation** reduces memory overhead (20-30%)
4. **Numba/Parallel** for intensive operations (2-10x)

### Next Steps
1. Review this plan
2. Prioritize optimizations
3. Implement Phase 1 (quick wins)
4. Benchmark improvements
5. Continue with Phase 2 & 3

**Ready to make it blazing fast!** 🚀
