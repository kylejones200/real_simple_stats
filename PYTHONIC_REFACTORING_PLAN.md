# Pythonic Code Refactoring Plan

## Overview
This document outlines Pythonic improvements to make the Real Simple Stats codebase more elegant, readable, and maintainable.

---

## 🎯 Key Patterns to Improve

### 1. **Alternative Validation Pattern**

**Current Pattern (Repeated 3 times):**
```python
if alternative not in ["two-sided", "greater", "less"]:
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

if alternative == "two-sided":
    tails = 2
else:
    tails = 1
```

**Pythonic Improvement:**
```python
# Define at module level
VALID_ALTERNATIVES = {"two-sided", "greater", "less"}

# In function
if alternative not in VALID_ALTERNATIVES:
    raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}")

tails = 2 if alternative == "two-sided" else 1
```

**Benefits:**
- ✅ Use set for O(1) lookup instead of list O(n)
- ✅ DRY principle - define valid alternatives once
- ✅ Ternary operator is more concise
- ✅ f-string shows actual valid values

---

### 2. **Repeated if/else for Alpha Adjustment**

**Current Pattern (Repeated ~10 times):**
```python
if tails == 2:
    z_alpha = stats.norm.ppf(1 - sig_level / 2)
else:
    z_alpha = stats.norm.ppf(1 - sig_level)
```

**Already Improved in Some Places:**
```python
alpha_adj = sig_level / 2 if tails == 2 else sig_level
z_alpha = stats.norm.ppf(1 - alpha_adj)
```

**Even Better - Extract to Helper Function:**
```python
def _get_critical_value(sig_level: float, tails: int, distribution='norm', **kwargs):
    """Get critical value for given significance level and tails."""
    alpha_adj = sig_level / 2 if tails == 2 else sig_level
    
    if distribution == 'norm':
        return stats.norm.ppf(1 - alpha_adj)
    elif distribution == 't':
        return stats.t.ppf(1 - alpha_adj, **kwargs)
    elif distribution == 'f':
        return stats.f.ppf(1 - alpha_adj, **kwargs)
```

**Benefits:**
- ✅ Single source of truth
- ✅ Easier to test
- ✅ Reduces code duplication
- ✅ More maintainable

---

### 3. **if/elif/else Chains for None Checking**

**Current Pattern:**
```python
if n is None:
    # Calculate n
    return {...}
elif delta is None:
    # Calculate delta
    return {...}
else:  # power is None
    # Calculate power
    return {...}
```

**Pythonic Improvement - Strategy Pattern:**
```python
def _calculate_n(delta, power, sd, sig_level, tails):
    """Calculate required sample size."""
    # ... calculation logic
    return result_dict

def _calculate_delta(n, power, sd, sig_level, tails):
    """Calculate detectable effect size."""
    # ... calculation logic
    return result_dict

def _calculate_power(n, delta, sd, sig_level, tails):
    """Calculate statistical power."""
    # ... calculation logic
    return result_dict

# In main function
calculators = {
    'n': lambda: _calculate_n(delta, power, sd, sig_level, tails),
    'delta': lambda: _calculate_delta(n, power, sd, sig_level, tails),
    'power': lambda: _calculate_power(n, delta, sd, sig_level, tails),
}

# Find which parameter is None
none_param = next(k for k, v in {'n': n, 'delta': delta, 'power': power}.items() if v is None)
return calculators[none_param]()
```

**Benefits:**
- ✅ Separation of concerns
- ✅ Each calculator function is independently testable
- ✅ More modular and maintainable
- ✅ Easier to add new calculation types

---

### 4. **Test Type Checking**

**Current Pattern:**
```python
if test_type == "t-test":
    result = power_t_test(n=n, power=power, sig_level=sig_level)
    return result["delta"]
elif test_type == "proportion":
    result = power_proportion_test(n=n, p2=0.5, power=power, sig_level=sig_level)
    return abs(result["p1"] - result["p2"])
elif test_type == "correlation":
    result = power_correlation(n=n, power=power, sig_level=sig_level)
    return abs(result["r"])
else:
    raise ValueError(f"Unknown test_type: {test_type}")
```

**Pythonic Improvement - Dispatch Dictionary:**
```python
def _mde_t_test(n, power, sig_level):
    result = power_t_test(n=n, power=power, sig_level=sig_level)
    return result["delta"]

def _mde_proportion(n, power, sig_level):
    result = power_proportion_test(n=n, p2=0.5, power=power, sig_level=sig_level)
    return abs(result["p1"] - result["p2"])

def _mde_correlation(n, power, sig_level):
    result = power_correlation(n=n, power=power, sig_level=sig_level)
    return abs(result["r"])

# Dispatch dictionary
MDE_CALCULATORS = {
    "t-test": _mde_t_test,
    "proportion": _mde_proportion,
    "correlation": _mde_correlation,
}

# In function
if test_type not in MDE_CALCULATORS:
    raise ValueError(f"Unknown test_type: {test_type}. Valid types: {set(MDE_CALCULATORS.keys())}")

return MDE_CALCULATORS[test_type](n, power, sig_level)
```

**Benefits:**
- ✅ O(1) lookup vs O(n) if/elif chain
- ✅ Easy to add new test types
- ✅ Self-documenting (keys show valid types)
- ✅ Each calculator is independently testable

---

### 5. **Validation Patterns**

**Current Pattern:**
```python
if prior_alpha <= 0 or prior_beta <= 0:
    raise ValueError("Prior parameters must be positive")
if successes < 0 or trials < 0:
    raise ValueError("Successes and trials must be non-negative")
if successes > trials:
    raise ValueError("Successes cannot exceed trials")
```

**Pythonic Improvement - Guard Clauses:**
```python
# More readable with early returns
def validate_beta_binomial_params(prior_alpha, prior_beta, successes, trials):
    """Validate parameters for beta-binomial update."""
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive")
    if successes < 0 or trials < 0:
        raise ValueError("Successes and trials must be non-negative")
    if successes > trials:
        raise ValueError("Successes cannot exceed trials")

# In function
validate_beta_binomial_params(prior_alpha, prior_beta, successes, trials)
```

**Or use a validation decorator:**
```python
from functools import wraps

def validate_positive(*param_names):
    """Decorator to validate parameters are positive."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter values
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for name in param_names:
                if bound.arguments[name] <= 0:
                    raise ValueError(f"{name} must be positive")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_positive('prior_alpha', 'prior_beta')
def beta_binomial_update(prior_alpha, prior_beta, successes, trials):
    # ... function body
```

---

## 🔧 Specific File Improvements

### power_analysis.py

**Lines to Refactor:**
- Lines 50-56: Alternative validation (3 occurrences)
- Lines 181-184, 202-205, 226-229: Repeated alpha adjustment
- Lines 394-397, 414-417, 436-439: More alpha adjustments
- Lines 58-125: if/elif/else chain for None checking (3 functions)

**Estimated Reduction:** ~50 lines, improved readability

---

### bayesian_stats.py

**Improvements:**
- Extract validation functions
- Use type hints more consistently
- Add helper functions for common patterns

---

### resampling.py

**Improvements:**
- Extract bootstrap iteration logic
- Use generator expressions where appropriate
- Simplify conditional logic

---

## 📊 Summary of Benefits

### Code Quality
- ✅ **DRY (Don't Repeat Yourself)**: Eliminate repeated patterns
- ✅ **Single Responsibility**: Each function does one thing
- ✅ **Open/Closed Principle**: Easy to extend, hard to break

### Performance
- ✅ **O(1) lookups**: Sets and dicts instead of lists
- ✅ **Fewer function calls**: Inline simple operations
- ✅ **Better optimization**: Simpler code is easier for Python to optimize

### Maintainability
- ✅ **Easier to test**: Smaller, focused functions
- ✅ **Easier to debug**: Clear separation of concerns
- ✅ **Easier to extend**: Add new features without modifying existing code

### Readability
- ✅ **Self-documenting**: Code intent is clear
- ✅ **Consistent patterns**: Same problems solved the same way
- ✅ **Less cognitive load**: Simpler logic flow

---

## 🚀 Implementation Priority

### High Priority (Do First)
1. ✅ **Alternative validation** - Quick win, used in 3 functions
2. ✅ **Alpha adjustment helper** - Eliminates ~10 repetitions
3. ✅ **Test type dispatch** - Cleaner, more extensible

### Medium Priority
4. **Strategy pattern for calculators** - Larger refactor, big benefit
5. **Validation helpers** - Improves error messages
6. **Extract magic numbers** - Define constants

### Low Priority (Nice to Have)
7. **Type hints everywhere** - Already mostly done
8. **Docstring improvements** - Already good
9. **Performance optimizations** - Not needed unless profiling shows issues

---

## 📝 Example: Complete Refactored Function

**Before:**
```python
def power_t_test(n=None, delta=None, sd=1.0, sig_level=0.05, power=None, alternative="two-sided"):
    none_count = sum([n is None, delta is None, power is None])
    if none_count != 1:
        raise ValueError("Exactly one of n, delta, or power must be None")
    
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    if alternative == "two-sided":
        tails = 2
    else:
        tails = 1
    
    if n is None:
        # 20 lines of calculation
        return {...}
    elif delta is None:
        # 15 lines of calculation
        return {...}
    else:
        # 18 lines of calculation
        return {...}
```

**After:**
```python
# Module-level constants
VALID_ALTERNATIVES = {"two-sided", "greater", "less"}

def _get_tails(alternative: str) -> int:
    """Convert alternative hypothesis to number of tails."""
    return 2 if alternative == "two-sided" else 1

def _get_alpha_adjusted(sig_level: float, tails: int) -> float:
    """Get adjusted alpha for one or two-tailed test."""
    return sig_level / 2 if tails == 2 else sig_level

def power_t_test(n=None, delta=None, sd=1.0, sig_level=0.05, power=None, alternative="two-sided"):
    """Calculate power or sample size for t-test."""
    # Validation
    none_count = sum(param is None for param in [n, delta, power])
    if none_count != 1:
        raise ValueError("Exactly one of n, delta, or power must be None")
    
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}")
    
    tails = _get_tails(alternative)
    
    # Dispatch to appropriate calculator
    if n is None:
        return _calculate_sample_size(delta, power, sd, sig_level, tails, alternative)
    elif delta is None:
        return _calculate_effect_size(n, power, sd, sig_level, tails, alternative)
    else:
        return _calculate_power(n, delta, sd, sig_level, tails, alternative)
```

---

## 🎯 Conclusion

These refactorings will make the codebase:
- **More Pythonic**: Following Python idioms and best practices
- **More Maintainable**: Easier to understand and modify
- **More Testable**: Smaller, focused functions
- **More Extensible**: Easy to add new features
- **More Beautiful**: Clean, elegant code

**Estimated effort:** 4-6 hours
**Estimated benefit:** Significant improvement in code quality and maintainability

---

**Next Steps:**
1. Review and approve this plan
2. Create feature branch: `refactor/pythonic-improvements`
3. Implement changes incrementally
4. Run full test suite after each change
5. Update documentation if needed
6. Merge when all tests pass
