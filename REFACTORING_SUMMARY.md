# Pythonic Code Refactoring Summary

## 📋 Overview

I've reviewed the Real Simple Stats codebase and created a comprehensive plan for making it more Pythonic and beautiful. The code is already quite good, but there are several patterns that can be improved.

---

## 🎯 Key Findings

### What's Already Good ✅
- **Type hints**: Consistently used throughout
- **Docstrings**: Comprehensive Google-style documentation
- **Naming**: Clear, descriptive function and variable names
- **Some refactoring already done**: Lines 66, 86, 108 in power_analysis.py already use ternary operators

### What Can Be Improved 🔧

#### 1. **Repeated if/else Chains** (High Priority)
**Pattern found:** Alternative validation repeated 3 times
```python
# Current (repeated 3x)
if alternative not in ["two-sided", "greater", "less"]:
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

if alternative == "two-sided":
    tails = 2
else:
    tails = 1
```

**Pythonic solution:**
```python
# Module-level constant
VALID_ALTERNATIVES = {"two-sided", "greater", "less"}

# Use set for O(1) lookup and ternary operator
if alternative not in VALID_ALTERNATIVES:
    raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}")

tails = 2 if alternative == "two-sided" else 1
```

**Benefits:**
- ✅ Set lookup is O(1) vs list O(n)
- ✅ DRY principle - define once
- ✅ More concise with ternary
- ✅ f-string shows actual values

---

#### 2. **Repeated Alpha Adjustment** (High Priority)
**Pattern found:** ~10 occurrences
```python
# Current
if tails == 2:
    z_alpha = stats.norm.ppf(1 - sig_level / 2)
else:
    z_alpha = stats.norm.ppf(1 - sig_level)
```

**Pythonic solution:**
```python
# Extract to helper function
def _get_alpha_adjusted(sig_level: float, tails: int) -> float:
    """Get adjusted alpha for one or two-tailed test."""
    return sig_level / 2 if tails == 2 else sig_level

# Use it
alpha_adj = _get_alpha_adjusted(sig_level, tails)
z_alpha = stats.norm.ppf(1 - alpha_adj)
```

**Benefits:**
- ✅ Single source of truth
- ✅ Easier to test
- ✅ Reduces duplication
- ✅ More maintainable

---

#### 3. **if/elif/else for None Checking** (Medium Priority)
**Pattern found:** Calculator dispatch in power functions
```python
# Current
if n is None:
    # 20 lines of calculation
    return {...}
elif delta is None:
    # 15 lines of calculation
    return {...}
else:  # power is None
    # 18 lines of calculation
    return {...}
```

**Pythonic solution - Strategy Pattern:**
```python
# Extract to separate functions
def _calculate_t_test_n(delta, power, tails, base_params):
    """Calculate required sample size."""
    # ... calculation logic
    return result_dict

def _calculate_t_test_delta(n, power, tails, base_params):
    """Calculate detectable effect size."""
    # ... calculation logic
    return result_dict

def _calculate_t_test_power(n, delta, tails, base_params):
    """Calculate statistical power."""
    # ... calculation logic
    return result_dict

# Dispatch
if n is None:
    return _calculate_t_test_n(delta, power, tails, base_params)
elif delta is None:
    return _calculate_t_test_delta(n, power, tails, base_params)
else:
    return _calculate_t_test_power(n, delta, tails, base_params)
```

**Benefits:**
- ✅ Separation of concerns
- ✅ Each function independently testable
- ✅ More modular
- ✅ Easier to maintain

---

#### 4. **Test Type Dispatch** (High Priority)
**Pattern found:** if/elif/else chain for test types
```python
# Current
if test_type == "t-test":
    result = power_t_test(...)
    return result["delta"]
elif test_type == "proportion":
    result = power_proportion_test(...)
    return abs(result["p1"] - result["p2"])
elif test_type == "correlation":
    result = power_correlation(...)
    return abs(result["r"])
else:
    raise ValueError(f"Unknown test_type: {test_type}")
```

**Pythonic solution - Dispatch Dictionary:**
```python
# Define calculators
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

# Use it
if test_type not in MDE_CALCULATORS:
    raise ValueError(f"Unknown test_type: {test_type}. Valid: {set(MDE_CALCULATORS.keys())}")

return MDE_CALCULATORS[test_type](n, power, sig_level)
```

**Benefits:**
- ✅ O(1) lookup vs O(n) if/elif
- ✅ Easy to extend
- ✅ Self-documenting
- ✅ More testable

---

## 📊 Impact Analysis

### Files Reviewed
1. ✅ `power_analysis.py` - Main target, ~50 lines can be improved
2. ✅ `bayesian_stats.py` - Good structure, minor improvements possible
3. ✅ `resampling.py` - Clean code, few improvements needed
4. ✅ Other modules - Generally well-written

### Estimated Improvements
- **Lines reduced**: ~50-80 lines
- **Readability**: Significantly improved
- **Maintainability**: Much easier to extend
- **Performance**: Marginal improvement (O(1) lookups)
- **Testability**: Each helper function independently testable

---

## 🚀 Implementation

### Created Files
1. **`PYTHONIC_REFACTORING_PLAN.md`** - Detailed refactoring plan with examples
2. **`power_analysis_refactored.py`** - Fully refactored version showing best practices

### Refactored Version Highlights
- ✅ Module-level constants (`VALID_ALTERNATIVES`, `VALID_TEST_TYPES`)
- ✅ Helper functions (`_get_tails`, `_get_alpha_adjusted`, `_validate_alternative`)
- ✅ Strategy pattern (separate calculator functions)
- ✅ Dispatch dictionary (`MDE_CALCULATORS`)
- ✅ Consistent validation (`_validate_none_count`)
- ✅ Cleaner, more readable code
- ✅ Better separation of concerns

### Code Comparison

**Before (original):**
```python
def power_t_test(...):
    # 125 lines with nested if/elif/else
    # Repeated validation logic
    # Repeated alpha adjustment
    # Mixed concerns
```

**After (refactored):**
```python
# Helper functions (reusable)
def _get_tails(alternative): ...
def _get_alpha_adjusted(sig_level, tails): ...
def _validate_alternative(alternative): ...
def _validate_none_count(params, expected): ...

# Main function (clean dispatch)
def power_t_test(...):
    # Validation
    _validate_none_count({"n": n, "delta": delta, "power": power})
    _validate_alternative(alternative)
    
    # Dispatch
    if n is None:
        return _calculate_t_test_n(...)
    elif delta is None:
        return _calculate_t_test_delta(...)
    else:
        return _calculate_t_test_power(...)

# Separate calculator functions (focused, testable)
def _calculate_t_test_n(...): ...
def _calculate_t_test_delta(...): ...
def _calculate_t_test_power(...): ...
```

---

## 💡 Recommendations

### Immediate Actions (High Priority)
1. ✅ **Review** `power_analysis_refactored.py` - See the improvements in action
2. ✅ **Test** refactored version - Ensure all tests still pass
3. ✅ **Decide** whether to apply these changes

### If Approved
1. Replace `power_analysis.py` with refactored version
2. Run full test suite
3. Apply similar patterns to other modules
4. Update documentation if needed

### Benefits of Refactoring
- **Readability**: Code is more elegant and easier to understand
- **Maintainability**: Easier to modify and extend
- **Testability**: Each function can be tested independently
- **Performance**: Minor improvements from O(1) lookups
- **Pythonic**: Follows Python best practices and idioms

---

## 🎓 Key Pythonic Principles Applied

1. **DRY (Don't Repeat Yourself)**: Extract repeated patterns
2. **Single Responsibility**: Each function does one thing
3. **Open/Closed**: Easy to extend, hard to break
4. **Explicit is better than implicit**: Clear function names
5. **Flat is better than nested**: Reduce nesting depth
6. **Use built-ins**: Sets, dicts, ternary operators
7. **Extract helpers**: Small, focused functions

---

## 📈 Next Steps

### Option 1: Apply Refactoring
1. Backup current code
2. Replace with refactored version
3. Run tests: `pytest tests/ --cov=real_simple_stats`
4. Commit if all tests pass

### Option 2: Incremental Refactoring
1. Apply one improvement at a time
2. Test after each change
3. Commit incrementally

### Option 3: Keep as Reference
1. Use refactored version as reference
2. Apply patterns as needed
3. Gradual improvement over time

---

## 🎯 Conclusion

The Real Simple Stats codebase is **already well-written**, but these refactorings will make it:
- ✨ **More Pythonic** - Following Python idioms
- 🎨 **More Beautiful** - Elegant and clean
- 🔧 **More Maintainable** - Easier to modify
- 🧪 **More Testable** - Better separation of concerns
- 📚 **More Professional** - Industry best practices

**The refactored version is ready to review in `power_analysis_refactored.py`!**

---

**Created**: January 5, 2025  
**Files**: 2 new documentation files, 1 refactored module  
**Status**: ✅ Ready for review
