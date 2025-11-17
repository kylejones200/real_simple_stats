"""Statistical assumptions checking for hypothesis tests.

This module provides functions to check assumptions for various
statistical tests, helping users ensure their data meets test requirements.
"""

from typing import Sequence, List, Dict, Optional, Any
import math

from . import descriptive_statistics as desc

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def check_t_test_assumptions(
    data: Sequence[float],
    group2: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check assumptions for one-sample or two-sample t-test.
    
    Assumptions:
    1. Normality (data should be approximately normally distributed)
    2. Independence (observations should be independent)
    3. For two-sample: Equal variances (homoscedasticity)
    
    Args:
        data: First sample (or only sample for one-sample test)
        group2: Second sample (if None, assumes one-sample test)
        verbose: If True, print detailed results
        
    Returns:
        Dictionary with assumption check results
    """
    results = {}
    
    if verbose:
        test_type = "Two-sample" if group2 is not None else "One-sample"
        print("=" * 70)
        print(f"{test_type} T-Test: Assumptions Check")
        print("=" * 70)
    
    # Check 1: Normality
    if verbose:
        print("\n1. Normality Assumption")
        print("-" * 70)
        print("The data should be approximately normally distributed.")
    
    n1 = len(data)
    normality1 = _check_normality(data, n1, verbose, "Group 1")
    results["normality_group1"] = normality1
    
    if group2 is not None:
        n2 = len(group2)
        normality2 = _check_normality(group2, n2, verbose, "Group 2")
        results["normality_group2"] = normality2
        results["normality_overall"] = normality1["passed"] and normality2["passed"]
    else:
        results["normality_overall"] = normality1["passed"]
    
    # Check 2: Independence
    if verbose:
        print("\n2. Independence Assumption")
        print("-" * 70)
        print("Observations should be independent (not correlated).")
        print("✓ This assumption cannot be tested statistically.")
        print("  Ensure your sampling method produces independent observations:")
        print("  - Random sampling")
        print("  - No repeated measures on same subject")
        print("  - No clustering or grouping effects")
    
    results["independence"] = {  # type: ignore[assignment]
        "passed": None,  # Cannot be tested
        "note": "Must be ensured through study design"
    }
    
    # Check 3: Equal variances (for two-sample test)
    if group2 is not None:
        if verbose:
            print("\n3. Equal Variances (Homoscedasticity)")
            print("-" * 70)
            print("For two-sample t-test, variances should be approximately equal.")
        
        var1 = desc.sample_variance(data)
        var2 = desc.sample_variance(group2)
        variance_ratio = max(var1, var2) / min(var1, var2)
        
        if verbose:
            print(f"Group 1 variance: {var1:.4f}")
            print(f"Group 2 variance: {var2:.4f}")
            print(f"Variance ratio: {variance_ratio:.4f}")
        
        # Rule of thumb: ratio < 2 suggests equal variances
        equal_var = variance_ratio < 2.0
        
        if verbose:
            if equal_var:
                print("✓ Variances appear equal (ratio < 2.0)")
                print("  Standard two-sample t-test is appropriate")
            else:
                print("⚠️  Variances may be unequal (ratio ≥ 2.0)")
                print("  Consider using Welch's t-test (unequal variances)")
        
        results["equal_variances"] = {
            "passed": equal_var,
            "variance_ratio": variance_ratio,
            "recommendation": "Welch's t-test" if not equal_var else "Standard t-test"
        }
    else:
        results["equal_variances"] = None  # type: ignore[assignment]
    
    # Overall assessment
    if verbose:
        print("\n" + "=" * 70)
        print("Overall Assessment")
        print("=" * 70)
        
        all_passed = results["normality_overall"]
        if group2 is not None:
            all_passed = all_passed and results["equal_variances"]["passed"]
        
        if all_passed:
            print("✓ All testable assumptions are met.")
            print("  The t-test is appropriate for your data.")
        else:
            print("⚠️  Some assumptions may be violated.")
            if not results["normality_overall"]:
                print("  - Normality: Consider non-parametric alternative (Mann-Whitney U)")
            if group2 is not None and not results["equal_variances"]["passed"]:
                print("  - Equal variances: Use Welch's t-test")
            print("\nNote: T-tests are robust to minor violations, especially with larger samples.")
    
    results["all_passed"] = all_passed if group2 is None else (
        results["normality_overall"] and results["equal_variances"]["passed"]
    )
    
    return results


def _check_normality(
    data: Sequence[float],
    n: int,
    verbose: bool,
    group_name: str = "Data"
) -> Dict[str, Any]:
    """Check normality using multiple methods."""
    result: Dict[str, Any] = {
        "passed": False,
        "methods": {},
        "recommendation": ""
    }
    
    # Method 1: Sample size rule
    if n >= 30:
        size_ok = True
        size_note = "Large sample (n ≥ 30): Central Limit Theorem applies"
    else:
        size_ok = False
        size_note = "Small sample (n < 30): Normality assumption is more critical"
    
    result["methods"]["sample_size"] = {
        "passed": size_ok,
        "note": size_note
    }
    
    if verbose:
        print(f"\n{group_name}:")
        print(f"  Sample size: n = {n}")
        print(f"  {size_note}")
    
    # Method 2: Skewness check
    mean_val = desc.mean(data)
    std_val = desc.sample_std_dev(data) if n > 1 else 0
    
    if std_val > 0:
        # Calculate skewness
        skewness = sum(((x - mean_val) / std_val) ** 3 for x in data) / n
        skew_ok = abs(skewness) < 1.0  # Rule of thumb: |skewness| < 1
        
        result["methods"]["skewness"] = {
            "passed": skew_ok,
            "value": skewness,
            "note": "|skewness| < 1 suggests approximate normality"
        }
        
        if verbose:
            print(f"  Skewness: {skewness:.4f}")
            if skew_ok:
                print(f"  ✓ Skewness is acceptable (|{skewness:.4f}| < 1)")
            else:
                print(f"  ⚠️  Data may be skewed (|{skewness:.4f}| ≥ 1)")
    else:
        result["methods"]["skewness"] = {
            "passed": None,
            "note": "Cannot calculate (std = 0)"
        }
    
    # Method 3: Mean vs Median (for symmetry)
    median_val = desc.median(data)
    mean_median_diff = abs(mean_val - median_val)
    mean_median_ratio = mean_median_diff / std_val if std_val > 0 else 0
    
    symmetry_ok = mean_median_ratio < 0.2  # Rule of thumb
    
    result["methods"]["symmetry"] = {
        "passed": symmetry_ok,
        "mean": mean_val,
        "median": median_val,
        "difference": mean_median_diff,
        "note": "Mean ≈ Median suggests symmetry"
    }
    
    if verbose:
        print(f"  Mean: {mean_val:.4f}, Median: {median_val:.4f}")
        print(f"  Difference: {mean_median_diff:.4f}")
        if symmetry_ok:
            print(f"  ✓ Mean and median are close (suggests symmetry)")
        else:
            print(f"  ⚠️  Mean and median differ (may indicate skewness)")
    
    # Method 4: Shapiro-Wilk test (if scipy available and n <= 5000)
    if SCIPY_AVAILABLE and 3 <= n <= 5000:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            shapiro_ok = shapiro_p > 0.05  # Not significant = normal
            
            result["methods"]["shapiro_wilk"] = {
                "passed": shapiro_ok,
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "note": "Shapiro-Wilk test (p > 0.05 suggests normality)"
            }
            
            if verbose:
                print(f"  Shapiro-Wilk test: W = {shapiro_stat:.4f}, p = {shapiro_p:.6f}")
                if shapiro_ok:
                    print(f"  ✓ Shapiro-Wilk test suggests normality (p > 0.05)")
                else:
                    print(f"  ⚠️  Shapiro-Wilk test suggests non-normality (p ≤ 0.05)")
        except Exception:
            result["methods"]["shapiro_wilk"] = {
                "passed": None,
                "note": "Test failed to run"
            }
    else:
        if not SCIPY_AVAILABLE:
            note = "SciPy not available"
        elif n < 3:
            note = "Sample too small (n < 3)"
        else:
            note = "Sample too large (n > 5000)"
        
        result["methods"]["shapiro_wilk"] = {
            "passed": None,
            "note": note
        }
    
    # Overall normality assessment
    passed_methods = sum(
        1 for m in result["methods"].values()
        if isinstance(m, dict) and m.get("passed") is True
    )
    total_methods = sum(
        1 for m in result["methods"].values()
        if isinstance(m, dict) and m.get("passed") is not None
    )
    
    if total_methods > 0:
        result["passed"] = passed_methods >= (total_methods / 2)  # Majority rule
    else:
        result["passed"] = size_ok  # Fall back to sample size
    
    if result["passed"]:
        result["recommendation"] = "Normality assumption appears met"
    else:
        result["recommendation"] = "Consider non-parametric test or transformation"
    
    return result


def check_regression_assumptions(
    x: Sequence[float],
    y: Sequence[float],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check assumptions for linear regression.
    
    Assumptions:
    1. Linearity (relationship between x and y is linear)
    2. Independence (residuals are independent)
    3. Homoscedasticity (constant variance of residuals)
    4. Normality of residuals
    
    Args:
        x: Independent variable
        y: Dependent variable
        verbose: If True, print detailed results
        
    Returns:
        Dictionary with assumption check results
    """
    results: Dict[str, Any] = {}
    
    if verbose:
        print("=" * 70)
        print("Linear Regression: Assumptions Check")
        print("=" * 70)
    
    # Calculate regression to get residuals
    from . import linear_regression_utils as lr
    
    slope, intercept, r_value, p_value, std_err = lr.linear_regression(x, y)
    
    # Calculate residuals
    residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
    
    # Check 1: Linearity
    if verbose:
        print("\n1. Linearity Assumption")
        print("-" * 70)
        print("The relationship between x and y should be linear.")
    
    r_squared = r_value ** 2
    linearity_ok = r_squared > 0.5  # Rule of thumb
    
    if verbose:
        print(f"Correlation coefficient (r): {r_value:.4f}")
        print(f"R²: {r_squared:.4f}")
        if linearity_ok:
            print("✓ Strong linear relationship (R² > 0.5)")
            print("  Consider plotting x vs y to visually confirm linearity")
        else:
            print("⚠️  Weak linear relationship (R² ≤ 0.5)")
            print("  Consider: transformation, polynomial regression, or non-linear model")
    
    results["linearity"] = {
        "passed": linearity_ok,
        "r": r_value,
        "r_squared": r_squared,
        "recommendation": "Plot x vs y to visually confirm" if linearity_ok else "Consider non-linear model"
    }
    
    # Check 2: Independence
    if verbose:
        print("\n2. Independence Assumption")
        print("-" * 70)
        print("Residuals should be independent (no autocorrelation).")
        print("✓ This assumption cannot be tested statistically.")
        print("  Ensure your data collection method produces independent observations.")
    
    results["independence"] = {
        "passed": None,
        "note": "Must be ensured through study design"
    }
    
    # Check 3: Homoscedasticity (constant variance)
    if verbose:
        print("\n3. Homoscedasticity (Constant Variance)")
        print("-" * 70)
        print("Residuals should have constant variance across all x values.")
    
    # Simple check: variance of residuals in first half vs second half
    n = len(residuals)
    mid = n // 2
    var_first = desc.sample_variance(residuals[:mid]) if mid > 1 else 0
    var_second = desc.sample_variance(residuals[mid:]) if (n - mid) > 1 else 0
    
    if var_first > 0 and var_second > 0:
        variance_ratio = max(var_first, var_second) / min(var_first, var_second)
        homoscedasticity_ok = variance_ratio < 2.0
    else:
        variance_ratio = None
        homoscedasticity_ok = True  # Can't test
    
    if verbose:
        if variance_ratio is not None:
            print(f"Variance of residuals (first half): {var_first:.4f}")
            print(f"Variance of residuals (second half): {var_second:.4f}")
            print(f"Variance ratio: {variance_ratio:.4f}")
            if homoscedasticity_ok:
                print("✓ Variances appear constant (ratio < 2.0)")
                print("  Consider plotting residuals vs x to visually confirm")
            else:
                print("⚠️  Variances may not be constant (ratio ≥ 2.0)")
                print("  Consider: transformation, weighted regression, or robust methods")
        else:
            print("⚠️  Cannot test (sample too small)")
    
    results["homoscedasticity"] = {
        "passed": homoscedasticity_ok,
        "variance_ratio": variance_ratio,
        "recommendation": "Plot residuals vs x" if homoscedasticity_ok else "Consider transformation"
    }
    
    # Check 4: Normality of residuals
    if verbose:
        print("\n4. Normality of Residuals")
        print("-" * 70)
        print("Residuals should be approximately normally distributed.")
    
    normality = _check_normality(residuals, len(residuals), verbose, "Residuals")
    results["normality"] = normality
    
    # Overall assessment
    if verbose:
        print("\n" + "=" * 70)
        print("Overall Assessment")
        print("=" * 70)
        
        all_passed = (
            results["linearity"]["passed"] and
            results["homoscedasticity"]["passed"] and
            results["normality"]["passed"]
        )
        
        if all_passed:
            print("✓ All testable assumptions are met.")
            print("  Linear regression is appropriate for your data.")
        else:
            print("⚠️  Some assumptions may be violated.")
            if not results["linearity"]["passed"]:
                print("  - Linearity: Consider non-linear model")
            if not results["homoscedasticity"]["passed"]:
                print("  - Homoscedasticity: Consider transformation")
            if not results["normality"]["passed"]:
                print("  - Normality: Regression is robust to this violation")
    
    results["all_passed"] = (
        results["linearity"]["passed"] and
        results["homoscedasticity"]["passed"] and
        results["normality"]["passed"]
    )
    
    return results

