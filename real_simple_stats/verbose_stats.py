"""Verbose statistical functions for educational purposes.

This module provides step-by-step output for statistical calculations,
helping users understand the math behind the results.
"""

import math
from collections.abc import Sequence

from scipy.stats import t as t_dist

from . import descriptive_statistics as desc
from . import hypothesis_testing as ht
from . import linear_regression_utils as lr


def t_test_verbose(
    data: Sequence[float],
    mu_null: float,
    alpha: float = 0.05,
    test_type: str = "two-tailed",
    verbose: bool = True,
) -> tuple[float, float, float, bool]:
    """Perform a one-sample t-test with step-by-step output.

    Args:
        data: Sample data
        mu_null: Null hypothesis mean
        alpha: Significance level
        test_type: 'two-tailed', 'greater', or 'less'
        verbose: If True, print step-by-step calculations

    Returns:
        Tuple of (t_statistic, p_value, critical_value, reject_null)
    """
    if verbose:
        print("=" * 70)
        print("One-Sample T-Test: Step-by-Step Calculation")
        print("=" * 70)

    # Step 1: Calculate sample statistics
    n = len(data)
    sample_mean = desc.mean(data)
    sample_std = desc.sample_std_dev(data)

    if verbose:
        print("\nStep 1: Calculate Sample Statistics")
        print("-" * 70)
        print(f"Sample size: n = {n}")
        print(f"Sample mean: x̄ = {sample_mean:.4f}")
        print(f"Sample standard deviation: s = {sample_std:.4f}")
        print("\nFormula: x̄ = Σx / n")
        print("         s = √[Σ(x - x̄)² / (n - 1)]")

    # Step 2: Calculate standard error
    standard_error = sample_std / math.sqrt(n)

    if verbose:
        print("\nStep 2: Calculate Standard Error")
        print("-" * 70)
        print("Standard error: SE = s / √n")
        print(f"                SE = {sample_std:.4f} / √{n}")
        print(f"                SE = {sample_std:.4f} / {math.sqrt(n):.4f}")
        print(f"                SE = {standard_error:.4f}")

    # Step 3: Calculate t-statistic
    t_stat = (sample_mean - mu_null) / standard_error

    if verbose:
        print("\nStep 3: Calculate T-Statistic")
        print("-" * 70)
        print("Formula: t = (x̄ - μ₀) / SE")
        print(
            f"         t = ({sample_mean:.4f} - {mu_null:.4f}) / {standard_error:.4f}"
        )
        print(f"         t = {sample_mean - mu_null:.4f} / {standard_error:.4f}")
        print(f"         t = {t_stat:.4f}")

    # Step 4: Calculate degrees of freedom
    df = n - 1

    if verbose:
        print("\nStep 4: Degrees of Freedom")
        print("-" * 70)
        print(f"df = n - 1 = {n} - 1 = {df}")

    # Step 5: Find critical value
    if test_type == "two-tailed":
        t_critical = ht.critical_value_t(alpha, df, "two-tailed")
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    elif test_type == "greater":
        t_critical = ht.critical_value_t(alpha, df, "greater")
        p_value = 1 - t_dist.cdf(t_stat, df)
    else:  # less
        t_critical = -ht.critical_value_t(alpha, df, "less")  # Negative for left-tailed
        p_value = t_dist.cdf(t_stat, df)

    if verbose:
        print("\nStep 5: Critical Value")
        print("-" * 70)
        print(f"Test type: {test_type}")
        print(f"Significance level: α = {alpha}")
        print(f"Degrees of freedom: df = {df}")
        if test_type == "two-tailed":
            print(f"Critical value: t_{{{alpha / 2:.3f}, {df}}} = ±{t_critical:.4f}")
        elif test_type == "less":
            print(f"Critical value: t_{{{alpha:.3f}, {df}}} = {t_critical:.4f}")
        else:
            print(f"Critical value: t_{{{alpha:.3f}, {df}}} = {t_critical:.4f}")

    # Step 6: Calculate p-value
    if verbose:
        print("\nStep 6: Calculate P-Value")
        print("-" * 70)
        if test_type == "two-tailed":
            print(f"P-value = 2 × P(t > |{t_stat:.4f}|)")
            print(f"        = 2 × P(t > {abs(t_stat):.4f})")
        elif test_type == "greater":
            print(f"P-value = P(t > {t_stat:.4f})")
        else:
            print(f"P-value = P(t < {t_stat:.4f})")
        print(f"        = {p_value:.6f}")

    # Step 7: Make decision
    reject = p_value < alpha

    if verbose:
        print("\nStep 7: Decision")
        print("-" * 70)
        print("Compare p-value to significance level:")
        print(f"  p-value = {p_value:.6f}")
        print(f"  α = {alpha}")
        if reject:
            print(f"\nSince {p_value:.6f} < {alpha}, we REJECT H₀")
            print(
                "Conclusion: There is sufficient evidence to reject the null hypothesis"
            )
        else:
            print(f"\nSince {p_value:.6f} ≥ {alpha}, we FAIL TO REJECT H₀")
            print(
                "Conclusion: There is insufficient evidence to reject the null hypothesis"
            )

        print("\n" + "=" * 70)

    return t_stat, p_value, t_critical, reject


def regression_verbose(
    x: Sequence[float],
    y: Sequence[float],
    verbose: bool = True,
) -> tuple[float, float, float, float, float]:
    """Perform linear regression with step-by-step output.

    Args:
        x: Independent variable
        y: Dependent variable
        verbose: If True, print step-by-step calculations

    Returns:
        Tuple of (slope, intercept, r_value, p_value, std_err)
    """
    if verbose:
        print("=" * 70)
        print("Linear Regression: Step-by-Step Calculation")
        print("=" * 70)

    n = len(x)
    x_mean = desc.mean(x)
    y_mean = desc.mean(y)

    if verbose:
        print("\nStep 1: Calculate Means")
        print("-" * 70)
        print(f"Sample size: n = {n}")
        print(f"x̄ (mean of x) = {x_mean:.4f}")
        print(f"ȳ (mean of y) = {y_mean:.4f}")
        print("\nFormula: x̄ = Σx / n")
        print("         ȳ = Σy / n")

    # Step 2: Calculate sums for slope
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if verbose:
        print("\nStep 2: Calculate Components for Slope")
        print("-" * 70)
        print("Numerator: Σ(x - x̄)(y - ȳ)")
        print(f"  = Σ(x - {x_mean:.4f})(y - {y_mean:.4f})")
        print(f"  = {numerator:.4f}")
        print("\nDenominator: Σ(x - x̄)²")
        print(f"  = Σ(x - {x_mean:.4f})²")
        print(f"  = {denominator:.4f}")

    # Step 3: Calculate slope
    slope = numerator / denominator

    if verbose:
        print("\nStep 3: Calculate Slope (b)")
        print("-" * 70)
        print("Formula: b = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²")
        print(f"         b = {numerator:.4f} / {denominator:.4f}")
        print(f"         b = {slope:.4f}")

    # Step 4: Calculate intercept
    intercept = y_mean - slope * x_mean

    if verbose:
        print("\nStep 4: Calculate Intercept (a)")
        print("-" * 70)
        print("Formula: a = ȳ - b·x̄")
        print(f"         a = {y_mean:.4f} - ({slope:.4f} × {x_mean:.4f})")
        print(f"         a = {y_mean:.4f} - {slope * x_mean:.4f}")
        print(f"         a = {intercept:.4f}")

    # Step 5: Get full regression results (for correlation, p-value, etc.)
    slope_full, intercept_full, r_value, p_value, std_err = lr.linear_regression(x, y)

    if verbose:
        print("\nStep 5: Regression Equation")
        print("-" * 70)
        print("y = a + bx")
        print(f"y = {intercept:.4f} + {slope:.4f}x")

        print("\nStep 6: Correlation and Model Fit")
        print("-" * 70)
        r_squared = r_value**2
        print(f"Correlation coefficient (r): {r_value:.4f}")
        print(f"Coefficient of determination (R²): {r_squared:.4f}")
        print(f"Interpretation: {r_squared:.1%} of variance in y is explained by x")

        print("\nStep 7: Statistical Significance")
        print("-" * 70)
        print(f"P-value: {p_value:.6f}")
        if p_value < 0.05:
            print("Since p < 0.05, the relationship is statistically significant")
        else:
            print("Since p ≥ 0.05, the relationship is not statistically significant")

        print(f"\nStandard error: {std_err:.4f}")
        print("\n" + "=" * 70)

    return slope_full, intercept_full, r_value, p_value, std_err


def mean_verbose(
    data: Sequence[float],
    verbose: bool = True,
) -> float:
    """Calculate mean with step-by-step output.

    Args:
        data: Input data
        verbose: If True, print step-by-step calculations

    Returns:
        Mean value
    """
    if verbose:
        print("=" * 70)
        print("Mean Calculation: Step-by-Step")
        print("=" * 70)
        print(f"\nData: {data}")
        print("\nStep 1: Sum all values")
        print("-" * 70)
        print("Sum = ", end="")
        print(" + ".join(str(x) for x in data))
        total = sum(data)
        print(f"     = {total}")

        print("\nStep 2: Count the number of values")
        print("-" * 70)
        n = len(data)
        print(f"n = {n}")

        print("\nStep 3: Divide sum by count")
        print("-" * 70)
        mean_val = total / n
        print("Mean = Sum / n")
        print(f"     = {total} / {n}")
        print(f"     = {mean_val:.4f}")
        print("\n" + "=" * 70)
    else:
        mean_val = desc.mean(data)

    return mean_val
