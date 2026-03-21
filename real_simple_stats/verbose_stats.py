"""Verbose statistical functions for educational purposes.

This module provides step-by-step output for statistical calculations,
helping users understand the math behind the results.
"""

import logging
import math
from collections.abc import Sequence

from scipy.stats import t as t_dist

from . import descriptive_statistics as desc

logger = logging.getLogger(__name__)
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
        logger.info("=" * 70)
        logger.info("One-Sample T-Test: Step-by-Step Calculation")
        logger.info("=" * 70)

    # Step 1: Calculate sample statistics
    n = len(data)
    sample_mean = desc.mean(data)
    sample_std = desc.sample_std_dev(data)

    if verbose:
        logger.info("\nStep 1: Calculate Sample Statistics")
        logger.info("-" * 70)
        logger.info("Sample size: n = %s", n)
        logger.info("Sample mean: x̄ = %.4f", sample_mean)
        logger.info("Sample standard deviation: s = %.4f", sample_std)
        logger.info("\nFormula: x̄ = Σx / n")
        logger.info("         s = √[Σ(x - x̄)² / (n - 1)]")

    # Step 2: Calculate standard error
    standard_error = sample_std / math.sqrt(n)

    if verbose:
        logger.info("\nStep 2: Calculate Standard Error")
        logger.info("-" * 70)
        logger.info("Standard error: SE = s / √n")
        logger.info("                SE = %.4f / √%s", sample_std, n)
        logger.info("                SE = %.4f / %.4f", sample_std, math.sqrt(n))
        logger.info("                SE = %.4f", standard_error)

    # Step 3: Calculate t-statistic
    t_stat = (sample_mean - mu_null) / standard_error

    if verbose:
        logger.info("\nStep 3: Calculate T-Statistic")
        logger.info("-" * 70)
        logger.info("Formula: t = (x̄ - μ₀) / SE")
        logger.info(
            "         t = (%.4f - %.4f) / %.4f",
            sample_mean,
            mu_null,
            standard_error,
        )
        logger.info("         t = %.4f / %.4f", sample_mean - mu_null, standard_error)
        logger.info("         t = %.4f", t_stat)

    # Step 4: Calculate degrees of freedom
    df = n - 1

    if verbose:
        logger.info("\nStep 4: Degrees of Freedom")
        logger.info("-" * 70)
        logger.info("df = n - 1 = %s - 1 = %s", n, df)

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
        logger.info("\nStep 5: Critical Value")
        logger.info("-" * 70)
        logger.info("Test type: %s", test_type)
        logger.info("Significance level: α = %s", alpha)
        logger.info("Degrees of freedom: df = %s", df)
        if test_type == "two-tailed":
            logger.info("Critical value: t_{%s, %s} = ±%.4f", alpha / 2, df, t_critical)
        elif test_type == "less":
            logger.info("Critical value: t_{%s, %s} = %.4f", alpha, df, t_critical)
        else:
            logger.info("Critical value: t_{%s, %s} = %.4f", alpha, df, t_critical)

    # Step 6: Calculate p-value
    if verbose:
        logger.info("\nStep 6: Calculate P-Value")
        logger.info("-" * 70)
        if test_type == "two-tailed":
            logger.info("P-value = 2 × P(t > |%.4f|)", t_stat)
            logger.info("        = 2 × P(t > %.4f)", abs(t_stat))
        elif test_type == "greater":
            logger.info("P-value = P(t > %.4f)", t_stat)
        else:
            logger.info("P-value = P(t < %.4f)", t_stat)
        logger.info("        = %.6f", p_value)

    # Step 7: Make decision
    reject = p_value < alpha

    if verbose:
        logger.info("\nStep 7: Decision")
        logger.info("-" * 70)
        logger.info("Compare p-value to significance level:")
        logger.info("  p-value = %.6f", p_value)
        logger.info("  α = %s", alpha)
        if reject:
            logger.info("\nSince %.6f < %s, we REJECT H₀", p_value, alpha)
            logger.info(
                "Conclusion: There is sufficient evidence to reject the null hypothesis"
            )
        else:
            logger.info("\nSince %.6f ≥ %s, we FAIL TO REJECT H₀", p_value, alpha)
            logger.info(
                "Conclusion: There is insufficient evidence to reject the null hypothesis"
            )

        logger.info("\n" + "=" * 70)

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
        logger.info("=" * 70)
        logger.info("Linear Regression: Step-by-Step Calculation")
        logger.info("=" * 70)

    n = len(x)
    x_mean = desc.mean(x)
    y_mean = desc.mean(y)

    if verbose:
        logger.info("\nStep 1: Calculate Means")
        logger.info("-" * 70)
        logger.info("Sample size: n = %s", n)
        logger.info("x̄ (mean of x) = %.4f", x_mean)
        logger.info("ȳ (mean of y) = %.4f", y_mean)
        logger.info("\nFormula: x̄ = Σx / n")
        logger.info("         ȳ = Σy / n")

    # Step 2: Calculate sums for slope
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if verbose:
        logger.info("\nStep 2: Calculate Components for Slope")
        logger.info("-" * 70)
        logger.info("Numerator: Σ(x - x̄)(y - ȳ)")
        logger.info("  = Σ(x - %.4f)(y - %.4f)", x_mean, y_mean)
        logger.info("  = %.4f", numerator)
        logger.info("\nDenominator: Σ(x - x̄)²")
        logger.info("  = Σ(x - %.4f)²", x_mean)
        logger.info("  = %.4f", denominator)

    # Step 3: Calculate slope
    slope = numerator / denominator

    if verbose:
        logger.info("\nStep 3: Calculate Slope (b)")
        logger.info("-" * 70)
        logger.info("Formula: b = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²")
        logger.info("         b = %.4f / %.4f", numerator, denominator)
        logger.info("         b = %.4f", slope)

    # Step 4: Calculate intercept
    intercept = y_mean - slope * x_mean

    if verbose:
        logger.info("\nStep 4: Calculate Intercept (a)")
        logger.info("-" * 70)
        logger.info("Formula: a = ȳ - b·x̄")
        logger.info("         a = %.4f - (%.4f × %.4f)", y_mean, slope, x_mean)
        logger.info("         a = %.4f - %.4f", y_mean, slope * x_mean)
        logger.info("         a = %.4f", intercept)

    # Step 5: Get full regression results (for correlation, p-value, etc.)
    slope_full, intercept_full, r_value, p_value, std_err = lr.linear_regression(x, y)

    if verbose:
        logger.info("\nStep 5: Regression Equation")
        logger.info("-" * 70)
        logger.info("y = a + bx")
        logger.info("y = %.4f + %.4f x", intercept, slope)

        logger.info("\nStep 6: Correlation and Model Fit")
        logger.info("-" * 70)
        r_squared = r_value**2
        logger.info("Correlation coefficient (r): %.4f", r_value)
        logger.info("Coefficient of determination (R²): %.4f", r_squared)
        logger.info("Interpretation: %.1f%% of variance in y is explained by x", r_squared * 100)

        logger.info("\nStep 7: Statistical Significance")
        logger.info("-" * 70)
        logger.info("P-value: %.6f", p_value)
        if p_value < 0.05:
            logger.info("Since p < 0.05, the relationship is statistically significant")
        else:
            logger.info("Since p ≥ 0.05, the relationship is not statistically significant")

        logger.info("\nStandard error: %.4f", std_err)
        logger.info("\n" + "=" * 70)

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
        logger.info("=" * 70)
        logger.info("Mean Calculation: Step-by-Step")
        logger.info("=" * 70)
        logger.info("\nData: %s", data)
        logger.info("\nStep 1: Sum all values")
        logger.info("-" * 70)
        logger.info("Sum = %s", " + ".join(str(x) for x in data))
        total = sum(data)
        logger.info("     = %s", total)

        logger.info("\nStep 2: Count the number of values")
        logger.info("-" * 70)
        n = len(data)
        logger.info("n = %s", n)

        logger.info("\nStep 3: Divide sum by count")
        logger.info("-" * 70)
        mean_val = total / n
        logger.info("Mean = Sum / n")
        logger.info("     = %s / %s", total, n)
        logger.info("     = %.4f", mean_val)
        logger.info("\n" + "=" * 70)
    else:
        mean_val = desc.mean(data)

    return mean_val
