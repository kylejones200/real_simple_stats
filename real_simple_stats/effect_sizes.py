"""Effect size calculations for statistical tests.

This module provides functions for calculating various effect size measures
including Cohen's d, eta-squared, Cramér's V, and odds ratios.
"""

from typing import List, Tuple
import numpy as np
from scipy import stats


def cohens_d(group1: List[float], group2: List[float], pooled: bool = True) -> float:
    """Calculate Cohen's d effect size for two groups.

    Args:
        group1: First group data
        group2: Second group data
        pooled: Use pooled standard deviation (default: True)

    Returns:
        Cohen's d effect size

    Raises:
        ValueError: If groups are too small

    Examples:
        >>> group1 = [1, 2, 3, 4, 5]
        >>> group2 = [3, 4, 5, 6, 7]
        >>> d = cohens_d(group1, group2)
        >>> abs(d) > 0
        True
    """
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Both groups must contain at least 2 values")

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    if pooled:
        # Pooled standard deviation
        n1 = len(group1)
        n2 = len(group2)
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        d = (mean1 - mean2) / pooled_std
    else:
        # Use standard deviation of control group (group2)
        std2 = np.std(group2, ddof=1)

        if std2 == 0:
            return 0.0

        d = (mean1 - mean2) / std2

    return float(d)


def hedges_g(group1: List[float], group2: List[float]) -> float:
    """Calculate Hedges' g effect size (bias-corrected Cohen's d).

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Hedges' g effect size

    Raises:
        ValueError: If groups are too small

    Examples:
        >>> group1 = [1, 2, 3, 4, 5]
        >>> group2 = [3, 4, 5, 6, 7]
        >>> g = hedges_g(group1, group2)
        >>> abs(g) > 0
        True
    """
    d = cohens_d(group1, group2, pooled=True)

    n1 = len(group1)
    n2 = len(group2)
    df = n1 + n2 - 2

    # Correction factor
    correction = 1 - (3 / (4 * df - 1))

    return float(d * correction)


def glass_delta(group1: List[float], group2: List[float]) -> float:
    """Calculate Glass's delta effect size.

    Uses only the control group's standard deviation.

    Args:
        group1: Treatment group data
        group2: Control group data

    Returns:
        Glass's delta effect size

    Raises:
        ValueError: If groups are too small

    Examples:
        >>> treatment = [5, 6, 7, 8, 9]
        >>> control = [1, 2, 3, 4, 5]
        >>> delta = glass_delta(treatment, control)
        >>> delta > 0
        True
    """
    return cohens_d(group1, group2, pooled=False)


def eta_squared(groups: List[List[float]]) -> float:
    """Calculate eta-squared effect size for ANOVA.

    Eta-squared represents the proportion of variance explained.

    Args:
        groups: List of groups (each group is a list of values)

    Returns:
        Eta-squared value (0 to 1)

    Raises:
        ValueError: If there are fewer than 2 groups or groups are too small

    Examples:
        >>> groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> eta2 = eta_squared(groups)
        >>> 0 <= eta2 <= 1
        True
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")
    if any(len(group) < 1 for group in groups):
        raise ValueError("All groups must contain at least 1 value")

    # Flatten all data
    all_data = np.concatenate([np.array(group) for group in groups])
    grand_mean = np.mean(all_data)

    # Calculate between-group sum of squares
    ss_between = 0
    for group in groups:
        group_mean = np.mean(group)
        ss_between += len(group) * (group_mean - grand_mean) ** 2

    # Calculate total sum of squares
    ss_total = np.sum((all_data - grand_mean) ** 2)

    if ss_total == 0:
        return 0.0

    eta2 = ss_between / ss_total

    return float(eta2)


def partial_eta_squared(groups: List[List[float]]) -> float:
    """Calculate partial eta-squared effect size.

    Args:
        groups: List of groups (each group is a list of values)

    Returns:
        Partial eta-squared value (0 to 1)

    Raises:
        ValueError: If there are fewer than 2 groups or groups are too small

    Examples:
        >>> groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> peta2 = partial_eta_squared(groups)
        >>> 0 <= peta2 <= 1
        True
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")
    if any(len(group) < 1 for group in groups):
        raise ValueError("All groups must contain at least 1 value")

    # Flatten all data
    all_data = np.concatenate([np.array(group) for group in groups])
    grand_mean = np.mean(all_data)

    # Calculate between-group sum of squares
    ss_between = 0
    for group in groups:
        group_mean = np.mean(group)
        ss_between += len(group) * (group_mean - grand_mean) ** 2

    # Calculate within-group sum of squares
    ss_within = 0
    for group in groups:
        group_mean = np.mean(group)
        ss_within += np.sum((np.array(group) - group_mean) ** 2)

    if ss_between + ss_within == 0:
        return 0.0

    partial_eta2 = ss_between / (ss_between + ss_within)

    return float(partial_eta2)


def omega_squared(groups: List[List[float]]) -> float:
    """Calculate omega-squared effect size (less biased than eta-squared).

    Args:
        groups: List of groups (each group is a list of values)

    Returns:
        Omega-squared value

    Raises:
        ValueError: If there are fewer than 2 groups or groups are too small

    Examples:
        >>> groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> omega2 = omega_squared(groups)
        >>> omega2 >= 0
        True
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")
    if any(len(group) < 2 for group in groups):
        raise ValueError("All groups must contain at least 2 values")

    k = len(groups)  # Number of groups
    n_total = sum(len(group) for group in groups)

    # Flatten all data
    all_data = np.concatenate([np.array(group) for group in groups])
    grand_mean = np.mean(all_data)

    # Calculate between-group sum of squares
    ss_between = 0
    for group in groups:
        group_mean = np.mean(group)
        ss_between += len(group) * (group_mean - grand_mean) ** 2

    # Calculate within-group sum of squares
    ss_within = 0
    for group in groups:
        group_mean = np.mean(group)
        ss_within += np.sum((np.array(group) - group_mean) ** 2)

    # Calculate total sum of squares
    ss_total = np.sum((all_data - grand_mean) ** 2)

    # Mean square within
    ms_within = ss_within / (n_total - k)

    if ss_total + ms_within == 0:
        return 0.0

    omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)

    return float(max(0, omega2))  # Can't be negative


def cramers_v(contingency_table: List[List[int]]) -> float:
    """Calculate Cramér's V effect size for chi-square test.

    Args:
        contingency_table: 2D contingency table

    Returns:
        Cramér's V value (0 to 1)

    Raises:
        ValueError: If table dimensions are invalid

    Examples:
        >>> table = [[10, 20], [30, 40]]
        >>> v = cramers_v(table)
        >>> 0 <= v <= 1
        True
    """
    if len(contingency_table) < 2:
        raise ValueError("Contingency table must have at least 2 rows")
    if any(len(row) < 2 for row in contingency_table):
        raise ValueError("Contingency table must have at least 2 columns")

    table = np.array(contingency_table)

    # Perform chi-square test
    chi2, _, _, _ = stats.chi2_contingency(table)

    n = np.sum(table)
    min_dim = min(table.shape[0], table.shape[1]) - 1

    if n == 0 or min_dim == 0:
        return 0.0

    v = np.sqrt(chi2 / (n * min_dim))

    return float(v)


def phi_coefficient(contingency_table: List[List[int]]) -> float:
    """Calculate phi coefficient for 2x2 contingency table.

    Args:
        contingency_table: 2x2 contingency table

    Returns:
        Phi coefficient (-1 to 1)

    Raises:
        ValueError: If table is not 2x2

    Examples:
        >>> table = [[10, 20], [30, 40]]
        >>> phi = phi_coefficient(table)
        >>> -1 <= phi <= 1
        True
    """
    if len(contingency_table) != 2 or any(len(row) != 2 for row in contingency_table):
        raise ValueError("Phi coefficient requires a 2x2 contingency table")

    table = np.array(contingency_table)
    a, b = table[0]
    c, d = table[1]

    numerator = (a * d) - (b * c)
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

    if denominator == 0:
        return 0.0

    phi = numerator / denominator

    return float(phi)


def odds_ratio(contingency_table: List[List[int]]) -> Tuple[float, Tuple[float, float]]:
    """Calculate odds ratio and 95% confidence interval for 2x2 table.

    Args:
        contingency_table: 2x2 contingency table [[a, b], [c, d]]

    Returns:
        Tuple of (odds_ratio, (ci_lower, ci_upper))

    Raises:
        ValueError: If table is not 2x2 or contains zeros

    Examples:
        >>> table = [[10, 20], [30, 40]]
        >>> or_value, ci = odds_ratio(table)
        >>> or_value > 0
        True
    """
    if len(contingency_table) != 2 or any(len(row) != 2 for row in contingency_table):
        raise ValueError("Odds ratio requires a 2x2 contingency table")

    table = np.array(contingency_table)
    a, b = table[0]
    c, d = table[1]

    # Add small constant to avoid division by zero
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_value = (a * d) / (b * c)

    # Calculate 95% confidence interval using log transformation
    log_or = np.log(or_value)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)

    return float(or_value), (float(ci_lower), float(ci_upper))


def relative_risk(
    contingency_table: List[List[int]],
) -> Tuple[float, Tuple[float, float]]:
    """Calculate relative risk and 95% confidence interval for 2x2 table.

    Args:
        contingency_table: 2x2 contingency table [[a, b], [c, d]]

    Returns:
        Tuple of (relative_risk, (ci_lower, ci_upper))

    Raises:
        ValueError: If table is not 2x2

    Examples:
        >>> table = [[10, 90], [30, 70]]
        >>> rr, ci = relative_risk(table)
        >>> rr > 0
        True
    """
    if len(contingency_table) != 2 or any(len(row) != 2 for row in contingency_table):
        raise ValueError("Relative risk requires a 2x2 contingency table")

    table = np.array(contingency_table)
    a, b = table[0]
    c, d = table[1]

    # Risk in exposed group
    risk1 = a / (a + b) if (a + b) > 0 else 0

    # Risk in unexposed group
    risk2 = c / (c + d) if (c + d) > 0 else 0

    if risk2 == 0:
        raise ValueError("Cannot calculate relative risk when control risk is zero")

    rr = risk1 / risk2

    # Calculate 95% confidence interval using log transformation
    log_rr = np.log(rr)
    se_log_rr = np.sqrt((b / (a * (a + b))) + (d / (c * (c + d))))

    ci_lower = np.exp(log_rr - 1.96 * se_log_rr)
    ci_upper = np.exp(log_rr + 1.96 * se_log_rr)

    return float(rr), (float(ci_lower), float(ci_upper))


def cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h for comparing two proportions.

    Args:
        p1: First proportion (0 to 1)
        p2: Second proportion (0 to 1)

    Returns:
        Cohen's h effect size

    Raises:
        ValueError: If proportions are not between 0 and 1

    Examples:
        >>> h = cohens_h(0.7, 0.5)
        >>> h > 0
        True
    """
    if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
        raise ValueError("Proportions must be between 0 and 1")

    # Arcsine transformation
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))

    h = phi1 - phi2

    return float(h)


def interpret_effect_size(effect_size: float, measure: str) -> str:
    """Interpret effect size magnitude according to Cohen's conventions.

    Args:
        effect_size: Effect size value
        measure: Type of effect size ('d', 'r', 'eta_squared', 'cramers_v')

    Returns:
        Interpretation string

    Raises:
        ValueError: If measure type is unknown

    Examples:
        >>> interpret_effect_size(0.5, 'd')
        'medium'
    """
    abs_effect = abs(effect_size)

    if measure == "d":  # Cohen's d
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    elif measure == "r":  # Correlation coefficient
        if abs_effect < 0.1:
            return "negligible"
        elif abs_effect < 0.3:
            return "small"
        elif abs_effect < 0.5:
            return "medium"
        else:
            return "large"

    elif measure == "eta_squared":
        if abs_effect < 0.01:
            return "negligible"
        elif abs_effect < 0.06:
            return "small"
        elif abs_effect < 0.14:
            return "medium"
        else:
            return "large"

    elif measure == "cramers_v":
        if abs_effect < 0.1:
            return "negligible"
        elif abs_effect < 0.3:
            return "small"
        elif abs_effect < 0.5:
            return "medium"
        else:
            return "large"

    else:
        raise ValueError(f"Unknown measure: {measure}")


__all__ = [
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "eta_squared",
    "partial_eta_squared",
    "omega_squared",
    "cramers_v",
    "phi_coefficient",
    "odds_ratio",
    "relative_risk",
    "cohens_h",
    "interpret_effect_size",
]
