from typing import List, Tuple
from scipy.stats import chi2

# --- CHI-SQUARE CORE UTILITIES ---

def chi_square_statistic(observed: List[int], expected: List[int]) -> float:
    """
    Compute the chi-square statistic: Σ((O - E)^2 / E)
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected lists must be the same length.")
    return sum((o - e) ** 2 / e for o, e in zip(observed, expected))

def critical_chi_square_value(alpha: float, df: int) -> float:
    """
    Return the critical chi-square value (right-tailed only).

    Args:
        alpha: Significance level (e.g., 0.05)
        df: Degrees of freedom

    Returns:
        Right-tailed critical value from the chi-square distribution.
    """
    return chi2.ppf(1 - alpha, df)

def reject_null_chi_square(chi_stat: float, critical_value: float) -> bool:
    """
    Determine whether to reject H0 based on test statistic and critical value.

    Args:
        chi_stat: The chi-square statistic.
        critical_value: The critical value for the given alpha and df.

    Returns:
        True if the null hypothesis should be rejected.
    """
    return chi_stat > critical_value