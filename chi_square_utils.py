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

def critical_chi_square_value(alpha: float, df: int, tail: str = "right") -> float:
    """
    Find the critical chi-square value.
    
    Parameters:
        alpha: significance level (e.g., 0.05)
        df: degrees of freedom
        tail: 'right' or 'left'
    """
    if tail == "right":
        return chi2.ppf(1 - alpha, df)
    elif tail == "left":
        return chi2.ppf(alpha, df)
    else:
        raise ValueError("tail must be 'right' or 'left'")

def reject_null_chi_square(chi_stat: float, critical_value: float, tail: str = "right") -> bool:
    """
    Determine whether to reject H0 based on the test statistic and critical value.
    """
    if tail == "right":
        return chi_stat > critical_value
    elif tail == "left":
        return chi_stat < critical_value
    else:
        raise ValueError("tail must be 'right' or 'left'")

# Example usage
if __name__ == "__main__":
    observed = [20, 30, 25]
    expected = [25, 25, 25]
    
    chi_stat = chi_square_statistic(observed, expected)
    print("Chi-square statistic:", chi_stat)

    alpha = 0.05
    df = len(observed) - 1
    critical_right = critical_chi_square_value(alpha, df, tail="right")
    critical_left = critical_chi_square_value(alpha, df, tail="left")

    print("Right-tailed critical value (α=0.05, df=2):", critical_right)
    print("Left-tailed critical value (α=0.05, df=2):", critical_left)
    print("Reject H0 (right-tailed):", reject_null_chi_square(chi_stat, critical_right, tail="right"))
