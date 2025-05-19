import math
from scipy.stats import norm, t, f
from typing import Tuple

# --- HYPOTHESIS TESTING BASICS ---

def state_null_hypothesis(description: str) -> str:
    return f"H0: {description}"

def state_alternate_hypothesis(description: str) -> str:
    return f"H1: {description}"

def is_right_tailed(test_statistic: float, critical_value: float) -> bool:
    return test_statistic > critical_value

def is_left_tailed(test_statistic: float, critical_value: float) -> bool:
    return test_statistic < -abs(critical_value)

def is_two_tailed(test_statistic: float, critical_value: float) -> bool:
    return abs(test_statistic) > critical_value

def p_value_method(test_statistic: float, test_type: str = "two-tailed") -> float:
    """Returns the p-value based on the test type."""
    if test_type == "two-tailed":
        return 2 * (1 - norm.cdf(abs(test_statistic)))
    elif test_type == "right-tailed":
        return 1 - norm.cdf(test_statistic)
    elif test_type == "left-tailed":
        return norm.cdf(test_statistic)
    else:
        raise ValueError("Invalid test_type")

def reject_null(p_value: float, alpha: float) -> bool:
    return p_value < alpha

# --- T-TEST AND F-TEST ---

def t_score(sample_mean: float, population_mean: float, sample_std: float, n: int) -> float:
    return (sample_mean - population_mean) / (sample_std / math.sqrt(n))

def f_test(var1: float, var2: float) -> float:
    """Conduct F-test: variance1 / variance2"""
    return var1 / var2

def critical_value_z(alpha: float, test_type: str = "two-tailed") -> float:
    if test_type == "two-tailed":
        return norm.ppf(1 - alpha / 2)
    return norm.ppf(1 - alpha)

def critical_value_t(alpha: float, df: int, test_type: str = "two-tailed") -> float:
    if test_type == "two-tailed":
        return t.ppf(1 - alpha / 2, df)
    return t.ppf(1 - alpha, df)

def critical_value_f(alpha: float, dfn: int, dfd: int) -> float:
    return f.ppf(1 - alpha, dfn, dfd)

# Example usage
if __name__ == "__main__":
    # Hypotheses
    print(state_null_hypothesis("μ = 100"))
    print(state_alternate_hypothesis("μ ≠ 100"))

    # Tail tests
    print("Is right-tailed:", is_right_tailed(2.1, 1.96))
    print("Is left-tailed:", is_left_tailed(-2.2, 1.96))
    print("Is two-tailed:", is_two_tailed(2.3, 1.96))

    # P-value and decision
    z = 2.05
    p = p_value_method(z, "two-tailed")
    print("P-value:", p)
    print("Reject H0 at alpha=0.05:", reject_null(p, 0.05))

    # T-test
    t_stat = t_score(sample_mean=104, population_mean=100, sample_std=10, n=25)
    print("T-score:", t_stat)
    print("Critical t (df=24):", critical_value_t(0.05, 24))

    # F-test
    f_stat = f_test(var1=36, var2=25)
    print("F statistic:", f_stat)
    print("Critical F (df1=9, df2=11):", critical_value_f(0.05, 9, 11))

    # Critical z values
    print("Critical Z (alpha=0.05):", critical_value_z(0.05))
