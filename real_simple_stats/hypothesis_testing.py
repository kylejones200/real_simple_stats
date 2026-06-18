import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency, f, f_oneway, norm, t

logger = logging.getLogger(__name__)

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
        return 2 * (1 - float(norm.cdf(abs(test_statistic))))
    elif test_type == "right-tailed":
        return 1 - float(norm.cdf(test_statistic))
    elif test_type == "left-tailed":
        return float(norm.cdf(test_statistic))
    else:
        raise ValueError("Invalid test_type")


def reject_null(p_value: float, alpha: float) -> bool:
    return p_value < alpha


# --- T-TEST AND F-TEST ---


def t_score(
    sample_mean: float, population_mean: float, sample_std: float, n: int
) -> float:
    return (sample_mean - population_mean) / (sample_std / math.sqrt(n))


def f_test(var1: float, var2: float) -> float:
    """Conduct F-test: variance1 / variance2"""
    return var1 / var2


def critical_value_z(alpha: float, test_type: str = "two-tailed") -> float:
    if test_type == "two-tailed":
        return float(norm.ppf(1 - alpha / 2))
    return float(norm.ppf(1 - alpha))


def critical_value_t(alpha: float, df: int, test_type: str = "two-tailed") -> float:
    if test_type == "two-tailed":
        return float(t.ppf(1 - alpha / 2, df))
    return float(t.ppf(1 - alpha, df))


def critical_value_f(alpha: float, dfn: int, dfd: int) -> float:
    return float(f.ppf(1 - alpha, dfn, dfd))


def one_way_anova(
    *groups: Sequence[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-way ANOVA — test whether multiple group means are equal.

    Tests the null hypothesis that all k groups have the same population mean.
    When the F-statistic is large (the between-group variance dwarfs the
    within-group variance) we reject H₀ and conclude at least one group
    differs.

    A significant p-value only tells you *some* group differs — follow up with
    post-hoc tests (Tukey's HSD, Bonferroni) to find *which* ones.

    Args:
        *groups: Two or more sequences of numeric observations (the groups).
        alpha: Significance level (default 0.05).

    Returns:
        dict with keys:
            f_stat: F-statistic (between-group variance / within-group variance).
            p_value: p-value under the F(k-1, N-k) distribution.
            df_between: Degrees of freedom for the numerator (k − 1).
            df_within: Degrees of freedom for the denominator (N − k).
            eta_squared: Effect size η² = SS_between / SS_total. 0.01 small,
                0.06 medium, 0.14 large (Cohen 1988).
            reject_null: True if p < alpha.
            group_means: Mean of each group.
            group_ns: Sample size of each group.
            n_groups: Number of groups (k).
            n_total: Total observations (N).

    Raises:
        ValueError: If fewer than 2 groups or any group has fewer than 2 observations.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> g1 = rng.normal(0, 1, 30)
        >>> g2 = rng.normal(1, 1, 30)
        >>> g3 = rng.normal(2, 1, 30)
        >>> r = one_way_anova(g1, g2, g3)
        >>> r["reject_null"]
        True
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups.")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    for i, a in enumerate(arrays):
        if len(a) < 2:
            raise ValueError(f"Group {i} has fewer than 2 observations.")

    f_stat, p_value = f_oneway(*arrays)
    f_stat, p_value = float(f_stat), float(p_value)

    k = len(arrays)
    group_means = [float(a.mean()) for a in arrays]
    group_ns = [len(a) for a in arrays]
    n_total = sum(group_ns)

    grand_mean = float(np.concatenate(arrays).mean())
    ss_between = sum(
        n * (m - grand_mean) ** 2
        for n, m in zip(group_ns, group_means)
    )
    ss_total = float(sum(((a - grand_mean) ** 2).sum() for a in arrays))
    eta_squared = ss_between / ss_total if ss_total > 0 else float("nan")

    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "df_between": k - 1,
        "df_within": n_total - k,
        "eta_squared": eta_squared,
        "reject_null": p_value < alpha,
        "group_means": group_means,
        "group_ns": group_ns,
        "n_groups": k,
        "n_total": n_total,
    }


def chi_square_independence(
    observed: Sequence[Sequence[int]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Chi-square test of independence for a contingency table.

    Tests whether two categorical variables are independent.  If p < α we
    reject independence and conclude the variables are associated.

    Cramér's V measures the *strength* of that association (0 = none,
    1 = perfect), scaled so it's comparable across tables of different sizes:

        V = sqrt(χ² / (n × (min(r, c) − 1)))

    Note: The test assumes expected cell counts ≥ 5.  Warn the user if this
    is violated.

    Args:
        observed: 2-D contingency table (rows × columns) of counts.
        alpha: Significance level (default 0.05).

    Returns:
        dict with keys:
            chi2: Chi-square statistic.
            p_value: p-value.
            dof: Degrees of freedom = (rows − 1) × (cols − 1).
            expected: Expected frequencies under independence.
            cramers_v: Effect size Cramér's V (0–1).
            reject_null: True if p < alpha.
            low_expected_cells: Number of cells with expected count < 5.
            interpretation: Plain-English summary.

    Raises:
        ValueError: If the table has fewer than 2 rows or 2 columns.

    Example:
        >>> table = [[25, 15], [20, 30]]
        >>> r = chi_square_independence(table)
        >>> r["reject_null"]
        True
        >>> 0 <= r["cramers_v"] <= 1
        True
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2:
        raise ValueError("observed must be a 2-D array with at least 2 rows and 2 columns.")

    chi2, p_value, dof, expected = chi2_contingency(obs)
    chi2, p_value, dof = float(chi2), float(p_value), int(dof)

    n = float(obs.sum())
    min_dim = min(obs.shape) - 1
    cramers_v = float(math.sqrt(chi2 / (n * min_dim))) if n > 0 and min_dim > 0 else 0.0

    low_expected = int((expected < 5).sum())

    if p_value < alpha:
        interp = (
            f"Reject independence (p = {p_value:.4f}). "
            f"The variables are associated (Cramér's V = {cramers_v:.3f})."
        )
    else:
        interp = (
            f"Fail to reject independence (p = {p_value:.4f}). "
            "No significant association detected."
        )
    if low_expected > 0:
        interp += (
            f" Warning: {low_expected} cell(s) have expected count < 5 — "
            "consider Fisher's exact test or collapsing categories."
        )

    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "expected": expected,
        "cramers_v": cramers_v,
        "reject_null": p_value < alpha,
        "low_expected_cells": low_expected,
        "interpretation": interp,
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Hypotheses
    logger.info("%s", state_null_hypothesis("μ = 100"))
    logger.info("%s", state_alternate_hypothesis("μ ≠ 100"))

    # Tail tests
    logger.info("Is right-tailed: %s", is_right_tailed(2.1, 1.96))
    logger.info("Is left-tailed: %s", is_left_tailed(-2.2, 1.96))
    logger.info("Is two-tailed: %s", is_two_tailed(2.3, 1.96))

    # P-value and decision
    z = 2.05
    p = p_value_method(z, "two-tailed")
    logger.info("P-value: %s", p)
    logger.info("Reject H0 at alpha=0.05: %s", reject_null(p, 0.05))

    # T-test
    t_stat = t_score(sample_mean=104, population_mean=100, sample_std=10, n=25)
    logger.info("T-score: %s", t_stat)
    logger.info("Critical t (df=24): %s", critical_value_t(0.05, 24))

    # F-test
    f_stat = f_test(var1=36, var2=25)
    logger.info("F statistic: %s", f_stat)
    logger.info("Critical F (df1=9, df2=11): %s", critical_value_f(0.05, 9, 11))

    # Critical z values
    logger.info("Critical Z (alpha=0.05): %s", critical_value_z(0.05))
