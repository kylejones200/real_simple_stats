import logging
import math
from collections.abc import Sequence
from typing import Any

from . import _rss

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
        return 2 * _rss.norm_sf(abs(test_statistic))
    elif test_type == "right-tailed":
        return _rss.norm_sf(test_statistic)
    elif test_type == "left-tailed":
        return _rss.norm_cdf(test_statistic)
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
        return _rss.norm_ppf(1 - alpha / 2)
    return _rss.norm_ppf(1 - alpha)


def critical_value_t(alpha: float, df: int, test_type: str = "two-tailed") -> float:
    if test_type == "two-tailed":
        return _rss.t_ppf(1 - alpha / 2, df)
    return _rss.t_ppf(1 - alpha, df)


def critical_value_f(alpha: float, dfn: int, dfd: int) -> float:
    return _rss.f_ppf(1 - alpha, dfn, dfd)


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
        >>> from real_simple_stats import Rng
        >>> rng = Rng(0)
        >>> g1 = rng.normal(0, 1, 30)
        >>> g2 = rng.normal(1, 1, 30)
        >>> g3 = rng.normal(2, 1, 30)
        >>> r = one_way_anova(g1, g2, g3)
        >>> r["reject_null"]
        True
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups.")
    arrays: list[list[float]] = [[float(v) for v in g] for g in groups]
    for i, a in enumerate(arrays):
        if len(a) < 2:
            raise ValueError(f"Group {i} has fewer than 2 observations.")

    f_stat, p_value = _rss.f_oneway(arrays)

    k = len(arrays)
    group_means = [_rss.mean(a) for a in arrays]
    group_ns = [len(a) for a in arrays]
    n_total = sum(group_ns)

    grand_mean = _rss.mean([v for a in arrays for v in a])
    ss_between = sum(
        n * (m - grand_mean) ** 2
        for n, m in zip(group_ns, group_means)
    )
    ss_total = sum((v - grand_mean) ** 2 for a in arrays for v in a)
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
        >>> round(r["p_value"], 4)   # just above 0.05 once Yates-corrected
        0.0562
        >>> r["reject_null"]
        False
        >>> 0 <= r["cramers_v"] <= 1
        True
    """
    try:
        obs: list[list[float]] = [[float(v) for v in row] for row in observed]
    except TypeError as exc:
        raise ValueError("observed must be a 2-D table of numbers.") from exc
    n_rows = len(obs)
    n_cols = len(obs[0]) if n_rows else 0
    if n_rows < 2 or n_cols < 2 or any(len(r) != n_cols for r in obs):
        raise ValueError("observed must be a 2-D array with at least 2 rows and 2 columns.")

    flat = [v for row in obs for v in row]
    chi2, p_value, dof, expected_flat = _rss.chi2_contingency(n_rows, n_cols, flat, True)
    dof = int(dof)
    expected = [expected_flat[i * n_cols : (i + 1) * n_cols] for i in range(n_rows)]

    n = sum(flat)
    min_dim = min(n_rows, n_cols) - 1
    cramers_v = float(math.sqrt(chi2 / (n * min_dim))) if n > 0 and min_dim > 0 else 0.0

    low_expected = sum(1 for row in expected for v in row if v < 5)

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


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Rank values from 1 upward, giving tied values their average rank.

    Ties matter here: both nonparametric tests below correct their variance
    using the tie structure, and using ordinal ranks instead would inflate
    significance.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0  # average of the 1-based positions
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _tie_correction_sum(values: Sequence[float]) -> float:
    """Sum of (t^3 - t) over each group of t tied values."""
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return sum(t**3 - t for t in counts.values() if t > 1)


def one_sample_t_test(
    data: Sequence[float], mu: float = 0.0
) -> tuple[float, float]:
    """Test whether a sample mean differs from a hypothesized value.

    Args:
        data: The observed sample (at least 2 values)
        mu: The hypothesized population mean

    Returns:
        Tuple of (t_statistic, two_sided_p_value)

    Raises:
        ValueError: If fewer than 2 values are given, or they are all identical

    Example:
        >>> t, p = one_sample_t_test([5.1, 4.9, 5.3, 5.0, 5.2], mu=5.0)
        >>> round(t, 4), round(p, 4)
        (1.4142, 0.2302)
    """
    if len(data) < 2:
        raise ValueError("One-sample t-test requires at least 2 values")
    t_stat, p_value = _rss.ttest_1samp(data, float(mu))
    if math.isnan(t_stat):
        raise ValueError("t-test is undefined when all values are identical")
    return t_stat, p_value


def two_sample_t_test(
    group1: Sequence[float], group2: Sequence[float], equal_var: bool = True
) -> tuple[float, float]:
    """Compare the means of two independent groups.

    Args:
        group1: First sample (at least 2 values)
        group2: Second sample (at least 2 values)
        equal_var: True for Student's pooled-variance test, False for Welch's,
            which does not assume the two groups share a variance

    Returns:
        Tuple of (t_statistic, two_sided_p_value)

    Raises:
        ValueError: If either group has fewer than 2 values

    Example:
        >>> t, p = two_sample_t_test([1, 2, 3, 4], [6, 7, 8, 9])
        >>> p < 0.01
        True
    """
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Two-sample t-test requires at least 2 values per group")
    return _rss.ttest_ind(group1, group2, bool(equal_var))


def paired_t_test(
    before: Sequence[float], after: Sequence[float]
) -> tuple[float, float]:
    """Compare two measurements taken on the same subjects.

    Equivalent to a one-sample t-test on the differences.

    Args:
        before: Measurements before the intervention
        after: Measurements after, in the same subject order

    Returns:
        Tuple of (t_statistic, two_sided_p_value)

    Raises:
        ValueError: If the samples differ in length or have fewer than 2 pairs

    Example:
        >>> t, p = paired_t_test([10, 12, 11, 13], [12, 15, 13, 16])
        >>> t < 0    # values rose, so the before-after difference is negative
        True
    """
    if len(before) != len(after):
        raise ValueError("Paired t-test requires samples of the same length")
    if len(before) < 2:
        raise ValueError("Paired t-test requires at least 2 pairs")
    return _rss.ttest_rel(before, after)


def z_test(
    data: Sequence[float], mu: float, sigma: float
) -> tuple[float, float]:
    """Test a sample mean against a hypothesized mean with a *known* variance.

    Use this only when the population standard deviation is genuinely known.
    When it is estimated from the sample -- the usual case -- use
    :func:`one_sample_t_test` instead.

    Args:
        data: The observed sample
        mu: Hypothesized population mean
        sigma: Known population standard deviation (positive)

    Returns:
        Tuple of (z_statistic, two_sided_p_value)

    Raises:
        ValueError: If the sample is empty or sigma is not positive

    Example:
        >>> z, p = z_test([102, 98, 105, 101, 99], mu=100, sigma=15)
        >>> round(z, 4)
        0.1491
    """
    if not data:
        raise ValueError("z-test requires at least one value")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    n = len(data)
    z_stat = (_rss.mean(data) - mu) / (sigma / math.sqrt(n))
    return z_stat, 2 * _rss.norm_sf(abs(z_stat))


def one_proportion_z_test(
    p_hat: float, n: int, p0: float
) -> tuple[float, float]:
    """Test an observed proportion against a hypothesized one.

    The standard error uses the hypothesized proportion ``p0``, which is the
    convention for a null-hypothesis test.

    Args:
        p_hat: Observed sample proportion, in [0, 1]
        n: Sample size (positive)
        p0: Hypothesized population proportion, strictly between 0 and 1

    Returns:
        Tuple of (z_statistic, two_sided_p_value)

    Raises:
        ValueError: If any argument is out of range

    Example:
        >>> z, p = one_proportion_z_test(p_hat=0.6, n=50, p0=0.5)
        >>> round(z, 4)
        1.4142
    """
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError("p_hat must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be strictly between 0 and 1")
    z_stat = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
    return z_stat, 2 * _rss.norm_sf(abs(z_stat))


def mann_whitney_u(
    group1: Sequence[float], group2: Sequence[float]
) -> tuple[float, float]:
    """Nonparametric test for whether one group tends to exceed the other.

    The rank-based alternative to :func:`two_sample_t_test`: it assumes no
    particular distribution, only that the observations are independent. The
    p-value comes from the normal approximation with a continuity correction
    and a tie correction, matching ``scipy.stats.mannwhitneyu`` with
    ``method="asymptotic"``.

    Args:
        group1: First sample (non-empty)
        group2: Second sample (non-empty)

    Returns:
        Tuple of (U_statistic_for_group1, two_sided_p_value)

    Raises:
        ValueError: If either group is empty

    Example:
        >>> u, p = mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        >>> u
        0.0
        >>> p < 0.05
        True
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Mann-Whitney U requires both groups to be non-empty")

    pooled = list(group1) + list(group2)
    ranks = _average_ranks(pooled)
    rank_sum_1 = sum(ranks[:n1])
    u1 = rank_sum_1 - n1 * (n1 + 1) / 2.0

    n = n1 + n2
    mean_u = n1 * n2 / 2.0
    ties = _tie_correction_sum(pooled)
    variance = (n1 * n2 / 12.0) * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
    if variance <= 0:
        return u1, 1.0

    # Continuity correction of 0.5 toward the mean.
    numerator = abs(u1 - mean_u) - 0.5
    z = max(numerator, 0.0) / math.sqrt(variance)
    return u1, 2 * _rss.norm_sf(z)


def wilcoxon_signed_rank(
    before: Sequence[float],
    after: Sequence[float],
    correction: bool = True,
) -> tuple[float, float]:
    """Nonparametric test for a shift between paired measurements.

    The rank-based alternative to :func:`paired_t_test`. Pairs with zero
    difference are discarded, as is conventional, and the p-value comes from
    the normal approximation with a tie correction.

    Args:
        before: Measurements before the intervention
        after: Measurements after, in the same subject order
        correction: Apply the continuity correction. Defaults to True, which
            is the textbook treatment and matches this library's
            :func:`mann_whitney_u`. Note that ``scipy.stats.wilcoxon``
            defaults the other way, so pass ``correction=False`` to reproduce
            SciPy's default output exactly.

    Returns:
        Tuple of (W_statistic, two_sided_p_value), where W is the smaller of
        the positive and negative signed-rank sums

    Raises:
        ValueError: If the samples differ in length, or every difference is zero

    Example:
        >>> w, p = wilcoxon_signed_rank([10, 12, 11, 13, 9], [12, 15, 13, 16, 12])
        >>> w
        0.0
    """
    if len(before) != len(after):
        raise ValueError("Wilcoxon signed-rank requires samples of the same length")

    diffs = [float(a) - float(b) for a, b in zip(before, after)]
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        raise ValueError("All differences are zero; the test is undefined")

    n = len(nonzero)
    ranks = _average_ranks([abs(d) for d in nonzero])
    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(nonzero, ranks) if d < 0)
    w = float(min(w_plus, w_minus))

    mean_w = n * (n + 1) / 4.0
    ties = _tie_correction_sum([abs(d) for d in nonzero])
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - ties / 48.0
    if variance <= 0:
        return w, 1.0

    shift = 0.5 if correction else 0.0
    z = (w - mean_w + shift) / math.sqrt(variance)
    return w, 2 * _rss.norm_cdf(z)
