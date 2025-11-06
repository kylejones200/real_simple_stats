"""Statistical power analysis and sample size calculations.

This module provides functions for calculating statistical power and
required sample sizes for various statistical tests.
"""

from typing import Optional, Dict
import numpy as np
from scipy import stats, optimize


def power_t_test(
    n: Optional[int] = None,
    delta: Optional[float] = None,
    sd: float = 1.0,
    sig_level: float = 0.05,
    power: Optional[float] = None,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Calculate power or sample size for t-test.

    Provide exactly 3 of: n, delta, power. The 4th will be calculated.

    Args:
        n: Sample size per group
        delta: True difference in means
        sd: Standard deviation (default: 1.0)
        sig_level: Significance level (default: 0.05)
        power: Statistical power (1 - beta)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Dictionary with all parameters including the calculated one

    Raises:
        ValueError: If wrong number of parameters provided

    Examples:
        >>> # Calculate required sample size
        >>> result = power_t_test(delta=0.5, power=0.8)
        >>> result['n'] > 0
        True
    """
    # Count how many parameters are None
    none_count = sum([n is None, delta is None, power is None])

    if none_count != 1:
        raise ValueError("Exactly one of n, delta, or power must be None")

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alternative == "two-sided":
        tails = 2
    else:
        tails = 1

    if n is None:
        # Calculate required sample size
        if delta == 0:
            raise ValueError("delta cannot be zero when calculating sample size")

        effect_size = abs(delta) / sd

        # Critical value
        alpha_adj = sig_level / 2 if tails == 2 else sig_level
        z_alpha = stats.norm.ppf(1 - alpha_adj)

        z_beta = stats.norm.ppf(power)

        # Approximate sample size formula
        n_calculated = ((z_alpha + z_beta) ** 2) / (effect_size**2)
        n_calculated = int(np.ceil(n_calculated))

        return {
            "n": n_calculated,
            "delta": delta,
            "sd": sd,
            "sig_level": sig_level,
            "power": power,
            "alternative": alternative,
        }

    elif delta is None:
        # Calculate detectable effect size
        alpha_adj = sig_level / 2 if tails == 2 else sig_level
        t_crit = stats.t.ppf(1 - alpha_adj, df=n - 1)

        t_beta = stats.t.ppf(power, df=n - 1)

        effect_size = (t_crit + abs(t_beta)) / np.sqrt(n)
        delta_calculated = effect_size * sd

        return {
            "n": n,
            "delta": delta_calculated,
            "sd": sd,
            "sig_level": sig_level,
            "power": power,
            "alternative": alternative,
        }

    else:  # power is None
        # Calculate power
        effect_size = abs(delta) / sd
        ncp = effect_size * np.sqrt(n)  # Non-centrality parameter

        alpha_adj = sig_level / 2 if tails == 2 else sig_level
        t_crit = stats.t.ppf(1 - alpha_adj, df=n - 1)

        # Power is probability of rejecting null when alternative is true
        power_calculated = 1 - stats.nct.cdf(t_crit, df=n - 1, nc=ncp)

        if tails == 2:
            # Add lower tail for two-sided test
            power_calculated += stats.nct.cdf(-t_crit, df=n - 1, nc=ncp)

        return {
            "n": n,
            "delta": delta,
            "sd": sd,
            "sig_level": sig_level,
            "power": float(power_calculated),
            "alternative": alternative,
        }


def power_proportion_test(
    n: Optional[int] = None,
    p1: Optional[float] = None,
    p2: float = 0.5,
    sig_level: float = 0.05,
    power: Optional[float] = None,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Calculate power or sample size for proportion test.

    Args:
        n: Sample size per group
        p1: Proportion in group 1
        p2: Proportion in group 2 (default: 0.5)
        sig_level: Significance level (default: 0.05)
        power: Statistical power
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Dictionary with all parameters

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> result = power_proportion_test(p1=0.6, p2=0.5, power=0.8)
        >>> result['n'] > 0
        True
    """
    none_count = sum([n is None, p1 is None, power is None])

    if none_count != 1:
        raise ValueError("Exactly one of n, p1, or power must be None")

    if not 0 < p2 < 1:
        raise ValueError("p2 must be between 0 and 1")

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alternative == "two-sided":
        tails = 2
    else:
        tails = 1

    if n is None:
        # Calculate required sample size
        if p1 is None or p1 == p2:
            raise ValueError("p1 must be different from p2")

        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        if tails == 2:
            z_alpha = stats.norm.ppf(1 - sig_level / 2)
        else:
            z_alpha = stats.norm.ppf(1 - sig_level)

        z_beta = stats.norm.ppf(power)

        n_calculated = ((z_alpha + z_beta) / h) ** 2
        n_calculated = int(np.ceil(n_calculated))

        return {
            "n": n_calculated,
            "p1": p1,
            "p2": p2,
            "sig_level": sig_level,
            "power": power,
            "alternative": alternative,
        }

    elif p1 is None:
        # Calculate detectable proportion difference
        if tails == 2:
            z_alpha = stats.norm.ppf(1 - sig_level / 2)
        else:
            z_alpha = stats.norm.ppf(1 - sig_level)

        z_beta = stats.norm.ppf(power)

        h = (z_alpha + z_beta) / np.sqrt(n)
        phi1 = np.arcsin(np.sqrt(p2)) + h / 2
        p1_calculated = np.sin(phi1) ** 2

        return {
            "n": n,
            "p1": float(p1_calculated),
            "p2": p2,
            "sig_level": sig_level,
            "power": power,
            "alternative": alternative,
        }

    else:  # power is None
        # Calculate power
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        if tails == 2:
            z_alpha = stats.norm.ppf(1 - sig_level / 2)
        else:
            z_alpha = stats.norm.ppf(1 - sig_level)

        z_beta = abs(h) * np.sqrt(n) - z_alpha
        power_calculated = stats.norm.cdf(z_beta)

        return {
            "n": n,
            "p1": p1,
            "p2": p2,
            "sig_level": sig_level,
            "power": float(power_calculated),
            "alternative": alternative,
        }


def power_anova(
    n_groups: int,
    n_per_group: Optional[int] = None,
    effect_size: Optional[float] = None,
    sig_level: float = 0.05,
    power: Optional[float] = None,
) -> Dict[str, float]:
    """Calculate power or sample size for one-way ANOVA.

    Args:
        n_groups: Number of groups
        n_per_group: Sample size per group
        effect_size: Effect size (f)
        sig_level: Significance level (default: 0.05)
        power: Statistical power

    Returns:
        Dictionary with all parameters

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> result = power_anova(n_groups=3, effect_size=0.25, power=0.8)
        >>> result['n_per_group'] > 0
        True
    """
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")

    none_count = sum([n_per_group is None, effect_size is None, power is None])

    if none_count != 1:
        raise ValueError("Exactly one of n_per_group, effect_size, or power must be None")

    if n_per_group is None:
        # Calculate required sample size per group
        if effect_size <= 0:
            raise ValueError("effect_size must be positive")

        # Use iterative approach
        def power_func(n):
            df1 = n_groups - 1
            df2 = n_groups * (n - 1)
            ncp = n * n_groups * (effect_size**2)
            f_crit = stats.f.ppf(1 - sig_level, df1, df2)
            return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

        # Find n that gives desired power
        n_calculated = int(
            optimize.brentq(lambda n: power_func(n) - power, 2, 10000)
        )

        return {
            "n_groups": n_groups,
            "n_per_group": n_calculated,
            "effect_size": effect_size,
            "sig_level": sig_level,
            "power": power,
        }

    elif effect_size is None:
        # Calculate detectable effect size
        df1 = n_groups - 1
        df2 = n_groups * (n_per_group - 1)
        f_crit = stats.f.ppf(1 - sig_level, df1, df2)

        # Use iterative approach
        def power_func(es):
            ncp = n_per_group * n_groups * (es**2)
            return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

        effect_size_calculated = optimize.brentq(
            lambda es: power_func(es) - power, 0.01, 10
        )

        return {
            "n_groups": n_groups,
            "n_per_group": n_per_group,
            "effect_size": float(effect_size_calculated),
            "sig_level": sig_level,
            "power": power,
        }

    else:  # power is None
        # Calculate power
        df1 = n_groups - 1
        df2 = n_groups * (n_per_group - 1)
        ncp = n_per_group * n_groups * (effect_size**2)
        f_crit = stats.f.ppf(1 - sig_level, df1, df2)

        power_calculated = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

        return {
            "n_groups": n_groups,
            "n_per_group": n_per_group,
            "effect_size": effect_size,
            "sig_level": sig_level,
            "power": float(power_calculated),
        }


def power_correlation(
    n: Optional[int] = None,
    r: Optional[float] = None,
    sig_level: float = 0.05,
    power: Optional[float] = None,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Calculate power or sample size for correlation test.

    Args:
        n: Sample size
        r: Population correlation coefficient
        sig_level: Significance level (default: 0.05)
        power: Statistical power
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Dictionary with all parameters

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> result = power_correlation(r=0.3, power=0.8)
        >>> result['n'] > 0
        True
    """
    none_count = sum([n is None, r is None, power is None])

    if none_count != 1:
        raise ValueError("Exactly one of n, r, or power must be None")

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if alternative == "two-sided":
        tails = 2
    else:
        tails = 1

    if n is None:
        # Calculate required sample size
        if abs(r) >= 1:
            raise ValueError("r must be between -1 and 1")

        # Fisher's z transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r))

        if tails == 2:
            z_alpha = stats.norm.ppf(1 - sig_level / 2)
        else:
            z_alpha = stats.norm.ppf(1 - sig_level)

        z_beta = stats.norm.ppf(power)

        n_calculated = ((z_alpha + z_beta) / z_r) ** 2 + 3
        n_calculated = int(np.ceil(n_calculated))

        return {
            "n": n_calculated,
            "r": r,
            "sig_level": sig_level,
            "power": power,
            "alternative": alternative,
        }

    elif r is None:
        # Calculate detectable correlation
        if tails == 2:
            z_alpha = stats.norm.ppf(1 - sig_level / 2)
        else:
            z_alpha = stats.norm.ppf(1 - sig_level)

        z_beta = stats.norm.ppf(power)

        z_r = (z_alpha + z_beta) / np.sqrt(n - 3)
        r_calculated = (np.exp(2 * z_r) - 1) / (np.exp(2 * z_r) + 1)

        return {
            "n": n,
            "r": float(r_calculated),
            "sig_level": sig_level,
            "power": power,
            "alternative": alternative,
        }

    else:  # power is None
        # Calculate power
        z_r = 0.5 * np.log((1 + r) / (1 - r))

        if tails == 2:
            z_alpha = stats.norm.ppf(1 - sig_level / 2)
        else:
            z_alpha = stats.norm.ppf(1 - sig_level)

        z_beta = z_r * np.sqrt(n - 3) - z_alpha
        power_calculated = stats.norm.cdf(z_beta)

        return {
            "n": n,
            "r": r,
            "sig_level": sig_level,
            "power": float(power_calculated),
            "alternative": alternative,
        }


def minimum_detectable_effect(
    n: int, sig_level: float = 0.05, power: float = 0.8, test_type: str = "t-test"
) -> float:
    """Calculate minimum detectable effect size.

    Args:
        n: Sample size (per group for t-test)
        sig_level: Significance level (default: 0.05)
        power: Statistical power (default: 0.8)
        test_type: Type of test ('t-test', 'proportion', 'correlation')

    Returns:
        Minimum detectable effect size

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> mde = minimum_detectable_effect(50, test_type='t-test')
        >>> mde > 0
        True
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0 < sig_level < 1:
        raise ValueError("sig_level must be between 0 and 1")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")

    if test_type == "t-test":
        result = power_t_test(n=n, power=power, sig_level=sig_level)
        return result["delta"]

    elif test_type == "proportion":
        result = power_proportion_test(n=n, p2=0.5, power=power, sig_level=sig_level)
        return abs(result["p1"] - result["p2"])

    elif test_type == "correlation":
        result = power_correlation(n=n, power=power, sig_level=sig_level)
        return abs(result["r"])

    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def sample_size_summary(
    effect_size: float, power: float = 0.8, sig_level: float = 0.05
) -> Dict[str, int]:
    """Get sample size requirements for multiple test types.

    Args:
        effect_size: Standardized effect size
        power: Statistical power (default: 0.8)
        sig_level: Significance level (default: 0.05)

    Returns:
        Dictionary with sample sizes for different tests

    Examples:
        >>> summary = sample_size_summary(0.5, power=0.8)
        >>> all(n > 0 for n in summary.values())
        True
    """
    results = {}

    # T-test
    t_result = power_t_test(delta=effect_size, power=power, sig_level=sig_level)
    results["t_test_per_group"] = t_result["n"]

    # ANOVA (3 groups)
    anova_result = power_anova(
        n_groups=3, effect_size=effect_size, power=power, sig_level=sig_level
    )
    results["anova_3groups_per_group"] = anova_result["n_per_group"]

    # Correlation (if effect size is reasonable for correlation)
    if abs(effect_size) < 1:
        corr_result = power_correlation(
            r=effect_size, power=power, sig_level=sig_level
        )
        results["correlation_total"] = corr_result["n"]

    return results


__all__ = [
    "power_t_test",
    "power_proportion_test",
    "power_anova",
    "power_correlation",
    "minimum_detectable_effect",
    "sample_size_summary",
]
