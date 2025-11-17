"""Statistical power analysis and sample size calculations.

This module provides functions for calculating statistical power and
required sample sizes for various statistical tests.

Refactored for Pythonic elegance and maintainability.
"""

from typing import Optional, Dict, Callable
from functools import lru_cache
import numpy as np
from scipy import stats, optimize

# Module-level constants
VALID_ALTERNATIVES = {"two-sided", "greater", "less"}
VALID_TEST_TYPES = {"t-test", "proportion", "correlation"}


def _get_tails(alternative: str) -> int:
    """Convert alternative hypothesis to number of tails.

    Args:
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        2 for two-sided, 1 for one-sided
    """
    return 2 if alternative == "two-sided" else 1


def _get_alpha_adjusted(sig_level: float, tails: int) -> float:
    """Get adjusted alpha for one or two-tailed test.

    Args:
        sig_level: Significance level
        tails: Number of tails (1 or 2)

    Returns:
        Adjusted significance level
    """
    return sig_level / 2 if tails == 2 else sig_level


@lru_cache(maxsize=256)
def _cached_norm_ppf(alpha: float) -> float:
    """Cache normal distribution critical values.

    Args:
        alpha: Significance level

    Returns:
        Critical value from standard normal distribution
    """
    return stats.norm.ppf(1 - alpha)


@lru_cache(maxsize=512)
def _cached_t_ppf(alpha: float, df: int) -> float:
    """Cache t distribution critical values.

    Args:
        alpha: Significance level
        df: Degrees of freedom

    Returns:
        Critical value from t distribution
    """
    return stats.t.ppf(1 - alpha, df=df)


@lru_cache(maxsize=512)
def _cached_f_ppf(alpha: float, dfn: int, dfd: int) -> float:
    """Cache F distribution critical values.

    Args:
        alpha: Significance level
        dfn: Numerator degrees of freedom
        dfd: Denominator degrees of freedom

    Returns:
        Critical value from F distribution
    """
    return stats.f.ppf(1 - alpha, dfn, dfd)


def _validate_alternative(alternative: str) -> None:
    """Validate alternative hypothesis parameter.

    Args:
        alternative: Alternative hypothesis type

    Raises:
        ValueError: If alternative is not valid
    """
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}")


def _validate_none_count(params: Dict[str, Optional[float]], expected: int = 1) -> None:
    """Validate that exactly the expected number of parameters are None.

    Args:
        params: Dictionary of parameter names to values
        expected: Expected number of None values

    Raises:
        ValueError: If wrong number of parameters are None
    """
    none_count = sum(v is None for v in params.values())
    if none_count != expected:
        param_names = ", ".join(params.keys())
        raise ValueError(f"Exactly {expected} of {param_names} must be None")


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
    # Validation
    _validate_none_count({"n": n, "delta": delta, "power": power})
    _validate_alternative(alternative)

    tails = _get_tails(alternative)
    base_params = {
        "sd": sd,
        "sig_level": sig_level,
        "alternative": alternative,
    }

    # Dispatch to appropriate calculator
    if n is None:
        return _calculate_t_test_n(delta, power, tails, base_params)
    elif delta is None:
        return _calculate_t_test_delta(n, power, tails, base_params)
    else:  # power is None
        return _calculate_t_test_power(n, delta, tails, base_params)


def _calculate_t_test_n(
    delta: float, power: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate required sample size for t-test."""
    if delta == 0:
        raise ValueError("delta cannot be zero when calculating sample size")

    effect_size = abs(delta) / base_params["sd"]
    alpha_adj = _get_alpha_adjusted(base_params["sig_level"], tails)

    z_alpha = _cached_norm_ppf(alpha_adj)
    z_beta = _cached_norm_ppf(1 - power)

    n_calculated = int(np.ceil(((z_alpha + z_beta) ** 2) / (effect_size**2)))

    return {
        "n": n_calculated,
        "delta": delta,
        "power": power,
        **base_params,
    }


def _calculate_t_test_delta(
    n: int, power: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate detectable effect size for t-test."""
    alpha_adj = _get_alpha_adjusted(base_params["sig_level"], tails)
    t_crit = _cached_t_ppf(alpha_adj, df=n - 1)
    t_beta = _cached_t_ppf(1 - power, df=n - 1)

    effect_size = (t_crit + abs(t_beta)) / np.sqrt(n)
    delta_calculated = effect_size * base_params["sd"]

    sign = -1.0 if base_params["alternative"] == "less" else 1.0

    return {
        "n": n,
        "delta": float(sign * delta_calculated),
        "power": power,
        **base_params,
    }


def _calculate_t_test_power(
    n: int, delta: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate statistical power for t-test."""
    effect_size = delta / base_params["sd"]
    ncp = effect_size * np.sqrt(n)
    df = n - 1
    sig_level = base_params["sig_level"]
    alternative = base_params["alternative"]

    if alternative == "greater":
        t_crit = stats.t.ppf(1 - sig_level, df=df)
        power_calculated = 1 - stats.nct.cdf(t_crit, df=df, nc=ncp)
    elif alternative == "less":
        t_crit = stats.t.ppf(sig_level, df=df)
        power_calculated = stats.nct.cdf(t_crit, df=df, nc=ncp)
    else:
        t_crit = stats.t.ppf(1 - sig_level / 2, df=df)
        power_calculated = (1 - stats.nct.cdf(t_crit, df=df, nc=ncp)) + stats.nct.cdf(
            -t_crit, df=df, nc=ncp
        )

    return {
        "n": n,
        "delta": delta,
        "power": float(power_calculated),
        **base_params,
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
    # Validation
    _validate_none_count({"n": n, "p1": p1, "power": power})
    _validate_alternative(alternative)

    if not 0 < p2 < 1:
        raise ValueError("p2 must be between 0 and 1")

    tails = _get_tails(alternative)
    base_params = {
        "p2": p2,
        "sig_level": sig_level,
        "alternative": alternative,
    }

    # Dispatch to appropriate calculator
    if n is None:
        return _calculate_proportion_n(p1, power, tails, base_params)
    elif p1 is None:
        return _calculate_proportion_p1(n, power, tails, base_params)
    else:  # power is None
        return _calculate_proportion_power(n, p1, tails, base_params)


def _calculate_proportion_n(
    p1: float, power: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate required sample size for proportion test."""
    if p1 is None or p1 == base_params["p2"]:
        raise ValueError("p1 must be different from p2")

    # Cohen's h effect size
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(base_params["p2"])))

    alpha_adj = _get_alpha_adjusted(base_params["sig_level"], tails)
    z_alpha = stats.norm.ppf(1 - alpha_adj)
    z_beta = stats.norm.ppf(power)

    n_calculated = int(np.ceil(((z_alpha + z_beta) / h) ** 2))

    return {
        "n": n_calculated,
        "p1": p1,
        "power": power,
        **base_params,
    }


def _calculate_proportion_p1(
    n: int, power: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate detectable proportion difference."""
    alpha_adj = _get_alpha_adjusted(base_params["sig_level"], tails)
    z_alpha = stats.norm.ppf(1 - alpha_adj)
    z_beta = stats.norm.ppf(power)

    h = (z_alpha + z_beta) / np.sqrt(n)
    if base_params["alternative"] == "less":
        h *= -1
    phi1 = np.arcsin(np.sqrt(base_params["p2"])) + h / 2
    p1_calculated = np.sin(phi1) ** 2

    return {
        "n": n,
        "p1": float(p1_calculated),
        "power": power,
        **base_params,
    }


def _calculate_proportion_power(
    n: int, p1: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate statistical power for proportion test."""
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(base_params["p2"])))
    mean_shift = h * np.sqrt(n)
    sig_level = base_params["sig_level"]
    alternative = base_params["alternative"]

    if alternative == "greater":
        z_alpha = stats.norm.ppf(1 - sig_level)
        power_calculated = stats.norm.sf(z_alpha - mean_shift)
    elif alternative == "less":
        z_alpha = stats.norm.ppf(sig_level)
        power_calculated = stats.norm.cdf(z_alpha - mean_shift)
    else:
        z_alpha = stats.norm.ppf(1 - sig_level / 2)
        power_calculated = stats.norm.sf(z_alpha - mean_shift) + stats.norm.cdf(
            -z_alpha - mean_shift
        )

    return {
        "n": n,
        "p1": p1,
        "power": float(power_calculated),
        **base_params,
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

    _validate_none_count(
        {"n_per_group": n_per_group, "effect_size": effect_size, "power": power}
    )

    base_params = {
        "n_groups": n_groups,
        "sig_level": sig_level,
    }

    # Dispatch to appropriate calculator
    if n_per_group is None:
        return _calculate_anova_n(effect_size, power, base_params)
    elif effect_size is None:
        return _calculate_anova_effect(n_per_group, power, base_params)
    else:  # power is None
        return _calculate_anova_power(n_per_group, effect_size, base_params)


def _calculate_anova_n(
    effect_size: float, power: float, base_params: Dict
) -> Dict[str, float]:
    """Calculate required sample size per group for ANOVA."""
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")

    n_groups = base_params["n_groups"]
    sig_level = base_params["sig_level"]

    def power_func(n):
        df1 = n_groups - 1
        df2 = n_groups * (n - 1)
        ncp = n * n_groups * (effect_size**2)
        f_crit = stats.f.ppf(1 - sig_level, df1, df2)
        return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

    n_calculated = int(optimize.brentq(lambda n: power_func(n) - power, 2, 10000))

    return {
        "n_per_group": n_calculated,
        "effect_size": effect_size,
        "power": power,
        **base_params,
    }


def _calculate_anova_effect(
    n_per_group: int, power: float, base_params: Dict
) -> Dict[str, float]:
    """Calculate detectable effect size for ANOVA."""
    n_groups = base_params["n_groups"]
    sig_level = base_params["sig_level"]

    df1 = n_groups - 1
    df2 = n_groups * (n_per_group - 1)
    f_crit = stats.f.ppf(1 - sig_level, df1, df2)

    def power_func(es):
        ncp = n_per_group * n_groups * (es**2)
        return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

    effect_size_calculated = optimize.brentq(
        lambda es: power_func(es) - power, 0.01, 10
    )

    return {
        "n_per_group": n_per_group,
        "effect_size": float(effect_size_calculated),
        "power": power,
        **base_params,
    }


def _calculate_anova_power(
    n_per_group: int, effect_size: float, base_params: Dict
) -> Dict[str, float]:
    """Calculate statistical power for ANOVA."""
    n_groups = base_params["n_groups"]
    sig_level = base_params["sig_level"]

    df1 = n_groups - 1
    df2 = n_groups * (n_per_group - 1)
    ncp = n_per_group * n_groups * (effect_size**2)
    f_crit = stats.f.ppf(1 - sig_level, df1, df2)

    power_calculated = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

    return {
        "n_per_group": n_per_group,
        "effect_size": effect_size,
        "power": float(power_calculated),
        **base_params,
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
    _validate_none_count({"n": n, "r": r, "power": power})
    _validate_alternative(alternative)

    tails = _get_tails(alternative)
    base_params = {
        "sig_level": sig_level,
        "alternative": alternative,
    }

    # Dispatch to appropriate calculator
    if n is None:
        return _calculate_correlation_n(r, power, tails, base_params)
    elif r is None:
        return _calculate_correlation_r(n, power, tails, base_params)
    else:  # power is None
        return _calculate_correlation_power(n, r, tails, base_params)


def _calculate_correlation_n(
    r: float, power: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate required sample size for correlation test."""
    if abs(r) >= 1:
        raise ValueError("r must be between -1 and 1")

    z_r = 0.5 * np.log((1 + r) / (1 - r))  # Fisher's z transformation

    alpha_adj = _get_alpha_adjusted(base_params["sig_level"], tails)
    z_alpha = stats.norm.ppf(1 - alpha_adj)
    z_beta = stats.norm.ppf(power)

    n_calculated = int(np.ceil(((z_alpha + z_beta) / z_r) ** 2 + 3))

    return {
        "n": n_calculated,
        "r": r,
        "power": power,
        **base_params,
    }


def _calculate_correlation_r(
    n: int, power: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate detectable correlation coefficient."""
    alpha_adj = _get_alpha_adjusted(base_params["sig_level"], tails)
    z_alpha = stats.norm.ppf(1 - alpha_adj)
    z_beta = stats.norm.ppf(power)

    z_r = (z_alpha + z_beta) / np.sqrt(n - 3)
    r_calculated = (np.exp(2 * z_r) - 1) / (np.exp(2 * z_r) + 1)

    sign = -1.0 if base_params["alternative"] == "less" else 1.0

    return {
        "n": n,
        "r": float(sign * r_calculated),
        "power": power,
        **base_params,
    }


def _calculate_correlation_power(
    n: int, r: float, tails: int, base_params: Dict
) -> Dict[str, float]:
    """Calculate statistical power for correlation test."""
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    sig_level = base_params["sig_level"]
    alternative = base_params["alternative"]

    if alternative == "greater":
        z_alpha = stats.norm.ppf(1 - sig_level)
        power_calculated = stats.norm.sf(z_alpha - z_r * np.sqrt(n - 3))
    elif alternative == "less":
        z_alpha = stats.norm.ppf(sig_level)
        power_calculated = stats.norm.cdf(z_alpha - z_r * np.sqrt(n - 3))
    else:
        z_alpha = stats.norm.ppf(1 - sig_level / 2)
        mean_shift = z_r * np.sqrt(n - 3)
        power_calculated = stats.norm.sf(z_alpha - mean_shift) + stats.norm.cdf(
            -z_alpha - mean_shift
        )

    return {
        "n": n,
        "r": r,
        "power": float(power_calculated),
        **base_params,
    }


# Dispatch dictionary for minimum detectable effect calculators
def _mde_t_test(n: int, power: float, sig_level: float) -> float:
    """Calculate MDE for t-test."""
    result = power_t_test(n=n, power=power, sig_level=sig_level)
    return result["delta"]


def _mde_proportion(n: int, power: float, sig_level: float) -> float:
    """Calculate MDE for proportion test."""
    result = power_proportion_test(n=n, p2=0.5, power=power, sig_level=sig_level)
    return abs(result["p1"] - result["p2"])


def _mde_correlation(n: int, power: float, sig_level: float) -> float:
    """Calculate MDE for correlation test."""
    result = power_correlation(n=n, power=power, sig_level=sig_level)
    return abs(result["r"])


MDE_CALCULATORS: Dict[str, Callable[[int, float, float], float]] = {
    "t-test": _mde_t_test,
    "proportion": _mde_proportion,
    "correlation": _mde_correlation,
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

    if test_type not in MDE_CALCULATORS:
        raise ValueError(
            f"Unknown test_type: {test_type}. "
            f"Valid types: {set(MDE_CALCULATORS.keys())}"
        )

    return MDE_CALCULATORS[test_type](n, power, sig_level)


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
        corr_result = power_correlation(r=effect_size, power=power, sig_level=sig_level)
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
