"""Pandas compatibility layer for Real Simple Stats.

This module provides seamless integration with pandas DataFrames and Series,
allowing you to use Real Simple Stats functions directly on pandas objects.
"""

from typing import Any, Sequence, Union

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def _extract_values(data: Any) -> Sequence[float]:
    """Extract numeric values from various input types.

    Args:
        data: Input data (list, array, Series, DataFrame column, etc.)

    Returns:
        Sequence of float values

    Raises:
        ValueError: If data cannot be converted to numeric sequence
    """
    if PANDAS_AVAILABLE and isinstance(data, pd.Series):
        return data.dropna().tolist()
    elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        raise ValueError(
            "DataFrame passed. Please select a column first (e.g., df['column'])"
        )
    elif hasattr(data, "tolist"):
        # NumPy array
        return data.tolist()
    elif hasattr(data, "values"):
        # Something with .values attribute (like pandas Series)
        return data.values.tolist()
    else:
        # Assume it's already a list or sequence
        return list(data)


def _maybe_return_series(result: Any, original: Any) -> Any:
    """Return result as pandas Series if original was a Series.

    Args:
        result: The computed result
        original: The original input data

    Returns:
        Result, possibly as pandas Series
    """
    if (
        PANDAS_AVAILABLE
        and isinstance(original, pd.Series)
        and isinstance(result, (list, tuple))
    ):
        return pd.Series(result, index=original.index[: len(result)])
    return result


# Convenience functions that wrap common operations
def mean(data: Union[Sequence[float], "pd.Series"]) -> float:
    """Calculate mean, accepting pandas Series."""
    from real_simple_stats import descriptive_statistics as desc

    values = _extract_values(data)
    return desc.mean(values)


def median(data: Union[Sequence[float], "pd.Series"]) -> float:
    """Calculate median, accepting pandas Series."""
    from real_simple_stats import descriptive_statistics as desc

    values = _extract_values(data)
    return desc.median(values)


def standard_deviation(data: Union[Sequence[float], "pd.Series"]) -> float:
    """Calculate standard deviation, accepting pandas Series."""
    from real_simple_stats import descriptive_statistics as desc

    values = _extract_values(data)
    return desc.sample_std_dev(values)


def sample_std_dev(data: Union[Sequence[float], "pd.Series"]) -> float:
    """Calculate sample standard deviation, accepting pandas Series."""
    from real_simple_stats import descriptive_statistics as desc

    values = _extract_values(data)
    return desc.sample_std_dev(values)


def variance(data: Union[Sequence[float], "pd.Series"]) -> float:
    """Calculate variance, accepting pandas Series."""
    from real_simple_stats import descriptive_statistics as desc

    values = _extract_values(data)
    return desc.sample_variance(values)


def five_number_summary(data: Union[Sequence[float], "pd.Series"]) -> dict:
    """Calculate five-number summary, accepting pandas Series."""
    from real_simple_stats import descriptive_statistics as desc

    values = _extract_values(data)
    return desc.five_number_summary(values)


def one_sample_t_test(data: Union[Sequence[float], "pd.Series"], mu: float) -> tuple:
    """Perform one-sample t-test, accepting pandas Series.

    Note: This is a placeholder. real_simple_stats doesn't have a direct
    one_sample_t_test function. Use t_score and calculate p-value manually.
    """
    from scipy.stats import t as t_dist

    from real_simple_stats import descriptive_statistics as desc
    from real_simple_stats import hypothesis_testing as ht

    values = _extract_values(data)
    n = len(values)
    sample_mean = desc.mean(values)
    sample_std = desc.sample_std_dev(values)
    t_stat = ht.t_score(sample_mean, mu, sample_std, n)
    df = n - 1
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    return t_stat, p_value


def two_sample_t_test(
    data1: Union[Sequence[float], "pd.Series"],
    data2: Union[Sequence[float], "pd.Series"],
) -> tuple:
    """Perform two-sample t-test, accepting pandas Series.

    Note: This is a placeholder. real_simple_stats doesn't have a direct
    two_sample_t_test function. Use scipy.stats.ttest_ind for now.
    """
    from scipy.stats import ttest_ind

    values1 = _extract_values(data1)
    values2 = _extract_values(data2)
    t_stat, p_value = ttest_ind(values1, values2)
    return t_stat, p_value


def linear_regression(
    x: Union[Sequence[float], "pd.Series"], y: Union[Sequence[float], "pd.Series"]
) -> tuple:
    """Perform linear regression, accepting pandas Series."""
    from real_simple_stats import linear_regression_utils as lr

    x_values = _extract_values(x)
    y_values = _extract_values(y)
    return lr.linear_regression(x_values, y_values)
