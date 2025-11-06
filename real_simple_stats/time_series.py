"""Time series analysis functions.

This module provides functions for analyzing time series data including
moving averages, autocorrelation, and trend analysis.
"""

from typing import List, Tuple
import numpy as np
from scipy import stats


def moving_average(
    data: List[float], window_size: int, method: str = "simple"
) -> List[float]:
    """Calculate moving average of a time series.

    Args:
        data: Time series data
        window_size: Size of the moving window
        method: Type of moving average ('simple', 'exponential', 'weighted')

    Returns:
        List of moving average values

    Raises:
        ValueError: If window_size is invalid or method is unknown

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> moving_average(data, 3)
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    if window_size > len(data):
        raise ValueError("Window size cannot exceed data length")

    if method == "simple":
        return _simple_moving_average(data, window_size)
    elif method == "exponential":
        return _exponential_moving_average(data, window_size)
    elif method == "weighted":
        return _weighted_moving_average(data, window_size)
    else:
        raise ValueError(f"Unknown method: {method}")


def _simple_moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate simple moving average (SMA)."""
    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        result.append(sum(window) / window_size)
    return result


def _exponential_moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate exponential moving average (EMA)."""
    alpha = 2 / (window_size + 1)
    ema = [data[0]]

    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])

    return ema


def _weighted_moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate weighted moving average (WMA)."""
    weights = np.arange(1, window_size + 1)
    weights = weights / weights.sum()

    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        result.append(np.dot(window, weights))

    return result


def autocorrelation(data: List[float], max_lag: int = None) -> List[float]:
    """Calculate autocorrelation function (ACF).

    Args:
        data: Time series data
        max_lag: Maximum lag to calculate (default: len(data) - 1)

    Returns:
        List of autocorrelation coefficients for each lag

    Raises:
        ValueError: If data is too short or max_lag is invalid

    Examples:
        >>> data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        >>> acf = autocorrelation(data, max_lag=3)
        >>> len(acf)
        4
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values")

    if max_lag is None:
        max_lag = len(data) - 1
    elif max_lag < 0 or max_lag >= len(data):
        raise ValueError(f"max_lag must be between 0 and {len(data) - 1}")

    data_array = np.array(data)
    mean = np.mean(data_array)
    var = np.var(data_array)

    if var == 0:
        return [1.0] + [0.0] * max_lag

    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            numerator = np.sum((data_array[:-lag] - mean) * (data_array[lag:] - mean))
            denominator = len(data_array) * var
            acf.append(numerator / denominator)

    return acf


def partial_autocorrelation(data: List[float], max_lag: int = None) -> List[float]:
    """Calculate partial autocorrelation function (PACF).

    Args:
        data: Time series data
        max_lag: Maximum lag to calculate (default: min(len(data)//2, 10))

    Returns:
        List of partial autocorrelation coefficients

    Raises:
        ValueError: If data is too short

    Examples:
        >>> data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        >>> pacf = partial_autocorrelation(data, max_lag=3)
        >>> len(pacf)
        4
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values")

    if max_lag is None:
        max_lag = min(len(data) // 2, 10)

    acf_values = autocorrelation(data, max_lag)
    pacf = [1.0]  # PACF at lag 0 is always 1

    for k in range(1, max_lag + 1):
        if k == 1:
            pacf.append(acf_values[1])
        else:
            # Durbin-Levinson algorithm
            numerator = acf_values[k]
            for j in range(1, k):
                numerator -= pacf[j] * acf_values[k - j]

            denominator = 1.0
            for j in range(1, k):
                denominator -= pacf[j] * acf_values[j]

            pacf.append(numerator / denominator if denominator != 0 else 0.0)

    return pacf


def linear_trend(data: List[float]) -> Tuple[float, float, float]:
    """Fit a linear trend to time series data.

    Args:
        data: Time series data

    Returns:
        Tuple of (slope, intercept, r_squared)

    Raises:
        ValueError: If data is too short

    Examples:
        >>> data = [1, 2, 3, 4, 5]
        >>> slope, intercept, r2 = linear_trend(data)
        >>> round(slope, 2)
        1.0
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values")

    x = np.arange(len(data))
    y = np.array(data)

    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_squared = r_value**2

    return float(slope), float(intercept), float(r_squared)


def detrend(data: List[float], method: str = "linear") -> List[float]:
    """Remove trend from time series data.

    Args:
        data: Time series data
        method: Detrending method ('linear' or 'mean')

    Returns:
        Detrended data

    Raises:
        ValueError: If method is unknown or data is too short

    Examples:
        >>> data = [1, 2, 3, 4, 5]
        >>> detrended = detrend(data, method='linear')
        >>> len(detrended)
        5
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values")

    if method == "linear":
        slope, intercept, _ = linear_trend(data)
        x = np.arange(len(data))
        trend = slope * x + intercept
        return (np.array(data) - trend).tolist()
    elif method == "mean":
        mean = np.mean(data)
        return (np.array(data) - mean).tolist()
    else:
        raise ValueError(f"Unknown method: {method}")


def seasonal_decompose(
    data: List[float], period: int
) -> Tuple[List[float], List[float], List[float]]:
    """Decompose time series into trend, seasonal, and residual components.

    Args:
        data: Time series data
        period: Length of seasonal cycle

    Returns:
        Tuple of (trend, seasonal, residual) components

    Raises:
        ValueError: If period is invalid or data is too short

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 2
        >>> trend, seasonal, residual = seasonal_decompose(data, period=12)
        >>> len(trend) == len(data)
        True
    """
    if period < 2:
        raise ValueError("Period must be at least 2")
    if len(data) < 2 * period:
        raise ValueError(f"Data must contain at least {2 * period} values")

    # Calculate trend using centered moving average
    trend = []
    half_period = period // 2

    for i in range(len(data)):
        if i < half_period or i >= len(data) - half_period:
            trend.append(np.nan)
        else:
            window = data[i - half_period : i + half_period + 1]
            trend.append(np.mean(window))

    # Calculate seasonal component
    detrended = np.array(data) - np.array(trend)
    seasonal_avg = np.zeros(period)

    for i in range(period):
        season_values = [
            detrended[j]
            for j in range(i, len(data), period)
            if not np.isnan(detrended[j])
        ]
        if season_values:
            seasonal_avg[i] = np.mean(season_values)

    # Center seasonal component
    seasonal_avg -= np.mean(seasonal_avg)

    # Repeat seasonal pattern
    seasonal = [seasonal_avg[i % period] for i in range(len(data))]

    # Calculate residual
    residual = (np.array(data) - np.array(trend) - np.array(seasonal)).tolist()

    return trend, seasonal, residual


def difference(data: List[float], lag: int = 1, order: int = 1) -> List[float]:
    """Calculate differenced time series.

    Args:
        data: Time series data
        lag: Lag for differencing
        order: Number of times to difference

    Returns:
        Differenced time series

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> data = [1, 2, 4, 7, 11]
        >>> difference(data, lag=1, order=1)
        [1, 2, 3, 4]
    """
    if lag < 1:
        raise ValueError("Lag must be at least 1")
    if order < 1:
        raise ValueError("Order must be at least 1")
    if len(data) <= lag:
        raise ValueError("Data length must exceed lag")

    result = data.copy()

    for _ in range(order):
        if len(result) <= lag:
            raise ValueError("Insufficient data for differencing order")
        result = [result[i] - result[i - lag] for i in range(lag, len(result))]

    return result


__all__ = [
    "moving_average",
    "autocorrelation",
    "partial_autocorrelation",
    "linear_trend",
    "detrend",
    "seasonal_decompose",
    "difference",
]
