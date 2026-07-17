"""Time series analysis functions.

This module provides functions for analyzing time series data including
moving averages, autocorrelation, and trend analysis.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats


def moving_average(
    data: list[float], window_size: int, method: str = "simple"
) -> list[float]:
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


def _simple_moving_average(data: list[float], window_size: int) -> list[float]:
    """Calculate simple moving average (SMA)."""
    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        result.append(sum(window) / window_size)
    return result


def _exponential_moving_average(data: list[float], window_size: int) -> list[float]:
    """Calculate exponential moving average (EMA)."""
    alpha = 2 / (window_size + 1)
    ema = [data[0]]

    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])

    return ema


def _weighted_moving_average(data: list[float], window_size: int) -> list[float]:
    """Calculate weighted moving average (WMA)."""
    weights = np.arange(1, window_size + 1)
    weights = weights / weights.sum()

    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        result.append(np.dot(window, weights))

    return result


def autocorrelation(data: list[float], max_lag: int = None) -> list[float]:
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


def partial_autocorrelation(data: list[float], max_lag: int = None) -> list[float]:
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


def linear_trend(data: list[float]) -> tuple[float, float, float]:
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


def detrend(data: list[float], method: str = "linear") -> list[float]:
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
    data: list[float], period: int
) -> tuple[list[float], list[float], list[float]]:
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


def difference(data: list[float], lag: int = 1, order: int = 1) -> list[float]:
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


def mean_absolute_scaled_error(
    actual: list[float],
    forecast: list[float],
) -> float:
    """Compute the Mean Absolute Scaled Error (MASE).

    Scales forecast errors against a naïve one-step-ahead forecast (predict
    today = yesterday).  MASE < 1 means the model beats the naïve baseline;
    MASE > 1 means it doesn't.

    MASE is scale-independent and works on a single series — unlike percentage
    errors it handles zero values, and unlike MSE it doesn't overweight outliers.

    Args:
        actual: Array of true values (length n).
        forecast: Array of predicted values (length n).

    Returns:
        MASE value (float).  Lower is better; 1.0 = naïve baseline.

    Raises:
        ValueError: If arrays have different lengths or fewer than 2 elements.

    Example:
        >>> actual   = [10, 12, 14, 16, 18]
        >>> forecast = [10, 12, 14, 16, 18]   # perfect forecast
        >>> mean_absolute_scaled_error(actual, forecast)
        0.0
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if len(a) != len(f):
        raise ValueError("actual and forecast must have the same length.")
    if len(a) < 2:
        raise ValueError("Need at least 2 observations.")

    mae_forecast = float(np.mean(np.abs(a - f)))
    mae_naive = float(np.mean(np.abs(np.diff(a))))
    if mae_naive == 0:
        return 0.0 if mae_forecast == 0 else float("inf")
    return mae_forecast / mae_naive


def exponential_smoothing(
    data: list[float],
    alpha: float,
) -> list[float]:
    """Simple Exponential Smoothing (SES) — level only.

    Each smoothed value is a weighted average of the current observation and
    the previous smoothed value:

        s_t = α · x_t + (1 − α) · s_{t-1}

    Small α gives a smoother series (longer memory); α=1 returns the raw data.

    Args:
        data: Time series values.
        alpha: Smoothing factor in (0, 1].

    Returns:
        List of smoothed values (same length as data).

    Example:
        >>> s = exponential_smoothing([1, 3, 5, 7, 9], alpha=0.3)
        >>> len(s) == 5 and s[0] == 1.0
        True
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    if len(data) == 0:
        return []

    result = [float(data[0])]
    for x in data[1:]:
        result.append(alpha * float(x) + (1.0 - alpha) * result[-1])
    return result


def double_exponential_smoothing(
    data: list[float],
    alpha: float,
    beta: float,
) -> dict[str, list[float]]:
    """Holt's Double Exponential Smoothing — level + trend.

    Extends SES with a trend component so the method can follow a linear
    trajectory.  The level and trend are updated together:

        l_t = α · x_t + (1 − α) · (l_{t-1} + b_{t-1})
        b_t = β · (l_t − l_{t-1}) + (1 − β) · b_{t-1}
        ŷ_{t+h} = l_t + h · b_t

    Args:
        data: Time series values (length ≥ 2).
        alpha: Level smoothing factor in (0, 1).
        beta: Trend smoothing factor in (0, 1).

    Returns:
        dict with keys:
            smoothed: In-sample fitted values.
            level: Level component at each time step.
            trend: Trend component at each time step.

    Example:
        >>> r = double_exponential_smoothing([1, 3, 5, 7, 9], alpha=0.8, beta=0.2)
        >>> len(r["smoothed"]) == 5
        True
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1).")
    if len(data) < 2:
        raise ValueError("Need at least 2 observations.")

    n = len(data)
    level = [float(data[0])]
    trend = [float(data[1]) - float(data[0])]
    smoothed = [level[0]]

    for i in range(1, n):
        x = float(data[i])
        l_prev, b_prev = level[-1], trend[-1]
        lvl = alpha * x + (1.0 - alpha) * (l_prev + b_prev)
        b = beta * (lvl - l_prev) + (1.0 - beta) * b_prev
        level.append(lvl)
        trend.append(b)
        smoothed.append(lvl)

    return {"smoothed": smoothed, "level": level, "trend": trend}


def rolling_statistics(
    data: list[float],
    window: int,
) -> dict[str, list[float]]:
    """Compute rolling window statistics for a time series.

    Each output value at position t uses observations in [t-window+1, t]
    (past values only — no future leakage).  Positions before the first
    full window use all available data (``min_periods=1`` behaviour).

    Args:
        data: Time series values.
        window: Rolling window size (number of periods).

    Returns:
        dict with keys:
            mean: Rolling mean.
            std: Rolling standard deviation (ddof=1, or 0 for single-element windows).
            minimum: Rolling minimum.
            maximum: Rolling maximum.
            expanding_mean: Expanding (cumulative) mean.

    Raises:
        ValueError: If window < 1 or data is empty.

    Example:
        >>> r = rolling_statistics([1, 2, 3, 4, 5], window=3)
        >>> r["mean"]
        [1.0, 1.5, 2.0, 3.0, 4.0]
    """
    if window < 1:
        raise ValueError("window must be at least 1.")
    if len(data) == 0:
        raise ValueError("data must not be empty.")

    x = np.asarray(data, dtype=float)
    n = len(x)
    roll_mean, roll_std, roll_min, roll_max, exp_mean = [], [], [], [], []

    running_sum = 0.0
    for i in range(n):
        lo = max(0, i - window + 1)
        window_vals = x[lo : i + 1]
        roll_mean.append(float(window_vals.mean()))
        roll_std.append(float(window_vals.std(ddof=1)) if len(window_vals) > 1 else 0.0)
        roll_min.append(float(window_vals.min()))
        roll_max.append(float(window_vals.max()))
        running_sum += x[i]
        exp_mean.append(running_sum / (i + 1))

    return {
        "mean": roll_mean,
        "std": roll_std,
        "minimum": roll_min,
        "maximum": roll_max,
        "expanding_mean": exp_mean,
    }


def detect_change_points(
    data: Sequence[float],
    n_breaks: int = 1,
    min_size: int = 5,
) -> dict[str, list]:
    """Detect change points in a time series via binary segmentation.

    Finds the n_breaks positions where the series mean shifts most, using a
    greedy binary segmentation algorithm: at each step, split the current
    segment at the point that maximises the reduction in within-segment
    variance.

    Args:
        data: Time series values (length ≥ 2 * min_size).
        n_breaks: Number of change points to find (default 1).
        min_size: Minimum segment length on either side of a break (default 5).

    Returns:
        dict with keys:
            change_points: Indices (0-based) of the detected change points.
                A change point at index k means the break is *between*
                positions k-1 and k.
            segment_means: Mean of each segment defined by the change points.
            rss_reduction: Total variance reduction achieved (higher = stronger
                evidence of a real break).

    Raises:
        ValueError: If data is too short for the requested breaks and min_size.

    Example:
        >>> data = [1.0] * 20 + [5.0] * 20
        >>> r = detect_change_points(data, n_breaks=1)
        >>> r["change_points"]
        [20]
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 2 * min_size:
        raise ValueError(
            f"Data too short ({n}) for min_size={min_size}. "
            f"Need at least {2 * min_size} observations."
        )

    def _best_split(seg: np.ndarray) -> tuple[int, float]:
        """Return the best split index within seg and its variance reduction."""
        m = len(seg)
        best_idx, best_gain = min_size, -np.inf
        total_var = float(np.var(seg, ddof=0) * m)
        for k in range(min_size, m - min_size + 1):
            left, right = seg[:k], seg[k:]
            reduced = (
                float(np.var(left, ddof=0) * len(left))
                + float(np.var(right, ddof=0) * len(right))
            )
            gain = total_var - reduced
            if gain > best_gain:
                best_gain, best_idx = gain, k
        return best_idx, best_gain

    # Greedy binary segmentation
    # Each segment is tracked as (global_start, global_end)
    segments: list[tuple[int, int]] = [(0, n)]
    change_points: list[int] = []
    total_rss_reduction = 0.0

    for _ in range(n_breaks):
        best_global: tuple[int, float, tuple[int, int]] | None = None
        for start, end in segments:
            seg = x[start:end]
            if len(seg) < 2 * min_size:
                continue
            local_idx, gain = _best_split(seg)
            if best_global is None or gain > best_global[1]:
                best_global = (start + local_idx, gain, (start, end))

        if best_global is None:
            break

        cp, gain, (start, end) = best_global
        change_points.append(cp)
        total_rss_reduction += gain
        segments.remove((start, end))
        segments.extend([(start, cp), (cp, end)])
        segments.sort()

    change_points.sort()

    # Segment means
    boundaries = [0] + change_points + [n]
    segment_means = [
        float(x[boundaries[i] : boundaries[i + 1]].mean())
        for i in range(len(boundaries) - 1)
    ]

    return {
        "change_points": change_points,
        "segment_means": segment_means,
        "rss_reduction": float(total_rss_reduction),
    }


__all__ = [
    "moving_average",
    "autocorrelation",
    "partial_autocorrelation",
    "linear_trend",
    "detrend",
    "seasonal_decompose",
    "difference",
    "mean_absolute_scaled_error",
    "exponential_smoothing",
    "double_exponential_smoothing",
    "rolling_statistics",
    "detect_change_points",
]
