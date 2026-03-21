"""Demo: Time series - moving averages, autocorrelation, trend."""

import logging

from real_simple_stats.time_series import (
    autocorrelation,
    linear_trend,
    moving_average,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample time series (e.g., daily values)
data = [10, 12, 11, 13, 14, 15, 13, 16, 18, 17, 19, 20, 18, 21, 22, 23]

# Moving average (window=3)
ma = moving_average(data, window_size=3, method="simple")
logger.info("Simple moving average (k=3): %s ...", [f"{x:.1f}" for x in ma[:5]])

# Linear trend
slope, intercept, r_squared = linear_trend(data)
logger.info("\nLinear trend: y = %.2f + %.2f t", intercept, slope)
logger.info("R²: %.4f", r_squared)

# Autocorrelation (first few lags)
acf = autocorrelation(data, max_lag=5)
logger.info("\nAutocorrelation (lags 0-4): %s", [f"{x:.3f}" for x in acf])
