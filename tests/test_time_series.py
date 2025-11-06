"""Tests for time series analysis functions."""

import pytest
import numpy as np
from real_simple_stats.time_series import (
    moving_average,
    autocorrelation,
    partial_autocorrelation,
    linear_trend,
    detrend,
    seasonal_decompose,
    difference,
)


class TestMovingAverage:
    def test_simple_moving_average(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = moving_average(data, 3, method="simple")
        assert len(result) == 8
        assert result[0] == pytest.approx(2.0)
        assert result[-1] == pytest.approx(9.0)

    def test_exponential_moving_average(self):
        data = [1, 2, 3, 4, 5]
        result = moving_average(data, 3, method="exponential")
        assert len(result) == 5
        assert result[0] == 1.0

    def test_weighted_moving_average(self):
        data = [1, 2, 3, 4, 5]
        result = moving_average(data, 3, method="weighted")
        assert len(result) == 3

    def test_invalid_window_size(self):
        data = [1, 2, 3]
        with pytest.raises(ValueError):
            moving_average(data, 0)
        with pytest.raises(ValueError):
            moving_average(data, 10)

    def test_unknown_method(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            moving_average(data, 3, method="unknown")


class TestAutocorrelation:
    def test_autocorrelation_lag_zero(self):
        data = [1, 2, 3, 4, 5]
        result = autocorrelation(data, max_lag=0)
        assert result[0] == pytest.approx(1.0)

    def test_autocorrelation_multiple_lags(self):
        data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        result = autocorrelation(data, max_lag=3)
        assert len(result) == 4
        assert result[0] == pytest.approx(1.0)

    def test_autocorrelation_constant_data(self):
        data = [5, 5, 5, 5, 5]
        result = autocorrelation(data, max_lag=2)
        # Constant data should have zero variance
        assert result[1] == pytest.approx(0.0)

    def test_invalid_data(self):
        with pytest.raises(ValueError):
            autocorrelation([1])


class TestPartialAutocorrelation:
    def test_partial_autocorrelation(self):
        data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        result = partial_autocorrelation(data, max_lag=3)
        assert len(result) == 4
        assert result[0] == pytest.approx(1.0)

    def test_invalid_data(self):
        with pytest.raises(ValueError):
            partial_autocorrelation([1])


class TestLinearTrend:
    def test_linear_trend_perfect(self):
        data = [1, 2, 3, 4, 5]
        slope, intercept, r2 = linear_trend(data)
        assert slope == pytest.approx(1.0)
        assert r2 == pytest.approx(1.0)

    def test_linear_trend_flat(self):
        data = [5, 5, 5, 5, 5]
        slope, intercept, r2 = linear_trend(data)
        assert slope == pytest.approx(0.0)

    def test_invalid_data(self):
        with pytest.raises(ValueError):
            linear_trend([1])


class TestDetrend:
    def test_detrend_linear(self):
        data = [1, 2, 3, 4, 5]
        result = detrend(data, method="linear")
        assert len(result) == 5
        # Detrended data should have mean close to 0
        assert np.mean(result) == pytest.approx(0.0, abs=1e-10)

    def test_detrend_mean(self):
        data = [1, 2, 3, 4, 5]
        result = detrend(data, method="mean")
        assert len(result) == 5
        assert np.mean(result) == pytest.approx(0.0)

    def test_unknown_method(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            detrend(data, method="unknown")


class TestSeasonalDecompose:
    def test_seasonal_decompose(self):
        # Create data with clear seasonal pattern
        data = [1, 2, 3, 4] * 6
        trend, seasonal, residual = seasonal_decompose(data, period=4)
        assert len(trend) == len(data)
        assert len(seasonal) == len(data)
        assert len(residual) == len(data)

    def test_invalid_period(self):
        data = [1, 2, 3, 4, 5, 6]
        with pytest.raises(ValueError):
            seasonal_decompose(data, period=1)
        with pytest.raises(ValueError):
            seasonal_decompose(data, period=10)


class TestDifference:
    def test_difference_first_order(self):
        data = [1, 2, 4, 7, 11]
        result = difference(data, lag=1, order=1)
        assert result == [1, 2, 3, 4]

    def test_difference_second_order(self):
        data = [1, 2, 4, 7, 11]
        result = difference(data, lag=1, order=2)
        assert result == [1, 1, 1]

    def test_invalid_parameters(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            difference(data, lag=0)
        with pytest.raises(ValueError):
            difference(data, lag=1, order=0)
        with pytest.raises(ValueError):
            difference(data, lag=10)
