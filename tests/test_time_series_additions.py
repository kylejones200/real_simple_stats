"""Tests for the book-content additions to time_series.py.

Covers the five new functions added in Phase 4:
  - mean_absolute_scaled_error
  - exponential_smoothing
  - double_exponential_smoothing
  - rolling_statistics
  - detect_change_points
"""

import numpy as np
import pytest

from real_simple_stats.time_series import (
    detect_change_points,
    double_exponential_smoothing,
    exponential_smoothing,
    mean_absolute_scaled_error,
    rolling_statistics,
)


# ---------------------------------------------------------------------------
# mean_absolute_scaled_error
# ---------------------------------------------------------------------------


class TestMASE:
    def test_perfect_forecast_is_zero(self):
        actual = [10.0, 12.0, 14.0, 16.0, 18.0]
        assert mean_absolute_scaled_error(actual, actual) == pytest.approx(0.0)

    def test_naive_forecast_less_than_or_equal_one(self):
        # Naïve shift forecast (f[t] = x[t-1]) has errors == differences of x,
        # except at t=0 where error is 0. So MASE ≤ 1 (no worse than baseline).
        actual = [1.0, 3.0, 5.0, 7.0, 9.0]
        naive = [actual[0]] + actual[:-1]
        mase = mean_absolute_scaled_error(actual, naive)
        assert mase <= 1.0

    def test_better_than_naive_lt_one(self):
        actual = [10.0, 12.0, 14.0, 16.0, 18.0]
        # Perfect forecast beats naïve
        assert mean_absolute_scaled_error(actual, actual) < 1.0

    def test_worse_than_naive_gt_one(self):
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        terrible = [5.0, 4.0, 3.0, 2.0, 1.0]  # opposite direction
        assert mean_absolute_scaled_error(actual, terrible) > 1.0

    def test_constant_actual_returns_zero_or_inf(self):
        actual = [5.0, 5.0, 5.0, 5.0]
        result = mean_absolute_scaled_error(actual, actual)
        assert result == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mean_absolute_scaled_error([1, 2, 3], [1, 2])

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            mean_absolute_scaled_error([1.0], [1.0])

    def test_returns_float(self):
        actual = [1.0, 2.0, 3.0, 4.0]
        forecast = [1.1, 2.1, 3.1, 4.1]
        result = mean_absolute_scaled_error(actual, forecast)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# exponential_smoothing
# ---------------------------------------------------------------------------


class TestExponentialSmoothing:
    def test_same_length_as_input(self):
        s = exponential_smoothing([1, 2, 3, 4, 5], alpha=0.3)
        assert len(s) == 5

    def test_first_element_unchanged(self):
        s = exponential_smoothing([7.0, 2.0, 5.0], alpha=0.5)
        assert s[0] == pytest.approx(7.0)

    def test_alpha_one_returns_original(self):
        data = [1.0, 3.0, 5.0, 7.0]
        s = exponential_smoothing(data, alpha=1.0)
        for a, b in zip(data, s):
            assert a == pytest.approx(b)

    def test_alpha_small_very_smooth(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 5, 100).tolist()
        s_small = exponential_smoothing(data, alpha=0.05)
        s_large = exponential_smoothing(data, alpha=0.9)
        var_small = float(np.var(s_small))
        var_large = float(np.var(s_large))
        assert var_small < var_large

    def test_empty_returns_empty(self):
        assert exponential_smoothing([], alpha=0.5) == []

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            exponential_smoothing([1, 2, 3], alpha=0.0)
        with pytest.raises(ValueError):
            exponential_smoothing([1, 2, 3], alpha=1.5)

    def test_increasing_series_smoothed_lags(self):
        data = list(range(1, 11))
        s = exponential_smoothing(data, alpha=0.3)
        # Smoothed values should be below the raw data (lagging behind)
        assert all(s[i] <= data[i] + 1e-6 for i in range(1, 10))


# ---------------------------------------------------------------------------
# double_exponential_smoothing
# ---------------------------------------------------------------------------


class TestDoubleExponentialSmoothing:
    def test_output_keys(self):
        r = double_exponential_smoothing([1, 3, 5, 7, 9], alpha=0.8, beta=0.2)
        assert set(r.keys()) == {"smoothed", "level", "trend"}

    def test_same_length(self):
        data = list(range(1, 21))
        r = double_exponential_smoothing(data, alpha=0.5, beta=0.3)
        assert len(r["smoothed"]) == 20
        assert len(r["level"]) == 20
        assert len(r["trend"]) == 20

    def test_linear_series_trend_positive(self):
        data = list(range(1, 21))
        r = double_exponential_smoothing(data, alpha=0.8, beta=0.5)
        # Trend should be positive throughout for a rising series
        assert all(b > 0 for b in r["trend"][1:])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            double_exponential_smoothing([1, 2, 3], alpha=0.0, beta=0.3)
        with pytest.raises(ValueError):
            double_exponential_smoothing([1, 2, 3], alpha=1.0, beta=0.3)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            double_exponential_smoothing([1, 2, 3], alpha=0.5, beta=0.0)
        with pytest.raises(ValueError):
            double_exponential_smoothing([1, 2, 3], alpha=0.5, beta=1.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            double_exponential_smoothing([1.0], alpha=0.5, beta=0.3)

    def test_first_level_equals_first_data(self):
        data = [5.0, 8.0, 11.0, 14.0]
        r = double_exponential_smoothing(data, alpha=0.6, beta=0.4)
        assert r["level"][0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# rolling_statistics
# ---------------------------------------------------------------------------


class TestRollingStatistics:
    def test_output_keys(self):
        r = rolling_statistics([1, 2, 3, 4, 5], window=3)
        assert set(r.keys()) == {"mean", "std", "minimum", "maximum", "expanding_mean"}

    def test_same_length_as_input(self):
        data = list(range(10))
        r = rolling_statistics(data, window=4)
        for v in r.values():
            assert len(v) == 10

    def test_mean_known_values(self):
        r = rolling_statistics([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        # [1], [1,2], [1,2,3], [2,3,4], [3,4,5]
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]
        for a, b in zip(r["mean"], expected):
            assert a == pytest.approx(b)

    def test_minimum_maximum_monotone(self):
        data = [5.0, 3.0, 8.0, 2.0, 7.0]
        r = rolling_statistics(data, window=3)
        for lo, hi in zip(r["minimum"], r["maximum"]):
            assert lo <= hi

    def test_expanding_mean_increasing_for_rising_series(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = rolling_statistics(data, window=3)
        em = r["expanding_mean"]
        assert all(em[i] <= em[i + 1] for i in range(len(em) - 1))

    def test_window_one_returns_raw(self):
        data = [4.0, 1.0, 7.0]
        r = rolling_statistics(data, window=1)
        assert r["mean"] == pytest.approx(data)
        assert r["minimum"] == pytest.approx(data)
        assert r["maximum"] == pytest.approx(data)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            rolling_statistics([1, 2, 3], window=0)

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            rolling_statistics([], window=2)

    def test_std_zero_for_single_element_window(self):
        r = rolling_statistics([1.0, 2.0, 3.0], window=3)
        assert r["std"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# detect_change_points
# ---------------------------------------------------------------------------


class TestDetectChangePoints:
    def test_obvious_single_break(self):
        data = [1.0] * 20 + [10.0] * 20
        r = detect_change_points(data, n_breaks=1)
        assert r["change_points"] == [20]

    def test_two_breaks(self):
        data = [1.0] * 20 + [10.0] * 20 + [1.0] * 20
        r = detect_change_points(data, n_breaks=2)
        assert len(r["change_points"]) == 2
        cps = r["change_points"]
        assert abs(cps[0] - 20) <= 2
        assert abs(cps[1] - 40) <= 2

    def test_segment_means_correct(self):
        data = [0.0] * 20 + [5.0] * 20
        r = detect_change_points(data, n_breaks=1)
        means = r["segment_means"]
        assert len(means) == 2
        assert means[0] == pytest.approx(0.0)
        assert means[1] == pytest.approx(5.0)

    def test_rss_reduction_positive_for_real_break(self):
        data = [0.0] * 20 + [5.0] * 20
        r = detect_change_points(data, n_breaks=1)
        assert r["rss_reduction"] > 0

    def test_output_keys(self):
        r = detect_change_points([0.0] * 20 + [5.0] * 20, n_breaks=1)
        assert set(r.keys()) == {"change_points", "segment_means", "rss_reduction"}

    def test_change_points_sorted(self):
        data = [0.0] * 15 + [5.0] * 15 + [0.0] * 15
        r = detect_change_points(data, n_breaks=2)
        cps = r["change_points"]
        assert cps == sorted(cps)

    def test_n_segment_means_equals_n_breaks_plus_one(self):
        data = [0.0] * 10 + [5.0] * 10 + [0.0] * 10 + [5.0] * 10
        r = detect_change_points(data, n_breaks=3)
        assert len(r["segment_means"]) == len(r["change_points"]) + 1

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            detect_change_points([1.0, 2.0, 3.0], n_breaks=1, min_size=5)

    def test_constant_series_returns_empty_or_breaks(self):
        # Constant data — no real break; should not crash
        data = [3.0] * 30
        r = detect_change_points(data, n_breaks=1)
        assert "change_points" in r
