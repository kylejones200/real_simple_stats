"""Comprehensive tests for linear regression utilities."""

import pytest
import numpy as np
from real_simple_stats.linear_regression_utils import (
    prepare_scatter_data,
    pearson_correlation,
    coefficient_of_determination,
    linear_regression,
    regression_equation,
    manual_slope_intercept,
)


class TestPrepareScatterData:
    def test_prepare_scatter_data_returns_same(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        result_x, result_y = prepare_scatter_data(x, y)
        assert result_x == x
        assert result_y == y

    def test_prepare_scatter_data_floats(self):
        x = [1.5, 2.5, 3.5]
        y = [2.1, 4.3, 5.2]
        result_x, result_y = prepare_scatter_data(x, y)
        assert result_x == x
        assert result_y == y

    def test_prepare_scatter_data_empty(self):
        x = []
        y = []
        result_x, result_y = prepare_scatter_data(x, y)
        assert result_x == []
        assert result_y == []


class TestPearsonCorrelation:
    def test_perfect_positive_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = pearson_correlation(x, y)
        assert r == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        r = pearson_correlation(x, y)
        assert r == pytest.approx(-1.0)

    def test_no_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [3, 3, 3, 3, 3]
        r = pearson_correlation(x, y)
        # Correlation with constant should be NaN, but numpy handles it
        assert np.isnan(r) or r == pytest.approx(0.0, abs=0.1)

    def test_moderate_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        r = pearson_correlation(x, y)
        assert -1 <= r <= 1

    def test_correlation_with_floats(self):
        x = [1.5, 2.5, 3.5, 4.5, 5.5]
        y = [2.1, 4.3, 5.2, 7.8, 9.1]
        r = pearson_correlation(x, y)
        assert -1 <= r <= 1

    def test_correlation_negative_values(self):
        x = [-5, -4, -3, -2, -1]
        y = [-10, -8, -6, -4, -2]
        r = pearson_correlation(x, y)
        assert r == pytest.approx(1.0)


class TestCoefficientOfDetermination:
    def test_r_squared_perfect_fit(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r2 = coefficient_of_determination(x, y)
        assert r2 == pytest.approx(1.0)

    def test_r_squared_range(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        r2 = coefficient_of_determination(x, y)
        assert 0 <= r2 <= 1

    def test_r_squared_no_relationship(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 3, 4, 2, 5]
        r2 = coefficient_of_determination(x, y)
        assert 0 <= r2 <= 1

    def test_r_squared_negative_correlation(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        r2 = coefficient_of_determination(x, y)
        # R² is always positive (it's r^2)
        assert r2 == pytest.approx(1.0)


class TestLinearRegression:
    def test_linear_regression_perfect_fit(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        slope, intercept, r_value, p_value, std_err = linear_regression(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0)
        assert r_value == pytest.approx(1.0)
        assert p_value < 0.05  # Significant
        assert std_err >= 0

    def test_linear_regression_with_intercept(self):
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]  # y = 2x + 1
        slope, intercept, r_value, p_value, std_err = linear_regression(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)
        assert r_value == pytest.approx(1.0)

    def test_linear_regression_negative_slope(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        slope, intercept, r_value, p_value, std_err = linear_regression(x, y)

        assert slope == pytest.approx(-2.0)
        assert intercept == pytest.approx(12.0)
        assert r_value == pytest.approx(-1.0)

    def test_linear_regression_returns_five_values(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        result = linear_regression(x, y)

        assert len(result) == 5
        assert all(isinstance(v, float) for v in result)

    def test_linear_regression_with_noise(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9]
        slope, intercept, r_value, p_value, std_err = linear_regression(x, y)

        # Should be close to y = 2x
        assert slope == pytest.approx(2.0, abs=0.1)
        assert r_value > 0.99  # Very high correlation
        assert p_value < 0.001  # Highly significant

    def test_linear_regression_floats(self):
        x = [1.5, 2.5, 3.5, 4.5, 5.5]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, intercept, r_value, p_value, std_err = linear_regression(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0)


class TestRegressionEquation:
    def test_regression_equation_simple(self):
        slope = 2.0
        intercept = 1.0
        x = 5.0
        y_pred = regression_equation(x, slope, intercept)
        assert y_pred == pytest.approx(11.0)  # 2*5 + 1

    def test_regression_equation_zero_slope(self):
        slope = 0.0
        intercept = 5.0
        x = 10.0
        y_pred = regression_equation(x, slope, intercept)
        assert y_pred == pytest.approx(5.0)

    def test_regression_equation_negative_slope(self):
        slope = -3.0
        intercept = 10.0
        x = 2.0
        y_pred = regression_equation(x, slope, intercept)
        assert y_pred == pytest.approx(4.0)  # -3*2 + 10

    def test_regression_equation_zero_intercept(self):
        slope = 2.5
        intercept = 0.0
        x = 4.0
        y_pred = regression_equation(x, slope, intercept)
        assert y_pred == pytest.approx(10.0)

    def test_regression_equation_negative_x(self):
        slope = 2.0
        intercept = 3.0
        x = -5.0
        y_pred = regression_equation(x, slope, intercept)
        assert y_pred == pytest.approx(-7.0)  # 2*(-5) + 3

    def test_regression_equation_floats(self):
        slope = 1.5
        intercept = 2.5
        x = 3.5
        y_pred = regression_equation(x, slope, intercept)
        assert y_pred == pytest.approx(7.75)  # 1.5*3.5 + 2.5


class TestManualSlopeIntercept:
    def test_manual_slope_intercept_perfect_fit(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        slope, intercept = manual_slope_intercept(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0)

    def test_manual_slope_intercept_with_intercept(self):
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]  # y = 2x + 1
        slope, intercept = manual_slope_intercept(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_manual_slope_intercept_negative_slope(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        slope, intercept = manual_slope_intercept(x, y)

        assert slope == pytest.approx(-2.0)
        assert intercept == pytest.approx(12.0)

    def test_manual_slope_intercept_matches_linregress(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]

        manual_slope, manual_intercept = manual_slope_intercept(x, y)
        scipy_slope, scipy_intercept, _, _, _ = linear_regression(x, y)

        assert manual_slope == pytest.approx(scipy_slope)
        assert manual_intercept == pytest.approx(scipy_intercept)

    def test_manual_slope_intercept_floats(self):
        x = [1.5, 2.5, 3.5, 4.5, 5.5]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, intercept = manual_slope_intercept(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0)

    def test_manual_slope_intercept_negative_values(self):
        x = [-5, -4, -3, -2, -1]
        y = [-10, -8, -6, -4, -2]
        slope, intercept = manual_slope_intercept(x, y)

        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0)

    def test_manual_slope_intercept_returns_two_values(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 5, 4, 5]
        result = manual_slope_intercept(x, y)

        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


class TestIntegration:
    def test_full_regression_workflow(self):
        """Test a complete regression analysis workflow."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9]

        # Calculate correlation
        r = pearson_correlation(x, y)
        assert r > 0.99

        # Calculate R²
        r2 = coefficient_of_determination(x, y)
        assert r2 > 0.98

        # Perform regression
        slope, intercept, r_value, p_value, std_err = linear_regression(x, y)
        assert slope == pytest.approx(2.0, abs=0.1)
        assert p_value < 0.001

        # Make predictions
        y_pred = regression_equation(11, slope, intercept)
        assert y_pred == pytest.approx(22.0, abs=0.5)

        # Verify manual calculation matches
        manual_slope, manual_intercept = manual_slope_intercept(x, y)
        assert manual_slope == pytest.approx(slope)
        assert manual_intercept == pytest.approx(intercept)
