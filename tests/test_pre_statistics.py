"""Comprehensive tests for pre-statistics utilities."""

import math

import pytest

from real_simple_stats.pre_statistics import (
    decimal_to_percent,
    factorial,
    mean,
    median,
    mode,
    order_of_operations_example,
    percent_to_decimal,
    round_to_decimal_places,
    weighted_mean,
)


class TestPercentToDecimal:
    def test_percent_to_decimal_basic(self):
        assert percent_to_decimal(75) == pytest.approx(0.75)

    def test_percent_to_decimal_hundred(self):
        assert percent_to_decimal(100) == pytest.approx(1.0)

    def test_percent_to_decimal_zero(self):
        assert percent_to_decimal(0) == pytest.approx(0.0)

    def test_percent_to_decimal_fifty(self):
        assert percent_to_decimal(50) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "percent,expected",
        [
            (25, 0.25),
            (33.33, 0.3333),
            (66.67, 0.6667),
            (150, 1.5),
        ],
    )
    def test_percent_to_decimal_parametrized(self, percent, expected):
        assert percent_to_decimal(percent) == pytest.approx(expected, abs=0.01)


class TestDecimalToPercent:
    def test_decimal_to_percent_basic(self):
        assert decimal_to_percent(0.75) == pytest.approx(75.0)

    def test_decimal_to_percent_one(self):
        assert decimal_to_percent(1.0) == pytest.approx(100.0)

    def test_decimal_to_percent_zero(self):
        assert decimal_to_percent(0.0) == pytest.approx(0.0)

    def test_decimal_to_percent_half(self):
        assert decimal_to_percent(0.5) == pytest.approx(50.0)

    @pytest.mark.parametrize(
        "decimal,expected",
        [
            (0.25, 25.0),
            (0.3333, 33.33),
            (0.6667, 66.67),
            (1.5, 150.0),
        ],
    )
    def test_decimal_to_percent_parametrized(self, decimal, expected):
        assert decimal_to_percent(decimal) == pytest.approx(expected, abs=0.01)

    def test_percent_decimal_roundtrip(self):
        """Test that converting back and forth gives original value."""
        original = 75.0
        decimal = percent_to_decimal(original)
        back_to_percent = decimal_to_percent(decimal)
        assert back_to_percent == pytest.approx(original)


class TestRoundToDecimalPlaces:
    def test_round_to_decimal_places_two(self):
        assert round_to_decimal_places(0.1284, 2) == pytest.approx(0.13)

    def test_round_to_decimal_places_zero(self):
        assert round_to_decimal_places(3.7, 0) == pytest.approx(4.0)

    def test_round_to_decimal_places_one(self):
        assert round_to_decimal_places(3.14159, 1) == pytest.approx(3.1)

    def test_round_to_decimal_places_three(self):
        assert round_to_decimal_places(3.14159, 3) == pytest.approx(3.142)

    @pytest.mark.parametrize(
        "value,places,expected",
        [
            (1.2345, 2, 1.23),
            (1.2355, 2, 1.24),
            (9.9999, 2, 10.00),
            (0.0001, 3, 0.000),
        ],
    )
    def test_round_to_decimal_places_parametrized(self, value, places, expected):
        assert round_to_decimal_places(value, places) == pytest.approx(expected)

    def test_round_negative_number(self):
        assert round_to_decimal_places(-3.14159, 2) == pytest.approx(-3.14)


class TestOrderOfOperationsExample:
    def test_order_of_operations_example_returns_float(self):
        result = order_of_operations_example()
        assert isinstance(result, float)

    def test_order_of_operations_example_deterministic(self):
        # Should always return the same value
        result1 = order_of_operations_example()
        result2 = order_of_operations_example()
        assert result1 == result2

    def test_order_of_operations_example_reasonable_value(self):
        # Just verify it returns a reasonable number
        result = order_of_operations_example()
        assert -100 < result < 100  # Reasonable range


class TestMean:
    def test_mean_basic(self):
        assert mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)

    def test_mean_all_same(self):
        assert mean([5, 5, 5, 5]) == pytest.approx(5.0)

    def test_mean_negative_numbers(self):
        assert mean([-5, -3, -1, 1, 3, 5]) == pytest.approx(0.0)

    def test_mean_floats(self):
        assert mean([1.5, 2.5, 3.5]) == pytest.approx(2.5)

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([10, 20, 30], 20.0),
            ([1, 1, 1, 1], 1.0),
            ([0, 0, 10], 10 / 3),
        ],
    )
    def test_mean_parametrized(self, values, expected):
        assert mean(values) == pytest.approx(expected)

    def test_mean_single_value(self):
        assert mean([42]) == pytest.approx(42.0)


class TestMode:
    def test_mode_single_mode(self):
        result = mode([1, 2, 2, 3, 4])
        assert result == [2]

    def test_mode_multiple_modes(self):
        result = mode([1, 1, 2, 2, 3])
        assert set(result) == {1, 2}

    def test_mode_all_same(self):
        result = mode([5, 5, 5, 5])
        assert result == [5]

    def test_mode_no_repeats(self):
        # When all values appear once, all are modes
        result = mode([1, 2, 3, 4, 5])
        assert set(result) == {1, 2, 3, 4, 5}

    def test_mode_with_strings(self):
        result = mode(["a", "b", "b", "c"])
        assert result == ["b"]

    def test_mode_from_example(self):
        data = [2, 19, 44, 44, 44, 51, 56, 78, 86, 99, 99]
        result = mode(data)
        assert result == [44]


class TestMedian:
    def test_median_odd_length(self):
        assert median([1, 2, 3, 4, 5]) == pytest.approx(3)

    def test_median_even_length(self):
        assert median([1, 2, 3, 4]) == pytest.approx(2.5)

    def test_median_unsorted(self):
        assert median([5, 1, 3, 2, 4]) == pytest.approx(3)

    def test_median_single_value(self):
        assert median([42]) == pytest.approx(42)

    def test_median_two_values(self):
        assert median([1, 3]) == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([1, 2, 3], 2),
            ([1, 2, 3, 4], 2.5),
            ([10, 20, 30, 40, 50], 30),
            ([5, 5, 5, 5], 5),
        ],
    )
    def test_median_parametrized(self, values, expected):
        assert median(values) == pytest.approx(expected)

    def test_median_negative_numbers(self):
        assert median([-5, -3, -1, 1, 3]) == pytest.approx(-1)

    def test_median_from_example(self):
        data = [2, 19, 44, 44, 44, 51, 56, 78, 86, 99, 99]
        result = median(data)
        assert result == pytest.approx(51)


class TestWeightedMean:
    def test_weighted_mean_equal_weights(self):
        # Equal weights should give same result as regular mean
        values = [80, 85, 90]
        weights = [1, 1, 1]
        result = weighted_mean(values, weights)
        assert result == pytest.approx(85.0)

    def test_weighted_mean_different_weights(self):
        values = [80, 80, 85]
        weights = [0.4, 0.4, 0.2]
        result = weighted_mean(values, weights)
        # (80*0.4 + 80*0.4 + 85*0.2) / (0.4 + 0.4 + 0.2) = 81
        assert result == pytest.approx(81.0)

    def test_weighted_mean_single_value(self):
        result = weighted_mean([100], [1])
        assert result == pytest.approx(100.0)

    @pytest.mark.parametrize(
        "values,weights,expected",
        [
            ([70, 80, 90], [1, 2, 1], 80.0),
            ([50, 100], [1, 1], 75.0),
            ([10, 20, 30], [0.5, 0.3, 0.2], 17.0),
        ],
    )
    def test_weighted_mean_parametrized(self, values, weights, expected):
        assert weighted_mean(values, weights) == pytest.approx(expected)

    def test_weighted_mean_high_weight_dominates(self):
        # Value with high weight should dominate
        values = [10, 100]
        weights = [9, 1]
        result = weighted_mean(values, weights)
        # Should be closer to 10 than to 100
        assert result < 30


class TestFactorial:
    def test_factorial_zero(self):
        assert factorial(0) == 1

    def test_factorial_one(self):
        assert factorial(1) == 1

    def test_factorial_five(self):
        assert factorial(5) == 120

    def test_factorial_ten(self):
        assert factorial(10) == 3628800

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (4, 24),
            (5, 120),
            (6, 720),
        ],
    )
    def test_factorial_parametrized(self, n, expected):
        assert factorial(n) == expected

    def test_factorial_uses_math_factorial(self):
        # Verify it matches math.factorial
        for n in range(10):
            assert factorial(n) == math.factorial(n)


class TestIntegration:
    def test_percentage_workflow(self):
        """Test converting between percentages and decimals."""
        # Start with a percentage
        percent = 75.0

        # Convert to decimal
        decimal = percent_to_decimal(percent)
        assert decimal == pytest.approx(0.75)

        # Convert back to percent
        back_to_percent = decimal_to_percent(decimal)
        assert back_to_percent == pytest.approx(percent)

    def test_descriptive_stats_workflow(self):
        """Test calculating multiple descriptive statistics."""
        data = [2, 19, 44, 44, 44, 51, 56, 78, 86, 99, 99]

        # Calculate mean
        mean_val = mean(data)
        assert mean_val > 0

        # Calculate median
        median_val = median(data)
        assert median_val == 51

        # Calculate mode
        mode_val = mode(data)
        assert 44 in mode_val

    def test_weighted_average_workflow(self):
        """Test weighted average calculation."""
        # Exam scores with different weights
        exam1, exam2, final = 80, 85, 90
        weight1, weight2, weight_final = 0.3, 0.3, 0.4

        scores = [exam1, exam2, final]
        weights = [weight1, weight2, weight_final]

        final_grade = weighted_mean(scores, weights)

        # Verify it's between min and max
        assert min(scores) <= final_grade <= max(scores)

        # Round to 2 decimal places
        final_grade_rounded = round_to_decimal_places(final_grade, 2)
        assert isinstance(final_grade_rounded, float)

    def test_rounding_workflow(self):
        """Test rounding in a practical context."""
        # Calculate something that needs rounding
        raw_value = 3.14159265359

        # Round to different precisions
        rounded_1 = round_to_decimal_places(raw_value, 1)
        rounded_2 = round_to_decimal_places(raw_value, 2)
        rounded_3 = round_to_decimal_places(raw_value, 3)

        assert rounded_1 == pytest.approx(3.1)
        assert rounded_2 == pytest.approx(3.14)
        assert rounded_3 == pytest.approx(3.142)
