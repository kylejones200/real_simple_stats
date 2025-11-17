import math
import unittest

from real_simple_stats.descriptive_statistics import (
    coefficient_of_variation,
    five_number_summary,
    interquartile_range,
    is_continuous,
    is_discrete,
    mean,
    median,
    sample_std_dev,
    sample_variance,
)


class TestDescriptiveStatistics(unittest.TestCase):
    def setUp(self):
        self.sample_data = [1, 2, 3, 4, 5]
        self.float_data = [1.1, 2.2, 3.3, 4.4, 5.5]
        self.unsorted_data = [5, 1, 3, 2, 4]

    def test_median_odd_length(self):
        """Test median calculation for odd-length list."""
        result = median([1, 2, 3, 4, 5])
        self.assertEqual(result, 3)

    def test_median_even_length(self):
        """Test median calculation for even-length list."""
        result = median([1, 2, 3, 4])
        self.assertEqual(result, 2.5)

    def test_median_unsorted(self):
        """Test median works with unsorted data."""
        result = median(self.unsorted_data)
        self.assertEqual(result, 3)

    def test_sample_variance(self):
        """Test sample variance calculation."""
        # For [1,2,3,4,5]: mean=3, variance=2.5
        result = sample_variance(self.sample_data)
        self.assertAlmostEqual(result, 2.5, places=6)

    def test_sample_std_dev(self):
        """Test sample standard deviation calculation."""
        result = sample_std_dev(self.sample_data)
        expected = math.sqrt(2.5)
        self.assertAlmostEqual(result, expected, places=6)

    def test_five_number_summary(self):
        """Test five number summary calculation."""
        result = five_number_summary(self.sample_data)
        expected = {"min": 1, "Q1": 1.5, "median": 3, "Q3": 4.5, "max": 5}
        self.assertEqual(result, expected)

    def test_is_discrete(self):
        """Test discrete variable detection."""
        self.assertTrue(is_discrete([1, 2, 3, 4, 5]))
        self.assertTrue(is_discrete([1.0, 2.0, 3.0]))
        self.assertFalse(is_discrete([1.1, 2.2, 3.3]))

    def test_is_continuous(self):
        """Test continuous variable detection."""
        self.assertFalse(is_continuous([1, 2, 3, 4, 5]))
        self.assertTrue(is_continuous([1.1, 2.2, 3.3]))

    def test_interquartile_range(self):
        """Test IQR calculation."""
        result = interquartile_range(self.sample_data)
        self.assertEqual(result, 3.0)  # Q3(4.5) - Q1(1.5) = 3

    def test_coefficient_of_variation(self):
        """Test coefficient of variation calculation."""
        result = coefficient_of_variation(self.sample_data)
        # CV = std_dev / mean = sqrt(2.5) / 3
        expected = math.sqrt(2.5) / 3
        self.assertAlmostEqual(result, expected, places=6)

    def test_empty_list_handling(self):
        """Test that functions handle empty lists appropriately."""
        with self.assertRaises(ValueError):
            median([])
        with self.assertRaises(ValueError):
            mean([])

    def test_single_value_list(self):
        """Test functions with single-value lists."""
        single_val = [5]
        self.assertEqual(median(single_val), 5)
        self.assertEqual(mean(single_val), 5)
        # Sample variance requires at least 2 values
        with self.assertRaises(ValueError):
            sample_variance(single_val)

    def test_five_number_summary_empty(self):
        """Test five number summary with empty list."""
        with self.assertRaises(ValueError):
            five_number_summary([])

    def test_five_number_summary_single_value(self):
        """Test five number summary with single value."""
        result = five_number_summary([5])
        expected = {"min": 5, "Q1": 5, "median": 5, "Q3": 5, "max": 5}
        self.assertEqual(result, expected)

    def test_five_number_summary_two_values(self):
        """Test five number summary with two values."""
        result = five_number_summary([1, 2])
        expected = {"min": 1, "Q1": 1, "median": 1.5, "Q3": 2, "max": 2}
        self.assertEqual(result, expected)

    def test_five_number_summary_three_values(self):
        """Test five number summary with three values."""
        result = five_number_summary([1, 2, 3])
        expected = {"min": 1, "Q1": 1, "median": 2, "Q3": 3, "max": 3}
        self.assertEqual(result, expected)

    def test_five_number_summary_four_values(self):
        """Test five number summary with four values."""
        result = five_number_summary([1, 2, 3, 4])
        # For [1,2,3,4]: median=2.5, lower_half=[1,2] Q1=1.5, upper_half=[3,4] Q3=3.5
        expected = {"min": 1, "Q1": 1.5, "median": 2.5, "Q3": 3.5, "max": 4}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
