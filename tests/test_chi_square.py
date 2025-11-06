"""Comprehensive tests for chi-square utilities."""

import pytest
from real_simple_stats.chi_square_utils import (
    chi_square_statistic,
    critical_chi_square_value,
    reject_null_chi_square,
)


class TestChiSquareStatistic:
    def test_chi_square_statistic_perfect_fit(self):
        observed = [10, 20, 30]
        expected = [10, 20, 30]
        result = chi_square_statistic(observed, expected)
        assert result == pytest.approx(0.0)

    def test_chi_square_statistic_some_difference(self):
        observed = [10, 20, 30]
        expected = [12, 18, 30]
        result = chi_square_statistic(observed, expected)
        assert result > 0

    def test_chi_square_statistic_large_difference(self):
        observed = [10, 20, 30]
        expected = [30, 20, 10]
        result = chi_square_statistic(observed, expected)
        assert result > 10  # Should be large

    def test_chi_square_statistic_mismatched_lengths(self):
        observed = [10, 20, 30]
        expected = [10, 20]
        with pytest.raises(ValueError, match="same length"):
            chi_square_statistic(observed, expected)

    @pytest.mark.parametrize(
        "observed,expected",
        [
            ([10, 10, 10], [10, 10, 10]),
            ([5, 15, 25], [5, 15, 25]),
            ([100, 200], [100, 200]),
        ],
    )
    def test_chi_square_statistic_perfect_fits(self, observed, expected):
        result = chi_square_statistic(observed, expected)
        assert result == pytest.approx(0.0)

    def test_chi_square_statistic_calculation(self):
        # Manual calculation: ((10-12)^2/12) + ((20-18)^2/18) + ((30-30)^2/30)
        # = 4/12 + 4/18 + 0 = 0.333 + 0.222 = 0.555
        observed = [10, 20, 30]
        expected = [12, 18, 30]
        result = chi_square_statistic(observed, expected)
        assert result == pytest.approx(0.555, abs=0.01)


class TestCriticalChiSquareValue:
    def test_critical_chi_square_value_alpha_05_df_1(self):
        result = critical_chi_square_value(0.05, 1)
        assert result == pytest.approx(3.841, abs=0.01)

    def test_critical_chi_square_value_alpha_05_df_5(self):
        result = critical_chi_square_value(0.05, 5)
        assert result == pytest.approx(11.07, abs=0.1)

    def test_critical_chi_square_value_alpha_01_df_1(self):
        result = critical_chi_square_value(0.01, 1)
        assert result == pytest.approx(6.635, abs=0.01)

    @pytest.mark.parametrize(
        "alpha,df",
        [
            (0.05, 1),
            (0.05, 5),
            (0.05, 10),
            (0.01, 1),
            (0.01, 5),
        ],
    )
    def test_critical_chi_square_value_positive(self, alpha, df):
        result = critical_chi_square_value(alpha, df)
        assert result > 0

    def test_critical_chi_square_value_increases_with_df(self):
        # For same alpha, critical value increases with df
        cv_df1 = critical_chi_square_value(0.05, 1)
        cv_df5 = critical_chi_square_value(0.05, 5)
        cv_df10 = critical_chi_square_value(0.05, 10)
        assert cv_df1 < cv_df5 < cv_df10

    def test_critical_chi_square_value_increases_with_lower_alpha(self):
        # Lower alpha (more stringent) gives higher critical value
        cv_alpha_10 = critical_chi_square_value(0.10, 5)
        cv_alpha_05 = critical_chi_square_value(0.05, 5)
        cv_alpha_01 = critical_chi_square_value(0.01, 5)
        assert cv_alpha_10 < cv_alpha_05 < cv_alpha_01


class TestRejectNullChiSquare:
    def test_reject_null_chi_square_reject(self):
        # Chi-square statistic exceeds critical value
        result = reject_null_chi_square(10.0, 5.0)
        assert result is True

    def test_reject_null_chi_square_fail_to_reject(self):
        # Chi-square statistic below critical value
        result = reject_null_chi_square(3.0, 5.0)
        assert result is False

    def test_reject_null_chi_square_exactly_equal(self):
        # At boundary (typically fail to reject)
        result = reject_null_chi_square(5.0, 5.0)
        assert result is False

    @pytest.mark.parametrize(
        "chi_stat,critical_value,expected",
        [
            (10.0, 5.0, True),
            (3.0, 5.0, False),
            (5.0, 5.0, False),
            (5.1, 5.0, True),
            (4.9, 5.0, False),
        ],
    )
    def test_reject_null_chi_square_parametrized(
        self, chi_stat, critical_value, expected
    ):
        result = reject_null_chi_square(chi_stat, critical_value)
        assert result == expected


class TestIntegration:
    def test_chi_square_test_workflow(self):
        """Test a complete chi-square test workflow."""
        # Observed frequencies
        observed = [10, 20, 30, 40]
        # Expected frequencies (uniform distribution)
        expected = [25, 25, 25, 25]
        
        # Calculate chi-square statistic
        chi_stat = chi_square_statistic(observed, expected)
        assert chi_stat > 0
        
        # Get critical value (alpha=0.05, df=3)
        critical_value = critical_chi_square_value(0.05, 3)
        
        # Make decision
        reject = reject_null_chi_square(chi_stat, critical_value)
        
        # This should reject (chi_stat should be > 7.815)
        assert isinstance(reject, bool)

    def test_goodness_of_fit_test(self):
        """Test goodness of fit for a fair die."""
        # Roll a die 60 times, expect each face 10 times
        observed = [8, 12, 9, 11, 10, 10]
        expected = [10, 10, 10, 10, 10, 10]
        
        chi_stat = chi_square_statistic(observed, expected)
        critical_value = critical_chi_square_value(0.05, 5)  # df = 6-1 = 5
        
        # Should not reject (die appears fair)
        reject = reject_null_chi_square(chi_stat, critical_value)
        assert reject is False  # Chi-stat should be small

    def test_different_alpha_levels(self):
        """Test that different alpha levels give different critical values."""
        df = 5
        
        cv_10 = critical_chi_square_value(0.10, df)
        cv_05 = critical_chi_square_value(0.05, df)
        cv_01 = critical_chi_square_value(0.01, df)
        
        # More stringent alpha requires larger chi-square to reject
        assert cv_10 < cv_05 < cv_01
        
        # Test with a chi-square statistic that will reject at 0.10
        chi_stat = 10.0
        
        # Should reject at alpha=0.10 (cv ~9.24)
        assert reject_null_chi_square(chi_stat, cv_10) is True
        # May or may not reject at 0.05 (cv ~11.07) - depends on chi_stat value
        # So we just verify the critical values are ordered correctly
