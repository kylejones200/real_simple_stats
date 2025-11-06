"""Comprehensive tests for sampling and confidence intervals."""

import pytest
from real_simple_stats.sampling_and_intervals import (
    sampling_distribution_mean,
    sampling_distribution_variance,
    clt_probability_greater_than,
    clt_probability_less_than,
    clt_probability_between,
    confidence_interval_known_std,
    confidence_interval_unknown_std,
    required_sample_size,
    slovins_formula,
)


class TestSamplingDistributionMean:
    def test_sampling_distribution_mean_positive(self):
        assert sampling_distribution_mean(100) == 100

    def test_sampling_distribution_mean_negative(self):
        assert sampling_distribution_mean(-50) == -50

    def test_sampling_distribution_mean_zero(self):
        assert sampling_distribution_mean(0) == 0

    def test_sampling_distribution_mean_float(self):
        assert sampling_distribution_mean(75.5) == 75.5


class TestSamplingDistributionVariance:
    def test_sampling_distribution_variance_basic(self):
        # Variance = (15^2) / 100 = 225 / 100 = 2.25
        result = sampling_distribution_variance(15, 100)
        assert result == pytest.approx(2.25)

    def test_sampling_distribution_variance_small_sample(self):
        # Variance = (10^2) / 4 = 100 / 4 = 25
        result = sampling_distribution_variance(10, 4)
        assert result == pytest.approx(25.0)

    def test_sampling_distribution_variance_large_sample(self):
        # Variance = (20^2) / 1000 = 400 / 1000 = 0.4
        result = sampling_distribution_variance(20, 1000)
        assert result == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "pop_std,sample_size,expected",
        [
            (10, 100, 1.0),
            (15, 225, 1.0),
            (20, 400, 1.0),
            (5, 25, 1.0),
        ],
    )
    def test_sampling_distribution_variance_parametrized(
        self, pop_std, sample_size, expected
    ):
        result = sampling_distribution_variance(pop_std, sample_size)
        assert result == pytest.approx(expected)


class TestCLTProbabilityGreaterThan:
    def test_clt_probability_greater_than_above_mean(self):
        # P(sample mean > 82) when pop mean = 80
        result = clt_probability_greater_than(82, 80, 10, 100)
        assert 0 < result < 0.5  # Should be less than 0.5 since 82 > 80

    def test_clt_probability_greater_than_below_mean(self):
        # P(sample mean > 78) when pop mean = 80
        result = clt_probability_greater_than(78, 80, 10, 100)
        assert 0.5 < result < 1.0  # Should be greater than 0.5

    def test_clt_probability_greater_than_at_mean(self):
        # P(sample mean > 80) when pop mean = 80
        result = clt_probability_greater_than(80, 80, 10, 100)
        assert result == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "x,mean,std_dev,n",
        [
            (85, 80, 10, 100),
            (75, 80, 15, 50),
            (100, 95, 20, 200),
        ],
    )
    def test_clt_probability_greater_than_range(self, x, mean, std_dev, n):
        result = clt_probability_greater_than(x, mean, std_dev, n)
        assert 0 <= result <= 1


class TestCLTProbabilityLessThan:
    def test_clt_probability_less_than_below_mean(self):
        # P(sample mean < 75) when pop mean = 80
        result = clt_probability_less_than(75, 80, 10, 100)
        assert 0 < result < 0.5

    def test_clt_probability_less_than_above_mean(self):
        # P(sample mean < 85) when pop mean = 80
        result = clt_probability_less_than(85, 80, 10, 100)
        assert 0.5 < result < 1.0

    def test_clt_probability_less_than_at_mean(self):
        # P(sample mean < 80) when pop mean = 80
        result = clt_probability_less_than(80, 80, 10, 100)
        assert result == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "x,mean,std_dev,n",
        [
            (75, 80, 10, 100),
            (85, 80, 15, 50),
            (90, 95, 20, 200),
        ],
    )
    def test_clt_probability_less_than_range(self, x, mean, std_dev, n):
        result = clt_probability_less_than(x, mean, std_dev, n)
        assert 0 <= result <= 1

    def test_clt_probability_complement(self):
        # P(X < x) + P(X > x) should be close to 1 (accounting for P(X = x) ≈ 0)
        x, mean, std_dev, n = 82, 80, 10, 100
        p_less = clt_probability_less_than(x, mean, std_dev, n)
        p_greater = clt_probability_greater_than(x, mean, std_dev, n)
        assert p_less + p_greater == pytest.approx(1.0, abs=0.001)


class TestCLTProbabilityBetween:
    def test_clt_probability_between_symmetric(self):
        # P(78 < sample mean < 82) when pop mean = 80
        result = clt_probability_between(78, 82, 80, 10, 100)
        assert 0 < result < 1

    def test_clt_probability_between_narrow_range(self):
        # Narrow range should give smaller probability
        result = clt_probability_between(79.5, 80.5, 80, 10, 100)
        assert 0 < result < 0.5

    def test_clt_probability_between_wide_range(self):
        # Wide range should give larger probability
        result = clt_probability_between(70, 90, 80, 10, 100)
        assert 0.5 < result <= 1.0  # Can be exactly 1.0 for very wide ranges

    @pytest.mark.parametrize(
        "x1,x2,mean,std_dev,n",
        [
            (78, 82, 80, 10, 100),
            (75, 85, 80, 15, 50),
            (90, 100, 95, 20, 200),
        ],
    )
    def test_clt_probability_between_range(self, x1, x2, mean, std_dev, n):
        result = clt_probability_between(x1, x2, mean, std_dev, n)
        assert 0 <= result <= 1

    def test_clt_probability_between_equals_difference(self):
        # P(x1 < X < x2) = P(X < x2) - P(X < x1)
        x1, x2, mean, std_dev, n = 78, 82, 80, 10, 100
        p_between = clt_probability_between(x1, x2, mean, std_dev, n)
        p_less_x2 = clt_probability_less_than(x2, mean, std_dev, n)
        p_less_x1 = clt_probability_less_than(x1, mean, std_dev, n)
        assert p_between == pytest.approx(p_less_x2 - p_less_x1)


class TestConfidenceIntervalKnownStd:
    def test_confidence_interval_known_std_95(self):
        # 95% CI for mean=100, std=15, n=36
        lower, upper = confidence_interval_known_std(100, 15, 36, 0.95)
        assert lower < 100 < upper
        assert upper - lower > 0  # Positive width

    def test_confidence_interval_known_std_99(self):
        # 99% CI should be wider than 95% CI
        lower_95, upper_95 = confidence_interval_known_std(100, 15, 36, 0.95)
        lower_99, upper_99 = confidence_interval_known_std(100, 15, 36, 0.99)

        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95

    def test_confidence_interval_known_std_larger_sample(self):
        # Larger sample should give narrower CI
        lower_small, upper_small = confidence_interval_known_std(100, 15, 36, 0.95)
        lower_large, upper_large = confidence_interval_known_std(100, 15, 144, 0.95)

        width_small = upper_small - lower_small
        width_large = upper_large - lower_large
        assert width_large < width_small

    @pytest.mark.parametrize(
        "mean,std_dev,n,confidence",
        [
            (100, 15, 36, 0.95),
            (50, 10, 25, 0.90),
            (200, 30, 100, 0.99),
        ],
    )
    def test_confidence_interval_known_std_contains_mean(
        self, mean, std_dev, n, confidence
    ):
        lower, upper = confidence_interval_known_std(mean, std_dev, n, confidence)
        assert lower < mean < upper

    def test_confidence_interval_known_std_symmetric(self):
        mean = 100
        lower, upper = confidence_interval_known_std(mean, 15, 36, 0.95)
        # CI should be symmetric around the mean
        assert abs(mean - lower) == pytest.approx(abs(upper - mean))


class TestConfidenceIntervalUnknownStd:
    def test_confidence_interval_unknown_std_95(self):
        # 95% CI using t-distribution
        lower, upper = confidence_interval_unknown_std(100, 15, 36, 0.95)
        assert lower < 100 < upper

    def test_confidence_interval_unknown_std_wider_than_known(self):
        # t-distribution CI should be slightly wider than z-distribution CI
        lower_z, upper_z = confidence_interval_known_std(100, 15, 36, 0.95)
        lower_t, upper_t = confidence_interval_unknown_std(100, 15, 36, 0.95)

        width_z = upper_z - lower_z
        width_t = upper_t - lower_t
        assert width_t >= width_z  # t-CI should be at least as wide

    def test_confidence_interval_unknown_std_small_sample(self):
        # Small sample should give wider CI
        lower_small, upper_small = confidence_interval_unknown_std(100, 15, 10, 0.95)
        lower_large, upper_large = confidence_interval_unknown_std(100, 15, 100, 0.95)

        width_small = upper_small - lower_small
        width_large = upper_large - lower_large
        assert width_small > width_large

    @pytest.mark.parametrize(
        "sample_mean,sample_std,n,confidence",
        [
            (100, 15, 36, 0.95),
            (50, 10, 25, 0.90),
            (200, 30, 100, 0.99),
        ],
    )
    def test_confidence_interval_unknown_std_contains_mean(
        self, sample_mean, sample_std, n, confidence
    ):
        lower, upper = confidence_interval_unknown_std(
            sample_mean, sample_std, n, confidence
        )
        assert lower < sample_mean < upper


class TestRequiredSampleSize:
    def test_required_sample_size_basic(self):
        # Calculate required sample size for 95% CI with width=10
        n = required_sample_size(0.95, 10, 15)
        assert n > 0
        assert isinstance(n, int)

    def test_required_sample_size_narrower_width(self):
        # Narrower width requires larger sample
        n_wide = required_sample_size(0.95, 20, 15)
        n_narrow = required_sample_size(0.95, 10, 15)
        assert n_narrow > n_wide

    def test_required_sample_size_higher_confidence(self):
        # Higher confidence requires larger sample
        n_95 = required_sample_size(0.95, 10, 15)
        n_99 = required_sample_size(0.99, 10, 15)
        assert n_99 > n_95

    @pytest.mark.parametrize(
        "confidence,width,std_dev",
        [
            (0.95, 10, 15),
            (0.90, 5, 10),
            (0.99, 20, 30),
        ],
    )
    def test_required_sample_size_positive(self, confidence, width, std_dev):
        n = required_sample_size(confidence, width, std_dev)
        assert n > 0

    def test_required_sample_size_larger_std(self):
        # Larger std dev requires larger sample
        n_small_std = required_sample_size(0.95, 10, 10)
        n_large_std = required_sample_size(0.95, 10, 20)
        assert n_large_std > n_small_std


class TestSlovinsFormula:
    def test_slovins_formula_basic(self):
        # N=1000, e=0.05
        n = slovins_formula(1000, 0.05)
        assert n > 0
        assert n < 1000  # Sample should be smaller than population
        assert isinstance(n, int)

    def test_slovins_formula_smaller_error(self):
        # Smaller error requires larger sample
        n_large_error = slovins_formula(1000, 0.10)
        n_small_error = slovins_formula(1000, 0.05)
        assert n_small_error > n_large_error

    def test_slovins_formula_larger_population(self):
        # Larger population requires larger sample
        n_small_pop = slovins_formula(500, 0.05)
        n_large_pop = slovins_formula(2000, 0.05)
        assert n_large_pop > n_small_pop

    @pytest.mark.parametrize(
        "N,e,expected_range",
        [
            (1000, 0.05, (200, 400)),
            (500, 0.10, (50, 150)),
            (2000, 0.03, (500, 1200)),
        ],
    )
    def test_slovins_formula_reasonable_range(self, N, e, expected_range):
        n = slovins_formula(N, e)
        assert expected_range[0] < n < expected_range[1]

    def test_slovins_formula_small_population(self):
        # For very small populations
        n = slovins_formula(100, 0.05)
        assert 0 < n <= 100

    def test_slovins_formula_large_error(self):
        # Large error margin gives small sample
        n = slovins_formula(1000, 0.20)
        assert n < 100


class TestIntegration:
    def test_clt_workflow(self):
        """Test a complete CLT analysis workflow."""
        pop_mean = 80
        pop_std = 10
        sample_size = 100

        # Sampling distribution properties
        mean_of_means = sampling_distribution_mean(pop_mean)
        var_of_means = sampling_distribution_variance(pop_std, sample_size)

        assert mean_of_means == pop_mean
        assert var_of_means == pytest.approx(1.0)

        # Probability calculations
        p_greater = clt_probability_greater_than(82, pop_mean, pop_std, sample_size)
        p_less = clt_probability_less_than(78, pop_mean, pop_std, sample_size)
        p_between = clt_probability_between(78, 82, pop_mean, pop_std, sample_size)

        assert 0 < p_greater < 0.5
        assert 0 < p_less < 0.5
        assert p_between > 0

    def test_confidence_interval_workflow(self):
        """Test confidence interval calculation workflow."""
        sample_mean = 100
        sample_std = 15
        n = 36

        # Calculate both types of CI
        ci_known = confidence_interval_known_std(sample_mean, sample_std, n, 0.95)
        ci_unknown = confidence_interval_unknown_std(sample_mean, sample_std, n, 0.95)

        # Both should contain the mean
        assert ci_known[0] < sample_mean < ci_known[1]
        assert ci_unknown[0] < sample_mean < ci_unknown[1]

        # t-CI should be slightly wider
        width_known = ci_known[1] - ci_known[0]
        width_unknown = ci_unknown[1] - ci_unknown[0]
        assert width_unknown >= width_known

    def test_sample_size_planning_workflow(self):
        """Test sample size planning workflow."""
        # Determine required sample size
        desired_width = 10
        confidence = 0.95
        pop_std = 15

        n_required = required_sample_size(confidence, desired_width, pop_std)

        # Verify the CI with this sample size has approximately the desired width
        ci = confidence_interval_known_std(100, pop_std, n_required, confidence)
        actual_width = ci[1] - ci[0]

        # Should be close to desired width (within 10%)
        assert abs(actual_width - desired_width) / desired_width < 0.1
