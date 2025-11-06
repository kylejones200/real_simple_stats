"""Comprehensive tests for probability distributions."""

import pytest
from real_simple_stats.probability_distributions import (
    poisson_pmf,
    poisson_cdf,
    geometric_pmf,
    geometric_cdf,
    exponential_pdf,
    exponential_cdf,
    negative_binomial_pmf,
    expected_value_poisson,
    variance_poisson,
    expected_value_geometric,
    variance_geometric,
    expected_value_exponential,
    variance_exponential,
)


class TestPoissonDistribution:
    def test_poisson_pmf_basic(self):
        # P(X=3) for Poisson(λ=2)
        result = poisson_pmf(3, 2)
        assert 0 <= result <= 1

    def test_poisson_pmf_zero(self):
        # P(X=0) should be e^(-λ)
        result = poisson_pmf(0, 2)
        assert result == pytest.approx(0.1353, abs=0.01)

    def test_poisson_cdf_basic(self):
        # P(X≤3) for Poisson(λ=2)
        result = poisson_cdf(3, 2)
        assert 0 <= result <= 1

    def test_poisson_cdf_increases(self):
        # CDF should be non-decreasing
        lam = 2
        cdf_0 = poisson_cdf(0, lam)
        cdf_1 = poisson_cdf(1, lam)
        cdf_2 = poisson_cdf(2, lam)
        assert cdf_0 <= cdf_1 <= cdf_2

    @pytest.mark.parametrize(
        "k,lam",
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (5, 3),
        ],
    )
    def test_poisson_pmf_range(self, k, lam):
        result = poisson_pmf(k, lam)
        assert 0 <= result <= 1


class TestGeometricDistribution:
    def test_geometric_pmf_basic(self):
        # P(X=4) for Geometric(p=0.2)
        result = geometric_pmf(4, 0.2)
        assert 0 <= result <= 1

    def test_geometric_pmf_first_trial(self):
        # P(X=1) = p
        p = 0.5
        result = geometric_pmf(1, p)
        assert result == pytest.approx(p)

    def test_geometric_cdf_basic(self):
        # P(X≤4) for Geometric(p=0.2)
        result = geometric_cdf(4, 0.2)
        assert 0 <= result <= 1

    def test_geometric_cdf_increases(self):
        # CDF should be non-decreasing
        p = 0.2
        cdf_1 = geometric_cdf(1, p)
        cdf_2 = geometric_cdf(2, p)
        cdf_3 = geometric_cdf(3, p)
        assert cdf_1 <= cdf_2 <= cdf_3

    @pytest.mark.parametrize(
        "k,p",
        [
            (1, 0.5),
            (2, 0.3),
            (5, 0.1),
        ],
    )
    def test_geometric_pmf_range(self, k, p):
        result = geometric_pmf(k, p)
        assert 0 <= result <= 1


class TestExponentialDistribution:
    def test_exponential_pdf_basic(self):
        # f(x=2) for Exponential(λ=0.5)
        result = exponential_pdf(2, 0.5)
        assert result >= 0

    def test_exponential_pdf_at_zero(self):
        # f(0) = λ
        lam = 0.5
        result = exponential_pdf(0, lam)
        assert result == pytest.approx(lam)

    def test_exponential_cdf_basic(self):
        # P(X≤2) for Exponential(λ=0.5)
        result = exponential_cdf(2, 0.5)
        assert 0 <= result <= 1

    def test_exponential_cdf_at_zero(self):
        # P(X≤0) = 0
        result = exponential_cdf(0, 0.5)
        assert result == pytest.approx(0.0)

    def test_exponential_cdf_increases(self):
        # CDF should be non-decreasing
        lam = 0.5
        cdf_1 = exponential_cdf(1, lam)
        cdf_2 = exponential_cdf(2, lam)
        cdf_3 = exponential_cdf(3, lam)
        assert cdf_1 <= cdf_2 <= cdf_3

    @pytest.mark.parametrize(
        "x,lam",
        [
            (1, 0.5),
            (2, 1.0),
            (5, 0.2),
        ],
    )
    def test_exponential_cdf_range(self, x, lam):
        result = exponential_cdf(x, lam)
        assert 0 <= result <= 1


class TestNegativeBinomialDistribution:
    def test_negative_binomial_pmf_basic(self):
        # P(k=3 failures before r=2 successes, p=0.5)
        result = negative_binomial_pmf(3, 2, 0.5)
        assert 0 <= result <= 1

    def test_negative_binomial_pmf_zero_failures(self):
        # P(0 failures) = p^r
        r = 2
        p = 0.5
        result = negative_binomial_pmf(0, r, p)
        assert result == pytest.approx(p**r)

    @pytest.mark.parametrize(
        "k,r,p",
        [
            (0, 1, 0.5),
            (1, 2, 0.3),
            (5, 3, 0.4),
        ],
    )
    def test_negative_binomial_pmf_range(self, k, r, p):
        result = negative_binomial_pmf(k, r, p)
        assert 0 <= result <= 1


class TestPoissonExpectations:
    def test_expected_value_poisson(self):
        # E[X] = λ
        lam = 4
        result = expected_value_poisson(lam)
        assert result == pytest.approx(lam)

    def test_variance_poisson(self):
        # Var[X] = λ
        lam = 4
        result = variance_poisson(lam)
        assert result == pytest.approx(lam)

    @pytest.mark.parametrize("lam", [1, 2, 5, 10])
    def test_poisson_mean_equals_variance(self, lam):
        # For Poisson, mean = variance = λ
        mean = expected_value_poisson(lam)
        var = variance_poisson(lam)
        assert mean == pytest.approx(var)


class TestGeometricExpectations:
    def test_expected_value_geometric(self):
        # E[X] = 1/p
        p = 0.2
        result = expected_value_geometric(p)
        assert result == pytest.approx(5.0)

    def test_variance_geometric(self):
        # Var[X] = (1-p)/p^2
        p = 0.2
        result = variance_geometric(p)
        expected = (1 - p) / (p**2)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75])
    def test_geometric_expectations_positive(self, p):
        mean = expected_value_geometric(p)
        var = variance_geometric(p)
        assert mean > 0
        assert var > 0


class TestExponentialExpectations:
    def test_expected_value_exponential(self):
        # E[X] = 1/λ
        lam = 0.5
        result = expected_value_exponential(lam)
        assert result == pytest.approx(2.0)

    def test_variance_exponential(self):
        # Var[X] = 1/λ^2
        lam = 0.5
        result = variance_exponential(lam)
        assert result == pytest.approx(4.0)

    @pytest.mark.parametrize("lam", [0.1, 0.5, 1.0, 2.0])
    def test_exponential_expectations_positive(self, lam):
        mean = expected_value_exponential(lam)
        var = variance_exponential(lam)
        assert mean > 0
        assert var > 0

    def test_exponential_mean_variance_relationship(self):
        # Var[X] = E[X]^2
        lam = 0.5
        mean = expected_value_exponential(lam)
        var = variance_exponential(lam)
        assert var == pytest.approx(mean**2)


class TestIntegration:
    def test_poisson_workflow(self):
        """Test complete Poisson distribution workflow."""
        lam = 3
        
        # Calculate probabilities
        pmf_2 = poisson_pmf(2, lam)
        cdf_2 = poisson_cdf(2, lam)
        
        # CDF should be >= PMF
        assert cdf_2 >= pmf_2
        
        # Check expectations
        mean = expected_value_poisson(lam)
        var = variance_poisson(lam)
        assert mean == var == lam

    def test_geometric_workflow(self):
        """Test complete geometric distribution workflow."""
        p = 0.25
        
        # Calculate probabilities
        pmf_3 = geometric_pmf(3, p)
        cdf_3 = geometric_cdf(3, p)
        
        # CDF should be >= PMF
        assert cdf_3 >= pmf_3
        
        # Check expectations
        mean = expected_value_geometric(p)
        var = variance_geometric(p)
        assert mean == pytest.approx(4.0)
        assert var > 0

    def test_exponential_workflow(self):
        """Test complete exponential distribution workflow."""
        lam = 0.5
        x = 2
        
        # Calculate probabilities
        pdf_x = exponential_pdf(x, lam)
        cdf_x = exponential_cdf(x, lam)
        
        # PDF should be non-negative
        assert pdf_x >= 0
        # CDF should be between 0 and 1
        assert 0 <= cdf_x <= 1
        
        # Check expectations
        mean = expected_value_exponential(lam)
        var = variance_exponential(lam)
        assert mean == pytest.approx(2.0)
        assert var == pytest.approx(4.0)

    def test_distribution_comparison(self):
        """Compare properties of different distributions."""
        # Poisson with λ=4
        poisson_mean = expected_value_poisson(4)
        poisson_var = variance_poisson(4)
        
        # Geometric with p=0.25
        geom_mean = expected_value_geometric(0.25)
        geom_var = variance_geometric(0.25)
        
        # Exponential with λ=0.25
        exp_mean = expected_value_exponential(0.25)
        exp_var = variance_exponential(0.25)
        
        # All should have positive mean and variance
        assert all(m > 0 for m in [poisson_mean, geom_mean, exp_mean])
        assert all(v > 0 for v in [poisson_var, geom_var, exp_var])
