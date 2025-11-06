"""Tests for power analysis and sample size calculations."""

import pytest
from real_simple_stats.power_analysis import (
    power_t_test,
    power_proportion_test,
    power_anova,
    power_correlation,
    minimum_detectable_effect,
    sample_size_summary,
)


class TestPowerTTest:
    def test_calculate_sample_size(self):
        result = power_t_test(delta=0.5, power=0.8, sig_level=0.05)
        assert result["n"] > 0
        assert result["delta"] == 0.5
        assert result["power"] == 0.8

    def test_calculate_power(self):
        result = power_t_test(n=50, delta=0.5, sig_level=0.05)
        assert 0 < result["power"] < 1
        assert result["n"] == 50

    def test_calculate_effect_size(self):
        result = power_t_test(n=50, power=0.8, sig_level=0.05)
        assert result["delta"] > 0
        assert result["n"] == 50

    def test_invalid_parameters(self):
        # Must provide exactly 3 parameters
        with pytest.raises(ValueError):
            power_t_test(n=50, delta=0.5, power=0.8)
        with pytest.raises(ValueError):
            power_t_test()


class TestPowerProportionTest:
    def test_calculate_sample_size(self):
        result = power_proportion_test(p1=0.6, p2=0.5, power=0.8)
        assert result["n"] > 0
        assert result["p1"] == 0.6

    def test_calculate_power(self):
        result = power_proportion_test(n=100, p1=0.6, p2=0.5)
        assert 0 < result["power"] < 1

    def test_calculate_proportion(self):
        result = power_proportion_test(n=100, p2=0.5, power=0.8)
        assert 0 < result["p1"] < 1

    def test_invalid_proportion(self):
        with pytest.raises(ValueError):
            power_proportion_test(p1=0.6, p2=1.5, power=0.8)


class TestPowerANOVA:
    def test_calculate_sample_size(self):
        result = power_anova(n_groups=3, effect_size=0.25, power=0.8)
        assert result["n_per_group"] > 0
        assert result["n_groups"] == 3

    def test_calculate_power(self):
        result = power_anova(n_groups=3, n_per_group=30, effect_size=0.25)
        assert 0 < result["power"] < 1

    def test_calculate_effect_size(self):
        result = power_anova(n_groups=3, n_per_group=30, power=0.8)
        assert result["effect_size"] > 0

    def test_invalid_groups(self):
        with pytest.raises(ValueError):
            power_anova(n_groups=1, effect_size=0.25, power=0.8)


class TestPowerCorrelation:
    def test_calculate_sample_size(self):
        result = power_correlation(r=0.3, power=0.8)
        assert result["n"] > 0
        assert result["r"] == 0.3

    def test_calculate_power(self):
        result = power_correlation(n=100, r=0.3)
        assert 0 < result["power"] < 1

    def test_calculate_correlation(self):
        result = power_correlation(n=100, power=0.8)
        assert 0 < abs(result["r"]) < 1

    def test_invalid_correlation(self):
        with pytest.raises(ValueError):
            power_correlation(r=1.5, power=0.8)


class TestMinimumDetectableEffect:
    def test_t_test_mde(self):
        mde = minimum_detectable_effect(50, test_type="t-test")
        assert mde > 0

    def test_proportion_mde(self):
        mde = minimum_detectable_effect(100, test_type="proportion")
        assert mde > 0

    def test_correlation_mde(self):
        mde = minimum_detectable_effect(100, test_type="correlation")
        assert 0 < mde < 1

    def test_invalid_test_type(self):
        with pytest.raises(ValueError):
            minimum_detectable_effect(50, test_type="unknown")

    def test_invalid_sample_size(self):
        with pytest.raises(ValueError):
            minimum_detectable_effect(1, test_type="t-test")


class TestSampleSizeSummary:
    def test_sample_size_summary(self):
        summary = sample_size_summary(0.5, power=0.8)
        assert "t_test_per_group" in summary
        assert "anova_3groups_per_group" in summary
        assert all(n > 0 for n in summary.values())

    def test_small_effect_size(self):
        summary = sample_size_summary(0.2, power=0.8)
        # Small effect size should require larger sample
        assert summary["t_test_per_group"] > 100
