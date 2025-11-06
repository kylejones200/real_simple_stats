"""Tests for effect size calculations."""

import pytest
from real_simple_stats.effect_sizes import (
    cohens_d,
    hedges_g,
    eta_squared,
    cramers_v,
    phi_coefficient,
    odds_ratio,
    relative_risk,
    cohens_h,
    interpret_effect_size,
)


class TestCohensD:
    def test_cohens_d_equal_groups(self):
        group1 = [1, 2, 3, 4, 5]
        group2 = [1, 2, 3, 4, 5]
        d = cohens_d(group1, group2)
        assert d == pytest.approx(0.0)

    def test_cohens_d_different_groups(self):
        group1 = [1, 2, 3, 4, 5]
        group2 = [3, 4, 5, 6, 7]
        d = cohens_d(group1, group2)
        assert d != 0

    def test_cohens_d_pooled(self):
        group1 = [1, 2, 3]
        group2 = [4, 5, 6]
        d = cohens_d(group1, group2, pooled=True)
        assert abs(d) > 0

    def test_invalid_groups(self):
        with pytest.raises(ValueError):
            cohens_d([1], [2, 3])


class TestHedgesG:
    def test_hedges_g(self):
        group1 = [1, 2, 3, 4, 5]
        group2 = [3, 4, 5, 6, 7]
        g = hedges_g(group1, group2)
        d = cohens_d(group1, group2)
        # Hedges' g should be slightly smaller than Cohen's d
        assert abs(g) < abs(d)


class TestEtaSquared:
    def test_eta_squared_three_groups(self):
        groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        eta2 = eta_squared(groups)
        assert 0 <= eta2 <= 1

    def test_eta_squared_identical_groups(self):
        groups = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        eta2 = eta_squared(groups)
        assert eta2 == pytest.approx(0.0)

    def test_invalid_groups(self):
        with pytest.raises(ValueError):
            eta_squared([[1, 2, 3]])


class TestCramersV:
    def test_cramers_v_2x2(self):
        table = [[10, 20], [30, 40]]
        v = cramers_v(table)
        assert 0 <= v <= 1

    def test_cramers_v_3x3(self):
        table = [[10, 20, 30], [15, 25, 35], [20, 30, 40]]
        v = cramers_v(table)
        assert 0 <= v <= 1

    def test_invalid_table(self):
        with pytest.raises(ValueError):
            cramers_v([[10]])


class TestPhiCoefficient:
    def test_phi_coefficient(self):
        table = [[10, 20], [30, 40]]
        phi = phi_coefficient(table)
        assert -1 <= phi <= 1

    def test_invalid_table(self):
        with pytest.raises(ValueError):
            phi_coefficient([[10, 20, 30], [40, 50, 60]])


class TestOddsRatio:
    def test_odds_ratio(self):
        table = [[10, 20], [30, 40]]
        or_value, ci = odds_ratio(table)
        assert or_value > 0
        assert ci[0] < or_value < ci[1]

    def test_odds_ratio_with_zeros(self):
        table = [[0, 20], [30, 40]]
        or_value, ci = odds_ratio(table)
        # Should handle zeros with continuity correction
        assert or_value > 0


class TestRelativeRisk:
    def test_relative_risk(self):
        table = [[10, 90], [30, 70]]
        rr, ci = relative_risk(table)
        assert rr > 0
        assert ci[0] < rr < ci[1]

    def test_invalid_table(self):
        with pytest.raises(ValueError):
            relative_risk([[10, 20, 30], [40, 50, 60]])


class TestCohensH:
    def test_cohens_h_equal_proportions(self):
        h = cohens_h(0.5, 0.5)
        assert h == pytest.approx(0.0)

    def test_cohens_h_different_proportions(self):
        h = cohens_h(0.7, 0.5)
        assert h != 0

    def test_invalid_proportions(self):
        with pytest.raises(ValueError):
            cohens_h(1.5, 0.5)
        with pytest.raises(ValueError):
            cohens_h(0.5, -0.1)


class TestInterpretEffectSize:
    def test_interpret_cohens_d(self):
        assert interpret_effect_size(0.1, "d") == "negligible"
        assert interpret_effect_size(0.3, "d") == "small"
        assert interpret_effect_size(0.6, "d") == "medium"
        assert interpret_effect_size(1.0, "d") == "large"

    def test_interpret_correlation(self):
        assert interpret_effect_size(0.05, "r") == "negligible"
        assert interpret_effect_size(0.2, "r") == "small"
        assert interpret_effect_size(0.4, "r") == "medium"
        assert interpret_effect_size(0.6, "r") == "large"

    def test_interpret_eta_squared(self):
        assert interpret_effect_size(0.005, "eta_squared") == "negligible"
        assert interpret_effect_size(0.03, "eta_squared") == "small"
        assert interpret_effect_size(0.10, "eta_squared") == "medium"
        assert interpret_effect_size(0.20, "eta_squared") == "large"

    def test_unknown_measure(self):
        with pytest.raises(ValueError):
            interpret_effect_size(0.5, "unknown")
