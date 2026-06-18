"""Tests for the Phase 4 additions to hypothesis_testing.py.

Covers:
  - one_way_anova
  - chi_square_independence
"""

import numpy as np
import pytest

from real_simple_stats.hypothesis_testing import (
    chi_square_independence,
    one_way_anova,
)


# ---------------------------------------------------------------------------
# one_way_anova
# ---------------------------------------------------------------------------


class TestOneWayAnova:
    def test_reject_when_means_differ(self):
        rng = np.random.default_rng(0)
        g1 = rng.normal(0, 1, 40)
        g2 = rng.normal(2, 1, 40)
        g3 = rng.normal(4, 1, 40)
        r = one_way_anova(g1, g2, g3)
        assert r["reject_null"] is True

    def test_fail_to_reject_same_mean(self):
        rng = np.random.default_rng(1)
        g1 = rng.normal(5, 1, 50)
        g2 = rng.normal(5, 1, 50)
        r = one_way_anova(g1, g2)
        assert r["reject_null"] is False

    def test_output_keys(self):
        rng = np.random.default_rng(2)
        r = one_way_anova(rng.normal(0, 1, 20), rng.normal(1, 1, 20))
        expected = {
            "f_stat", "p_value", "df_between", "df_within",
            "eta_squared", "reject_null", "group_means", "group_ns",
            "n_groups", "n_total",
        }
        assert expected <= set(r.keys())

    def test_df_between(self):
        rng = np.random.default_rng(3)
        groups = [rng.normal(i, 1, 20) for i in range(4)]
        r = one_way_anova(*groups)
        assert r["df_between"] == 3  # k - 1 = 3

    def test_df_within(self):
        rng = np.random.default_rng(4)
        g1, g2 = rng.normal(0, 1, 15), rng.normal(1, 1, 25)
        r = one_way_anova(g1, g2)
        assert r["df_within"] == 38  # N - k = 40 - 2

    def test_n_groups_count(self):
        rng = np.random.default_rng(5)
        groups = [rng.normal(0, 1, 10) for _ in range(5)]
        r = one_way_anova(*groups)
        assert r["n_groups"] == 5

    def test_n_total(self):
        g1 = np.ones(10)
        g2 = np.ones(20)
        r = one_way_anova(g1, g2)
        assert r["n_total"] == 30

    def test_eta_squared_in_range(self):
        rng = np.random.default_rng(6)
        g1 = rng.normal(0, 1, 30)
        g2 = rng.normal(3, 1, 30)
        r = one_way_anova(g1, g2)
        assert 0.0 <= r["eta_squared"] <= 1.0

    def test_eta_squared_large_for_big_effect(self):
        g1 = np.zeros(50)
        g2 = np.ones(50) * 100
        r = one_way_anova(g1, g2)
        assert r["eta_squared"] > 0.9

    def test_f_stat_positive(self):
        rng = np.random.default_rng(7)
        r = one_way_anova(rng.normal(0, 1, 20), rng.normal(1, 1, 20))
        assert r["f_stat"] >= 0

    def test_group_means_match(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        r = one_way_anova(g1, g2)
        assert r["group_means"][0] == pytest.approx(2.0)
        assert r["group_means"][1] == pytest.approx(5.0)

    def test_fewer_than_two_groups_raises(self):
        with pytest.raises(ValueError):
            one_way_anova(np.array([1.0, 2.0, 3.0]))

    def test_group_with_one_obs_raises(self):
        with pytest.raises(ValueError):
            one_way_anova(np.array([1.0]), np.array([2.0, 3.0]))

    def test_accepts_lists(self):
        r = one_way_anova([1, 2, 3], [4, 5, 6])
        assert "f_stat" in r


# ---------------------------------------------------------------------------
# chi_square_independence
# ---------------------------------------------------------------------------


class TestChiSquareIndependence:
    def test_reject_associated_table(self):
        # Strong association
        table = [[40, 5], [5, 40]]
        r = chi_square_independence(table)
        assert r["reject_null"] is True

    def test_fail_to_reject_independent_table(self):
        # Uniform distribution — no association
        table = [[25, 25], [25, 25]]
        r = chi_square_independence(table)
        assert r["reject_null"] is False

    def test_output_keys(self):
        r = chi_square_independence([[10, 20], [30, 40]])
        expected = {
            "chi2", "p_value", "dof", "expected",
            "cramers_v", "reject_null", "low_expected_cells", "interpretation",
        }
        assert expected <= set(r.keys())

    def test_dof_correct(self):
        r = chi_square_independence([[10, 20, 30], [15, 25, 35]])
        assert r["dof"] == 2  # (2-1)*(3-1)

    def test_cramers_v_in_range(self):
        r = chi_square_independence([[30, 10], [10, 30]])
        assert 0.0 <= r["cramers_v"] <= 1.0

    def test_cramers_v_near_one_for_perfect_assoc(self):
        # Perfect dependence: one cell per row; scipy applies Yates' correction
        # so result is just below 1.0
        table = [[50, 0], [0, 50]]
        r = chi_square_independence(table)
        assert r["cramers_v"] >= 0.95

    def test_cramers_v_zero_for_no_assoc(self):
        table = [[25, 25], [25, 25]]
        r = chi_square_independence(table)
        assert r["cramers_v"] == pytest.approx(0.0, abs=1e-9)

    def test_expected_shape_matches_observed(self):
        table = [[10, 20, 15], [5, 10, 20]]
        r = chi_square_independence(table)
        assert r["expected"].shape == (2, 3)

    def test_expected_row_sums_match(self):
        table = [[10, 20], [30, 40]]
        r = chi_square_independence(table)
        obs = np.array(table, dtype=float)
        assert r["expected"].sum(axis=1) == pytest.approx(obs.sum(axis=1))

    def test_low_expected_cells_counted(self):
        # Very sparse table — some cells will have expected < 5
        table = [[1, 0], [0, 1]]
        r = chi_square_independence(table)
        assert r["low_expected_cells"] >= 0  # may be > 0

    def test_p_value_in_range(self):
        r = chi_square_independence([[10, 20], [30, 40]])
        assert 0.0 <= r["p_value"] <= 1.0

    def test_chi2_non_negative(self):
        r = chi_square_independence([[10, 20], [30, 40]])
        assert r["chi2"] >= 0.0

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError):
            chi_square_independence([[1, 2, 3]])

    def test_too_few_cols_raises(self):
        with pytest.raises(ValueError):
            chi_square_independence([[1], [2]])

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            chi_square_independence([1, 2, 3])

    def test_interpretation_contains_reject_word(self):
        table = [[40, 5], [5, 40]]
        r = chi_square_independence(table)
        assert "Reject" in r["interpretation"] or "reject" in r["interpretation"]

    def test_3x3_table(self):
        table = [[10, 5, 2], [3, 15, 4], [1, 3, 20]]
        r = chi_square_independence(table)
        assert r["dof"] == 4  # (3-1)*(3-1)
        assert r["cramers_v"] >= 0.0
