"""Tests for the causal inference module."""

import numpy as np
import pytest

from real_simple_stats.causal_inference import (
    difference_in_differences,
    panel_fixed_effects,
    regression_discontinuity,
    synthetic_control,
)

# ---------------------------------------------------------------------------
# Difference-in-Differences
# ---------------------------------------------------------------------------


class TestDiD:
    def _make_data(self, did_effect=5.0, seed=0):
        rng = np.random.default_rng(seed)
        n = 200
        post = np.repeat([0, 0, 1, 1], n // 4)
        treated = np.tile([0, 1], n // 2)
        noise = rng.normal(0, 1, n)
        outcome = 10 + 2 * post + 3 * treated + did_effect * post * treated + noise
        return outcome, post, treated

    def test_recovers_true_effect(self):
        y, post, treated = self._make_data(did_effect=5.0)
        r = difference_in_differences(y, post, treated)
        assert abs(r["did_estimate"] - 5.0) < 0.5

    def test_zero_effect_not_significant(self):
        y, post, treated = self._make_data(did_effect=0.0)
        r = difference_in_differences(y, post, treated)
        assert not r["reject_null"]

    def test_nonzero_effect_significant(self):
        y, post, treated = self._make_data(did_effect=10.0)
        r = difference_in_differences(y, post, treated)
        assert r["reject_null"]

    def test_ci_contains_true_value(self):
        y, post, treated = self._make_data(did_effect=5.0)
        r = difference_in_differences(y, post, treated)
        lo, hi = r["ci"]
        assert lo < 5.0 < hi

    def test_coefficients_keys(self):
        y, post, treated = self._make_data()
        r = difference_in_differences(y, post, treated)
        for key in ("intercept", "post", "treated", "did"):
            assert key in r["coefficients"]

    def test_r_squared_between_zero_one(self):
        y, post, treated = self._make_data()
        r = difference_in_differences(y, post, treated)
        assert 0 <= r["r_squared"] <= 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            difference_in_differences([1, 2, 3], [0, 1], [0, 1, 0])

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            difference_in_differences([1, 2, 3], [0, 1, 1], [0, 0, 1])

    def test_exact_simple_case(self):
        # Exact case: control pre=100, post=102; treated pre=100, post=107
        # DiD = (107-100) - (102-100) = 5
        outcome = [100, 100, 102, 107]
        post    = [  0,   0,   1,   1]
        treated = [  0,   1,   0,   1]
        r = difference_in_differences(outcome, post, treated)
        assert r["did_estimate"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Regression Discontinuity
# ---------------------------------------------------------------------------


class TestRDD:
    def _make_data(self, effect=3.0, n=500, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3, 3, n)
        y = 1.0 + 0.5 * x + effect * (x >= 0) + rng.normal(0, 0.5, n)
        return y, x

    def test_recovers_effect(self):
        y, x = self._make_data(effect=3.0)
        r = regression_discontinuity(y, x, cutoff=0.0)
        assert abs(r["effect"] - 3.0) < 0.5

    def test_zero_effect_not_significant(self):
        y, x = self._make_data(effect=0.0)
        r = regression_discontinuity(y, x, cutoff=0.0)
        assert not r["reject_null"]

    def test_large_effect_significant(self):
        y, x = self._make_data(effect=5.0)
        r = regression_discontinuity(y, x, cutoff=0.0)
        assert r["reject_null"]

    def test_bandwidth_reduces_n_used(self):
        y, x = self._make_data()
        r_full = regression_discontinuity(y, x, cutoff=0.0)
        r_bw = regression_discontinuity(y, x, cutoff=0.0, bandwidth=1.0)
        assert r_bw["n_used"] < r_full["n_used"]
        assert r_bw["n_total"] == r_full["n_total"]

    def test_degree_2_still_recovers(self):
        y, x = self._make_data(effect=3.0, n=800)
        r = regression_discontinuity(y, x, cutoff=0.0, degree=2)
        assert abs(r["effect"] - 3.0) < 0.8

    def test_invalid_degree_raises(self):
        y, x = self._make_data()
        with pytest.raises(ValueError, match="degree"):
            regression_discontinuity(y, x, cutoff=0.0, degree=0)

    def test_result_keys(self):
        y, x = self._make_data()
        r = regression_discontinuity(y, x, cutoff=0.0)
        for key in ("effect", "se", "t_stat", "p_value", "ci", "n_used"):
            assert key in r


# ---------------------------------------------------------------------------
# Synthetic Control
# ---------------------------------------------------------------------------


class TestSyntheticControl:
    def _make_data(self, ate=2.0, T=30, N=10, n_pre=20, seed=42):
        rng = np.random.default_rng(seed)
        Y = rng.normal(size=(T, N))
        # Treated unit is the first column of Y plus a post-treatment effect
        y = Y[:, 0] + np.concatenate(
            [np.zeros(n_pre), np.full(T - n_pre, ate)]
        )
        return y, Y[:, 1:], n_pre

    def test_detects_positive_ate(self):
        y, Y_ctrl, n_pre = self._make_data(ate=2.0)
        r = synthetic_control(y, Y_ctrl, n_pre=n_pre)
        assert r["ate_post"] > 1.0

    def test_zero_effect_near_zero(self):
        y, Y_ctrl, n_pre = self._make_data(ate=0.0)
        r = synthetic_control(y, Y_ctrl, n_pre=n_pre)
        assert abs(r["ate_post"]) < 1.0

    def test_weights_sum_to_one(self):
        y, Y_ctrl, n_pre = self._make_data()
        r = synthetic_control(y, Y_ctrl, n_pre=n_pre)
        assert r["weights"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self):
        y, Y_ctrl, n_pre = self._make_data()
        r = synthetic_control(y, Y_ctrl, n_pre=n_pre)
        assert np.all(r["weights"] >= -1e-9)

    def test_pre_rmse_small_with_good_donors(self):
        rng = np.random.default_rng(0)
        T, N, n_pre = 40, 20, 30
        # Treated unit IS a linear combo of controls (perfect synthetic possible)
        Y = rng.normal(size=(T, N))
        w_true = np.zeros(N)
        w_true[:3] = [0.5, 0.3, 0.2]
        y = Y @ w_true
        r = synthetic_control(y, Y, n_pre=n_pre)
        assert r["pre_fit_rmse"] < 0.1

    def test_output_shapes(self):
        y, Y_ctrl, n_pre = self._make_data()
        T = len(y)
        r = synthetic_control(y, Y_ctrl, n_pre=n_pre)
        assert len(r["synthetic"]) == T
        assert len(r["gap"]) == T
        assert len(r["weights"]) == Y_ctrl.shape[1]

    def test_invalid_n_pre_raises(self):
        y, Y_ctrl, n_pre = self._make_data()
        with pytest.raises(ValueError, match="n_pre"):
            synthetic_control(y, Y_ctrl, n_pre=len(y))

    def test_shape_mismatch_raises(self):
        y, Y_ctrl, n_pre = self._make_data()
        with pytest.raises(ValueError):
            synthetic_control(y, Y_ctrl[:-1], n_pre=n_pre)


# ---------------------------------------------------------------------------
# Panel Fixed Effects
# ---------------------------------------------------------------------------


class TestPanelFE:
    def _make_data(self, slope=2.0, seed=0):
        rng = np.random.default_rng(seed)
        n_entities, n_periods = 30, 10
        entity = np.repeat(np.arange(n_entities), n_periods)
        x = rng.normal(size=n_entities * n_periods)
        entity_effect = np.repeat(rng.normal(0, 5, n_entities), n_periods)
        y = slope * x + entity_effect + rng.normal(0, 0.5, n_entities * n_periods)
        return y, x, entity

    def test_recovers_slope(self):
        y, x, entity = self._make_data(slope=2.0)
        r = panel_fixed_effects(y, x, entity)
        assert abs(r["coefficients"][0] - 2.0) < 0.3

    def test_significant_slope(self):
        y, x, entity = self._make_data(slope=2.0)
        r = panel_fixed_effects(y, x, entity)
        assert r["p_values"][0] < 0.05

    def test_n_entities_correct(self):
        y, x, entity = self._make_data()
        r = panel_fixed_effects(y, x, entity)
        assert r["n_entities"] == 30

    def test_df_residual(self):
        y, x, entity = self._make_data()
        r = panel_fixed_effects(y, x, entity)
        # df = n - k - n_entities = 300 - 1 - 30 = 269
        assert r["df_residual"] == 269

    def test_multiple_predictors(self):
        rng = np.random.default_rng(42)
        n_entities, n_periods = 20, 15
        entity = np.repeat(np.arange(n_entities), n_periods)
        X = rng.normal(size=(n_entities * n_periods, 2))
        y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + np.repeat(
            rng.normal(0, 3, n_entities), n_periods
        ) + rng.normal(size=n_entities * n_periods)
        r = panel_fixed_effects(y, X, entity)
        assert len(r["coefficients"]) == 2
        assert abs(r["coefficients"][0] - 1.5) < 0.4
        assert abs(r["coefficients"][1] + 0.8) < 0.4

    def test_length_mismatch_raises(self):
        y, x, entity = self._make_data()
        with pytest.raises(ValueError, match="same length"):
            panel_fixed_effects(y, x[:-1], entity)
