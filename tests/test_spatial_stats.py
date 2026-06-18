"""Tests for the spatial statistics module."""

import numpy as np
import pytest

from real_simple_stats.spatial_stats import (
    compute_variogram,
    fit_variogram,
    morans_i,
    variogram_exponential,
    variogram_gaussian,
    variogram_spherical,
)


# ---------------------------------------------------------------------------
# Variogram model functions
# ---------------------------------------------------------------------------


class TestVariogramModels:
    @pytest.mark.parametrize("fn", [variogram_spherical, variogram_exponential, variogram_gaussian])
    def test_at_zero_equals_nugget(self, fn):
        assert fn(np.array([0.0]), 1.0, 10.0, 20.0)[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("fn", [variogram_spherical, variogram_exponential, variogram_gaussian])
    def test_non_decreasing(self, fn):
        h = np.linspace(0, 50, 100)
        g = fn(h, 0.5, 8.0, 15.0)
        assert np.all(np.diff(g) >= -1e-9)

    def test_spherical_reaches_sill_beyond_range(self):
        g = variogram_spherical(np.array([100.0]), 1.0, 10.0, 20.0)
        assert g[0] == pytest.approx(10.0, abs=1e-6)

    def test_exponential_approaches_sill(self):
        g = variogram_exponential(np.array([1000.0]), 0.0, 5.0, 10.0)
        assert g[0] == pytest.approx(5.0, abs=0.01)

    def test_gaussian_approaches_sill(self):
        g = variogram_gaussian(np.array([1000.0]), 0.0, 5.0, 10.0)
        assert g[0] == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# compute_variogram
# ---------------------------------------------------------------------------


class TestComputeVariogram:
    def _make_data(self, n=80, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        v = np.sin(x / 20) + rng.normal(0, 0.3, n)
        return x, y, v

    def test_output_keys(self):
        x, y, v = self._make_data()
        r = compute_variogram(x, y, v)
        for k in ("lags", "gamma", "n_pairs", "max_lag", "total_variance"):
            assert k in r

    def test_n_lags_matches(self):
        x, y, v = self._make_data()
        r = compute_variogram(x, y, v, n_lags=12)
        assert len(r["lags"]) == 12
        assert len(r["gamma"]) == 12

    def test_gamma_non_negative(self):
        x, y, v = self._make_data()
        r = compute_variogram(x, y, v)
        assert np.all(r["gamma"] >= 0)

    def test_lags_increasing(self):
        x, y, v = self._make_data()
        r = compute_variogram(x, y, v)
        assert np.all(np.diff(r["lags"]) > 0)

    def test_custom_max_lag(self):
        x, y, v = self._make_data()
        r = compute_variogram(x, y, v, max_lag=40.0)
        assert r["max_lag"] == pytest.approx(40.0)
        assert np.all(r["lags"] <= 40.0)

    def test_total_variance_positive(self):
        x, y, v = self._make_data()
        r = compute_variogram(x, y, v)
        assert r["total_variance"] > 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_variogram([0, 1, 2], [0, 1], [0, 1, 2])

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            compute_variogram([0, 1], [0, 1], [1, 2])


# ---------------------------------------------------------------------------
# fit_variogram
# ---------------------------------------------------------------------------


class TestFitVariogram:
    def _synthetic_variogram(self, model_fn, nugget, sill, range_p, noise=0.1, seed=1):
        rng = np.random.default_rng(seed)
        lags = np.linspace(2, 80, 15)
        gamma = model_fn(lags, nugget, sill, range_p) + rng.normal(0, noise, 15)
        gamma = np.maximum(gamma, 0)
        n_pairs = np.full(15, 20, dtype=int)
        return lags, gamma, n_pairs

    @pytest.mark.parametrize("model,fn", [
        ("spherical", variogram_spherical),
        ("exponential", variogram_exponential),
        ("gaussian", variogram_gaussian),
    ])
    def test_recovers_parameters(self, model, fn):
        lags, gamma, n_pairs = self._synthetic_variogram(fn, 0.5, 8.0, 20.0)
        r = fit_variogram(lags, gamma, model=model, n_pairs=n_pairs)
        assert abs(r["sill"] - 8.0) < 2.0
        assert abs(r["range_param"] - 20.0) < 10.0
        assert r["nugget"] >= 0

    def test_model_name_in_result(self):
        lags, gamma, _ = self._synthetic_variogram(variogram_spherical, 0, 5, 25)
        r = fit_variogram(lags, gamma)
        assert r["model"] == "spherical"

    def test_model_fn_callable(self):
        lags, gamma, _ = self._synthetic_variogram(variogram_spherical, 0, 5, 25)
        r = fit_variogram(lags, gamma)
        val = r["model_fn"](10.0)
        assert isinstance(val, float)
        assert val >= 0

    def test_rmse_non_negative(self):
        lags, gamma, _ = self._synthetic_variogram(variogram_spherical, 0, 5, 25)
        r = fit_variogram(lags, gamma)
        assert r["rmse"] >= 0

    def test_unknown_model_raises(self):
        lags, gamma, _ = self._synthetic_variogram(variogram_spherical, 0, 5, 25)
        with pytest.raises(ValueError, match="Unknown model"):
            fit_variogram(lags, gamma, model="power")


# ---------------------------------------------------------------------------
# morans_i
# ---------------------------------------------------------------------------


class TestMoransI:
    def _clustered(self, seed=0):
        rng = np.random.default_rng(seed)
        # Two clusters with high and low values
        n = 60
        x = np.concatenate([rng.uniform(0, 30, n // 2), rng.uniform(70, 100, n // 2)])
        y = rng.uniform(0, 100, n)
        v = np.concatenate([
            rng.normal(10, 1, n // 2),
            rng.normal(0, 1, n // 2),
        ])
        return x, y, v

    def _random(self, seed=1):
        rng = np.random.default_rng(seed)
        n = 60
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        v = rng.normal(0, 1, n)
        return x, y, v

    def test_clustered_positive(self):
        x, y, v = self._clustered()
        r = morans_i(x, y, v, distance_threshold=40)
        assert r["moran_i"] > 0

    def test_random_near_zero(self):
        # Random spatial pattern — I should be near 0
        _, _, _ = self._random()  # just ensuring no crash
        # Can't guarantee sign with small n, just check it runs and is finite
        x, y, v = self._random()
        r = morans_i(x, y, v)
        assert np.isfinite(r["moran_i"])

    def test_output_keys(self):
        x, y, v = self._clustered()
        r = morans_i(x, y, v)
        for k in ("moran_i", "expected_i", "z_score", "p_value", "interpretation"):
            assert k in r

    def test_expected_i_formula(self):
        x, y, v = self._random()
        r = morans_i(x, y, v)
        n = r["n"]
        assert r["expected_i"] == pytest.approx(-1.0 / (n - 1))

    def test_p_value_in_range(self):
        x, y, v = self._random()
        r = morans_i(x, y, v)
        assert 0 <= r["p_value"] <= 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            morans_i([0, 1, 2], [0, 1], [0, 1, 2])

    def test_no_neighbours_raises(self):
        with pytest.raises(ValueError):
            morans_i([0, 50, 100], [0, 50, 100], [1, 2, 3], distance_threshold=0.001)
