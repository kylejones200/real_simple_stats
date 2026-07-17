"""Tests for the self-explaining statistical result wrappers added in Phase 5.

Covers the six new explained functions:
  - one_way_anova_explained
  - chi_square_independence_explained
  - difference_in_differences_explained
  - kaplan_meier_explained
  - morans_i_explained
  - detect_change_points_explained
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless matplotlib for all plot tests

from real_simple_stats.explain import (
    ExplainedResult,
    chi_square_independence_explained,
    detect_change_points_explained,
    difference_in_differences_explained,
    kaplan_meier_explained,
    morans_i_explained,
    one_way_anova_explained,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_explained_result(r: ExplainedResult) -> None:
    """Common structural checks that apply to every ExplainedResult."""
    assert isinstance(r, ExplainedResult)
    assert r.title
    assert r.question
    assert r.intuition
    assert r.interpretation
    assert isinstance(r.values, dict)
    assert len(r.values) >= 1
    text = str(r)
    assert "QUESTION" in text
    assert "WHAT THE TEST IS DOING" in text
    assert "INTERPRETATION" in text


# ---------------------------------------------------------------------------
# one_way_anova_explained
# ---------------------------------------------------------------------------


class TestOneWayAnovaExplained:
    def _groups(self, seed=0):
        rng = np.random.default_rng(seed)
        return (
            rng.normal(0, 1, 40),
            rng.normal(2, 1, 40),
            rng.normal(4, 1, 40),
        )

    def test_returns_explained_result(self):
        r = one_way_anova_explained(*self._groups())
        _assert_explained_result(r)

    def test_title(self):
        r = one_way_anova_explained(*self._groups())
        assert "ANOVA" in r.title

    def test_values_keys(self):
        r = one_way_anova_explained(*self._groups())
        for k in ("f_stat", "p_value", "eta_squared", "reject_null", "n_groups", "n_total"):
            assert k in r.values

    def test_attribute_access(self):
        r = one_way_anova_explained(*self._groups())
        assert r.f_stat > 0
        assert 0 <= r.p_value <= 1
        assert 0 <= r.eta_squared <= 1

    def test_reject_when_means_differ(self):
        r = one_way_anova_explained(*self._groups())
        assert r.reject_null is True
        assert r.decision == "Reject H₀"

    def test_fail_to_reject_same_mean(self):
        rng = np.random.default_rng(1)
        g1, g2 = rng.normal(5, 1, 60), rng.normal(5, 1, 60)
        r = one_way_anova_explained(g1, g2)
        assert r.reject_null is False
        assert "Fail to reject" in r.decision

    def test_caveats_mention_post_hoc(self):
        r = one_way_anova_explained(*self._groups())
        combined = " ".join(r.caveats)
        assert "post-hoc" in combined.lower() or "post hoc" in combined.lower()

    def test_n_groups_in_values(self):
        r = one_way_anova_explained(*self._groups())
        assert r.n_groups == 3

    def test_next_steps_mention_plot(self):
        r = one_way_anova_explained(*self._groups())
        assert any("plot()" in s for s in r.next_steps)

    def test_plot_returns_figure(self):
        import matplotlib.pyplot as plt
        r = one_way_anova_explained(*self._groups())
        fig = r.plot()
        assert fig is not None
        plt.close("all")

    def test_print_output_is_string(self):
        r = one_way_anova_explained(*self._groups())
        out = str(r)
        assert isinstance(out, str)
        assert len(out) > 100

    def test_markdown_repr(self):
        r = one_way_anova_explained(*self._groups())
        md = r._repr_markdown_()
        assert "###" in md
        assert "ANOVA" in md


# ---------------------------------------------------------------------------
# chi_square_independence_explained
# ---------------------------------------------------------------------------


class TestChiSquareIndependenceExplained:
    _ASSOC = [[40, 5], [5, 40]]
    _INDEP = [[25, 25], [25, 25]]

    def test_returns_explained_result(self):
        r = chi_square_independence_explained(self._ASSOC)
        _assert_explained_result(r)

    def test_title(self):
        r = chi_square_independence_explained(self._ASSOC)
        assert "Chi" in r.title or "chi" in r.title.lower()

    def test_values_keys(self):
        r = chi_square_independence_explained(self._ASSOC)
        for k in ("chi2", "p_value", "cramers_v", "reject_null", "dof"):
            assert k in r.values

    def test_attribute_access(self):
        r = chi_square_independence_explained(self._ASSOC)
        assert r.chi2 >= 0
        assert 0 <= r.p_value <= 1
        assert 0 <= r.cramers_v <= 1

    def test_reject_for_strong_association(self):
        r = chi_square_independence_explained(self._ASSOC)
        assert r.reject_null is True

    def test_no_reject_for_independence(self):
        r = chi_square_independence_explained(self._INDEP)
        assert r.reject_null is False

    def test_caveats_mention_causation(self):
        r = chi_square_independence_explained(self._ASSOC)
        combined = " ".join(r.caveats).lower()
        assert "causation" in combined or "causal" in combined

    def test_cramers_v_in_interpretation(self):
        r = chi_square_independence_explained(self._ASSOC)
        assert "V" in r.interpretation or "Cramér" in r.interpretation or "cramers" in r.interpretation.lower()

    def test_plot_returns_figure(self):
        import matplotlib.pyplot as plt
        r = chi_square_independence_explained([[10, 20], [30, 40]])
        fig = r.plot()
        assert fig is not None
        plt.close("all")

    def test_3x3_table(self):
        table = [[10, 5, 2], [3, 15, 4], [1, 3, 20]]
        r = chi_square_independence_explained(table)
        _assert_explained_result(r)
        assert r.dof == 4


# ---------------------------------------------------------------------------
# difference_in_differences_explained
# ---------------------------------------------------------------------------


class TestDiDExplained:
    def _data(self):
        rng = np.random.default_rng(0)
        n_each = 30
        # Control group: pre ~ 100, post ~ 102
        ctrl_pre = rng.normal(100, 3, n_each)
        ctrl_post = rng.normal(102, 3, n_each)
        # Treated group: pre ~ 100, post ~ 110 (treatment effect ≈ 8)
        trt_pre = rng.normal(100, 3, n_each)
        trt_post = rng.normal(110, 3, n_each)
        outcome = np.concatenate([ctrl_pre, ctrl_post, trt_pre, trt_post])
        post = np.array([0] * n_each + [1] * n_each + [0] * n_each + [1] * n_each)
        treated = np.array([0] * (2 * n_each) + [1] * (2 * n_each))
        return outcome, post, treated

    def test_returns_explained_result(self):
        y, post, treated = self._data()
        r = difference_in_differences_explained(y, post, treated)
        _assert_explained_result(r)

    def test_title(self):
        y, post, treated = self._data()
        r = difference_in_differences_explained(y, post, treated)
        assert "Difference" in r.title

    def test_values_keys(self):
        y, post, treated = self._data()
        r = difference_in_differences_explained(y, post, treated)
        for k in ("did_estimate", "se", "t_stat", "p_value", "ci", "reject_null"):
            assert k in r.values

    def test_attribute_access(self):
        y, post, treated = self._data()
        r = difference_in_differences_explained(y, post, treated)
        assert abs(r.did_estimate - 8.0) < 2.0
        assert r.reject_null is True

    def test_caveats_mention_parallel_trends(self):
        y, post, treated = self._data()
        r = difference_in_differences_explained(y, post, treated)
        combined = " ".join(r.caveats).lower()
        assert "parallel" in combined or "trends" in combined

    def test_plot_returns_figure(self):
        import matplotlib.pyplot as plt
        y, post, treated = self._data()
        r = difference_in_differences_explained(y, post, treated)
        fig = r.plot()
        assert fig is not None
        plt.close("all")

    def test_decision_in_str(self):
        y, post, treated = self._data()
        r = difference_in_differences_explained(y, post, treated)
        out = str(r)
        assert r.decision in out


# ---------------------------------------------------------------------------
# kaplan_meier_explained
# ---------------------------------------------------------------------------


class TestKaplanMeierExplained:
    def _data(self, seed=0):
        rng = np.random.default_rng(seed)
        n = 50
        durations = rng.exponential(scale=10, size=n)
        observed = (rng.uniform(size=n) < 0.7).astype(int)
        return durations.tolist(), observed.tolist()

    def test_returns_explained_result(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        _assert_explained_result(r)

    def test_title(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        assert "Kaplan" in r.title or "Survival" in r.title

    def test_values_keys(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        for k in ("n", "n_events", "n_censored", "final_survival_prob"):
            assert k in r.values

    def test_n_events_plus_censored_equals_n(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        assert r.n_events + r.n_censored == r.n

    def test_final_survival_in_range(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        assert 0 <= r.final_survival_prob <= 1

    def test_no_decision_field(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        assert r.decision is None

    def test_caveats_mention_censoring(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        combined = " ".join(r.caveats).lower()
        assert "censor" in combined

    def test_plot_returns_figure(self):
        import matplotlib.pyplot as plt
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        fig, ax = r.plot()
        assert fig is not None
        plt.close("all")

    def test_next_steps_mention_parametric(self):
        d, e = self._data()
        r = kaplan_meier_explained(d, e)
        combined = " ".join(r.next_steps).lower()
        assert "parametric" in combined or "weibull" in combined


# ---------------------------------------------------------------------------
# morans_i_explained
# ---------------------------------------------------------------------------


class TestMoransIExplained:
    def _clustered(self, seed=0):
        rng = np.random.default_rng(seed)
        n = 60
        x = np.concatenate([rng.uniform(0, 30, n // 2), rng.uniform(70, 100, n // 2)])
        y = rng.uniform(0, 100, n)
        v = np.concatenate([rng.normal(10, 1, n // 2), rng.normal(0, 1, n // 2)])
        return x, y, v

    def _random(self, seed=1):
        rng = np.random.default_rng(seed)
        n = 60
        return rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.normal(0, 1, n)

    def test_returns_explained_result(self):
        x, y, v = self._clustered()
        r = morans_i_explained(x, y, v, distance_threshold=40)
        _assert_explained_result(r)

    def test_title(self):
        x, y, v = self._clustered()
        r = morans_i_explained(x, y, v)
        assert "Moran" in r.title

    def test_values_keys(self):
        x, y, v = self._clustered()
        r = morans_i_explained(x, y, v)
        for k in ("moran_i", "expected_i", "z_score", "p_value", "n"):
            assert k in r.values

    def test_attribute_access(self):
        x, y, v = self._clustered()
        r = morans_i_explained(x, y, v, distance_threshold=40)
        assert np.isfinite(r.moran_i)
        assert 0 <= r.p_value <= 1

    def test_clustered_positive_i(self):
        x, y, v = self._clustered()
        r = morans_i_explained(x, y, v, distance_threshold=40)
        assert r.moran_i > 0

    def test_caveats_mention_local(self):
        x, y, v = self._clustered()
        r = morans_i_explained(x, y, v)
        combined = " ".join(r.caveats).lower()
        assert "local" in combined or "lisa" in combined

    def test_no_decision_field(self):
        x, y, v = self._random()
        r = morans_i_explained(x, y, v)
        assert r.decision is None

    def test_plot_returns_figure(self):
        import matplotlib.pyplot as plt
        x, y, v = self._clustered()
        r = morans_i_explained(x, y, v)
        fig = r.plot()
        assert fig is not None
        plt.close("all")

    def test_next_steps_mention_variogram(self):
        x, y, v = self._random()
        r = morans_i_explained(x, y, v)
        combined = " ".join(r.next_steps).lower()
        assert "variogram" in combined


# ---------------------------------------------------------------------------
# detect_change_points_explained
# ---------------------------------------------------------------------------


class TestDetectChangePointsExplained:
    def _step_series(self):
        return [0.0] * 20 + [5.0] * 20

    def test_returns_explained_result(self):
        r = detect_change_points_explained(self._step_series())
        _assert_explained_result(r)

    def test_title(self):
        r = detect_change_points_explained(self._step_series())
        assert "Change" in r.title

    def test_values_keys(self):
        r = detect_change_points_explained(self._step_series())
        for k in ("change_points", "n_breaks_found", "rss_reduction", "segment_means", "n"):
            assert k in r.values

    def test_detects_obvious_break(self):
        r = detect_change_points_explained(self._step_series())
        assert r.n_breaks_found == 1
        assert r.change_points == [20]

    def test_segment_means_correct(self):
        r = detect_change_points_explained(self._step_series())
        means = r.segment_means
        assert means[0] == pytest.approx(0.0)
        assert means[1] == pytest.approx(5.0)

    def test_rss_reduction_positive(self):
        r = detect_change_points_explained(self._step_series())
        assert r.rss_reduction > 0

    def test_decision_contains_break_info(self):
        r = detect_change_points_explained(self._step_series())
        assert r.decision is not None
        assert "1" in str(r.decision)

    def test_no_break_when_constant(self):
        data = [3.0] * 30
        r = detect_change_points_explained(data)
        assert "change_points" in r.values

    def test_caveats_mention_greedy(self):
        r = detect_change_points_explained(self._step_series())
        combined = " ".join(r.caveats).lower()
        assert "greedy" in combined

    def test_next_steps_mention_domain(self):
        r = detect_change_points_explained(self._step_series())
        combined = " ".join(r.next_steps).lower()
        assert "event" in combined or "domain" in combined or "known" in combined

    def test_plot_returns_figure(self):
        import matplotlib.pyplot as plt
        r = detect_change_points_explained(self._step_series())
        fig = r.plot()
        assert fig is not None
        plt.close("all")

    def test_two_breaks(self):
        data = [0.0] * 20 + [5.0] * 20 + [0.0] * 20
        r = detect_change_points_explained(data, n_breaks=2)
        assert r.n_breaks_found == 2
        assert len(r.segment_means) == 3

    def test_n_equals_data_length(self):
        data = self._step_series()
        r = detect_change_points_explained(data)
        assert r.n == len(data)
