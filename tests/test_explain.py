"""Tests for the self-explaining results module."""

import matplotlib

matplotlib.use("Agg")

import pytest

from real_simple_stats.explain import (
    ExplainedResult,
    one_sample_t_test_explained,
)


@pytest.fixture
def significant_result():
    # Mean clearly above 5.0
    return one_sample_t_test_explained(
        [5.4, 5.6, 5.9, 5.5, 5.7, 5.8, 5.6, 5.5], mu=5.0
    )


@pytest.fixture
def null_result():
    # Mean centered on 5.0
    return one_sample_t_test_explained([4.9, 5.1, 5.0, 4.95, 5.05, 5.0], mu=5.0)


class TestComputation:
    def test_returns_explained_result(self, significant_result):
        assert isinstance(significant_result, ExplainedResult)

    def test_numeric_attribute_access(self, significant_result):
        # The result doubles as a data object.
        assert significant_result.statistic > 0
        assert 0 <= significant_result.p_value <= 1
        assert significant_result.df == 7
        assert significant_result.reject_null is True
        lo, hi = significant_result.ci
        assert lo < hi

    def test_matches_scipy(self):
        from scipy import stats

        data = [5.4, 5.6, 5.9, 5.5, 5.7, 5.8, 5.6, 5.5]
        r = one_sample_t_test_explained(data, mu=5.0)
        t_ref, p_ref = stats.ttest_1samp(data, 5.0)
        assert r.statistic == pytest.approx(t_ref, rel=1e-6)
        assert r.p_value == pytest.approx(p_ref, rel=1e-6)

    def test_one_sided_pvalues_complement(self):
        data = [5.4, 5.6, 5.9, 5.5, 5.7]
        g = one_sample_t_test_explained(data, mu=5.0, alternative="greater")
        less = one_sample_t_test_explained(data, mu=5.0, alternative="less")
        assert g.p_value + less.p_value == pytest.approx(1.0, abs=1e-9)

    def test_alternative_aliases(self):
        data = [5.4, 5.6, 5.9, 5.5, 5.7]
        a = one_sample_t_test_explained(data, mu=5.0, alternative="two-tailed")
        b = one_sample_t_test_explained(data, mu=5.0, alternative="two-sided")
        assert a.p_value == pytest.approx(b.p_value)


class TestGuards:
    def test_rejects_tiny_sample(self):
        with pytest.raises(ValueError, match="at least 2"):
            one_sample_t_test_explained([1.0], mu=0)

    def test_rejects_unknown_alternative(self):
        with pytest.raises(ValueError, match="Unknown alternative"):
            one_sample_t_test_explained([1, 2, 3], mu=0, alternative="sideways")

    def test_unknown_attribute_raises(self, significant_result):
        with pytest.raises(AttributeError):
            _ = significant_result.does_not_exist


class TestNarrative:
    def test_explain_has_all_sections(self, significant_result):
        text = significant_result.explain()
        for section in [
            "QUESTION",
            "RESULT",
            "WHAT THE TEST IS DOING",
            "ASSUMPTIONS",
            "INTERPRETATION",
            "WHAT THIS DOES *NOT* MEAN",
            "NEXT STEPS",
        ]:
            assert section in text

    def test_str_equals_explain(self, significant_result):
        assert str(significant_result) == significant_result.explain()

    def test_markdown_repr(self, significant_result):
        md = significant_result._repr_markdown_()
        assert md.startswith("### One-Sample t-Test")
        assert "| quantity | value |" in md

    def test_misconception_guard_present(self, significant_result):
        text = significant_result.explain()
        # The p-value misinterpretation guard must always appear.
        assert "NOT the probability that H" in text

    def test_failing_to_reject_caveat_only_when_not_rejecting(
        self, significant_result, null_result
    ):
        assert any("Failing to reject" in c for c in null_result.caveats)
        assert not any("Failing to reject" in c for c in significant_result.caveats)

    def test_tiny_pvalue_not_rendered_as_zero(self, significant_result):
        # p is ~1e-5 here; prose should say "< 0.0001", never "0.0000".
        text = significant_result.explain()
        assert "< 0.0001" in text
        assert "p = 0.0000" not in text


class TestPlot:
    def test_plot_returns_fig_ax(self, significant_result):
        fig, ax = significant_result.plot()
        assert fig is not None
        assert ax is not None
