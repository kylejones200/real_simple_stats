"""Tests for the functions the README documents.

Every one of these was documented in the README but did not exist. They are
thin wrappers over kernels the Rust backend already provides, so the tests here
check the wrapper contract -- argument handling, validation, return shape --
and pin the numbers against SciPy where SciPy has an equivalent.
"""

import math

import pytest

import real_simple_stats as rss

st = pytest.importorskip("scipy.stats")
np = pytest.importorskip("numpy")


@pytest.fixture
def rng():
    return np.random.default_rng(20260909)


# --------------------------------------------------------------- descriptive


class TestShapeStatistics:
    def test_skewness_matches_scipy(self, rng):
        data = rng.gamma(2, 2, 400).tolist()
        assert rss.skewness(data) == pytest.approx(st.skew(data), rel=1e-12)

    def test_kurtosis_matches_scipy(self, rng):
        data = rng.standard_t(6, 400).tolist()
        assert rss.kurtosis(data) == pytest.approx(st.kurtosis(data), rel=1e-12)

    def test_symmetric_data_has_zero_skew(self):
        assert rss.skewness([1, 2, 3, 4, 5]) == pytest.approx(0.0, abs=1e-12)

    def test_right_tail_gives_positive_skew(self):
        assert rss.skewness([1, 1, 1, 2, 10]) > 0

    def test_constant_input_is_rejected(self):
        with pytest.raises(ValueError, match="identical"):
            rss.skewness([3, 3, 3, 3])
        with pytest.raises(ValueError, match="identical"):
            rss.kurtosis([3, 3, 3, 3])

    def test_too_few_values_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            rss.skewness([1])
        with pytest.raises(ValueError, match="at least 2"):
            rss.kurtosis([1])


class TestOutlierDetection:
    def test_finds_an_obvious_outlier(self):
        assert rss.detect_outliers_iqr([10, 12, 11, 13, 12, 100]) == [100.0]

    def test_clean_data_has_none(self):
        assert rss.detect_outliers_iqr([1, 2, 3, 4]) == []

    def test_larger_multiplier_is_more_permissive(self):
        data = [10, 12, 11, 13, 12, 30]
        assert len(rss.detect_outliers_iqr(data, multiplier=1.5)) >= len(
            rss.detect_outliers_iqr(data, multiplier=3.0)
        )

    def test_preserves_input_order(self):
        assert rss.detect_outliers_iqr([100, 10, 12, 11, 13, 12, -80]) == [100.0, -80.0]

    def test_validation(self):
        with pytest.raises(ValueError, match="empty"):
            rss.detect_outliers_iqr([])
        with pytest.raises(ValueError, match="non-negative"):
            rss.detect_outliers_iqr([1, 2, 3], multiplier=-1)


# ---------------------------------------------------------- hypothesis tests


class TestTTests:
    def test_one_sample_matches_scipy(self, rng):
        data = rng.normal(0.4, 1, 40).tolist()
        got = rss.one_sample_t_test(data, mu=0.0)
        assert got == pytest.approx(tuple(st.ttest_1samp(data, 0.0)), rel=1e-12)

    def test_two_sample_matches_scipy(self, rng):
        a, b = rng.normal(0, 1, 30).tolist(), rng.normal(1, 1.4, 35).tolist()
        assert rss.two_sample_t_test(a, b) == pytest.approx(
            tuple(st.ttest_ind(a, b)), rel=1e-12
        )

    def test_welch_matches_scipy(self, rng):
        a, b = rng.normal(0, 1, 30).tolist(), rng.normal(1, 1.4, 35).tolist()
        assert rss.two_sample_t_test(a, b, equal_var=False) == pytest.approx(
            tuple(st.ttest_ind(a, b, equal_var=False)), rel=1e-12
        )

    def test_paired_matches_scipy(self, rng):
        before = rng.normal(0, 1, 25)
        after = before + rng.normal(0.5, 0.4, 25)
        assert rss.paired_t_test(before.tolist(), after.tolist()) == pytest.approx(
            tuple(st.ttest_rel(before, after)), rel=1e-12
        )

    def test_paired_equals_one_sample_on_differences(self, rng):
        before = rng.normal(0, 1, 25)
        after = before + rng.normal(0.5, 0.4, 25)
        diffs = (before - after).tolist()
        assert rss.paired_t_test(before.tolist(), after.tolist()) == pytest.approx(
            rss.one_sample_t_test(diffs, 0.0), rel=1e-12
        )

    def test_validation(self):
        with pytest.raises(ValueError, match="at least 2"):
            rss.one_sample_t_test([1.0], 0.0)
        with pytest.raises(ValueError, match="identical"):
            rss.one_sample_t_test([2.0, 2.0, 2.0], 0.0)
        with pytest.raises(ValueError, match="same length"):
            rss.paired_t_test([1, 2, 3], [1, 2])
        with pytest.raises(ValueError, match="at least 2"):
            rss.two_sample_t_test([1.0], [1.0, 2.0])


class TestZTests:
    def test_z_test_formula(self):
        z, p = rss.z_test([102, 98, 105, 101, 99], mu=100, sigma=15)
        expected = (101.0 - 100) / (15 / math.sqrt(5))
        assert z == pytest.approx(expected, rel=1e-12)
        assert p == pytest.approx(2 * st.norm.sf(abs(expected)), rel=1e-12)

    def test_one_proportion_formula(self):
        z, p = rss.one_proportion_z_test(p_hat=0.6, n=50, p0=0.5)
        expected = (0.6 - 0.5) / math.sqrt(0.5 * 0.5 / 50)
        assert z == pytest.approx(expected, rel=1e-12)
        assert p == pytest.approx(2 * st.norm.sf(abs(expected)), rel=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            rss.z_test([1, 2, 3], mu=0, sigma=0)
        with pytest.raises(ValueError, match="at least one value"):
            rss.z_test([], mu=0, sigma=1)
        with pytest.raises(ValueError, match="p_hat"):
            rss.one_proportion_z_test(p_hat=1.5, n=10, p0=0.5)
        with pytest.raises(ValueError, match="n must be positive"):
            rss.one_proportion_z_test(p_hat=0.5, n=0, p0=0.5)
        with pytest.raises(ValueError, match="p0"):
            rss.one_proportion_z_test(p_hat=0.5, n=10, p0=1.0)


class TestNonparametric:
    def test_mann_whitney_matches_scipy(self, rng):
        a, b = rng.normal(0, 1, 22).tolist(), rng.normal(0.9, 1.3, 27).tolist()
        got = rss.mann_whitney_u(a, b)
        ref = st.mannwhitneyu(a, b, method="asymptotic")
        assert got[0] == pytest.approx(ref.statistic, rel=1e-12)
        assert got[1] == pytest.approx(ref.pvalue, rel=1e-10)

    def test_mann_whitney_handles_ties(self):
        a = [1, 2, 2, 3, 3, 3, 4]
        b = [2, 3, 3, 4, 4, 5, 5]
        got = rss.mann_whitney_u(a, b)
        ref = st.mannwhitneyu(a, b, method="asymptotic")
        assert got[0] == pytest.approx(ref.statistic, rel=1e-12)
        assert got[1] == pytest.approx(ref.pvalue, rel=1e-10)

    def test_mann_whitney_fully_separated_groups(self):
        u, p = rss.mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert u == 0.0
        assert p < 0.05

    def test_wilcoxon_matches_scipy_with_correction(self, rng):
        before = rng.normal(0, 1, 24)
        after = before + rng.normal(0.6, 0.5, 24)
        got = rss.wilcoxon_signed_rank(before.tolist(), after.tolist())
        ref = st.wilcoxon(before, after, method="approx", correction=True)
        assert got[0] == pytest.approx(ref.statistic, rel=1e-12)
        assert got[1] == pytest.approx(ref.pvalue, rel=1e-10)

    def test_wilcoxon_without_correction_matches_scipy_default(self, rng):
        before = rng.normal(0, 1, 24)
        after = before + rng.normal(0.6, 0.5, 24)
        got = rss.wilcoxon_signed_rank(before.tolist(), after.tolist(), correction=False)
        ref = st.wilcoxon(before, after, method="approx")
        assert got[1] == pytest.approx(ref.pvalue, rel=1e-10)

    def test_wilcoxon_discards_zero_differences(self):
        # Two pairs are unchanged; only the other three carry information.
        w, p = rss.wilcoxon_signed_rank([1, 2, 3, 4, 5], [1, 3, 4, 4, 7])
        assert w == 0.0
        assert 0.0 <= p <= 1.0

    def test_validation(self):
        with pytest.raises(ValueError, match="non-empty"):
            rss.mann_whitney_u([], [1, 2])
        with pytest.raises(ValueError, match="same length"):
            rss.wilcoxon_signed_rank([1, 2, 3], [1, 2])
        with pytest.raises(ValueError, match="all differences are zero|undefined"):
            rss.wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])


class TestAverageRanks:
    def test_matches_scipy_rankdata(self):
        from real_simple_stats.hypothesis_testing import _average_ranks

        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0]
        assert _average_ranks(values) == pytest.approx(list(st.rankdata(values)))

    def test_all_tied(self):
        from real_simple_stats.hypothesis_testing import _average_ranks

        assert _average_ranks([2.0, 2.0, 2.0]) == [2.0, 2.0, 2.0]


# -------------------------------------------------------------- correlation


class TestSpearman:
    def test_matches_scipy(self, rng):
        x = rng.normal(0, 1, 50).tolist()
        y = (np.array(x) ** 3 + rng.normal(0, 0.5, 50)).tolist()
        assert rss.spearman_correlation(x, y) == pytest.approx(
            st.spearmanr(x, y).statistic, rel=1e-12
        )

    def test_matches_scipy_with_ties(self):
        x = [1, 2, 2, 3, 4, 4, 4, 5]
        y = [5, 5, 6, 7, 8, 8, 9, 10]
        assert rss.spearman_correlation(x, y) == pytest.approx(
            st.spearmanr(x, y).statistic, rel=1e-12
        )

    def test_perfect_monotonic_but_nonlinear_is_exactly_one(self):
        # The point of a rank correlation: y = x^2 is monotonic, not linear.
        assert rss.spearman_correlation([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == 1.0
        assert rss.spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0

    def test_validation(self):
        with pytest.raises(ValueError, match="same length"):
            rss.spearman_correlation([1, 2, 3], [1, 2])
        with pytest.raises(ValueError, match="at least 2"):
            rss.spearman_correlation([1], [1])


class TestResiduals:
    def test_observed_minus_predicted(self):
        assert rss.calculate_residuals([2.0, 4.0, 6.0], [2.5, 3.5, 6.0]) == pytest.approx(
            [-0.5, 0.5, 0.0]
        )

    def test_perfect_fit_gives_zeros(self):
        assert rss.calculate_residuals([1, 2, 3], [1, 2, 3]) == [0.0, 0.0, 0.0]

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            rss.calculate_residuals([1, 2], [1])


# -------------------------------------------------------------- probability


class TestSimpleProbability:
    def test_basic(self):
        assert rss.simple_probability(favorable=3, total=10) == 0.3
        assert rss.simple_probability(0, 5) == 0.0
        assert rss.simple_probability(5, 5) == 1.0

    def test_validation(self):
        with pytest.raises(ValueError, match="total must be positive"):
            rss.simple_probability(1, 0)
        with pytest.raises(ValueError, match="non-negative"):
            rss.simple_probability(-1, 10)
        with pytest.raises(ValueError, match="cannot exceed"):
            rss.simple_probability(11, 10)


class TestBinomialCdf:
    def test_matches_scipy(self):
        for n, p in [(10, 0.5), (25, 0.3), (40, 0.85)]:
            for k in range(0, n + 1, 3):
                assert rss.binomial_cdf(n, k, p) == pytest.approx(
                    st.binom.cdf(k, n, p), rel=1e-11, abs=1e-15
                )

    def test_boundaries(self):
        assert rss.binomial_cdf(n=5, k=5, p=0.3) == pytest.approx(1.0)
        assert rss.binomial_cdf(n=5, k=-1, p=0.3) == 0.0

    def test_agrees_with_summed_pmf(self):
        total = sum(rss.binomial_probability(10, k, 0.4) for k in range(4))
        assert rss.binomial_cdf(10, 3, 0.4) == pytest.approx(total, rel=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="n must be non-negative"):
            rss.binomial_cdf(-1, 0, 0.5)
        with pytest.raises(ValueError, match="p must be between"):
            rss.binomial_cdf(10, 5, 1.5)
