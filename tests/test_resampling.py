"""Tests for resampling methods module."""

import numpy as np
from real_simple_stats.resampling import (
    bootstrap,
    bootstrap_hypothesis_test,
    permutation_test,
    jackknife,
    cross_validate,
    stratified_split,
)


class TestBootstrap:
    def test_bootstrap_basic(self):
        data = [1, 2, 3, 4, 5]
        result = bootstrap(data, np.mean, n_iterations=100)

        assert "statistic" in result
        assert "confidence_interval" in result
        assert "bootstrap_distribution" in result

    def test_bootstrap_confidence_interval(self):
        data = [1, 2, 3, 4, 5]
        result = bootstrap(data, np.mean, n_iterations=100)

        lower, upper = result["confidence_interval"]
        assert lower < upper
        assert lower <= result["statistic"] <= upper

    def test_bootstrap_with_seed(self):
        data = [1, 2, 3, 4, 5]
        result1 = bootstrap(data, np.mean, n_iterations=100, random_seed=42)
        result2 = bootstrap(data, np.mean, n_iterations=100, random_seed=42)

        assert result1["statistic"] == result2["statistic"]


class TestBootstrapHypothesisTest:
    def test_bootstrap_hypothesis_test_basic(self):
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]

        def diff_means(d1, d2):
            return np.mean(d1) - np.mean(d2)

        result = bootstrap_hypothesis_test(data1, data2, diff_means, n_iterations=100)

        assert "observed_statistic" in result
        assert "p_value" in result
        assert "bootstrap_distribution" in result

    def test_bootstrap_hypothesis_test_p_value_range(self):
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]

        result = bootstrap_hypothesis_test(
            data1, data2, lambda d1, d2: np.mean(d1) - np.mean(d2), n_iterations=100
        )

        assert 0 <= result["p_value"] <= 1


class TestPermutationTest:
    def test_permutation_test_basic(self):
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]

        def diff_means(d1, d2):
            return np.mean(d1) - np.mean(d2)

        result = permutation_test(data1, data2, diff_means, n_permutations=100)

        assert "observed_statistic" in result
        assert "p_value" in result
        assert "permutation_distribution" in result

    def test_permutation_test_p_value_range(self):
        data1 = [1, 2, 3]
        data2 = [4, 5, 6]

        result = permutation_test(
            data1, data2, lambda d1, d2: np.mean(d1) - np.mean(d2), n_permutations=100
        )

        assert 0 <= result["p_value"] <= 1

    def test_permutation_test_alternatives(self):
        data1 = [1, 2, 3]
        data2 = [4, 5, 6]

        for alt in ["two-sided", "less", "greater"]:
            result = permutation_test(
                data1,
                data2,
                lambda d1, d2: np.mean(d1) - np.mean(d2),
                n_permutations=100,
                alternative=alt,
            )
            assert 0 <= result["p_value"] <= 1


class TestJackknife:
    def test_jackknife_basic(self):
        data = [1, 2, 3, 4, 5]
        result = jackknife(data, np.mean)

        assert "statistic" in result
        assert "bias" in result
        assert "std_error" in result
        assert "jackknife_values" in result

    def test_jackknife_values_length(self):
        data = [1, 2, 3, 4, 5]
        result = jackknife(data, np.mean)

        assert len(result["jackknife_values"]) == len(data)

    def test_jackknife_std_error_positive(self):
        data = [1, 2, 3, 4, 5]
        result = jackknife(data, np.mean)

        assert result["std_error"] >= 0


class TestCrossValidate:
    def test_cross_validate_basic(self):
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]

        def simple_model(X_train, y_train, X_test):
            # Simple mean predictor
            mean_y = np.mean(y_train)
            return [mean_y] * len(X_test)

        result = cross_validate(X, y, simple_model, k_folds=3)

        assert "scores" in result
        assert "mean_score" in result
        assert "std_score" in result

    def test_cross_validate_k_folds(self):
        X = [[1], [2], [3], [4], [5], [6]]
        y = [1, 2, 3, 4, 5, 6]

        def simple_model(X_train, y_train, X_test):
            return [np.mean(y_train)] * len(X_test)

        result = cross_validate(X, y, simple_model, k_folds=3)

        assert len(result["scores"]) == 3


class TestStratifiedSplit:
    def test_stratified_split_basic(self):
        X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_stratified_split_proportions(self):
        X = [[i] for i in range(100)]
        y = [0] * 50 + [1] * 50

        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)

        # Check that class proportions are maintained
        train_prop = sum(y_train) / len(y_train)
        test_prop = sum(y_test) / len(y_test)

        assert abs(train_prop - 0.5) < 0.1
        assert abs(test_prop - 0.5) < 0.1

    def test_stratified_split_with_seed(self):
        X = [[i] for i in range(20)]
        y = [0] * 10 + [1] * 10

        result1 = stratified_split(X, y, test_size=0.2, random_seed=42)
        result2 = stratified_split(X, y, test_size=0.2, random_seed=42)

        assert len(result1[0]) == len(result2[0])
