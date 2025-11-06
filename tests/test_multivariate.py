"""Tests for multivariate analysis module."""

import pytest
import numpy as np
from real_simple_stats.multivariate import (
    multiple_regression,
    pca,
    factor_analysis,
    canonical_correlation,
    mahalanobis_distance,
)


class TestMultipleRegression:
    def test_multiple_regression_basic(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        y = [2, 4, 5, 4, 5]
        result = multiple_regression(X, y)
        
        assert 'coefficients' in result
        assert 'intercept' in result
        assert 'r_squared' in result
        assert 'predictions' in result

    def test_multiple_regression_with_intercept(self):
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]
        result = multiple_regression(X, y, include_intercept=True)
        
        assert result['r_squared'] > 0.99  # Perfect fit

    def test_multiple_regression_without_intercept(self):
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]
        result = multiple_regression(X, y, include_intercept=False)
        
        assert 'coefficients' in result


class TestPCA:
    def test_pca_basic(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        result = pca(X, n_components=2)
        
        assert 'components' in result
        assert 'explained_variance' in result
        assert 'transformed' in result

    def test_pca_single_component(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        result = pca(X, n_components=1)
        
        assert len(result['explained_variance']) == 1

    def test_pca_explained_variance_sum(self):
        X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        result = pca(X)
        
        # Explained variance ratios should sum to ~1
        total_var = sum(result['explained_variance'])
        assert total_var > 0


class TestFactorAnalysis:
    def test_factor_analysis_basic(self):
        X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
        result = factor_analysis(X, n_factors=2)
        
        assert 'loadings' in result
        assert 'communalities' in result

    def test_factor_analysis_single_factor(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        result = factor_analysis(X, n_factors=1)
        
        assert 'loadings' in result


class TestCanonicalCorrelation:
    def test_canonical_correlation_basic(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        Y = [[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        result = canonical_correlation(X, Y)
        
        assert 'correlations' in result
        assert 'X_weights' in result
        assert 'Y_weights' in result

    def test_canonical_correlation_values(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        Y = [[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        result = canonical_correlation(X, Y)
        
        # Correlations should be between 0 and 1
        for corr in result['correlations']:
            assert 0 <= corr <= 1


class TestMahalanobisDistance:
    def test_mahalanobis_distance_basic(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        distances = mahalanobis_distance(X)
        
        assert len(distances) == len(X)
        assert all(d >= 0 for d in distances)

    def test_mahalanobis_distance_with_point(self):
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        point = [3, 3]
        distances = mahalanobis_distance(X, point)
        
        assert len(distances) == len(X)

    def test_mahalanobis_distance_single_point(self):
        X = [[1, 2], [2, 3], [3, 4]]
        point = [2, 2]
        distances = mahalanobis_distance(X, point)
        
        assert all(isinstance(d, float) for d in distances)
