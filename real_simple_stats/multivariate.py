"""Multivariate statistical analysis functions.

This module provides functions for multivariate analysis including
multiple regression, PCA, and factor analysis.
"""

import math

from . import _matrix as M
from . import _rss


def multiple_regression(
    X: list[list[float]], y: list[float], include_intercept: bool = True
) -> dict[str, any]:
    """Perform multiple linear regression.

    Args:
        X: Independent variables (n_samples x n_features)
        y: Dependent variable (n_samples)
        include_intercept: Whether to include intercept term

    Returns:
        Dictionary containing:
            - coefficients: Regression coefficients
            - intercept: Intercept term (if included)
            - r_squared: R-squared value
            - adjusted_r_squared: Adjusted R-squared
            - f_statistic: F-statistic
            - p_value: P-value for F-test
            - residuals: Residual values
            - predictions: Predicted values

    Raises:
        ValueError: If dimensions don't match or data is insufficient

    Examples:
        >>> X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        >>> y = [2, 4, 5, 4, 5]
        >>> result = multiple_regression(X, y)
        >>> 'coefficients' in result
        True
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
    if len(X) < 2:
        raise ValueError("Need at least 2 samples")

    X_array = M.as_matrix(X)
    y_array = [float(v) for v in y]

    n_samples, n_features = M.shape(X_array)

    if n_samples <= n_features + 1:
        raise ValueError("Need more samples than features")

    if include_intercept:
        X_array = [[1.0, *row] for row in X_array]

    rows, cols = M.shape(X_array)
    coefficients = _rss.mat_lstsq(rows, cols, M.flatten(X_array), y_array)

    predictions = M.matvec(X_array, coefficients)
    residuals = [a - b for a, b in zip(y_array, predictions)]

    y_mean = _rss.mean(y_array)
    ss_total = sum((v - y_mean) ** 2 for v in y_array)
    ss_residual = sum(r * r for r in residuals)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    # Adjusted R-squared
    n = len(y_array)
    p = n_features
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # F-statistic
    ss_regression = ss_total - ss_residual
    df_regression = p
    df_residual = n - p - 1
    ms_regression = ss_regression / df_regression if df_regression > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 1

    f_statistic = ms_regression / ms_residual if ms_residual > 0 else 0
    p_value = _rss.f_sf(f_statistic, df_regression, df_residual)

    result = {
        "coefficients": (
            list(coefficients[1:]) if include_intercept else list(coefficients)
        ),
        "intercept": float(coefficients[0]) if include_intercept else None,
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "residuals": residuals,
        "predictions": predictions,
    }

    return result


def pca(X: list[list[float]], n_components: int | None = None) -> dict[str, any]:
    """Perform Principal Component Analysis (PCA).

    Args:
        X: Data matrix (n_samples x n_features)
        n_components: Number of components to keep (default: all)

    Returns:
        Dictionary containing:
            - components: Principal components
            - explained_variance: Variance explained by each component
            - explained_variance_ratio: Proportion of variance explained
            - transformed: Transformed data
            - mean: Mean of original data

    Raises:
        ValueError: If data is insufficient or n_components is invalid

    Examples:
        >>> X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        >>> result = pca(X, n_components=2)
        >>> len(result['components'])
        2
    """
    if len(X) < 2:
        raise ValueError("Need at least 2 samples")

    X_array = M.as_matrix(X)
    n_samples, n_features = M.shape(X_array)

    if n_components is None:
        n_components = min(n_samples, n_features)
    elif n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components must be between 1 and {min(n_samples, n_features)}"
        )

    X_centered, mean = M.center(X_array)
    cov_matrix = M.cov(X_centered, ddof=1)

    eigenvalues, eigenvectors = M.eigh_descending(cov_matrix)
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = M.take_columns(eigenvectors, n_components)

    transformed = M.matmul(X_centered, eigenvectors)

    total_variance = sum(eigenvalues)
    explained_variance_ratio = (
        [v / total_variance for v in eigenvalues]
        if total_variance > 0
        else list(eigenvalues)
    )

    return {
        "components": M.transpose(eigenvectors),
        "explained_variance": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "transformed": transformed,
        "mean": mean,
    }


def factor_analysis(
    X: list[list[float]], n_factors: int, max_iter: int = 100
) -> dict[str, any]:
    """Perform Factor Analysis.

    Args:
        X: Data matrix (n_samples x n_features)
        n_factors: Number of factors to extract
        max_iter: Maximum number of iterations

    Returns:
        Dictionary containing:
            - loadings: Factor loadings
            - communalities: Communalities for each variable
            - uniquenesses: Uniquenesses for each variable
            - transformed: Factor scores

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
        >>> result = factor_analysis(X, n_factors=2)
        >>> len(result['loadings'])
        2
    """
    if len(X) < 2:
        raise ValueError("Need at least 2 samples")

    X_array = M.as_matrix(X)
    n_samples, n_features = M.shape(X_array)

    if n_factors < 1 or n_factors > n_features:
        raise ValueError(f"n_factors must be between 1 and {n_features}")

    X_standardized, _mean, _std = M.standardize(X_array)
    corr_matrix = M.corr(X_standardized)

    uniquenesses = [0.5] * n_features
    loadings: M.Matrix = [[0.0] * n_factors for _ in range(n_features)]
    communalities = [0.0] * n_features

    # Principal-axis factoring: repeatedly re-estimate the communalities on the
    # diagonal and re-extract factors until the uniquenesses settle.
    for _ in range(max_iter):
        reduced_corr = [list(row) for row in corr_matrix]
        for i in range(n_features):
            reduced_corr[i][i] -= uniquenesses[i]

        eigenvalues, eigenvectors = M.eigh_descending(reduced_corr)
        eigenvalues = eigenvalues[:n_factors]
        eigenvectors = M.take_columns(eigenvectors, n_factors)

        scales = [math.sqrt(max(v, 0.0)) for v in eigenvalues]
        loadings = [
            [eigenvectors[i][j] * scales[j] for j in range(n_factors)]
            for i in range(n_features)
        ]

        communalities = [sum(v * v for v in row) for row in loadings]
        new_uniquenesses = [max(1.0 - c, 0.005) for c in communalities]

        if max(abs(a - b) for a, b in zip(uniquenesses, new_uniquenesses)) < 1e-6:
            uniquenesses = new_uniquenesses
            break

        uniquenesses = new_uniquenesses

    # Regression-method factor scores: X_std @ L @ (L'L)^-1
    lt_l = M.matmul(M.transpose(loadings), loadings)
    factor_scores = M.matmul(M.matmul(X_standardized, loadings), M.inv_or_pinv(lt_l))

    return {
        "loadings": M.transpose(loadings),
        "communalities": communalities,
        "uniquenesses": uniquenesses,
        "transformed": factor_scores,
    }


def canonical_correlation(X: list[list[float]], Y: list[list[float]]) -> dict[str, any]:
    """Perform Canonical Correlation Analysis (CCA).

    Args:
        X: First set of variables (n_samples x p_features)
        Y: Second set of variables (n_samples x q_features)

    Returns:
        Dictionary containing:
            - correlations: Canonical correlations
            - X_weights: Weights for X variables
            - Y_weights: Weights for Y variables

    Raises:
        ValueError: If dimensions don't match or data is insufficient

    Examples:
        >>> X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        >>> Y = [[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        >>> result = canonical_correlation(X, Y)
        >>> len(result['correlations']) > 0
        True
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have same number of samples")
    if len(X) < 2:
        raise ValueError("Need at least 2 samples")

    X_array = M.as_matrix(X)
    Y_array = M.as_matrix(Y)

    n_samples, p = M.shape(X_array)
    q = M.shape(Y_array)[1]

    X_centered, _ = M.center(X_array)
    Y_centered, _ = M.center(Y_array)

    denom = n_samples - 1
    xt = M.transpose(X_centered)
    yt = M.transpose(Y_centered)
    Cxx = [[v / denom for v in row] for row in M.matmul(xt, X_centered)]
    Cyy = [[v / denom for v in row] for row in M.matmul(yt, Y_centered)]
    Cxy = [[v / denom for v in row] for row in M.matmul(xt, Y_centered)]

    # Ridge the diagonals so a near-collinear block stays invertible.
    Cxx = M.add_ridge(Cxx, 1e-8)
    Cyy = M.add_ridge(Cyy, 1e-8)

    try:
        Cxx_inv_sqrt = M.sqrtm_spd(M.inv(Cxx))
        Cyy_inv_sqrt = M.sqrtm_spd(M.inv(Cyy))
    except ValueError as exc:
        raise ValueError(
            "Singular covariance matrix - check for collinearity"
        ) from exc

    mat = M.matmul(M.matmul(Cxx_inv_sqrt, Cxy), Cyy_inv_sqrt)
    u, sv, vt = M.svd(mat)

    correlations = [min(max(v, 0.0), 1.0) for v in sv[: min(p, q)]]
    x_weights = M.matmul(Cxx_inv_sqrt, u)
    y_weights = M.matmul(Cyy_inv_sqrt, M.transpose(vt))

    return {
        "correlations": correlations,
        "X_weights": x_weights,
        "Y_weights": y_weights,
    }


def mahalanobis_distance(
    X: list[list[float]], point: list[float] | None = None
) -> list[float]:
    """Calculate Mahalanobis distance from points to center of distribution.

    Args:
        X: Data matrix (n_samples x n_features)
        point: Reference point (default: mean of X)

    Returns:
        List of Mahalanobis distances

    Raises:
        ValueError: If data is insufficient or covariance is singular

    Examples:
        >>> X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        >>> distances = mahalanobis_distance(X)
        >>> len(distances)
        5
    """
    if len(X) < 2:
        raise ValueError("Need at least 2 samples")

    X_array = M.as_matrix(X)
    n_samples, n_features = M.shape(X_array)

    mean = M.col_means(X_array)
    cov_matrix = M.cov(X_array, ddof=1)

    if point is not None:
        if len(point) != n_features:
            raise ValueError("Point must have same number of features as X")
        center = [float(v) for v in point]
    else:
        center = mean

    # Ridge the diagonal for numerical stability, as the NumPy version did.
    cov_matrix = M.add_ridge(cov_matrix, 1e-8)

    try:
        cov_inv = M.inv(cov_matrix)
    except ValueError as exc:
        raise ValueError("Singular covariance matrix") from exc

    distances = []
    for row in X_array:
        diff = [a - b for a, b in zip(row, center)]
        quad = sum(d * v for d, v in zip(diff, M.matvec(cov_inv, diff)))
        distances.append(math.sqrt(max(quad, 0.0)))

    return distances


__all__ = [
    "multiple_regression",
    "pca",
    "factor_analysis",
    "canonical_correlation",
    "mahalanobis_distance",
]
