"""Multivariate statistical analysis functions.

This module provides functions for multivariate analysis including
multiple regression, PCA, and factor analysis.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import linalg, stats


def multiple_regression(
    X: List[List[float]], y: List[float], include_intercept: bool = True
) -> Dict[str, any]:
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

    X_array = np.array(X)
    y_array = np.array(y)

    n_samples, n_features = X_array.shape

    if n_samples <= n_features + 1:
        raise ValueError("Need more samples than features")

    # Add intercept column if requested
    if include_intercept:
        X_array = np.column_stack([np.ones(n_samples), X_array])

    # Calculate coefficients using normal equation: (X'X)^-1 X'y
    try:
        coefficients = np.linalg.lstsq(X_array, y_array, rcond=None)[0]
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix - features may be collinear")

    # Predictions and residuals
    predictions = X_array @ coefficients
    residuals = y_array - predictions

    # R-squared
    ss_total = np.sum((y_array - np.mean(y_array)) ** 2)
    ss_residual = np.sum(residuals**2)
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
    p_value = 1 - stats.f.cdf(f_statistic, df_regression, df_residual)

    result = {
        "coefficients": coefficients[1:].tolist() if include_intercept else coefficients.tolist(),
        "intercept": float(coefficients[0]) if include_intercept else None,
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "residuals": residuals.tolist(),
        "predictions": predictions.tolist(),
    }

    return result


def pca(
    X: List[List[float]], n_components: Optional[int] = None
) -> Dict[str, any]:
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

    X_array = np.array(X)
    n_samples, n_features = X_array.shape

    if n_components is None:
        n_components = min(n_samples, n_features)
    elif n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components must be between 1 and {min(n_samples, n_features)}"
        )

    # Center the data
    mean = np.mean(X_array, axis=0)
    X_centered = X_array - mean

    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # Transform data
    transformed = X_centered @ eigenvectors

    # Calculate explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance if total_variance > 0 else eigenvalues

    return {
        "components": eigenvectors.T.tolist(),
        "explained_variance": eigenvalues.tolist(),
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "transformed": transformed.tolist(),
        "mean": mean.tolist(),
    }


def factor_analysis(
    X: List[List[float]], n_factors: int, max_iter: int = 100
) -> Dict[str, any]:
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

    X_array = np.array(X)
    n_samples, n_features = X_array.shape

    if n_factors < 1 or n_factors > n_features:
        raise ValueError(f"n_factors must be between 1 and {n_features}")

    # Standardize data
    mean = np.mean(X_array, axis=0)
    std = np.std(X_array, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_standardized = (X_array - mean) / std

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X_standardized.T)

    # Initialize uniquenesses
    uniquenesses = np.ones(n_features) * 0.5

    # Iterative estimation
    for _ in range(max_iter):
        # Calculate reduced correlation matrix
        reduced_corr = corr_matrix - np.diag(uniquenesses)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(reduced_corr)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_factors]
        eigenvectors = eigenvectors[:, idx][:, :n_factors]

        # Calculate loadings
        loadings = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))

        # Update uniquenesses
        communalities = np.sum(loadings**2, axis=1)
        new_uniquenesses = 1 - communalities
        new_uniquenesses = np.maximum(new_uniquenesses, 0.005)  # Lower bound

        # Check convergence
        if np.max(np.abs(uniquenesses - new_uniquenesses)) < 1e-6:
            break

        uniquenesses = new_uniquenesses

    # Calculate factor scores using regression method
    factor_scores = X_standardized @ loadings @ np.linalg.inv(loadings.T @ loadings)

    return {
        "loadings": loadings.T.tolist(),
        "communalities": communalities.tolist(),
        "uniquenesses": uniquenesses.tolist(),
        "transformed": factor_scores.tolist(),
    }


def canonical_correlation(
    X: List[List[float]], Y: List[List[float]]
) -> Dict[str, any]:
    """Perform Canonical Correlation Analysis (CCA).

    Args:
        X: First set of variables (n_samples x p_features)
        Y: Second set of variables (n_samples x q_features)

    Returns:
        Dictionary containing:
            - correlations: Canonical correlations
            - x_weights: Weights for X variables
            - y_weights: Weights for Y variables

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

    X_array = np.array(X)
    Y_array = np.array(Y)

    n_samples = X_array.shape[0]
    p = X_array.shape[1]
    q = Y_array.shape[1]

    # Center the data
    X_centered = X_array - np.mean(X_array, axis=0)
    Y_centered = Y_array - np.mean(Y_array, axis=0)

    # Calculate covariance matrices
    Cxx = (X_centered.T @ X_centered) / (n_samples - 1)
    Cyy = (Y_centered.T @ Y_centered) / (n_samples - 1)
    Cxy = (X_centered.T @ Y_centered) / (n_samples - 1)

    # Add small regularization for numerical stability
    Cxx += np.eye(p) * 1e-8
    Cyy += np.eye(q) * 1e-8

    # Solve generalized eigenvalue problem
    try:
        Cxx_inv_sqrt = linalg.sqrtm(np.linalg.inv(Cxx))
        Cyy_inv_sqrt = linalg.sqrtm(np.linalg.inv(Cyy))

        M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt

        U, S, Vt = np.linalg.svd(M)

        # Canonical correlations
        correlations = S[: min(p, q)]

        # Canonical weights
        x_weights = Cxx_inv_sqrt @ U
        y_weights = Cyy_inv_sqrt @ Vt.T

    except np.linalg.LinAlgError:
        raise ValueError("Singular covariance matrix - check for collinearity")

    return {
        "correlations": correlations.tolist(),
        "x_weights": x_weights.tolist(),
        "y_weights": y_weights.tolist(),
    }


def mahalanobis_distance(
    X: List[List[float]], point: Optional[List[float]] = None
) -> List[float]:
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

    X_array = np.array(X)
    n_samples, n_features = X_array.shape

    # Calculate mean and covariance
    mean = np.mean(X_array, axis=0)
    cov = np.cov(X_array.T)

    # Use provided point or mean
    if point is not None:
        if len(point) != n_features:
            raise ValueError("Point must have same number of features as X")
        center = np.array(point)
    else:
        center = mean

    # Add regularization for numerical stability
    cov += np.eye(n_features) * 1e-8

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Singular covariance matrix")

    # Calculate distances
    distances = []
    for i in range(n_samples):
        diff = X_array[i] - center
        distance = np.sqrt(diff @ cov_inv @ diff)
        distances.append(float(distance))

    return distances


__all__ = [
    "multiple_regression",
    "pca",
    "factor_analysis",
    "canonical_correlation",
    "mahalanobis_distance",
]
