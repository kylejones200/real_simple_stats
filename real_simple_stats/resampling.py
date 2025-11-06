"""Resampling methods for statistical inference.

This module provides functions for bootstrap, permutation tests,
and cross-validation techniques.
"""

from typing import List, Tuple, Callable, Dict, Optional
import numpy as np
from scipy import stats

# Module-level constants
VALID_ALTERNATIVES = {"two-sided", "greater", "less"}


def bootstrap(
    data: List[float],
    statistic: Callable[[List[float]], float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Dict[str, any]:
    """Perform bootstrap resampling to estimate sampling distribution.

    Args:
        data: Original sample data
        statistic: Function to compute statistic (e.g., np.mean, np.median)
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level for interval
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - statistic: Original statistic value
            - bootstrap_distribution: Bootstrap distribution
            - mean: Mean of bootstrap distribution
            - std_error: Standard error
            - confidence_interval: Confidence interval

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> result = bootstrap(data, np.mean, n_iterations=100)
        >>> 'confidence_interval' in result
        True
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if random_seed is not None:
        np.random.seed(random_seed)

    data_array = np.array(data)
    n = len(data_array)

    # Calculate original statistic
    original_stat = statistic(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_iterations):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data_array, size=n, replace=True)
        bootstrap_stats.append(statistic(bootstrap_sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)

    return {
        "statistic": float(original_stat),
        "bootstrap_distribution": bootstrap_stats.tolist(),
        "mean": float(np.mean(bootstrap_stats)),
        "std_error": float(np.std(bootstrap_stats)),
        "confidence_interval": (float(ci_lower), float(ci_upper)),
    }


def bootstrap_hypothesis_test(
    data1: List[float],
    data2: List[float],
    statistic: Callable[[List[float], List[float]], float],
    n_iterations: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict[str, any]:
    """Perform bootstrap hypothesis test for difference between two groups.

    Args:
        data1: First sample
        data2: Second sample
        statistic: Function to compute test statistic (e.g., difference of means)
        n_iterations: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - observed_statistic: Observed test statistic
            - null_distribution: Bootstrap null distribution
            - p_value: Two-tailed p-value

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> data1 = [1, 2, 3, 4, 5]
        >>> data2 = [3, 4, 5, 6, 7]
        >>> result = bootstrap_hypothesis_test(data1, data2, lambda x, y: np.mean(x) - np.mean(y))
        >>> 'p_value' in result
        True
    """
    if len(data1) < 2 or len(data2) < 2:
        raise ValueError("Both samples must contain at least 2 values")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate observed statistic
    observed_stat = statistic(data1, data2)

    # Pool data under null hypothesis
    pooled_data = np.concatenate([data1, data2])
    n1 = len(data1)
    n2 = len(data2)

    # Bootstrap under null hypothesis
    null_distribution = []
    for _ in range(n_iterations):
        # Shuffle and split
        shuffled = np.random.permutation(pooled_data)
        bootstrap_sample1 = shuffled[:n1]
        bootstrap_sample2 = shuffled[n1:]
        null_distribution.append(statistic(bootstrap_sample1, bootstrap_sample2))

    null_distribution = np.array(null_distribution)

    # Calculate two-tailed p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))

    return {
        "observed_statistic": float(observed_stat),
        "null_distribution": null_distribution.tolist(),
        "p_value": float(p_value),
    }


def permutation_test(
    data1: List[float],
    data2: List[float],
    statistic: Callable[[List[float], List[float]], float],
    n_permutations: int = 1000,
    alternative: str = "two-sided",
    random_seed: Optional[int] = None,
) -> Dict[str, any]:
    """Perform permutation test for comparing two groups.

    Args:
        data1: First sample
        data2: Second sample
        statistic: Function to compute test statistic
        n_permutations: Number of permutations
        alternative: 'two-sided', 'greater', or 'less'
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - observed_statistic: Observed test statistic
            - permutation_distribution: Permutation distribution
            - p_value: P-value

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> data1 = [1, 2, 3, 4, 5]
        >>> data2 = [3, 4, 5, 6, 7]
        >>> result = permutation_test(data1, data2, lambda x, y: np.mean(x) - np.mean(y))
        >>> 0 <= result['p_value'] <= 1
        True
    """
    if len(data1) < 1 or len(data2) < 1:
        raise ValueError("Both samples must contain at least 1 value")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate observed statistic
    observed_stat = statistic(data1, data2)

    # Pool data
    pooled_data = np.concatenate([data1, data2])
    n1 = len(data1)

    # Generate permutation distribution
    permutation_stats = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(pooled_data)
        perm_sample1 = shuffled[:n1]
        perm_sample2 = shuffled[n1:]
        permutation_stats.append(statistic(perm_sample1, perm_sample2))

    permutation_stats = np.array(permutation_stats)

    # Calculate p-value based on alternative hypothesis
    if alternative == "two-sided":
        p_value = np.mean(np.abs(permutation_stats) >= np.abs(observed_stat))
    elif alternative == "greater":
        p_value = np.mean(permutation_stats >= observed_stat)
    else:  # less
        p_value = np.mean(permutation_stats <= observed_stat)

    return {
        "observed_statistic": float(observed_stat),
        "permutation_distribution": permutation_stats.tolist(),
        "p_value": float(p_value),
    }


def jackknife(
    data: List[float], statistic: Callable[[List[float]], float]
) -> Dict[str, any]:
    """Perform jackknife resampling to estimate bias and variance.

    Args:
        data: Original sample data
        statistic: Function to compute statistic

    Returns:
        Dictionary containing:
            - statistic: Original statistic value
            - jackknife_values: Jackknife statistics
            - bias: Estimated bias
            - std_error: Standard error

    Raises:
        ValueError: If data is insufficient

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> result = jackknife(data, np.mean)
        >>> 'std_error' in result
        True
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values")

    data_array = np.array(data)
    n = len(data_array)

    # Calculate original statistic
    original_stat = statistic(data)

    # Jackknife: leave one out
    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.delete(data_array, i)
        jackknife_stats.append(statistic(jackknife_sample))

    jackknife_stats = np.array(jackknife_stats)

    # Estimate bias
    jackknife_mean = np.mean(jackknife_stats)
    bias = (n - 1) * (jackknife_mean - original_stat)

    # Estimate standard error
    std_error = np.sqrt(((n - 1) / n) * np.sum((jackknife_stats - jackknife_mean) ** 2))

    return {
        "statistic": float(original_stat),
        "jackknife_values": jackknife_stats.tolist(),
        "bias": float(bias),
        "std_error": float(std_error),
    }


def cross_validate(
    X: List[List[float]],
    y: List[float],
    model_fn: Callable,
    k_folds: int = 5,
    random_seed: Optional[int] = None,
) -> Dict[str, any]:
    """Perform k-fold cross-validation.

    Args:
        X: Feature matrix (n_samples x n_features)
        y: Target values
        model_fn: Function that takes (X_train, y_train, X_test) and returns predictions
        k_folds: Number of folds
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - fold_scores: Score for each fold (MSE)
            - mean_score: Mean cross-validation score
            - std_score: Standard deviation of scores

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        >>> y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        >>> def simple_model(X_train, y_train, X_test):
        ...     return [np.mean(y_train)] * len(X_test)
        >>> result = cross_validate(X, y, simple_model, k_folds=5)
        >>> 'mean_score' in result
        True
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
    if len(X) < k_folds:
        raise ValueError("Number of samples must be at least k_folds")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")

    if random_seed is not None:
        np.random.seed(random_seed)

    X_array = np.array(X)
    y_array = np.array(y)
    n = len(X_array)

    # Shuffle indices
    indices = np.random.permutation(n)

    # Create folds
    fold_size = n // k_folds
    fold_scores = []

    for fold in range(k_folds):
        # Split into train and test
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k_folds - 1 else n

        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

        X_train = X_array[train_indices]
        y_train = y_array[train_indices]
        X_test = X_array[test_indices]
        y_test = y_array[test_indices]

        # Train and predict
        predictions = model_fn(X_train.tolist(), y_train.tolist(), X_test.tolist())

        # Calculate MSE
        mse = np.mean((np.array(predictions) - y_test) ** 2)
        fold_scores.append(float(mse))

    return {
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
    }


def stratified_split(
    X: List[List[float]],
    y: List[int],
    test_size: float = 0.2,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    """Split data into train and test sets with stratification.

    Args:
        X: Feature matrix
        y: Target labels (must be categorical)
        test_size: Proportion of data for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> X = [[i] for i in range(100)]
        >>> y = [0] * 50 + [1] * 50
        >>> X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
        >>> len(X_test)
        20
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if random_seed is not None:
        np.random.seed(random_seed)

    X_array = np.array(X)
    y_array = np.array(y)

    # Get unique classes
    classes = np.unique(y_array)

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    # Split each class proportionally
    for cls in classes:
        cls_indices = np.where(y_array == cls)[0]
        n_cls = len(cls_indices)
        n_test = int(n_cls * test_size)

        # Shuffle class indices
        shuffled_indices = np.random.permutation(cls_indices)

        test_indices = shuffled_indices[:n_test]
        train_indices = shuffled_indices[n_test:]

        X_train_list.append(X_array[train_indices])
        X_test_list.append(X_array[test_indices])
        y_train_list.append(y_array[train_indices])
        y_test_list.append(y_array[test_indices])

    # Concatenate all classes
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)

    # Shuffle final sets
    train_shuffle = np.random.permutation(len(X_train))
    test_shuffle = np.random.permutation(len(X_test))

    return (
        X_train[train_shuffle].tolist(),
        X_test[test_shuffle].tolist(),
        y_train[train_shuffle].tolist(),
        y_test[test_shuffle].tolist(),
    )


__all__ = [
    "bootstrap",
    "bootstrap_hypothesis_test",
    "permutation_test",
    "jackknife",
    "cross_validate",
    "stratified_split",
]
