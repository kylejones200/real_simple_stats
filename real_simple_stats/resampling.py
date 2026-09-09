"""Resampling methods: bootstrap, permutation tests, jackknife, and validation splits.

The heavy loops run in Rust and in parallel. When ``statistic`` is one of the
standard summaries the whole resample never enters Python at all; anything else
falls back to a Python loop driven by the same native generator, so custom
statistics still work, just more slowly.

Reproducibility: passing ``random_seed`` gives the same answer on every machine
and every run, and -- because each iteration draws from its own derived stream
rather than a shared one -- the same answer regardless of how many CPU cores do
the work. Sequences differ from 0.4.x, which used NumPy's generator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from . import _rss
from ._rss import Rng

__all__ = [
    "bootstrap",
    "bootstrap_hypothesis_test",
    "permutation_test",
    "jackknife",
    "cross_validate",
    "stratified_split",
]

#: Statistics the Rust backend can compute without calling back into Python.
_NATIVE_STATS = {"mean", "median", "std", "var", "min", "max", "sum"}

_MAX_SEED = 2**63 - 1


def _native_value(name: str, values: list[float]) -> float:
    """Evaluate a native statistic on the full sample."""
    if name == "var":
        return _rss.variance(values, 1)
    if name == "std":
        return _rss.std_dev(values, 1)
    return {
        "mean": _rss.mean,
        "median": _rss.median,
        "sum": _rss.sum_,
        "min": _rss.min_,
        "max": _rss.max_,
    }[name](values)


def _resolve_stat(statistic: Callable | str | None) -> str | None:
    """Map a statistic to a native kernel name, or None if it must run in Python.

    Recognises a plain string, this package's own functions, and anything whose
    ``__name__`` matches -- which covers ``statistics.mean``, ``numpy.mean`` for
    users who still have NumPy around, and most hand-written wrappers.
    """
    if statistic is None:
        return "mean"
    if isinstance(statistic, str):
        name = statistic
    else:
        name = getattr(statistic, "__name__", "")
        # Our own aliases carry longer names than the kernels.
        name = {
            "sample_std_dev": "std",
            "sample_variance": "var",
            "population_std_dev": None,
            "std_dev": "std",
            "variance": "var",
            "amin": "min",
            "amax": "max",
        }.get(name, name)
    return name if name in _NATIVE_STATS else None


def _seed_or_random(random_seed: int | None) -> int:
    if random_seed is not None:
        return int(random_seed) % _MAX_SEED
    return Rng().integers(0, _MAX_SEED, 1)[0]


def _as_floats(data: Sequence[float]) -> list[float]:
    return [float(v) for v in data]


def bootstrap(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | str = "mean",
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Estimate a sampling distribution by resampling with replacement.

    Args:
        data: The observed sample.
        statistic: A summary to bootstrap. Pass ``"mean"``, ``"median"``,
            ``"std"``, ``"var"``, ``"min"``, ``"max"`` or ``"sum"`` (or a
            function with one of those names) to use the native kernel; any
            other callable works but runs in Python.
        n_iterations: Number of bootstrap resamples.
        confidence_level: Coverage of the percentile interval.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary with ``statistic``, ``bootstrap_distribution``, ``mean``,
        ``std_error`` and ``confidence_interval``.

    Raises:
        ValueError: If the data is empty or the parameters are out of range.

    Example:
        >>> r = bootstrap([1, 2, 3, 4, 5], "mean", n_iterations=200, random_seed=1)
        >>> lo, hi = r["confidence_interval"]
        >>> lo <= r["statistic"] <= hi
        True
    """
    values = _as_floats(data)
    if not values:
        raise ValueError("Cannot bootstrap an empty dataset")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    seed = _seed_or_random(random_seed)
    native = _resolve_stat(statistic)

    if native is not None:
        dist = _rss.bootstrap_dist(values, native, n_iterations, seed)
        original_stat = _native_value(native, values)
    else:
        rng = Rng(seed)
        n = len(values)
        dist = [
            float(statistic(rng.choice(values, n, True)))  # type: ignore[operator]
            for _ in range(n_iterations)
        ]
        original_stat = float(statistic(values))  # type: ignore[operator]

    ci_lower, ci_upper = _rss.percentile_ci(dist, confidence_level)
    return {
        "statistic": float(original_stat),
        "bootstrap_distribution": dist,
        "mean": _rss.mean(dist),
        "std_error": _rss.std_dev(dist, 0),
        "confidence_interval": (ci_lower, ci_upper),
    }


def bootstrap_hypothesis_test(
    data1: Sequence[float],
    data2: Sequence[float],
    statistic: Callable[[Sequence[float], Sequence[float]], float] | str = "mean",
    n_iterations: int = 1000,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Test whether two groups differ, by resampling under a pooled null.

    Args:
        data1: First sample.
        data2: Second sample.
        statistic: Two-argument statistic, or ``"mean"`` for the difference in
            means (the native path).
        n_iterations: Number of resamples.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary with ``observed_statistic``, ``bootstrap_distribution`` and
        ``p_value``.

    Raises:
        ValueError: If either sample is empty.
    """
    a = _as_floats(data1)
    b = _as_floats(data2)
    if not a or not b:
        raise ValueError("Both datasets must be non-empty")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")

    seed = _seed_or_random(random_seed)

    if isinstance(statistic, str) or getattr(statistic, "__name__", "") == "mean":
        observed = _rss.mean(a) - _rss.mean(b)
        dist = _rss.permutation_dist(a, b, "mean", n_iterations, seed)
    else:
        observed = float(statistic(a, b))
        rng = Rng(seed)
        pooled = a + b
        n1 = len(a)
        dist = []
        for _ in range(n_iterations):
            shuffled = rng.shuffled(pooled)
            dist.append(float(statistic(shuffled[:n1], shuffled[n1:])))

    return {
        "observed_statistic": observed,
        "bootstrap_distribution": dist,
        "p_value": _rss.permutation_pvalue(dist, observed, "two-sided"),
    }


def permutation_test(
    data1: Sequence[float],
    data2: Sequence[float],
    statistic: Callable[[Sequence[float], Sequence[float]], float] | str = "mean",
    n_permutations: int = 1000,
    alternative: str = "two-sided",
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Compare two groups by permuting group labels.

    Args:
        data1: First sample.
        data2: Second sample.
        statistic: Two-argument statistic, or ``"mean"`` for the difference in
            means (the native path).
        n_permutations: Number of label permutations.
        alternative: ``"two-sided"``, ``"greater"`` or ``"less"``.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary with ``observed_statistic``, ``permutation_distribution``
        and ``p_value``.

    Raises:
        ValueError: If a sample is empty or ``alternative`` is unrecognised.

    Example:
        >>> r = permutation_test([1, 2, 3], [7, 8, 9], "mean", 200, random_seed=3)
        >>> 0 <= r["p_value"] <= 1
        True
    """
    a = _as_floats(data1)
    b = _as_floats(data2)
    if not a or not b:
        raise ValueError("Both datasets must be non-empty")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    seed = _seed_or_random(random_seed)

    if isinstance(statistic, str) or getattr(statistic, "__name__", "") == "mean":
        observed = _rss.mean(a) - _rss.mean(b)
        dist = _rss.permutation_dist(a, b, "mean", n_permutations, seed)
    else:
        observed = float(statistic(a, b))
        rng = Rng(seed)
        pooled = a + b
        n1 = len(a)
        dist = []
        for _ in range(n_permutations):
            shuffled = rng.shuffled(pooled)
            dist.append(float(statistic(shuffled[:n1], shuffled[n1:])))

    return {
        "observed_statistic": observed,
        "permutation_distribution": dist,
        "p_value": _rss.permutation_pvalue(dist, observed, alternative),
    }


def jackknife(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | str = "mean",
) -> dict[str, Any]:
    """Leave-one-out estimates of a statistic's bias and standard error.

    Args:
        data: The observed sample (at least 2 values).
        statistic: Summary to evaluate; native names take the fast path.

    Returns:
        Dictionary with ``statistic``, ``jackknife_values``, ``bias`` and
        ``std_error``.

    Raises:
        ValueError: If fewer than 2 values are supplied.

    Example:
        >>> r = jackknife([1, 2, 3, 4, 5], "mean")
        >>> len(r["jackknife_values"])
        5
    """
    values = _as_floats(data)
    n = len(values)
    if n < 2:
        raise ValueError("Jackknife requires at least 2 values")

    native = _resolve_stat(statistic)
    if native is not None:
        jack = _rss.jackknife_values(values, native)
        original = _native_value(native, values)
    else:
        jack = [
            float(statistic(values[:i] + values[i + 1 :]))  # type: ignore[operator]
            for i in range(n)
        ]
        original = float(statistic(values))  # type: ignore[operator]

    jack_mean = _rss.mean(jack)
    bias = (n - 1) * (jack_mean - original)
    std_error = math.sqrt(
        ((n - 1) / n) * sum((v - jack_mean) ** 2 for v in jack)
    )
    return {
        "statistic": float(original),
        "jackknife_values": jack,
        "bias": bias,
        "std_error": std_error,
    }


def cross_validate(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    model_fn: Callable,
    k_folds: int = 5,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Score a model by k-fold cross-validation, using mean squared error.

    Args:
        X: Feature matrix, one row per sample.
        y: Target values.
        model_fn: Callable ``(X_train, y_train, X_test) -> predictions``.
        k_folds: Number of folds (at least 2).
        random_seed: Seed for the shuffle.

    Returns:
        Dictionary with ``scores``, ``mean_score`` and ``std_score``.

    Raises:
        ValueError: If the shapes disagree or there are too few samples.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
    if len(X) < k_folds:
        raise ValueError("Number of samples must be at least k_folds")
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")

    rows = [list(map(float, r)) for r in X]
    targets = _as_floats(y)
    n = len(rows)

    order = Rng(_seed_or_random(random_seed)).permutation(n)
    fold_size = n // k_folds
    scores: list[float] = []

    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size if fold < k_folds - 1 else n
        test_idx = order[start:end]
        train_idx = order[:start] + order[end:]

        predictions = model_fn(
            [rows[i] for i in train_idx],
            [targets[i] for i in train_idx],
            [rows[i] for i in test_idx],
        )
        errors = [
            (float(p) - targets[i]) ** 2 for p, i in zip(predictions, test_idx)
        ]
        scores.append(_rss.mean(errors))

    return {
        "scores": scores,
        "mean_score": _rss.mean(scores),
        "std_score": _rss.std_dev(scores, 0),
    }


def stratified_split(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    test_size: float = 0.2,
    random_seed: int | None = None,
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Split into train and test sets, preserving each class's proportion.

    Args:
        X: Feature matrix.
        y: Categorical labels.
        test_size: Fraction of each class held out.
        random_seed: Seed for reproducibility.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If the shapes disagree or ``test_size`` is out of range.

    Example:
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

    rows = [list(map(float, r)) for r in X]
    labels = list(y)
    rng = Rng(_seed_or_random(random_seed))

    # Preserve first-appearance order so the split is deterministic given the seed.
    classes: list = []
    for label in labels:
        if label not in classes:
            classes.append(label)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in classes:
        cls_idx = [i for i, label in enumerate(labels) if label == cls]
        n_cls = len(cls_idx)
        n_test = int(n_cls * test_size)
        # Give a minority class at least one test sample, but never all of them.
        if n_test == 0 and n_cls > 1 and len(classes) > 1:
            n_test = 1
        if n_test >= n_cls and n_cls > 1:
            n_test = max(1, n_cls - 1)

        shuffled = [cls_idx[j] for j in rng.permutation(n_cls)]
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    train_idx = [train_idx[j] for j in rng.permutation(len(train_idx))]
    test_idx = [test_idx[j] for j in rng.permutation(len(test_idx))]

    return (
        [rows[i] for i in train_idx],
        [rows[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )
