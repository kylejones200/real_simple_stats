import math
from collections import Counter
from collections.abc import Sequence

from . import _rss

# --- Basic Descriptive Functions ---


def is_discrete(values: Sequence[float]) -> bool:
    """Determine if a variable is discrete (all values are integers).

    Args:
        values: List of numerical values to check

    Returns:
        True if all values are integers, False otherwise

    Example:
        >>> is_discrete([1.0, 2.0, 3.0])
        True
        >>> is_discrete([1.5, 2.0, 3.0])
        False
    """
    return all(float(v).is_integer() for v in values)


def is_continuous(values: Sequence[float]) -> bool:
    """Determine if a variable is continuous (contains non-integer values).

    Args:
        values: List of numerical values to check

    Returns:
        True if any values are non-integers, False if all are integers

    Example:
        >>> is_continuous([1.5, 2.0, 3.0])
        True
        >>> is_continuous([1.0, 2.0, 3.0])
        False
    """
    return not is_discrete(values)


def five_number_summary(values: Sequence[float]) -> dict[str, float]:
    """Return the five-number summary: min, Q1, median, Q3, max.

    Args:
        values: List of numerical values

    Returns:
        Dictionary with keys: min, Q1, median, Q3, max

    Raises:
        ValueError: If the input list is empty

    Example:
        >>> five_number_summary([1, 2, 3, 4, 5])
        {'min': 1.0, 'Q1': 1.5, 'median': 3.0, 'Q3': 4.5, 'max': 5.0}
        >>> five_number_summary([5])
        {'min': 5.0, 'Q1': 5.0, 'median': 5.0, 'Q3': 5.0, 'max': 5.0}
    """
    if not values:
        raise ValueError("Cannot calculate five-number summary of empty list")

    # The native kernel reproduces this library's Tukey (median-of-halves)
    # convention exactly, including the special cases at n <= 3. It is
    # deliberately not NumPy's linear-interpolation quantile.
    summary = _rss.five_number_summary(values)
    if summary is None:  # pragma: no cover - guarded by the emptiness check above
        raise ValueError("Cannot calculate five-number summary of empty list")
    minimum, q1, med, q3, maximum = summary
    return {"min": minimum, "Q1": q1, "median": med, "Q3": q3, "max": maximum}


def median(values: Sequence[float]) -> float:
    """Calculate the median (middle value) of a dataset.

    Args:
        values: List of numerical values

    Returns:
        The median value

    Raises:
        ValueError: If the input list is empty

    Example:
        >>> median([1, 2, 3, 4, 5])
        3.0
        >>> median([1, 2, 3, 4])
        2.5
    """
    if not values:
        raise ValueError("Cannot calculate median of empty list")
    return _rss.median(values)


def interquartile_range(values: Sequence[float]) -> float:
    summary = five_number_summary(values)
    return summary["Q3"] - summary["Q1"]


def sample_variance(values: Sequence[float]) -> float:
    """Calculate the sample variance of a dataset.

    Uses the sample variance formula with (n-1) degrees of freedom (Bessel's correction).

    Args:
        values: List of numerical values

    Returns:
        The sample variance

    Raises:
        ValueError: If fewer than 2 values are provided

    Example:
        >>> sample_variance([1, 2, 3, 4, 5])
        2.5
    """
    if len(values) < 2:
        raise ValueError("Sample variance requires at least 2 values")
    return _rss.variance(values, 1)


def sample_std_dev(values: Sequence[float]) -> float:
    """Calculate the sample standard deviation of a dataset.

    Args:
        values: List of numerical values

    Returns:
        The sample standard deviation (square root of sample variance)

    Raises:
        ValueError: If fewer than 2 values are provided

    Example:
        >>> sample_std_dev([1, 2, 3, 4, 5])
        1.5811388300841898
    """
    if len(values) < 2:
        raise ValueError("Sample standard deviation requires at least 2 values")
    return _rss.std_dev(values, 1)


def coefficient_of_variation(values: Sequence[float]) -> float:
    mean_val = mean(values)
    if mean_val == 0:
        raise ValueError("Cannot calculate coefficient of variation when mean is zero")
    return sample_std_dev(values) / mean_val


def mean(values: Sequence[float]) -> float:
    """Calculate the arithmetic mean (average) of a dataset.

    Args:
        values: List of numerical values

    Returns:
        The arithmetic mean

    Raises:
        ValueError: If the input list is empty

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return _rss.mean(values)


def draw_frequency_table(
    values: Sequence[str | int],
) -> dict[str | int, int]:
    """Generate a frequency table from a list of categorical or discrete values.

    Args:
        values: List of categorical or discrete values to count

    Returns:
        Dictionary mapping each unique value to its frequency

    Example:
        >>> draw_frequency_table(['A', 'B', 'A', 'C', 'B', 'A'])
        {'A': 3, 'B': 2, 'C': 1}
    """
    return dict(Counter(values))


def draw_cumulative_frequency_table(values: Sequence[int]) -> dict[int, int]:
    """Generate a cumulative frequency table from a list of discrete values.

    Args:
        values: List of discrete integer values

    Returns:
        Dictionary mapping each unique value to its cumulative frequency

    Example:
        >>> draw_cumulative_frequency_table([1, 2, 1, 3, 2, 1])
        {1: 3, 2: 5, 3: 6}
    """
    freq = Counter(values)
    sorted_keys = sorted(freq)
    cumulative: dict[int, int] = {}
    total = 0
    for k in sorted_keys:
        total += freq[k]
        cumulative[k] = total
    return cumulative


def detect_fake_statistics(
    survey_sponsor: str, is_voluntary: bool, correlation_not_causation: bool
) -> list[str]:
    """Detect potential issues with statistical claims or studies.

    Args:
        survey_sponsor: Organization sponsoring the survey/study
        is_voluntary: Whether the survey uses voluntary response sampling
        correlation_not_causation: Whether correlation is being presented as causation

    Returns:
        List of warning messages about potential statistical issues

    Example:
        >>> detect_fake_statistics("Diet Pill Company", True, True)
        ['Potential bias: Self-funded study', 'Warning: Voluntary response samples are biased', 'Warning: Correlation does not imply causation']
    """
    warnings: list[str] = []
    if survey_sponsor.lower() in {
        "diet pill company",
        "political campaign",
        "egg company",
    }:
        warnings.append("Potential bias: Self-funded study")
    if is_voluntary:
        warnings.append("Warning: Voluntary response samples are biased")
    if correlation_not_causation:
        warnings.append("Warning: Correlation does not imply causation")
    return warnings
