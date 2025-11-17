import math
from typing import Sequence, Dict, Union, List
from collections import Counter

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


def five_number_summary(values: Sequence[float]) -> Dict[str, float]:
    """Return the five-number summary: min, Q1, median, Q3, max.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with keys: min, Q1, median, Q3, max
        
    Raises:
        ValueError: If the input list is empty
        
    Example:
        >>> five_number_summary([1, 2, 3, 4, 5])
        {'min': 1, 'Q1': 1.5, 'median': 3, 'Q3': 4.5, 'max': 5}
        >>> five_number_summary([5])
        {'min': 5, 'Q1': 5, 'median': 5, 'Q3': 5, 'max': 5}
    """
    if not values:
        raise ValueError("Cannot calculate five-number summary of empty list")
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    # Handle edge cases for small samples
    if n == 1:
        # Single value: all statistics equal the value
        val = sorted_vals[0]
        return {
            "min": val,
            "Q1": val,
            "median": val,
            "Q3": val,
            "max": val,
        }
    
    # Calculate median
    mid = n // 2
    median_val = (
        sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    )
    
    # For n=2, Q1 and Q3 are the two values
    if n == 2:
        return {
            "min": sorted_vals[0],
            "Q1": sorted_vals[0],
            "median": median_val,
            "Q3": sorted_vals[1],
            "max": sorted_vals[1],
        }
    
    # For n=3, Q1 is min and Q3 is max
    if n == 3:
        return {
            "min": sorted_vals[0],
            "Q1": sorted_vals[0],
            "median": median_val,
            "Q3": sorted_vals[2],
            "max": sorted_vals[2],
        }
    
    # Standard calculation for n >= 4
    lower_half = sorted_vals[:mid]
    upper_half = sorted_vals[mid + 1 :] if n % 2 else sorted_vals[mid:]
    Q1 = median(lower_half)
    Q3 = median(upper_half)
    
    return {
        "min": sorted_vals[0],
        "Q1": Q1,
        "median": median_val,
        "Q3": Q3,
        "max": sorted_vals[-1],
    }


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
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


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
    m = sum(values) / len(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


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
    return math.sqrt(sample_variance(values))


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
    return sum(values) / len(values)


def draw_frequency_table(
    values: Sequence[Union[str, int]],
) -> Dict[Union[str, int], int]:
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


def draw_cumulative_frequency_table(values: Sequence[int]) -> Dict[int, int]:
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
    cumulative: Dict[int, int] = {}
    total = 0
    for k in sorted_keys:
        total += freq[k]
        cumulative[k] = total
    return cumulative


def detect_fake_statistics(
    survey_sponsor: str, is_voluntary: bool, correlation_not_causation: bool
) -> List[str]:
    """Detect potential issues with statistical claims or studies.

    Args:
        survey_sponsor: Organization sponsoring the survey/study
        is_voluntary: Whether the survey uses voluntary response sampling
        correlation_not_causation: Whether correlation is being presented as causation

    Returns:
        List of warning messages about potential statistical issues

    Example:
        >>> detect_fake_statistics("Diet Pill Company", True, True)
        ['Potential bias: Self-funded study', 'Warning: Voluntary response samples are biased',
         'Warning: Correlation does not imply causation']
    """
    warnings: List[str] = []
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
