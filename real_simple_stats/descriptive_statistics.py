import math
from typing import List, Dict
from collections import Counter

# --- Basic Descriptive Functions ---

def is_discrete(values: List[float]) -> bool:
    """Determine if a variable is discrete (all values are integers)."""
    return all(float(v).is_integer() for v in values)

def is_continuous(values: List[float]) -> bool:
    """Determine if a variable is continuous (contains non-integer values)."""
    return not is_discrete(values)

def five_number_summary(values: List[float]) -> Dict[str, float]:
    """Return the five-number summary: min, Q1, median, Q3, max."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    median_val = sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    lower_half = sorted_vals[:mid]
    upper_half = sorted_vals[mid + 1:] if n % 2 else sorted_vals[mid:]
    Q1 = median(lower_half)
    Q3 = median(upper_half)
    return {
        "min": sorted_vals[0],
        "Q1": Q1,
        "median": median_val,
        "Q3": Q3,
        "max": sorted_vals[-1]
    }

def median(values: List[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2

def interquartile_range(values: List[float]) -> float:
    summary = five_number_summary(values)
    return summary["Q3"] - summary["Q1"]

def sample_variance(values: List[float]) -> float:
    m = sum(values) / len(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)

def sample_std_dev(values: List[float]) -> float:
    return math.sqrt(sample_variance(values))

def coefficient_of_variation(values: List[float]) -> float:
    return (sample_std_dev(values) / mean(values)) * 100

def mean(values: List[float]) -> float:
    return sum(values) / len(values)

def draw_frequency_table(values: List[str]) -> Dict[str, int]:
    """Generate a frequency table from a list of categorical or discrete values."""
    return dict(Counter(values))

def draw_cumulative_frequency_table(values: List[int]) -> Dict[int, int]:
    freq = Counter(values)
    sorted_keys = sorted(freq)
    cumulative = {}
    total = 0
    for k in sorted_keys:
        total += freq[k]
        cumulative[k] = total
    return cumulative

def detect_fake_statistics(survey_sponsor: str, is_voluntary: bool, correlation_not_causation: bool) -> List[str]:
    warnings = []
    if survey_sponsor.lower() in {"diet pill company", "political campaign", "egg company"}:
        warnings.append("Potential bias: Self-funded study")
    if is_voluntary:
        warnings.append("Warning: Voluntary response samples are biased")
    if correlation_not_causation:
        warnings.append("Warning: Correlation does not imply causation")
    return warnings

# Example usage
if __name__ == "__main__":
    x = [1, 2, 5, 6, 7, 9, 12, 15, 18, 19, 27]
    print("Five-number summary:", five_number_summary(x))
    print("IQR:", interquartile_range(x))
    print("Sample variance:", sample_variance(x))
    print("Sample standard deviation:", sample_std_dev(x))
    print("Coefficient of variation:", coefficient_of_variation(x))

    blood_types = ['A', 'O', 'A', 'B', 'B', 'AB', 'B', 'B', 'O', 'A', 'O', 'O', 'O', 'AB', 'B', 'AB', 'AB', 'A', 'O', 'A']
    freq_table = draw_frequency_table(blood_types)
    print("Frequency table:", freq_table)

    values = [1, 1, 2, 2, 3, 3, 3, 4]
    print("Cumulative frequency:", draw_cumulative_frequency_table(values))

    print("Bias warnings:", detect_fake_statistics("diet pill company", True, True))
