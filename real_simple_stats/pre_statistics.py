import logging
import math

logger = logging.getLogger(__name__)


def percent_to_decimal(percent: float) -> float:
    """Convert percentage to decimal (e.g., 75% -> 0.75)."""
    return percent / 100


def decimal_to_percent(decimal: float) -> float:
    """Convert decimal to percentage (e.g., 0.75 -> 75%)."""
    return decimal * 100


def round_to_decimal_places(value: float, places: int) -> float:
    """Round a number to a given number of decimal places."""
    return round(value, places)


def order_of_operations_example():
    """Illustrates PEMDAS with an example."""
    # Equivalent to: 1.96 + ((5 - (3 * 7)) / (25 / 10))
    numerator = 5 - (3 * 7)
    denominator = 5**2 / math.sqrt(100)
    result = 1.96 + (numerator / denominator)
    return result


def mean(values):
    return sum(values) / len(values)


def mode(values):
    from collections import Counter

    freq = Counter(values)
    max_count = max(freq.values())
    return [k for k, v in freq.items() if v == max_count]


def median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def weighted_mean(values, weights):
    """Compute the weighted average: Σ(wx) / Σw"""
    return sum(w * x for w, x in zip(weights, values)) / sum(weights)


def factorial(n):
    return math.factorial(n)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Convert 75%% to decimal: %s", percent_to_decimal(75))
    logger.info("Convert 0.75 to percent: %s", decimal_to_percent(0.75))
    logger.info("Round 0.1284 to 2 decimal places: %s", round_to_decimal_places(0.1284, 2))
    logger.info("PEMDAS example result: %s", order_of_operations_example())

    data = [2, 19, 44, 44, 44, 51, 56, 78, 86, 99, 99]
    logger.info("Mean: %s", mean(data))
    logger.info("Median: %s", median(data))
    logger.info("Mode: %s", mode(data))

    scores = [80, 80, 85]
    weights = [0.4, 0.4, 0.2]
    logger.info("Weighted Mean: %s", weighted_mean(scores, weights))

    logger.info("Factorial of 5: %s", factorial(5))
