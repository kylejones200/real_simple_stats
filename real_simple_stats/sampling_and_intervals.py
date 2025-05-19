import math
from scipy.stats import norm, t
from typing import Tuple

# --- CENTRAL LIMIT THEOREM UTILITIES ---

def sampling_distribution_mean(pop_mean: float) -> float:
    return pop_mean

def sampling_distribution_variance(pop_std: float, sample_size: int) -> float:
    return (pop_std ** 2) / sample_size

def clt_probability_greater_than(x: float, mean: float, std_dev: float, n: int) -> float:
    """P(sample mean > x) using normal approximation"""
    z = (x - mean) / (std_dev / math.sqrt(n))
    return 1 - norm.cdf(z)

def clt_probability_less_than(x: float, mean: float, std_dev: float, n: int) -> float:
    """P(sample mean < x)"""
    z = (x - mean) / (std_dev / math.sqrt(n))
    return norm.cdf(z)

def clt_probability_between(x1: float, x2: float, mean: float, std_dev: float, n: int) -> float:
    """P(x1 < sample mean < x2)"""
    z1 = (x1 - mean) / (std_dev / math.sqrt(n))
    z2 = (x2 - mean) / (std_dev / math.sqrt(n))
    return norm.cdf(z2) - norm.cdf(z1)

# --- CONFIDENCE INTERVALS ---

def confidence_interval_known_std(mean: float, std_dev: float, n: int, confidence: float) -> Tuple[float, float]:
    """CI for known population standard deviation using Z-distribution."""
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    margin = z * (std_dev / math.sqrt(n))
    return (mean - margin, mean + margin)

def confidence_interval_unknown_std(sample_mean: float, sample_std: float, n: int, confidence: float) -> Tuple[float, float]:
    """CI for unknown population standard deviation using t-distribution."""
    alpha = 1 - confidence
    df = n - 1
    t_crit = t.ppf(1 - alpha / 2, df)
    margin = t_crit * (sample_std / math.sqrt(n))
    return (sample_mean - margin, sample_mean + margin)

def required_sample_size(confidence: float, width: float, std_dev: float) -> int:
    """Find sample size with known population std dev."""
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    return math.ceil(((z * std_dev) / (width / 2)) ** 2)

def slovins_formula(N: int, e: float) -> int:
    """Slovin’s formula: n = N / (1 + N * e^2)"""
    return int(N / (1 + N * (e ** 2)))

# Example usage
if __name__ == "__main__":
    print("Sampling distribution variance:", sampling_distribution_variance(15, 100))
    print("CLT P(mean > 82):", clt_probability_greater_than(82, 80, 10, 100))
    print("CLT P(mean < 75):", clt_probability_less_than(75, 80, 10, 100))
    print("CLT P(78 < mean < 82):", clt_probability_between(78, 82, 80, 10, 100))

    # Confidence intervals
    print("Confidence interval (known std):", confidence_interval_known_std(100, 15, 36, 0.95))
    print("Confidence interval (unknown std):", confidence_interval_unknown_std(100, 15, 36, 0.95))
    
    # Required sample size
    print("Required sample size for width=10, 95% confidence:", required_sample_size(0.95, 10, 15))
    
    # Slovin's formula
    print("Slovin's formula (N=1000, e=0.05):", slovins_formula(1000, 0.05))
