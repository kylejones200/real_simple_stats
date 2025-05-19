import math
from scipy.stats import binom
from typing import List

# --- BINOMIAL CORE FUNCTIONS ---

def is_binomial_experiment(trials: int, outcomes: List[str], probability: float) -> bool:
    """
    Checks if an experiment meets the binomial criteria:
    - fixed number of trials
    - each trial is independent
    - each trial has two possible outcomes
    - probability of success is constant
    """
    return isinstance(trials, int) and trials > 0 and len(outcomes) == 2 and 0 <= probability <= 1

def binomial_probability(n: int, k: int, p: float) -> float:
    """Computes probability of k successes in n binomial trials."""
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1 - p) ** (n - k))

def binomial_mean(n: int, p: float) -> float:
    return n * p

def binomial_variance(n: int, p: float) -> float:
    return n * p * (1 - p)

def binomial_std_dev(n: int, p: float) -> float:
    return math.sqrt(binomial_variance(n, p))

def expected_value_single(value: float, probability: float) -> float:
    """Expected value of a single outcome."""
    return value * probability

def expected_value_multiple(values: List[float], probabilities: List[float]) -> float:
    return sum(v * p for v, p in zip(values, probabilities))

# --- NORMAL APPROXIMATION AND CONTINUITY CORRECTION ---

def normal_approximation(n: int, p: float, k: int, use_continuity: bool = True) -> float:
    """Uses normal approximation with continuity correction to estimate binomial P(X ≤ k)."""
    mu = binomial_mean(n, p)
    sigma = binomial_std_dev(n, p)
    z = (k + 0.5 - mu) / sigma if use_continuity else (k - mu) / sigma
    from scipy.stats import norm
    return norm.cdf(z)

# Example usage
if __name__ == "__main__":
    # Binomial criteria check
    print("Is binomial:", is_binomial_experiment(10, ["pass", "fail"], 0.4))

    # Probability of exactly 3 successes in 10 trials with p=0.4
    print("P(X=3):", binomial_probability(10, 3, 0.4))

    # Mean, variance, std dev
    n, p = 10, 0.4
    print("Mean:", binomial_mean(n, p))
    print("Variance:", binomial_variance(n, p))
    print("Standard deviation:", binomial_std_dev(n, p))

    # Expected values
    print("Expected value (single):", expected_value_single(85, 0.2))
    print("Expected value (multiple):", expected_value_multiple([10, 20, 30], [0.1, 0.5, 0.4]))

    # Normal approximation with continuity
    print("Normal approximation P(X ≤ 3):", normal_approximation(n, p, 3))
