import math
from scipy.stats import norm
from typing import Tuple

# --- Z-SCORE CALCULATIONS ---

def z_score(x: float, mean: float, std_dev: float) -> float:
    """Calculate the z-score for a single value."""
    return (x - mean) / std_dev

def z_score_standard_error(sample_mean: float, population_mean: float, std_dev: float, sample_size: int) -> float:
    """Z-score using the standard error of the mean."""
    return (sample_mean - population_mean) / (std_dev / math.sqrt(sample_size))

# --- AREA UNDER THE NORMAL CURVE ---

def area_between_0_and_z(z: float) -> float:
    """Find area under normal curve between 0 and z (assumes standard normal)."""
    return norm.cdf(abs(z)) - 0.5

def area_in_tail(z: float) -> float:
    """Area to the right (or left) of a z-score."""
    return 1 - norm.cdf(z)

def area_between_z_scores(z1: float, z2: float) -> float:
    """Area between two z-scores."""
    return abs(norm.cdf(z2) - norm.cdf(z1))

def area_left_of_z(z: float) -> float:
    """Cumulative probability to the left of z."""
    return norm.cdf(z)

def area_right_of_z(z: float) -> float:
    """Cumulative probability to the right of z."""
    return 1 - norm.cdf(z)

def area_outside_range(z1: float, z2: float) -> float:
    """Area outside of range bounded by two z-scores (two-tailed)."""
    return 1 - area_between_z_scores(z1, z2)

# --- CHEBYSHEV'S THEOREM ---

def chebyshev_theorem(k: float) -> float:
    """Returns minimum proportion of values within k standard deviations of the mean."""
    if k <= 1:
        raise ValueError("k must be greater than 1")
    return 1 - (1 / k ** 2)

# Example usage
if __name__ == "__main__":
    print("Z-score of x=85 with mean=80, std_dev=5:", z_score(85, 80, 5))
    print("Z-score using SE:", z_score_standard_error(84, 80, 10, 100))

    print("Area between 0 and z=1.96:", area_between_0_and_z(1.96))
    print("Area in tail beyond z=2.0:", area_in_tail(2.0))
    print("Area between z=1 and z=2:", area_between_z_scores(1, 2))
    print("Area left of z=-1:", area_left_of_z(-1))
    print("Area right of z=1.5:", area_right_of_z(1.5))
    print("Area outside range -1.96 to 1.96:", area_outside_range(-1.96, 1.96))

    print("Chebyshev's Theorem (k=2):", chebyshev_theorem(2))
