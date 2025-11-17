import math

from scipy.stats import norm

# --- Z-SCORE CALCULATIONS ---


def z_score(x: float, mean: float, std_dev: float) -> float:
    """Calculate the z-score for a single value."""
    return (x - mean) / std_dev


def z_score_standard_error(
    sample_mean: float, population_mean: float, std_dev: float, sample_size: int
) -> float:
    """Z-score using the standard error of the mean."""
    return (sample_mean - population_mean) / (std_dev / math.sqrt(sample_size))


# --- AREA UNDER THE NORMAL CURVE ---


def area_between_0_and_z(z: float) -> float:
    """Find area under normal curve between 0 and z (assumes standard normal)."""
    return float(norm.cdf(abs(z))) - 0.5


def area_in_tail(z: float) -> float:
    """Area to the right (or left) of a z-score."""
    return 1 - float(norm.cdf(z))


def area_between_z_scores(z1: float, z2: float) -> float:
    """Area between two z-scores."""
    return abs(float(norm.cdf(z2)) - float(norm.cdf(z1)))


def area_left_of_z(z: float) -> float:
    """Cumulative probability to the left of z."""
    return float(norm.cdf(z))


def area_right_of_z(z: float) -> float:
    """Cumulative probability to the right of z."""
    return 1 - float(norm.cdf(z))


def area_outside_range(z1: float, z2: float) -> float:
    """Area outside of range bounded by two z-scores (two-tailed)."""
    return 1 - area_between_z_scores(z1, z2)


# --- CHEBYSHEV'S THEOREM ---


def chebyshev_theorem(k: float) -> float:
    """Returns minimum proportion of values within k standard deviations of the mean."""
    if k <= 1:
        raise ValueError("k must be greater than 1")
    return 1 - (1 / k**2)


# --- PDF AND CDF FOR NORMAL DISTRIBUTION ---


def normal_pdf(x: float, mean: float = 0.0, std_dev: float = 1.0) -> float:
    """Calculate the probability density function (PDF) for a normal distribution.

    Args:
        x: Value at which to evaluate the PDF
        mean: Mean of the normal distribution (default: 0.0)
        std_dev: Standard deviation of the normal distribution (default: 1.0)

    Returns:
        PDF value at x

    Raises:
        ValueError: If std_dev is not positive

    Example:
        >>> normal_pdf(0, mean=0, std_dev=1)
        0.3989422804014327
    """
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive")
    return float(norm.pdf(x, loc=mean, scale=std_dev))


def normal_cdf(x: float, mean: float = 0.0, std_dev: float = 1.0) -> float:
    """Calculate the cumulative distribution function (CDF) for a normal distribution.

    Args:
        x: Value at which to evaluate the CDF
        mean: Mean of the normal distribution (default: 0.0)
        std_dev: Standard deviation of the normal distribution (default: 1.0)

    Returns:
        CDF value at x (probability that X <= x)

    Raises:
        ValueError: If std_dev is not positive

    Example:
        >>> normal_cdf(0, mean=0, std_dev=1)
        0.5
        >>> normal_cdf(1.96, mean=0, std_dev=1)
        0.9750021048517795
    """
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive")
    return float(norm.cdf(x, loc=mean, scale=std_dev))


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
