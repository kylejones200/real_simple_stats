import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)
from . import _rss

# --- SCATTER PLOT PREP (data only, no plotting here) ---


def prepare_scatter_data(
    x: Sequence[float], y: Sequence[float]
) -> tuple[Sequence[float], Sequence[float]]:
    """Prepare data for plotting a scatter plot (returns as-is)."""
    return x, y


# --- CORRELATION ---


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Computes Pearson's correlation coefficient (r)."""
    return _rss.pearson_r(x, y)


def coefficient_of_determination(x: Sequence[float], y: Sequence[float]) -> float:
    """Returns R^2, the coefficient of determination."""
    r = pearson_correlation(x, y)
    return r**2


# --- LINEAR REGRESSION CALCULATIONS ---


def linear_regression(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float, float, float, float]:
    """
    Returns slope, intercept, r_value, p_value, std_err
    Formula: y = a + b*x
    """
    slope, intercept, rvalue, pvalue, stderr, _intercept_stderr = _rss.linregress(x, y)
    return (slope, intercept, rvalue, pvalue, stderr)


def regression_equation(x: float, slope: float, intercept: float) -> float:
    """Compute predicted y value using regression line."""
    return slope * x + intercept


# --- MANUAL SLOPE/INTERCEPT CALCULATION (for education/demo) ---


def manual_slope_intercept(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float]:
    """Computes slope and intercept manually."""
    x_mean = _rss.mean(x)
    y_mean = _rss.mean(y)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    slope = float(numerator / denominator)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    logger.info("Correlation (r): %s", pearson_correlation(x, y))
    logger.info("R²: %s", coefficient_of_determination(x, y))

    slope, intercept, r, p, stderr = linear_regression(x, y)
    logger.info("Slope: %s", slope)
    logger.info("Intercept: %s", intercept)
    logger.info("Regression equation for x=6: %s", regression_equation(6, slope, intercept))

    m_slope, m_intercept = manual_slope_intercept(x, y)
    logger.info("Manual slope: %s", m_slope)
    logger.info("Manual intercept: %s", m_intercept)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Correlation of the *ranks* of two variables.

    Spearman's rho measures monotonic association rather than linear
    association, so it is unaffected by outliers and does not assume the
    relationship is a straight line. It is Pearson's correlation applied to
    average ranks; tied values share their average rank.

    Args:
        x: First variable
        y: Second variable, the same length as x

    Returns:
        Spearman's rho, between -1 and 1

    Raises:
        ValueError: If the inputs differ in length or have fewer than 2 values

    Example:
        >>> spearman_correlation([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
        1.0
        >>> spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1])
        -1.0
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("Spearman correlation requires at least 2 values")

    from .hypothesis_testing import _average_ranks

    return _rss.pearson_r(_average_ranks(x), _average_ranks(y))


def calculate_residuals(
    y: Sequence[float], y_hat: Sequence[float]
) -> list[float]:
    """Differences between observed and predicted values.

    Residuals are ``observed - predicted``, so a positive residual means the
    model underpredicted that point. Plotting them against the predictions is
    the standard way to check a regression's assumptions.

    Args:
        y: Observed values
        y_hat: Predicted values, the same length as y

    Returns:
        List of residuals

    Raises:
        ValueError: If the inputs differ in length

    Example:
        >>> calculate_residuals([2.0, 4.0, 6.0], [2.5, 3.5, 6.0])
        [-0.5, 0.5, 0.0]
    """
    if len(y) != len(y_hat):
        raise ValueError("y and y_hat must have the same length")
    return [float(a) - float(b) for a, b in zip(y, y_hat)]
