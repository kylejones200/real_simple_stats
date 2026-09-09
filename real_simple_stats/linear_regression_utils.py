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
