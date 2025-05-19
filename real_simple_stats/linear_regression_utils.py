from typing import List, Tuple
import numpy as np
from scipy.stats import linregress

# --- SCATTER PLOT PREP (data only, no plotting here) ---

def prepare_scatter_data(x: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    """Prepare data for plotting a scatter plot (returns as-is)."""
    return x, y

# --- CORRELATION ---

def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Computes Pearson's correlation coefficient (r)."""
    return np.corrcoef(x, y)[0, 1]

def coefficient_of_determination(x: List[float], y: List[float]) -> float:
    """Returns R^2, the coefficient of determination."""
    r = pearson_correlation(x, y)
    return r ** 2

# --- LINEAR REGRESSION CALCULATIONS ---

def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float, float, float]:
    """
    Returns slope, intercept, r_value, p_value, std_err
    Formula: y = a + b*x
    """
    result = linregress(x, y)
    return result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr

def regression_equation(x: float, slope: float, intercept: float) -> float:
    """Compute predicted y value using regression line."""
    return slope * x + intercept

# --- MANUAL SLOPE/INTERCEPT CALCULATION (for education/demo) ---

def manual_slope_intercept(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Computes slope and intercept manually."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

# Example usage
if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    print("Correlation (r):", pearson_correlation(x, y))
    print("R²:", coefficient_of_determination(x, y))

    slope, intercept, r, p, stderr = linear_regression(x, y)
    print("Slope:", slope)
    print("Intercept:", intercept)
    print("Regression equation for x=6:", regression_equation(6, slope, intercept))

    m_slope, m_intercept = manual_slope_intercept(x, y)
    print("Manual slope:", m_slope)
    print("Manual intercept:", m_intercept)
