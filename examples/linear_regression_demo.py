"""Demo: Correlation and linear regression."""

import logging

from real_simple_stats.linear_regression_utils import (
    coefficient_of_determination,
    linear_regression,
    pearson_correlation,
    regression_equation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data: study hours vs exam score
hours = [2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [50, 55, 60, 65, 70, 75, 80, 85, 90]

# Pearson correlation
r = pearson_correlation(hours, scores)
logger.info("Pearson r: %.4f", r)

# R-squared
r_squared = coefficient_of_determination(hours, scores)
logger.info("R²: %.4f", r_squared)

# Linear regression
slope, intercept, r_val, r_sq, std_err = linear_regression(hours, scores)
logger.info("Slope: %.2f, Intercept: %.2f", slope, intercept)

# Predict for new value
predicted = regression_equation(5.5, slope, intercept)
logger.info("Predicted score for 5.5 hours: %.2f", predicted)
