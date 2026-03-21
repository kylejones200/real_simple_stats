"""Demo: Assumption checking for t-tests and regression."""

import logging

from real_simple_stats.assumptions import (
    check_regression_assumptions,
    check_t_test_assumptions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# T-test assumptions
data = [72, 68, 75, 71, 69, 74, 70, 73, 67, 72]
logger.info("T-test assumptions for single sample:")
check_t_test_assumptions(data, verbose=True)

# Two-group t-test
group1 = [22, 24, 26, 28, 30]
group2 = [20, 22, 24, 26, 28]
logger.info("\nT-test assumptions (two groups):")
check_t_test_assumptions(group1, group2=group2, verbose=True)

# Regression assumptions
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 4, 6, 7, 8, 9, 10, 11]
logger.info("\nRegression assumptions:")
check_regression_assumptions(x, y, verbose=True)
