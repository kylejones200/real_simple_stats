"""Demo: Multivariate methods - multiple regression and PCA."""

import logging

from real_simple_stats.multivariate import multiple_regression, pca

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multiple regression: predict y from X
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
y = [3, 5, 7, 9, 11, 13, 15, 17, 19]

result = multiple_regression(X, y, include_intercept=True)
logger.info("Multiple regression:")
logger.info("  Coefficients: %s", result["coefficients"])
logger.info("  R²: %.4f", result["r_squared"])
logger.info("  Predicted y for [5,6]: %.2f", result["predictions"][4])

# PCA: dimensionality reduction
pca_result = pca(X, n_components=2)
logger.info("\nPCA:")
logger.info("  Explained variance: %s", pca_result["explained_variance"])
logger.info(
    "  Total variance explained: %.4f", sum(pca_result["explained_variance"])
)
logger.info(
    "  Components shape: %s, %s",
    len(pca_result["components"]),
    len(pca_result["components"][0]),
)
