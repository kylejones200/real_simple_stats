"""Demo: Using PlotSmith with Real Simple Stats.

Requires: pip install real-simple-stats[plots]
"""

import logging

import numpy as np

from real_simple_stats import descriptive_statistics as desc
from real_simple_stats.linear_regression_utils import pearson_correlation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data for demos
data = [72, 68, 75, 71, 69, 74, 70, 73, 67, 72, 76, 71, 69, 74, 70]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2.1, 4.2, 5.8, 4.1, 6.2, 7.1, 8.3, 9.2, 10.1, 11.0]
observed = [20, 30, 25]
expected = [25, 25, 25]

try:
    from plotsmith import plot_histogram, plot_bar, plot_heatmap

    # Histogram of data (distribution)
    fig, ax = plot_histogram(data, bins=8, title="Score Distribution")
    fig.savefig("plotsmith_histogram.png", bbox_inches="tight", dpi=150)
    logger.info("Saved plotsmith_histogram.png")

    # Bar chart: observed vs expected (chi-square style)
    categories = [f"Cat {i+1}" for i in range(len(observed))]
    fig, ax = plot_bar(
        categories,
        observed,
        force_zero=True,
        title="Observed vs Expected",
    )
    # Add expected as second set (if PlotSmith supports multiple series)
    # Otherwise this shows observed; run plot_bar again for expected
    fig.savefig("plotsmith_observed.png", bbox_inches="tight", dpi=150)
    logger.info("Saved plotsmith_observed.png")

    # Correlation heatmap (e.g., from regression data)
    X = np.column_stack([x, y])
    corr = np.corrcoef(X.T)
    fig, ax = plot_heatmap(
        corr,
        annotate=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        title="Correlation Matrix",
    )
    fig.savefig("plotsmith_heatmap.png", bbox_inches="tight", dpi=150)
    logger.info("Saved plotsmith_heatmap.png")

    # Summary stats
    logger.info("\nData mean: %.2f", desc.mean(data))
    logger.info("Pearson r(x,y): %.4f", pearson_correlation(x, y))

except ImportError:
    logger.warning("PlotSmith not installed. Run: pip install real-simple-stats[plots]")
