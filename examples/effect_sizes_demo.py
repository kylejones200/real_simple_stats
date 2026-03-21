"""Demo: Effect sizes - Cohen's d, eta squared, odds ratio."""

import logging

from real_simple_stats.effect_sizes import (
    cohens_d,
    eta_squared,
    interpret_effect_size,
    odds_ratio,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Two independent groups
group1 = [22, 24, 26, 28, 30]
group2 = [18, 20, 22, 24, 26]
d = cohens_d(group1, group2)
logger.info("Cohen's d: %.3f", d)
logger.info("Interpretation: %s", interpret_effect_size(d, "d"))

# ANOVA effect size (multiple groups)
groups = [[10, 12, 14], [20, 22, 24], [30, 32, 34]]
eta2 = eta_squared(groups)
logger.info("\nEta squared (3 groups): %.4f", eta2)

# Odds ratio from 2x2 contingency table
# Rows: exposed / unexposed, Cols: disease / no disease
table = [[30, 70], [10, 90]]
or_val, (ci_low, ci_high) = odds_ratio(table)
logger.info("\nOdds ratio: %.2f (95%% CI: %.2f-%.2f)", or_val, ci_low, ci_high)
