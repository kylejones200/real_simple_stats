"""Recipe: Using Verbose Mode and Assumptions Checking

This recipe demonstrates the new educational features:
1. Step-by-step verbose calculations
2. Statistical assumptions checking
"""

import logging

from real_simple_stats import assumptions as assump
from real_simple_stats import verbose_stats as vs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("Educational Features: Verbose Mode & Assumptions Checking")
logger.info("=" * 70)

# ============================================================================
# Example 1: Verbose T-Test
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Example 1: Step-by-Step T-Test (Verbose Mode)")
logger.info("=" * 70)

blood_pressure = [118, 115, 122, 119, 117, 121, 116, 120, 118, 119]
mu_null = 120

t_stat, p_value, t_critical, reject = vs.t_test_verbose(
    blood_pressure, mu_null=mu_null, alpha=0.05, test_type="less", verbose=True
)

# ============================================================================
# Example 2: Verbose Regression
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Example 2: Step-by-Step Regression (Verbose Mode)")
logger.info("=" * 70)

study_hours = [5, 10, 15, 20, 25]
test_scores = [60, 65, 70, 75, 80]

slope, intercept, r, p, se = vs.regression_verbose(
    study_hours, test_scores, verbose=True
)

# ============================================================================
# Example 3: Verbose Mean (Simple Example)
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Example 3: Step-by-Step Mean Calculation (Verbose Mode)")
logger.info("=" * 70)

data = [10, 20, 30, 40, 50]
mean_val = vs.mean_verbose(data, verbose=True)

# ============================================================================
# Example 4: Check T-Test Assumptions
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Example 4: Checking T-Test Assumptions")
logger.info("=" * 70)

results = assump.check_t_test_assumptions(blood_pressure, verbose=True)

# ============================================================================
# Example 5: Check Regression Assumptions
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Example 5: Checking Regression Assumptions")
logger.info("=" * 70)

regression_results = assump.check_regression_assumptions(
    study_hours, test_scores, verbose=True
)

# ============================================================================
# Example 6: Two-Sample T-Test Assumptions
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Example 6: Checking Two-Sample T-Test Assumptions")
logger.info("=" * 70)

group_a = [78, 82, 85, 79, 83]
group_b = [72, 75, 78, 74, 76]

two_sample_results = assump.check_t_test_assumptions(
    group_a, group2=group_b, verbose=True
)

logger.info("\n" + "=" * 70)
logger.info("Summary")
logger.info("=" * 70)
logger.info("""
These educational features help you:
1. Understand the math behind statistical tests
2. Verify that your data meets test assumptions
3. Learn when to use alternative methods

Use verbose=True to see step-by-step calculations.
Use assumptions checking before running tests to ensure validity.
""")
