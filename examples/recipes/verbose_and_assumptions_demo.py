"""Recipe: Using Verbose Mode and Assumptions Checking

This recipe demonstrates the new educational features:
1. Step-by-step verbose calculations
2. Statistical assumptions checking
"""

from real_simple_stats import verbose_stats as vs
from real_simple_stats import assumptions as assump

print("=" * 70)
print("Educational Features: Verbose Mode & Assumptions Checking")
print("=" * 70)

# ============================================================================
# Example 1: Verbose T-Test
# ============================================================================
print("\n" + "=" * 70)
print("Example 1: Step-by-Step T-Test (Verbose Mode)")
print("=" * 70)

blood_pressure = [118, 115, 122, 119, 117, 121, 116, 120, 118, 119]
mu_null = 120

t_stat, p_value, t_critical, reject = vs.t_test_verbose(
    blood_pressure,
    mu_null=mu_null,
    alpha=0.05,
    test_type="less",
    verbose=True
)

# ============================================================================
# Example 2: Verbose Regression
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Step-by-Step Regression (Verbose Mode)")
print("=" * 70)

study_hours = [5, 10, 15, 20, 25]
test_scores = [60, 65, 70, 75, 80]

slope, intercept, r, p, se = vs.regression_verbose(
    study_hours,
    test_scores,
    verbose=True
)

# ============================================================================
# Example 3: Verbose Mean (Simple Example)
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Step-by-Step Mean Calculation (Verbose Mode)")
print("=" * 70)

data = [10, 20, 30, 40, 50]
mean_val = vs.mean_verbose(data, verbose=True)

# ============================================================================
# Example 4: Check T-Test Assumptions
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: Checking T-Test Assumptions")
print("=" * 70)

results = assump.check_t_test_assumptions(
    blood_pressure,
    verbose=True
)

# ============================================================================
# Example 5: Check Regression Assumptions
# ============================================================================
print("\n" + "=" * 70)
print("Example 5: Checking Regression Assumptions")
print("=" * 70)

regression_results = assump.check_regression_assumptions(
    study_hours,
    test_scores,
    verbose=True
)

# ============================================================================
# Example 6: Two-Sample T-Test Assumptions
# ============================================================================
print("\n" + "=" * 70)
print("Example 6: Checking Two-Sample T-Test Assumptions")
print("=" * 70)

group_a = [78, 82, 85, 79, 83]
group_b = [72, 75, 78, 74, 76]

two_sample_results = assump.check_t_test_assumptions(
    group_a,
    group2=group_b,
    verbose=True
)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
These educational features help you:
1. Understand the math behind statistical tests
2. Verify that your data meets test assumptions
3. Learn when to use alternative methods

Use verbose=True to see step-by-step calculations.
Use assumptions checking before running tests to ensure validity.
""")

