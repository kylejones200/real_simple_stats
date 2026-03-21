"""Recipe: Complete Linear Regression Analysis

This recipe demonstrates a full linear regression workflow:
1. Explore the relationship
2. Calculate correlation
3. Fit the regression model
4. Interpret coefficients
5. Assess model fit
6. Make predictions
"""

import logging

from real_simple_stats import descriptive_statistics as desc
from real_simple_stats import linear_regression_utils as lr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("Linear Regression Analysis: Study Hours vs Test Scores")
logger.info("=" * 70)

# ============================================================================
# Data: Study Hours (x) and Test Scores (y)
# ============================================================================
study_hours = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
test_scores = [60, 65, 70, 75, 80, 85, 88, 90, 92, 95]

logger.info("\nData:")
logger.info("Study Hours (x): %s", study_hours)
logger.info("Test Scores (y): %s", test_scores)

# ============================================================================
# Step 1: Exploratory Data Analysis
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Step 1: Exploratory Data Analysis")
logger.info("=" * 70)

mean_hours = desc.mean(study_hours)
mean_scores = desc.mean(test_scores)
std_hours = desc.sample_std_dev(study_hours)
std_scores = desc.sample_std_dev(test_scores)

logger.info("\nStudy Hours:")
logger.info(f"  Mean: {mean_hours:.1f} hours")
logger.info(f"  Std Dev: {std_hours:.2f} hours")

logger.info("\nTest Scores:")
logger.info(f"  Mean: {mean_scores:.1f} points")
logger.info(f"  Std Dev: {std_scores:.2f} points")

# ============================================================================
# Step 2: Correlation Analysis
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Step 2: Correlation Analysis")
logger.info("=" * 70)

r = lr.pearson_correlation(study_hours, test_scores)
r_squared = lr.coefficient_of_determination(study_hours, test_scores)

logger.info(f"\nPearson Correlation Coefficient (r): {r:.4f}")

# Interpret correlation strength
if abs(r) < 0.3:
    strength = "weak"
elif abs(r) < 0.7:
    strength = "moderate"
else:
    strength = "strong"

direction = "positive" if r > 0 else "negative"
logger.info(f"Interpretation: {strength} {direction} linear relationship")

logger.info(f"\nCoefficient of Determination (R²): {r_squared:.4f}")
logger.info(f"Interpretation: {r_squared:.1%} of the variance in test scores")
logger.info("                is explained by study hours")

# ============================================================================
# Step 3: Fit the Regression Model
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Step 3: Fit the Regression Model")
logger.info("=" * 70)

slope, intercept, r_value, p_value, std_err = lr.linear_regression(
    study_hours, test_scores
)

logger.info(f"\nRegression Equation: y = {intercept:.2f} + {slope:.2f}x")
logger.info("\nCoefficients:")
logger.info(f"  Intercept (a): {intercept:.2f}")
logger.info("    Interpretation: Predicted test score when study hours = 0")
logger.info(f"  Slope (b): {slope:.2f}")
logger.info("    Interpretation: For each additional hour of study,")
logger.info(f"                    test score increases by {slope:.2f} points")

# ============================================================================
# Step 4: Statistical Significance
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Step 4: Statistical Significance")
logger.info("=" * 70)

alpha = 0.05
logger.info("\nHypothesis Test:")
logger.info("  H₀: β = 0 (no linear relationship)")
logger.info("  H₁: β ≠ 0 (linear relationship exists)")
logger.info(f"  Significance level: α = {alpha}")

logger.info("\nResults:")
logger.info(f"  p-value: {p_value:.6f}")
logger.info(f"  Standard error: {std_err:.4f}")

if p_value < alpha:
    logger.info(f"\nDecision: Reject H0 (p = {p_value:.4f} < alpha = {alpha})")
    logger.info("  Conclusion: There is a statistically significant")
    logger.info("              linear relationship between study hours and test scores")
else:
    logger.info(f"\n✗ Decision: Fail to reject H₀ (p = {p_value:.4f} ≥ α = {alpha})")
    logger.info("  Conclusion: No significant linear relationship detected")

# ============================================================================
# Step 5: Model Fit Assessment
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Step 5: Model Fit Assessment")
logger.info("=" * 70)

logger.info(f"\nR² = {r_squared:.4f} ({r_squared:.1%})")

if r_squared < 0.3:
    fit_quality = "poor"
elif r_squared < 0.7:
    fit_quality = "moderate"
else:
    fit_quality = "good"

logger.info(f"Model fit: {fit_quality}")
logger.info("\nInterpretation:")
if r_squared > 0.7:
    logger.info("  The model explains most of the variation in test scores")
elif r_squared > 0.3:
    logger.info("  The model explains some variation, but other factors")
    logger.info("      may also be important")
else:
    logger.info("  ✗ The model explains little variation - consider other")
    logger.info("    predictors or check for non-linear relationships")

# ============================================================================
# Step 6: Predictions
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Step 6: Making Predictions")
logger.info("=" * 70)

# Predict for new values
new_hours = [12, 28, 55]
logger.info("\nPredictions for new study hours:")

for hours in new_hours:
    predicted_score = lr.regression_equation(hours, slope, intercept)
    logger.info(f"  {hours} hours → Predicted score: {predicted_score:.1f} points")

# ============================================================================
# Step 7: Manual Calculation (Educational)
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Step 7: Manual Calculation (Understanding the Math)")
logger.info("=" * 70)

manual_slope, manual_intercept = lr.manual_slope_intercept(study_hours, test_scores)

logger.info("\nManual Calculation:")
logger.info("  Slope (b) = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²")
logger.info("  Intercept (a) = ȳ - b·x̄")

logger.info("\nResults:")
logger.info(f"  Manual slope: {manual_slope:.4f}")
logger.info(f"  Manual intercept: {manual_intercept:.4f}")
logger.info("  (Should match regression results above)")

# Verify they match
if abs(manual_slope - slope) < 0.01 and abs(manual_intercept - intercept) < 0.01:
    logger.info("\nManual calculation matches regression function!")

# ============================================================================
# Summary and Interpretation
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("Summary")
logger.info("=" * 70)

logger.info(f"""
Key Findings:
1. Correlation: r = {r:.3f} ({strength} {direction} relationship)
2. Model: y = {intercept:.2f} + {slope:.2f}x
3. Fit: R² = {r_squared:.1%} ({fit_quality} fit)
4. Significance: {"Significant" if p_value < alpha else "Not significant"} 
   (p = {p_value:.4f})

Practical Interpretation:
- Each additional hour of study is associated with a {slope:.2f} point
  increase in test score, on average
- Study hours explain {r_squared:.1%} of the variation in test scores
- The relationship is {"statistically significant" if p_value < alpha else "not statistically significant"}

Limitations:
- Correlation does not imply causation
- Other factors (prior knowledge, test difficulty, etc.) may also matter
- Predictions outside the observed range (extrapolation) should be used cautiously
""")

logger.info("=" * 70)
