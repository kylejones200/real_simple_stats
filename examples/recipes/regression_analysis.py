"""Recipe: Complete Linear Regression Analysis

This recipe demonstrates a full linear regression workflow:
1. Explore the relationship
2. Calculate correlation
3. Fit the regression model
4. Interpret coefficients
5. Assess model fit
6. Make predictions
"""

from real_simple_stats import descriptive_statistics as desc
from real_simple_stats import linear_regression_utils as lr

print("=" * 70)
print("Linear Regression Analysis: Study Hours vs Test Scores")
print("=" * 70)

# ============================================================================
# Data: Study Hours (x) and Test Scores (y)
# ============================================================================
study_hours = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
test_scores = [60, 65, 70, 75, 80, 85, 88, 90, 92, 95]

print("\nData:")
print("Study Hours (x):", study_hours)
print("Test Scores (y):", test_scores)

# ============================================================================
# Step 1: Exploratory Data Analysis
# ============================================================================
print("\n" + "=" * 70)
print("Step 1: Exploratory Data Analysis")
print("=" * 70)

mean_hours = desc.mean(study_hours)
mean_scores = desc.mean(test_scores)
std_hours = desc.sample_std_dev(study_hours)
std_scores = desc.sample_std_dev(test_scores)

print("\nStudy Hours:")
print(f"  Mean: {mean_hours:.1f} hours")
print(f"  Std Dev: {std_hours:.2f} hours")

print("\nTest Scores:")
print(f"  Mean: {mean_scores:.1f} points")
print(f"  Std Dev: {std_scores:.2f} points")

# ============================================================================
# Step 2: Correlation Analysis
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Correlation Analysis")
print("=" * 70)

r = lr.pearson_correlation(study_hours, test_scores)
r_squared = lr.coefficient_of_determination(study_hours, test_scores)

print(f"\nPearson Correlation Coefficient (r): {r:.4f}")

# Interpret correlation strength
if abs(r) < 0.3:
    strength = "weak"
elif abs(r) < 0.7:
    strength = "moderate"
else:
    strength = "strong"

direction = "positive" if r > 0 else "negative"
print(f"Interpretation: {strength} {direction} linear relationship")

print(f"\nCoefficient of Determination (R²): {r_squared:.4f}")
print(f"Interpretation: {r_squared:.1%} of the variance in test scores")
print("                is explained by study hours")

# ============================================================================
# Step 3: Fit the Regression Model
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Fit the Regression Model")
print("=" * 70)

slope, intercept, r_value, p_value, std_err = lr.linear_regression(
    study_hours, test_scores
)

print(f"\nRegression Equation: y = {intercept:.2f} + {slope:.2f}x")
print("\nCoefficients:")
print(f"  Intercept (a): {intercept:.2f}")
print("    Interpretation: Predicted test score when study hours = 0")
print(f"  Slope (b): {slope:.2f}")
print("    Interpretation: For each additional hour of study,")
print(f"                    test score increases by {slope:.2f} points")

# ============================================================================
# Step 4: Statistical Significance
# ============================================================================
print("\n" + "=" * 70)
print("Step 4: Statistical Significance")
print("=" * 70)

alpha = 0.05
print("\nHypothesis Test:")
print("  H₀: β = 0 (no linear relationship)")
print("  H₁: β ≠ 0 (linear relationship exists)")
print(f"  Significance level: α = {alpha}")

print("\nResults:")
print(f"  p-value: {p_value:.6f}")
print(f"  Standard error: {std_err:.4f}")

if p_value < alpha:
    print(f"\n✓ Decision: Reject H₀ (p = {p_value:.4f} < α = {alpha})")
    print("  Conclusion: There is a statistically significant")
    print("              linear relationship between study hours and test scores")
else:
    print(f"\n✗ Decision: Fail to reject H₀ (p = {p_value:.4f} ≥ α = {alpha})")
    print("  Conclusion: No significant linear relationship detected")

# ============================================================================
# Step 5: Model Fit Assessment
# ============================================================================
print("\n" + "=" * 70)
print("Step 5: Model Fit Assessment")
print("=" * 70)

print(f"\nR² = {r_squared:.4f} ({r_squared:.1%})")

if r_squared < 0.3:
    fit_quality = "poor"
elif r_squared < 0.7:
    fit_quality = "moderate"
else:
    fit_quality = "good"

print(f"Model fit: {fit_quality}")
print("\nInterpretation:")
if r_squared > 0.7:
    print("  ✓ The model explains most of the variation in test scores")
elif r_squared > 0.3:
    print("  ⚠️  The model explains some variation, but other factors")
    print("      may also be important")
else:
    print("  ✗ The model explains little variation - consider other")
    print("    predictors or check for non-linear relationships")

# ============================================================================
# Step 6: Predictions
# ============================================================================
print("\n" + "=" * 70)
print("Step 6: Making Predictions")
print("=" * 70)

# Predict for new values
new_hours = [12, 28, 55]
print("\nPredictions for new study hours:")

for hours in new_hours:
    predicted_score = lr.regression_equation(hours, slope, intercept)
    print(f"  {hours} hours → Predicted score: {predicted_score:.1f} points")

# ============================================================================
# Step 7: Manual Calculation (Educational)
# ============================================================================
print("\n" + "=" * 70)
print("Step 7: Manual Calculation (Understanding the Math)")
print("=" * 70)

manual_slope, manual_intercept = lr.manual_slope_intercept(study_hours, test_scores)

print("\nManual Calculation:")
print("  Slope (b) = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²")
print("  Intercept (a) = ȳ - b·x̄")

print("\nResults:")
print(f"  Manual slope: {manual_slope:.4f}")
print(f"  Manual intercept: {manual_intercept:.4f}")
print("  (Should match regression results above)")

# Verify they match
if abs(manual_slope - slope) < 0.01 and abs(manual_intercept - intercept) < 0.01:
    print("\n✓ Manual calculation matches regression function!")

# ============================================================================
# Summary and Interpretation
# ============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(f"""
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

print("=" * 70)
