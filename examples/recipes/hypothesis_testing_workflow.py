"""Recipe: Complete Hypothesis Testing Workflow

This recipe demonstrates the full workflow for hypothesis testing:
1. State hypotheses
2. Choose significance level
3. Perform test
4. Make decision
5. Interpret results
"""

import logging

from real_simple_stats import hypothesis_testing as ht

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example: Testing if a new drug lowers blood pressure
# Null hypothesis: μ = 120 (no change)
# Alternative: μ < 120 (blood pressure decreases)

# Sample data: blood pressure readings after treatment
blood_pressure = [118, 115, 122, 119, 117, 121, 116, 120, 118, 119]

logger.info("=" * 60)
logger.info("Hypothesis Testing: Drug Effectiveness Study")
logger.info("=" * 60)

# Step 1: State the hypotheses
logger.info("\n1. Hypotheses")
logger.info("-" * 60)
logger.info("H₀ (Null): μ = 120 mmHg (drug has no effect)")
logger.info("H₁ (Alternative): μ < 120 mmHg (drug lowers blood pressure)")
logger.info("This is a one-tailed (left-tailed) test")

# Step 2: Set significance level
alpha = 0.05
logger.info(f"\n2. Significance Level: α = {alpha}")

# Step 3: Check assumptions
logger.info("\n3. Assumptions Check")
logger.info("-" * 60)
n = len(blood_pressure)
logger.info(f"Sample size: n = {n}")
if n < 30:
    logger.info("Small sample size - t-test is appropriate")
    logger.info("   (For n ≥ 30, z-test could be used)")
else:
    logger.info("Sample size is large enough")

# Check for normality (simplified - in practice, use Q-Q plot or Shapiro-Wilk)
from real_simple_stats import descriptive_statistics as desc

mean_bp = desc.mean(blood_pressure)
median_bp = desc.median(blood_pressure)
if abs(mean_bp - median_bp) < 2:
    logger.info("Data appears approximately normal (mean approx median)")
else:
    logger.info("Data may not be normally distributed")

# Step 4: Perform the test
logger.info("\n4. Statistical Test")
logger.info("-" * 60)
mu_null = 120
t_stat, p_value = ht.one_sample_t_test(blood_pressure, mu_null)

logger.info(f"Sample mean: {mean_bp:.2f} mmHg")
logger.info(f"Null hypothesis mean: {mu_null} mmHg")
logger.info(f"t-statistic: {t_stat:.4f}")
logger.info(f"p-value: {p_value:.6f}")

# Step 5: Make decision
logger.info("\n5. Decision")
logger.info("-" * 60)
if p_value < alpha:
    decision = "Reject H₀"
    conclusion = "There is sufficient evidence that the drug lowers blood pressure"
else:
    decision = "Fail to reject H₀"
    conclusion = "There is insufficient evidence that the drug lowers blood pressure"

logger.info(f"Since p-value ({p_value:.4f}) {'<' if p_value < alpha else '≥'} α ({alpha}),")
logger.info(f"Decision: {decision}")
logger.info(f"Conclusion: {conclusion}")

# Step 6: Calculate confidence interval
logger.info("\n6. Confidence Interval")
logger.info("-" * 60)
std_bp = desc.sample_std_dev(blood_pressure)
se = std_bp / (n**0.5)
df = n - 1
t_critical = ht.critical_value_t(alpha, df, test_type="one-tailed")
margin = t_critical * se
ci_lower = mean_bp - margin
ci_upper = mean_bp + margin

logger.info(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] mmHg")
logger.info("Interpretation: We're 95% confident the true mean blood pressure")
logger.info(f"after treatment is between {ci_lower:.2f} and {ci_upper:.2f} mmHg")

# Step 7: Effect size
logger.info("\n7. Effect Size")
logger.info("-" * 60)
effect_size = (mean_bp - mu_null) / std_bp
logger.info(f"Cohen's d: {effect_size:.4f}")

if abs(effect_size) < 0.2:
    effect = "negligible"
elif abs(effect_size) < 0.5:
    effect = "small"
elif abs(effect_size) < 0.8:
    effect = "medium"
else:
    effect = "large"

logger.info(f"Effect size: {effect}")
logger.info(
    f"Practical significance: The drug {'does' if abs(effect_size) >= 0.2 else 'may not'} "
    f"have a {'clinically' if abs(effect_size) >= 0.5 else 'practically'} significant effect"
)

# Step 8: Final summary
logger.info("\n8. Summary")
logger.info("-" * 60)
logger.info("• Statistical test: One-sample t-test (left-tailed)")
logger.info(f"• Sample mean: {mean_bp:.2f} mmHg (vs. null: {mu_null} mmHg)")
logger.info(f"• Test result: {decision} (p = {p_value:.4f})")
logger.info(f"• Effect size: {effect} (d = {effect_size:.4f})")
logger.info(f"• Conclusion: {conclusion}")
