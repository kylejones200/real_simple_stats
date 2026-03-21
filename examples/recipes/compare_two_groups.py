"""Recipe: Compare Two Groups with t-test and Effect Size

This recipe shows how to compare two groups using a t-test,
calculate effect size, and interpret the results.
"""

from real_simple_stats import effect_sizes as es
from real_simple_stats import hypothesis_testing as ht

# Example: Comparing test scores between two teaching methods
method_a_scores = [78, 82, 85, 79, 83, 88, 81, 84, 87, 80]
method_b_scores = [72, 75, 78, 74, 76, 79, 73, 77, 75, 74]

logger.info("=" * 60)
logger.info("Comparing Two Groups: Teaching Method A vs Method B")
logger.info("=" * 60)

# Step 1: Descriptive statistics for each group
logger.info("\n1. Descriptive Statistics")
logger.info("-" * 60)
import logging

from real_simple_stats import descriptive_statistics as desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mean_a = desc.mean(method_a_scores)
mean_b = desc.mean(method_b_scores)
std_a = desc.sample_std_dev(method_a_scores)
std_b = desc.sample_std_dev(method_b_scores)

logger.info(f"Method A: Mean = {mean_a:.2f}, SD = {std_a:.2f}, n = {len(method_a_scores)}")
logger.info(f"Method B: Mean = {mean_b:.2f}, SD = {std_b:.2f}, n = {len(method_b_scores)}")
logger.info(f"Difference: {mean_a - mean_b:.2f} points")

# Step 2: Two-sample t-test
logger.info("\n2. Hypothesis Test")
logger.info("-" * 60)
# Note: For two-sample t-test, use scipy.stats or calculate manually
# This is a simplified example - in practice you'd use scipy.stats.ttest_ind
from scipy import stats

t_stat, p_value = stats.ttest_ind(method_a_scores, method_b_scores)
logger.info(f"t-statistic: {t_stat:.4f}")
logger.info(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    logger.info("Result: Statistically significant (p < 0.05)")
    logger.info("Conclusion: There is a significant difference between the methods")
else:
    logger.info("Result: Not statistically significant (p >= 0.05)")
    logger.info("Conclusion: No significant difference detected")

# Step 3: Effect size (Cohen's d)
logger.info("\n3. Effect Size")
logger.info("-" * 60)
cohens_d = es.cohens_d(method_a_scores, method_b_scores)
logger.info(f"Cohen's d: {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    size = "negligible"
elif abs(cohens_d) < 0.5:
    size = "small"
elif abs(cohens_d) < 0.8:
    size = "medium"
else:
    size = "large"

logger.info(f"Effect size interpretation: {size}")

# Step 4: Confidence interval for the difference
logger.info("\n4. Confidence Interval for Mean Difference")
logger.info("-" * 60)
# Calculate 95% CI for difference
se_diff = ((std_a**2 / len(method_a_scores)) + (std_b**2 / len(method_b_scores))) ** 0.5
df = len(method_a_scores) + len(method_b_scores) - 2
t_critical = ht.critical_value_t(0.05, df, test_type="two-tailed")
margin = t_critical * se_diff
ci_lower = (mean_a - mean_b) - margin
ci_upper = (mean_a - mean_b) + margin

logger.info(f"95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
logger.info(
    f"Interpretation: We're 95% confident the true difference is between "
    f"{ci_lower:.2f} and {ci_upper:.2f} points"
)

# Step 5: Summary
logger.info("\n5. Summary")
logger.info("-" * 60)
logger.info(
    f"Method A appears to be {'better' if mean_a > mean_b else 'worse'} "
    f"than Method B by {abs(mean_a - mean_b):.2f} points on average."
)
logger.info(
    f"This difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
    f"(p = {p_value:.4f})"
)
logger.info(f"The effect size is {size} (d = {cohens_d:.4f})")
