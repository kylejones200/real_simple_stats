"""Recipe: Compare Two Groups with t-test and Effect Size

This recipe shows how to compare two groups using a t-test,
calculate effect size, and interpret the results.
"""

import real_simple_stats as rss
from real_simple_stats import hypothesis_testing as ht
from real_simple_stats import effect_sizes as es

# Example: Comparing test scores between two teaching methods
method_a_scores = [78, 82, 85, 79, 83, 88, 81, 84, 87, 80]
method_b_scores = [72, 75, 78, 74, 76, 79, 73, 77, 75, 74]

print("=" * 60)
print("Comparing Two Groups: Teaching Method A vs Method B")
print("=" * 60)

# Step 1: Descriptive statistics for each group
print("\n1. Descriptive Statistics")
print("-" * 60)
from real_simple_stats import descriptive_statistics as desc

mean_a = desc.mean(method_a_scores)
mean_b = desc.mean(method_b_scores)
std_a = desc.sample_std_dev(method_a_scores)
std_b = desc.sample_std_dev(method_b_scores)

print(f"Method A: Mean = {mean_a:.2f}, SD = {std_a:.2f}, n = {len(method_a_scores)}")
print(f"Method B: Mean = {mean_b:.2f}, SD = {std_b:.2f}, n = {len(method_b_scores)}")
print(f"Difference: {mean_a - mean_b:.2f} points")

# Step 2: Two-sample t-test
print("\n2. Hypothesis Test")
print("-" * 60)
# Note: For two-sample t-test, use scipy.stats or calculate manually
# This is a simplified example - in practice you'd use scipy.stats.ttest_ind
import numpy as np
from scipy import stats
t_stat, p_value = stats.ttest_ind(method_a_scores, method_b_scores)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("Result: Statistically significant (p < 0.05)")
    print("Conclusion: There is a significant difference between the methods")
else:
    print("Result: Not statistically significant (p >= 0.05)")
    print("Conclusion: No significant difference detected")

# Step 3: Effect size (Cohen's d)
print("\n3. Effect Size")
print("-" * 60)
cohens_d = es.cohens_d(method_a_scores, method_b_scores)
print(f"Cohen's d: {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    size = "negligible"
elif abs(cohens_d) < 0.5:
    size = "small"
elif abs(cohens_d) < 0.8:
    size = "medium"
else:
    size = "large"

print(f"Effect size interpretation: {size}")

# Step 4: Confidence interval for the difference
print("\n4. Confidence Interval for Mean Difference")
print("-" * 60)
# Calculate 95% CI for difference
se_diff = ((std_a**2 / len(method_a_scores)) + (std_b**2 / len(method_b_scores))) ** 0.5
df = len(method_a_scores) + len(method_b_scores) - 2
t_critical = ht.critical_value_t(0.05, df, test_type="two-tailed")
margin = t_critical * se_diff
ci_lower = (mean_a - mean_b) - margin
ci_upper = (mean_a - mean_b) + margin

print(f"95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Interpretation: We're 95% confident the true difference is between "
      f"{ci_lower:.2f} and {ci_upper:.2f} points")

# Step 5: Summary
print("\n5. Summary")
print("-" * 60)
print(f"Method A appears to be {'better' if mean_a > mean_b else 'worse'} "
      f"than Method B by {abs(mean_a - mean_b):.2f} points on average.")
print(f"This difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
      f"(p = {p_value:.4f})")
print(f"The effect size is {size} (d = {cohens_d:.4f})")

