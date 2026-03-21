"""Recipe: Professional vs Other Frogs - Jump Distance Comparison

This recipe demonstrates how to compare two groups (professional frogs vs other frogs)
using the frog jump dataset. This shows:

1. Group comparison with descriptive statistics
2. Two-sample t-test for mean difference
3. Effect size calculation (Cohen's d)
4. Confidence interval for the difference
5. Practical interpretation

Key Finding: Professional frogs jump significantly farther than other frogs!
"""

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

from real_simple_stats import descriptive_statistics as desc
from real_simple_stats import effect_sizes as es

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the frog jump data
data_file = Path(__file__).parent.parent / "data" / "froggy.csv"

logger.info("=" * 70)
logger.info("Professional Frogs vs Other Frogs: Jump Distance Comparison")
logger.info("=" * 70)

# Read and separate the data by frog type
pro_jumps = []
other_jumps = []

with open(data_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            distance = float(row["distance"])
            frog_type = row.get("frog_type", "").strip().lower()

            # Only include successful jumps (distance > 0)
            if distance > 0:
                if frog_type == "pro":
                    pro_jumps.append(distance)
                else:
                    other_jumps.append(distance)
        except (ValueError, KeyError):
            continue

logger.info("\nDataset Summary:")
logger.info(f"  Professional frogs: {len(pro_jumps)} successful jumps")
logger.info(f"  Other frogs: {len(other_jumps)} successful jumps")
logger.info(f"  Total: {len(pro_jumps) + len(other_jumps)} successful jumps")

# Step 1: Descriptive Statistics for Each Group
logger.info("\n" + "=" * 70)
logger.info("1. Descriptive Statistics by Group")
logger.info("=" * 70)

# Professional frogs
pro_mean = desc.mean(pro_jumps)
pro_median = desc.median(pro_jumps)
pro_std = desc.sample_std_dev(pro_jumps)
pro_cv = desc.coefficient_of_variation(pro_jumps) * 100
pro_summary = desc.five_number_summary(pro_jumps)

logger.info("\nProfessional Frogs:")
logger.info(f"  Sample size (n): {len(pro_jumps)}")
logger.info(f"  Mean: {pro_mean:.2f} cm")
logger.info(f"  Median: {pro_median:.2f} cm")
logger.info(f"  Standard deviation: {pro_std:.2f} cm")
logger.info(f"  Coefficient of variation: {pro_cv:.1f}%")
logger.info(f"  Range: {pro_summary['min']:.2f} - {pro_summary['max']:.2f} cm")
logger.info(f"  IQR: {pro_summary['Q3'] - pro_summary['Q1']:.2f} cm")

# Other frogs
other_mean = desc.mean(other_jumps)
other_median = desc.median(other_jumps)
other_std = desc.sample_std_dev(other_jumps)
other_cv = desc.coefficient_of_variation(other_jumps) * 100
other_summary = desc.five_number_summary(other_jumps)

logger.info("\nOther Frogs:")
logger.info(f"  Sample size (n): {len(other_jumps)}")
logger.info(f"  Mean: {other_mean:.2f} cm")
logger.info(f"  Median: {other_median:.2f} cm")
logger.info(f"  Standard deviation: {other_std:.2f} cm")
logger.info(f"  Coefficient of variation: {other_cv:.1f}%")
logger.info(f"  Range: {other_summary['min']:.2f} - {other_summary['max']:.2f} cm")
logger.info(f"  IQR: {other_summary['Q3'] - other_summary['Q1']:.2f} cm")

# Difference
mean_difference = pro_mean - other_mean
logger.info("\nDifference:")
logger.info(f"  Mean difference: {mean_difference:.2f} cm")
logger.info(f"  Professional frogs jump {mean_difference:.2f} cm farther on average")
logger.info(f"  That's {mean_difference / other_mean * 100:.1f}% farther than other frogs!")

# Step 2: Hypothesis Test
logger.info("\n" + "=" * 70)
logger.info("2. Hypothesis Test: Do Professional Frogs Jump Farther?")
logger.info("=" * 70)

logger.info("\nResearch Question: Is there a significant difference in jump distance")
logger.info("                  between professional and other frogs?")

logger.info("\nHypotheses:")
logger.info("  H₀: μ_pro = μ_other (no difference in mean jump distance)")
logger.info("  H₁: μ_pro > μ_other (professional frogs jump farther)")
logger.info("  This is a one-tailed (right-tailed) test")

alpha = 0.05
logger.info(f"  Significance level: α = {alpha}")

# Perform two-sample t-test using scipy (since real_simple_stats doesn't have it)
from scipy.stats import ttest_ind

t_stat, p_value_two_tailed = ttest_ind(pro_jumps, other_jumps, equal_var=False)
# For one-tailed test, divide p-value by 2 (since we expect pro > other)
p_value = p_value_two_tailed / 2

logger.info("\nTest Results:")
logger.info(f"  Professional mean: {pro_mean:.2f} cm")
logger.info(f"  Other mean: {other_mean:.2f} cm")
logger.info(f"  Mean difference: {mean_difference:.2f} cm")
logger.info(f"  t-statistic: {t_stat:.4f}")
logger.info(f"  p-value (one-tailed): {p_value:.6f}")

# Make decision
if p_value < alpha:
    decision = "Reject H₀"
    conclusion = "There is sufficient evidence that professional frogs jump farther"
    significance = "statistically significant"
else:
    decision = "Fail to reject H₀"
    conclusion = "There is insufficient evidence that professional frogs jump farther"
    significance = "not statistically significant"

logger.info(
    f"\nDecision: {decision} (p = {p_value:.4f} {'<' if p_value < alpha else '≥'} α = {alpha})"
)
logger.info(f"Conclusion: {conclusion}")
logger.info(f"The difference is {significance}!")

# Step 3: Effect Size
logger.info("\n" + "=" * 70)
logger.info("3. Effect Size: How Large is the Difference?")
logger.info("=" * 70)

cohens_d = es.cohens_d(pro_jumps, other_jumps)
logger.info(f"\nCohen's d: {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_size = "negligible"
elif abs(cohens_d) < 0.5:
    effect_size = "small"
elif abs(cohens_d) < 0.8:
    effect_size = "medium"
else:
    effect_size = "large"

logger.info(f"\nEffect Size Interpretation: {effect_size}")
logger.info("\nThis means:")
if abs(cohens_d) < 0.2:
    logger.info("  The difference, while statistically significant, is very small.")
    logger.info("  Professional frogs jump farther, but the practical difference is minimal.")
elif abs(cohens_d) < 0.5:
    logger.info("  There is a small but meaningful difference.")
    logger.info("  Professional frogs consistently jump farther than other frogs.")
elif abs(cohens_d) < 0.8:
    logger.info("  There is a medium-sized, practically important difference.")
    logger.info("  Professional frogs show substantially better jumping performance.")
else:
    logger.info("  There is a large, very important difference.")
    logger.info("  Professional frogs demonstrate dramatically better jumping ability.")

# Step 4: Confidence Interval for the Difference
logger.info("\n" + "=" * 70)
logger.info("4. Confidence Interval for Mean Difference")
logger.info("=" * 70)

# Calculate 95% CI for difference using Welch's t-test (unequal variances)
n1, n2 = len(pro_jumps), len(other_jumps)
s1, s2 = pro_std, other_std

# Standard error for difference (Welch's formula)
se_diff = np.sqrt((s1**2 / n1) + (s2**2 / n2))

# Degrees of freedom (Welch-Satterthwaite)
df_welch = ((s1**2 / n1 + s2**2 / n2) ** 2) / (
    (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
)

# Critical t-value
t_critical = t_dist.ppf(0.975, df_welch)  # 95% CI, two-tailed

# Margin of error
margin = t_critical * se_diff
ci_lower = mean_difference - margin
ci_upper = mean_difference + margin

logger.info(f"\n95% Confidence Interval for difference: [{ci_lower:.2f}, {ci_upper:.2f}] cm")
logger.info("\nInterpretation:")
logger.info("  We're 95% confident that professional frogs jump between")
logger.info(f"  {ci_lower:.2f} and {ci_upper:.2f} cm farther than other frogs, on average.")
if ci_lower > 0:
    logger.info("  This interval does NOT include zero,")
    logger.info("  which confirms the difference is statistically significant.")
else:
    logger.info("  This interval includes zero,")
    logger.info("  which suggests the difference may not be significant.")

# Step 5: Visual Summary
logger.info("\n" + "=" * 70)
logger.info("5. Summary Statistics")
logger.info("=" * 70)

logger.info(f"\n{'Metric':<30} {'Professional':<15} {'Other':<15} {'Difference':<15}")
logger.info("-" * 75)
logger.info(f"{'Sample size (n)':<30} {len(pro_jumps):<15} {len(other_jumps):<15} {'':<15}")
logger.info(
    f"{'Mean (cm)':<30} {pro_mean:<15.2f} {other_mean:<15.2f} {mean_difference:>+14.2f}"
)
logger.info(
    f"{'Median (cm)':<30} {pro_median:<15.2f} {other_median:<15.2f} {pro_median - other_median:>+14.2f}"
)
logger.info(f"{'Std Dev (cm)':<30} {pro_std:<15.2f} {other_std:<15.2f} {'':<15}")
logger.info(
    f"{'Coefficient of Variation (%)':<30} {pro_cv:<15.1f} {other_cv:<15.1f} {'':<15}"
)

# Step 6: Final Interpretation
logger.info("\n" + "=" * 70)
logger.info("6. Final Interpretation")
logger.info("=" * 70)

logger.info("\nKey Findings:")
logger.info(f"  Professional frogs jump {mean_difference:.2f} cm farther on average")
logger.info(f"  This difference is {significance} (p = {p_value:.4f})")
logger.info(f"  Effect size is {effect_size} (Cohen's d = {cohens_d:.4f})")
logger.info(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] cm")

logger.info("\nPractical Implications:")
if abs(cohens_d) >= 0.8:
    logger.info("  • The difference is large enough to be practically important")
    logger.info("  • Professional frogs show substantially better performance")
elif abs(cohens_d) >= 0.5:
    logger.info("  • The difference is meaningful in practice")
    logger.info("  • Professional frogs consistently outperform other frogs")
else:
    logger.info("  • While statistically significant, the practical difference is smaller")
    logger.info("  • Both groups show similar jumping ability overall")

logger.info("\nStatistical Conclusion:")
logger.info(
    f"  We have {'strong' if p_value < 0.01 else 'moderate' if p_value < 0.05 else 'weak'} evidence"
)
logger.info("  that professional frogs jump farther than other frogs.")
logger.info(
    f"  The mean difference of {mean_difference:.2f} cm is {'large' if abs(cohens_d) >= 0.8 else 'moderate' if abs(cohens_d) >= 0.5 else 'small but'} meaningful."
)

logger.info("\n" + "=" * 70)
logger.info("This analysis demonstrates how to compare two groups using")
logger.info("Real Simple Stats with real-world data from the BANA statistics book.")
logger.info("=" * 70)
