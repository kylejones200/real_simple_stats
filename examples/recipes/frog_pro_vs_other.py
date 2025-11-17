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
from pathlib import Path
from real_simple_stats import descriptive_statistics as desc
from real_simple_stats import hypothesis_testing as ht
from real_simple_stats import effect_sizes as es
from scipy.stats import t as t_dist
import numpy as np

# Load the frog jump data
data_file = Path(__file__).parent.parent / "data" / "froggy.csv"

print("=" * 70)
print("Professional Frogs vs Other Frogs: Jump Distance Comparison")
print("=" * 70)

# Read and separate the data by frog type
pro_jumps = []
other_jumps = []

with open(data_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            distance = float(row['distance'])
            frog_type = row.get('frog_type', '').strip().lower()
            
            # Only include successful jumps (distance > 0)
            if distance > 0:
                if frog_type == 'pro':
                    pro_jumps.append(distance)
                else:
                    other_jumps.append(distance)
        except (ValueError, KeyError):
            continue

print(f"\nDataset Summary:")
print(f"  Professional frogs: {len(pro_jumps)} successful jumps")
print(f"  Other frogs: {len(other_jumps)} successful jumps")
print(f"  Total: {len(pro_jumps) + len(other_jumps)} successful jumps")

# Step 1: Descriptive Statistics for Each Group
print("\n" + "=" * 70)
print("1. Descriptive Statistics by Group")
print("=" * 70)

# Professional frogs
pro_mean = desc.mean(pro_jumps)
pro_median = desc.median(pro_jumps)
pro_std = desc.sample_std_dev(pro_jumps)
pro_cv = desc.coefficient_of_variation(pro_jumps) * 100
pro_summary = desc.five_number_summary(pro_jumps)

print("\nProfessional Frogs:")
print(f"  Sample size (n): {len(pro_jumps)}")
print(f"  Mean: {pro_mean:.2f} cm")
print(f"  Median: {pro_median:.2f} cm")
print(f"  Standard deviation: {pro_std:.2f} cm")
print(f"  Coefficient of variation: {pro_cv:.1f}%")
print(f"  Range: {pro_summary['min']:.2f} - {pro_summary['max']:.2f} cm")
print(f"  IQR: {pro_summary['Q3'] - pro_summary['Q1']:.2f} cm")

# Other frogs
other_mean = desc.mean(other_jumps)
other_median = desc.median(other_jumps)
other_std = desc.sample_std_dev(other_jumps)
other_cv = desc.coefficient_of_variation(other_jumps) * 100
other_summary = desc.five_number_summary(other_jumps)

print("\nOther Frogs:")
print(f"  Sample size (n): {len(other_jumps)}")
print(f"  Mean: {other_mean:.2f} cm")
print(f"  Median: {other_median:.2f} cm")
print(f"  Standard deviation: {other_std:.2f} cm")
print(f"  Coefficient of variation: {other_cv:.1f}%")
print(f"  Range: {other_summary['min']:.2f} - {other_summary['max']:.2f} cm")
print(f"  IQR: {other_summary['Q3'] - other_summary['Q1']:.2f} cm")

# Difference
mean_difference = pro_mean - other_mean
print(f"\nDifference:")
print(f"  Mean difference: {mean_difference:.2f} cm")
print(f"  Professional frogs jump {mean_difference:.2f} cm farther on average")
print(f"  That's {mean_difference/other_mean*100:.1f}% farther than other frogs!")

# Step 2: Hypothesis Test
print("\n" + "=" * 70)
print("2. Hypothesis Test: Do Professional Frogs Jump Farther?")
print("=" * 70)

print("\nResearch Question: Is there a significant difference in jump distance")
print("                  between professional and other frogs?")

print(f"\nHypotheses:")
print(f"  H₀: μ_pro = μ_other (no difference in mean jump distance)")
print(f"  H₁: μ_pro > μ_other (professional frogs jump farther)")
print(f"  This is a one-tailed (right-tailed) test")

alpha = 0.05
print(f"  Significance level: α = {alpha}")

# Perform two-sample t-test using scipy (since real_simple_stats doesn't have it)
from scipy.stats import ttest_ind
t_stat, p_value_two_tailed = ttest_ind(pro_jumps, other_jumps, equal_var=False)
# For one-tailed test, divide p-value by 2 (since we expect pro > other)
p_value = p_value_two_tailed / 2

print(f"\nTest Results:")
print(f"  Professional mean: {pro_mean:.2f} cm")
print(f"  Other mean: {other_mean:.2f} cm")
print(f"  Mean difference: {mean_difference:.2f} cm")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value (one-tailed): {p_value:.6f}")

# Make decision
if p_value < alpha:
    decision = "Reject H₀"
    conclusion = "There is sufficient evidence that professional frogs jump farther"
    significance = "statistically significant"
else:
    decision = "Fail to reject H₀"
    conclusion = "There is insufficient evidence that professional frogs jump farther"
    significance = "not statistically significant"

print(f"\nDecision: {decision} (p = {p_value:.4f} {'<' if p_value < alpha else '≥'} α = {alpha})")
print(f"Conclusion: {conclusion}")
print(f"The difference is {significance}!")

# Step 3: Effect Size
print("\n" + "=" * 70)
print("3. Effect Size: How Large is the Difference?")
print("=" * 70)

cohens_d = es.cohens_d(pro_jumps, other_jumps)
print(f"\nCohen's d: {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_size = "negligible"
elif abs(cohens_d) < 0.5:
    effect_size = "small"
elif abs(cohens_d) < 0.8:
    effect_size = "medium"
else:
    effect_size = "large"

print(f"\nEffect Size Interpretation: {effect_size}")
print(f"\nThis means:")
if abs(cohens_d) < 0.2:
    print("  The difference, while statistically significant, is very small.")
    print("  Professional frogs jump farther, but the practical difference is minimal.")
elif abs(cohens_d) < 0.5:
    print("  There is a small but meaningful difference.")
    print("  Professional frogs consistently jump farther than other frogs.")
elif abs(cohens_d) < 0.8:
    print("  There is a medium-sized, practically important difference.")
    print("  Professional frogs show substantially better jumping performance.")
else:
    print("  There is a large, very important difference.")
    print("  Professional frogs demonstrate dramatically better jumping ability.")

# Step 4: Confidence Interval for the Difference
print("\n" + "=" * 70)
print("4. Confidence Interval for Mean Difference")
print("=" * 70)

# Calculate 95% CI for difference using Welch's t-test (unequal variances)
n1, n2 = len(pro_jumps), len(other_jumps)
s1, s2 = pro_std, other_std

# Standard error for difference (Welch's formula)
se_diff = np.sqrt((s1**2 / n1) + (s2**2 / n2))

# Degrees of freedom (Welch-Satterthwaite)
df_welch = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

# Critical t-value
t_critical = t_dist.ppf(0.975, df_welch)  # 95% CI, two-tailed

# Margin of error
margin = t_critical * se_diff
ci_lower = mean_difference - margin
ci_upper = mean_difference + margin

print(f"\n95% Confidence Interval for difference: [{ci_lower:.2f}, {ci_upper:.2f}] cm")
print(f"\nInterpretation:")
print(f"  We're 95% confident that professional frogs jump between")
print(f"  {ci_lower:.2f} and {ci_upper:.2f} cm farther than other frogs, on average.")
if ci_lower > 0:
    print(f"  This interval does NOT include zero,")
    print(f"  which confirms the difference is statistically significant.")
else:
    print(f"  This interval includes zero,")
    print(f"  which suggests the difference may not be significant.")

# Step 5: Visual Summary
print("\n" + "=" * 70)
print("5. Summary Statistics")
print("=" * 70)

print(f"\n{'Metric':<30} {'Professional':<15} {'Other':<15} {'Difference':<15}")
print("-" * 75)
print(f"{'Sample size (n)':<30} {len(pro_jumps):<15} {len(other_jumps):<15} {'':<15}")
print(f"{'Mean (cm)':<30} {pro_mean:<15.2f} {other_mean:<15.2f} {mean_difference:>+14.2f}")
print(f"{'Median (cm)':<30} {pro_median:<15.2f} {other_median:<15.2f} {pro_median-other_median:>+14.2f}")
print(f"{'Std Dev (cm)':<30} {pro_std:<15.2f} {other_std:<15.2f} {'':<15}")
print(f"{'Coefficient of Variation (%)':<30} {pro_cv:<15.1f} {other_cv:<15.1f} {'':<15}")

# Step 6: Final Interpretation
print("\n" + "=" * 70)
print("6. Final Interpretation")
print("=" * 70)

print(f"\nKey Findings:")
print(f"  ✓ Professional frogs jump {mean_difference:.2f} cm farther on average")
print(f"  ✓ This difference is {significance} (p = {p_value:.4f})")
print(f"  ✓ Effect size is {effect_size} (Cohen's d = {cohens_d:.4f})")
print(f"  ✓ 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] cm")

print(f"\nPractical Implications:")
if abs(cohens_d) >= 0.8:
    print(f"  • The difference is large enough to be practically important")
    print(f"  • Professional frogs show substantially better performance")
elif abs(cohens_d) >= 0.5:
    print(f"  • The difference is meaningful in practice")
    print(f"  • Professional frogs consistently outperform other frogs")
else:
    print(f"  • While statistically significant, the practical difference is smaller")
    print(f"  • Both groups show similar jumping ability overall")

print(f"\nStatistical Conclusion:")
print(f"  We have {'strong' if p_value < 0.01 else 'moderate' if p_value < 0.05 else 'weak'} evidence")
print(f"  that professional frogs jump farther than other frogs.")
print(f"  The mean difference of {mean_difference:.2f} cm is {'large' if abs(cohens_d) >= 0.8 else 'moderate' if abs(cohens_d) >= 0.5 else 'small but'} meaningful.")

print("\n" + "=" * 70)
print("This analysis demonstrates how to compare two groups using")
print("Real Simple Stats with real-world data from the BANA statistics book.")
print("=" * 70)

