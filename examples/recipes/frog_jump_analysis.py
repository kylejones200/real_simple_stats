"""Recipe: Frog Jump Distance Analysis

This recipe demonstrates a complete statistical analysis using the frog jump dataset
from the BANA statistics book. This is a real-world example that shows how to:

1. Load and explore data
2. Handle missing values and outliers
3. Calculate descriptive statistics
4. Perform hypothesis testing
5. Interpret results in context

Dataset: froggy.csv - Frog jump distances and related measurements
"""

import csv
from pathlib import Path
from real_simple_stats import descriptive_statistics as desc
from real_simple_stats import hypothesis_testing as ht

# Load the frog jump data
data_file = Path(__file__).parent.parent / "data" / "froggy.csv"

print("=" * 70)
print("Frog Jump Distance Analysis")
print("=" * 70)

# Read the data
jump_distances = []
with open(data_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            distance = float(row['distance'])
            # Filter to successful jumps (distance > 0)
            if distance > 0:
                jump_distances.append(distance)
        except (ValueError, KeyError):
            continue

print(f"\nDataset loaded: {len(jump_distances)} successful jumps")
print(f"Total jumps in dataset: {len(jump_distances)}")

# Step 1: Basic Descriptive Statistics
print("\n" + "=" * 70)
print("1. Descriptive Statistics")
print("=" * 70)

mean_distance = desc.mean(jump_distances)
median_distance = desc.median(jump_distances)
std_distance = desc.sample_std_dev(jump_distances)
variance_distance = desc.sample_variance(jump_distances)
cv = desc.coefficient_of_variation(jump_distances) * 100  # Convert to percentage

print(f"Sample size (n): {len(jump_distances)}")
print(f"Mean jump distance: {mean_distance:.2f} cm")
print(f"Median jump distance: {median_distance:.2f} cm")
print(f"Standard deviation: {std_distance:.2f} cm")
print(f"Variance: {variance_distance:.2f} cm²")
print(f"Coefficient of variation: {cv:.1f}%")

# Step 2: Five-Number Summary
print("\n" + "=" * 70)
print("2. Five-Number Summary")
print("=" * 70)

summary = desc.five_number_summary(jump_distances)
print(f"Minimum: {summary['min']:.2f} cm")
print(f"Q1 (25th percentile): {summary['Q1']:.2f} cm")
print(f"Median (50th percentile): {summary['median']:.2f} cm")
print(f"Q3 (75th percentile): {summary['Q3']:.2f} cm")
print(f"Maximum: {summary['max']:.2f} cm")

iqr = summary['Q3'] - summary['Q1']
print(f"Interquartile Range (IQR): {iqr:.2f} cm")

# Step 3: Distribution Shape
print("\n" + "=" * 70)
print("3. Distribution Shape")
print("=" * 70)

if mean_distance > median_distance:
    shape = "right-skewed (positive skew)"
elif mean_distance < median_distance:
    shape = "left-skewed (negative skew)"
else:
    shape = "approximately symmetric"

print(f"Mean vs Median: {shape}")
print(f"Difference: {abs(mean_distance - median_distance):.2f} cm")
print(f"Interpretation: The mean is {'higher' if mean_distance > median_distance else 'lower'} than the median,")
print(f"                suggesting the distribution is {shape}.")

# Step 4: Outlier Detection
print("\n" + "=" * 70)
print("4. Outlier Detection (IQR Method)")
print("=" * 70)

lower_bound = summary['Q1'] - 1.5 * iqr
upper_bound = summary['Q3'] + 1.5 * iqr

outliers = [x for x in jump_distances if x < lower_bound or x > upper_bound]
print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}] cm")
if outliers:
    print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(jump_distances)*100:.1f}%)")
    print(f"Outlier range: [{min(outliers):.2f}, {max(outliers):.2f}] cm")
else:
    print("No outliers detected using IQR method")

# Step 5: Hypothesis Test Example
print("\n" + "=" * 70)
print("5. Hypothesis Testing Example")
print("=" * 70)
print("Research Question: Is the average jump distance significantly different from 100 cm?")

# Set up hypotheses
mu_null = 100.0
alpha = 0.05
print(f"\nH₀: μ = {mu_null} cm (average jump distance is {mu_null} cm)")
print(f"H₁: μ ≠ {mu_null} cm (average jump distance differs from {mu_null} cm)")
print(f"Significance level: α = {alpha}")

# Perform one-sample t-test manually
n = len(jump_distances)
t_stat = ht.t_score(mean_distance, mu_null, std_distance, n)
df = n - 1
# Calculate p-value using scipy (two-tailed)
from scipy.stats import t as t_dist
p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))

print(f"\nTest Results:")
print(f"  Sample mean: {mean_distance:.2f} cm")
print(f"  Null hypothesis mean: {mu_null} cm")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

# Make decision
if p_value < alpha:
    decision = "Reject H₀"
    conclusion = f"There is sufficient evidence that the average jump distance differs from {mu_null} cm"
else:
    decision = "Fail to reject H₀"
    conclusion = f"There is insufficient evidence that the average jump distance differs from {mu_null} cm"

print(f"\nDecision: {decision} (p = {p_value:.4f} {'<' if p_value < alpha else '≥'} α = {alpha})")
print(f"Conclusion: {conclusion}")

# Step 6: Confidence Interval
print("\n" + "=" * 70)
print("6. Confidence Interval for Mean Jump Distance")
print("=" * 70)

n = len(jump_distances)
se = std_distance / (n ** 0.5)
df = n - 1
t_critical = ht.critical_value_t(alpha, df, test_type="two-tailed")
margin = t_critical * se
ci_lower = mean_distance - margin
ci_upper = mean_distance + margin

print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] cm")
print(f"Interpretation: We're 95% confident that the true mean jump distance")
print(f"                for the population is between {ci_lower:.2f} and {ci_upper:.2f} cm")

# Step 7: Summary
print("\n" + "=" * 70)
print("7. Summary")
print("=" * 70)
print(f"• Average jump distance: {mean_distance:.2f} cm (SD = {std_distance:.2f} cm)")
print(f"• Distribution: {shape}")
print(f"• Range: {summary['min']:.2f} to {summary['max']:.2f} cm")
print(f"• Middle 50% of jumps: {summary['Q1']:.2f} to {summary['Q3']:.2f} cm")
if outliers:
    print(f"• {len(outliers)} outlier(s) detected that may need investigation")
print(f"• Hypothesis test: Average jump distance is {'significantly different' if p_value < alpha else 'not significantly different'} from {mu_null} cm")
print(f"• 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] cm")

print("\n" + "=" * 70)
print("This analysis demonstrates a complete statistical workflow using")
print("Real Simple Stats with real-world data from the BANA statistics book.")
print("=" * 70)

