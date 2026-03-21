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
import logging
from pathlib import Path

from real_simple_stats import descriptive_statistics as desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from real_simple_stats import hypothesis_testing as ht

# Load the frog jump data
data_file = Path(__file__).parent.parent / "data" / "froggy.csv"

logger.info("=" * 70)
logger.info("Frog Jump Distance Analysis")
logger.info("=" * 70)

# Read the data
jump_distances = []
with open(data_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            distance = float(row["distance"])
            # Filter to successful jumps (distance > 0)
            if distance > 0:
                jump_distances.append(distance)
        except (ValueError, KeyError):
            continue

logger.info(f"\nDataset loaded: {len(jump_distances)} successful jumps")
logger.info(f"Total jumps in dataset: {len(jump_distances)}")

# Step 1: Basic Descriptive Statistics
logger.info("\n" + "=" * 70)
logger.info("1. Descriptive Statistics")
logger.info("=" * 70)

mean_distance = desc.mean(jump_distances)
median_distance = desc.median(jump_distances)
std_distance = desc.sample_std_dev(jump_distances)
variance_distance = desc.sample_variance(jump_distances)
cv = desc.coefficient_of_variation(jump_distances) * 100  # Convert to percentage

logger.info(f"Sample size (n): {len(jump_distances)}")
logger.info(f"Mean jump distance: {mean_distance:.2f} cm")
logger.info(f"Median jump distance: {median_distance:.2f} cm")
logger.info(f"Standard deviation: {std_distance:.2f} cm")
logger.info(f"Variance: {variance_distance:.2f} cm²")
logger.info(f"Coefficient of variation: {cv:.1f}%")

# Step 2: Five-Number Summary
logger.info("\n" + "=" * 70)
logger.info("2. Five-Number Summary")
logger.info("=" * 70)

summary = desc.five_number_summary(jump_distances)
logger.info(f"Minimum: {summary['min']:.2f} cm")
logger.info(f"Q1 (25th percentile): {summary['Q1']:.2f} cm")
logger.info(f"Median (50th percentile): {summary['median']:.2f} cm")
logger.info(f"Q3 (75th percentile): {summary['Q3']:.2f} cm")
logger.info(f"Maximum: {summary['max']:.2f} cm")

iqr = summary["Q3"] - summary["Q1"]
logger.info(f"Interquartile Range (IQR): {iqr:.2f} cm")

# Step 3: Distribution Shape
logger.info("\n" + "=" * 70)
logger.info("3. Distribution Shape")
logger.info("=" * 70)

if mean_distance > median_distance:
    shape = "right-skewed (positive skew)"
elif mean_distance < median_distance:
    shape = "left-skewed (negative skew)"
else:
    shape = "approximately symmetric"

logger.info(f"Mean vs Median: {shape}")
logger.info(f"Difference: {abs(mean_distance - median_distance):.2f} cm")
logger.info(
    f"Interpretation: The mean is {'higher' if mean_distance > median_distance else 'lower'} than the median,"
)
logger.info(f"                suggesting the distribution is {shape}.")

# Step 4: Outlier Detection
logger.info("\n" + "=" * 70)
logger.info("4. Outlier Detection (IQR Method)")
logger.info("=" * 70)

lower_bound = summary["Q1"] - 1.5 * iqr
upper_bound = summary["Q3"] + 1.5 * iqr

outliers = [x for x in jump_distances if x < lower_bound or x > upper_bound]
logger.info(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}] cm")
if outliers:
    logger.info(
        f"Outliers detected: {len(outliers)} ({len(outliers) / len(jump_distances) * 100:.1f}%)"
    )
    logger.info(f"Outlier range: [{min(outliers):.2f}, {max(outliers):.2f}] cm")
else:
    logger.info("No outliers detected using IQR method")

# Step 5: Hypothesis Test Example
logger.info("\n" + "=" * 70)
logger.info("5. Hypothesis Testing Example")
logger.info("=" * 70)
logger.info(
    "Research Question: Is the average jump distance significantly different from 100 cm?"
)

# Set up hypotheses
mu_null = 100.0
alpha = 0.05
logger.info(f"\nH₀: μ = {mu_null} cm (average jump distance is {mu_null} cm)")
logger.info(f"H₁: μ ≠ {mu_null} cm (average jump distance differs from {mu_null} cm)")
logger.info(f"Significance level: α = {alpha}")

# Perform one-sample t-test manually
n = len(jump_distances)
t_stat = ht.t_score(mean_distance, mu_null, std_distance, n)
df = n - 1
# Calculate p-value using scipy (two-tailed)
from scipy.stats import t as t_dist

p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))

logger.info("\nTest Results:")
logger.info(f"  Sample mean: {mean_distance:.2f} cm")
logger.info(f"  Null hypothesis mean: {mu_null} cm")
logger.info(f"  t-statistic: {t_stat:.4f}")
logger.info(f"  p-value: {p_value:.6f}")

# Make decision
if p_value < alpha:
    decision = "Reject H₀"
    conclusion = f"There is sufficient evidence that the average jump distance differs from {mu_null} cm"
else:
    decision = "Fail to reject H₀"
    conclusion = f"There is insufficient evidence that the average jump distance differs from {mu_null} cm"

logger.info(
    f"\nDecision: {decision} (p = {p_value:.4f} {'<' if p_value < alpha else '≥'} α = {alpha})"
)
logger.info(f"Conclusion: {conclusion}")

# Step 6: Confidence Interval
logger.info("\n" + "=" * 70)
logger.info("6. Confidence Interval for Mean Jump Distance")
logger.info("=" * 70)

n = len(jump_distances)
se = std_distance / (n**0.5)
df = n - 1
t_critical = ht.critical_value_t(alpha, df, test_type="two-tailed")
margin = t_critical * se
ci_lower = mean_distance - margin
ci_upper = mean_distance + margin

logger.info(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] cm")
logger.info("Interpretation: We're 95% confident that the true mean jump distance")
logger.info(
    f"                for the population is between {ci_lower:.2f} and {ci_upper:.2f} cm"
)

# Step 7: Summary
logger.info("\n" + "=" * 70)
logger.info("7. Summary")
logger.info("=" * 70)
logger.info(f"• Average jump distance: {mean_distance:.2f} cm (SD = {std_distance:.2f} cm)")
logger.info(f"• Distribution: {shape}")
logger.info(f"• Range: {summary['min']:.2f} to {summary['max']:.2f} cm")
logger.info(f"• Middle 50% of jumps: {summary['Q1']:.2f} to {summary['Q3']:.2f} cm")
if outliers:
    logger.info(f"• {len(outliers)} outlier(s) detected that may need investigation")
logger.info(
    f"• Hypothesis test: Average jump distance is {'significantly different' if p_value < alpha else 'not significantly different'} from {mu_null} cm"
)
logger.info(f"• 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] cm")

logger.info("\n" + "=" * 70)
logger.info("This analysis demonstrates a complete statistical workflow using")
logger.info("Real Simple Stats with real-world data from the BANA statistics book.")
logger.info("=" * 70)
