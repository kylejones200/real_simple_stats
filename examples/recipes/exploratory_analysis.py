"""Recipe: Exploratory Data Analysis (EDA)

This recipe demonstrates a complete exploratory data analysis workflow
using Real Simple Stats functions.
"""

import logging

from real_simple_stats import descriptive_statistics as desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example dataset: Daily sales data
sales_data = [
    120,
    135,
    142,
    118,
    145,
    132,
    128,
    140,
    138,
    125,
    150,
    148,
    155,
    142,
    160,
    145,
    152,
    158,
    162,
    150,
]

logger.info("=" * 60)
logger.info("Exploratory Data Analysis: Daily Sales Data")
logger.info("=" * 60)

# Step 1: Basic descriptive statistics
logger.info("\n1. Basic Statistics")
logger.info("-" * 60)
mean_sales = desc.mean(sales_data)
median_sales = desc.median(sales_data)
std_sales = desc.sample_std_dev(sales_data)
variance_sales = desc.sample_variance(sales_data)
cv = desc.coefficient_of_variation(sales_data)

logger.info(f"Sample size: {len(sales_data)}")
logger.info(f"Mean: {mean_sales:.2f}")
logger.info(f"Median: {median_sales:.2f}")
logger.info(f"Standard deviation: {std_sales:.2f}")
logger.info(f"Variance: {variance_sales:.2f}")
logger.info(f"Coefficient of variation: {cv:.2f}%")

# Step 2: Five-number summary
logger.info("\n2. Five-Number Summary")
logger.info("-" * 60)
summary = desc.five_number_summary(sales_data)
logger.info(f"Minimum: {summary['min']:.2f}")
logger.info(f"Q1 (25th percentile): {summary['Q1']:.2f}")
logger.info(f"Median (50th percentile): {summary['median']:.2f}")
logger.info(f"Q3 (75th percentile): {summary['Q3']:.2f}")
logger.info(f"Maximum: {summary['max']:.2f}")

# Calculate IQR
iqr = summary["Q3"] - summary["Q1"]
logger.info(f"Interquartile Range (IQR): {iqr:.2f}")

# Step 3: Outlier detection using IQR method
logger.info("\n3. Outlier Detection")
logger.info("-" * 60)
lower_bound = summary["Q1"] - 1.5 * iqr
upper_bound = summary["Q3"] + 1.5 * iqr

outliers = [x for x in sales_data if x < lower_bound or x > upper_bound]
logger.info(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
if outliers:
    logger.info(f"Outliers detected: {outliers}")
    logger.info(f"Number of outliers: {len(outliers)}")
else:
    logger.info("No outliers detected using IQR method")

# Step 4: Distribution shape
logger.info("\n4. Distribution Shape")
logger.info("-" * 60)
if mean_sales > median_sales:
    shape = "right-skewed (positive skew)"
elif mean_sales < median_sales:
    shape = "left-skewed (negative skew)"
else:
    shape = "approximately symmetric"

logger.info(f"Mean vs Median: {shape}")
logger.info(f"Difference: {abs(mean_sales - median_sales):.2f}")

# Step 5: Range and spread
logger.info("\n5. Range and Spread")
logger.info("-" * 60)
data_range = summary["max"] - summary["min"]
logger.info(f"Range: {data_range:.2f}")
logger.info(f"Standard deviation: {std_sales:.2f}")
logger.info(f"IQR: {iqr:.2f}")

# Step 6: Summary insights
logger.info("\n6. Key Insights")
logger.info("-" * 60)
logger.info(f"• Average daily sales: ${mean_sales:.2f}")
logger.info(f"• Sales vary by ${std_sales:.2f} on average (SD)")
logger.info(
    f"• Middle 50% of sales fall between ${summary['Q1']:.2f} and ${summary['Q3']:.2f}"
)
logger.info(f"• The distribution appears to be {shape}")
if outliers:
    logger.info(f"• {len(outliers)} outlier(s) detected that may need investigation")
else:
    logger.info("• No outliers detected - data appears consistent")
