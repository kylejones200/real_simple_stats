"""Recipe: Exploratory Data Analysis (EDA)

This recipe demonstrates a complete exploratory data analysis workflow
using Real Simple Stats functions.
"""

import real_simple_stats as rss
from real_simple_stats import descriptive_statistics as desc

# Example dataset: Daily sales data
sales_data = [120, 135, 142, 118, 145, 132, 128, 140, 138, 125,
              150, 148, 155, 142, 160, 145, 152, 158, 162, 150]

print("=" * 60)
print("Exploratory Data Analysis: Daily Sales Data")
print("=" * 60)

# Step 1: Basic descriptive statistics
print("\n1. Basic Statistics")
print("-" * 60)
mean_sales = desc.mean(sales_data)
median_sales = desc.median(sales_data)
std_sales = desc.sample_std_dev(sales_data)
variance_sales = desc.sample_variance(sales_data)
cv = desc.coefficient_of_variation(sales_data)

print(f"Sample size: {len(sales_data)}")
print(f"Mean: {mean_sales:.2f}")
print(f"Median: {median_sales:.2f}")
print(f"Standard deviation: {std_sales:.2f}")
print(f"Variance: {variance_sales:.2f}")
print(f"Coefficient of variation: {cv:.2f}%")

# Step 2: Five-number summary
print("\n2. Five-Number Summary")
print("-" * 60)
summary = desc.five_number_summary(sales_data)
print(f"Minimum: {summary['min']:.2f}")
print(f"Q1 (25th percentile): {summary['Q1']:.2f}")
print(f"Median (50th percentile): {summary['median']:.2f}")
print(f"Q3 (75th percentile): {summary['Q3']:.2f}")
print(f"Maximum: {summary['max']:.2f}")

# Calculate IQR
iqr = summary['Q3'] - summary['Q1']
print(f"Interquartile Range (IQR): {iqr:.2f}")

# Step 3: Outlier detection using IQR method
print("\n3. Outlier Detection")
print("-" * 60)
lower_bound = summary['Q1'] - 1.5 * iqr
upper_bound = summary['Q3'] + 1.5 * iqr

outliers = [x for x in sales_data if x < lower_bound or x > upper_bound]
print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
if outliers:
    print(f"Outliers detected: {outliers}")
    print(f"Number of outliers: {len(outliers)}")
else:
    print("No outliers detected using IQR method")

# Step 4: Distribution shape
print("\n4. Distribution Shape")
print("-" * 60)
if mean_sales > median_sales:
    shape = "right-skewed (positive skew)"
elif mean_sales < median_sales:
    shape = "left-skewed (negative skew)"
else:
    shape = "approximately symmetric"

print(f"Mean vs Median: {shape}")
print(f"Difference: {abs(mean_sales - median_sales):.2f}")

# Step 5: Range and spread
print("\n5. Range and Spread")
print("-" * 60)
data_range = summary['max'] - summary['min']
print(f"Range: {data_range:.2f}")
print(f"Standard deviation: {std_sales:.2f}")
print(f"IQR: {iqr:.2f}")

# Step 6: Summary insights
print("\n6. Key Insights")
print("-" * 60)
print(f"• Average daily sales: ${mean_sales:.2f}")
print(f"• Sales vary by ${std_sales:.2f} on average (SD)")
print(f"• Middle 50% of sales fall between ${summary['Q1']:.2f} and ${summary['Q3']:.2f}")
print(f"• The distribution appears to be {shape}")
if outliers:
    print(f"• {len(outliers)} outlier(s) detected that may need investigation")
else:
    print("• No outliers detected - data appears consistent")

