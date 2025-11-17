"""Recipe: Power Analysis for Study Planning

This recipe demonstrates how to use power analysis to plan a study:
1. Determine required sample size
2. Calculate power for a given sample size
3. Understand effect sizes
4. Make informed decisions about study design
"""

import real_simple_stats as rss
from real_simple_stats import power_analysis as pa

print("=" * 70)
print("Power Analysis: Planning a Study")
print("=" * 70)

# ============================================================================
# Scenario 1: Determine Required Sample Size
# ============================================================================
print("\n" + "=" * 70)
print("Scenario 1: How many participants do we need?")
print("=" * 70)

print("\nResearch Question: Does a new training program improve test scores?")
print("Expected effect: Mean improvement of 5 points (delta = 5)")
print("Standard deviation: 10 points (based on previous studies)")
print("Desired power: 0.80 (80% chance of detecting the effect)")
print("Significance level: 0.05 (5% chance of Type I error)")
print("Test type: Two-sided (we want to detect improvement or decline)")

# Calculate required sample size
result = pa.power_t_test(
    delta=5.0,      # Expected difference in means
    sd=10.0,        # Standard deviation
    power=0.80,     # Desired power
    sig_level=0.05, # Significance level
    alternative="two-sided"
)

n_required = int(result['n'])
print(f"\n✓ Required sample size: n = {n_required} per group")
print(f"  (Total participants needed: {n_required * 2} for two groups)")

# ============================================================================
# Scenario 2: Calculate Power for Existing Sample Size
# ============================================================================
print("\n" + "=" * 70)
print("Scenario 2: What's the power with our available sample?")
print("=" * 70)

print("\nWe can only recruit 30 participants per group.")
print("What's our power to detect a 5-point improvement?")

result2 = pa.power_t_test(
    n=30,           # Available sample size
    delta=5.0,      # Expected difference
    sd=10.0,        # Standard deviation
    sig_level=0.05,
    alternative="two-sided"
)

power = result2['power']
print(f"\n✓ Statistical power: {power:.1%}")
if power < 0.80:
    print(f"  ⚠️  Power is below the recommended 0.80 threshold")
    print(f"  Consider: Increasing sample size, larger effect size, or higher alpha")
else:
    print(f"  ✓ Power is adequate (≥ 0.80)")

# ============================================================================
# Scenario 3: Power Analysis for Proportion Test
# ============================================================================
print("\n" + "=" * 70)
print("Scenario 3: Sample Size for Proportion Test")
print("=" * 70)

print("\nResearch Question: Does a new ad campaign increase conversion rate?")
print("Current conversion rate: 10% (p₀ = 0.10)")
print("Expected new rate: 15% (p₁ = 0.15)")
print("Desired power: 0.80")
print("Significance level: 0.05")

result3 = pa.power_proportion_test(
    p1=0.15,        # Expected proportion
    p0=0.10,       # Null hypothesis proportion
    power=0.80,
    sig_level=0.05,
    alternative="two-sided"
)

n_prop = int(result3['n'])
print(f"\n✓ Required sample size: n = {n_prop} per group")

# ============================================================================
# Scenario 4: One-Sided Test (More Powerful)
# ============================================================================
print("\n" + "=" * 70)
print("Scenario 4: One-Sided vs Two-Sided Tests")
print("=" * 70)

print("\nComparing power for one-sided vs two-sided tests:")
print("(Same sample size: n = 30, delta = 5, sd = 10)")

# Two-sided
result_two = pa.power_t_test(
    n=30, delta=5.0, sd=10.0, sig_level=0.05, alternative="two-sided"
)

# One-sided (greater)
result_one = pa.power_t_test(
    n=30, delta=5.0, sd=10.0, sig_level=0.05, alternative="greater"
)

print(f"\nTwo-sided test power: {result_two['power']:.1%}")
print(f"One-sided test power:  {result_one['power']:.1%}")
print(f"\nNote: One-sided tests have higher power when you have a")
print("      directional hypothesis, but require stronger justification.")

# ============================================================================
# Scenario 5: Effect Size and Power Trade-offs
# ============================================================================
print("\n" + "=" * 70)
print("Scenario 5: Understanding Effect Size")
print("=" * 70)

print("\nHow does effect size affect required sample size?")
print("(Power = 0.80, sd = 10, two-sided test)")

effect_sizes = [2.0, 5.0, 10.0]
print("\nEffect Size | Required n | Interpretation")
print("-" * 50)

for delta in effect_sizes:
    result = pa.power_t_test(
        delta=delta, sd=10.0, power=0.80, sig_level=0.05, alternative="two-sided"
    )
    n = int(result['n'])
    
    # Cohen's d interpretation
    cohens_d = delta / 10.0
    if cohens_d < 0.2:
        interp = "negligible"
    elif cohens_d < 0.5:
        interp = "small"
    elif cohens_d < 0.8:
        interp = "medium"
    else:
        interp = "large"
    
    print(f"  {delta:4.1f}     |    {n:4d}    | {interp} effect")

print("\nKey Insight: Larger effects require smaller samples to detect.")
print("            Small effects require very large samples.")

# ============================================================================
# Summary and Best Practices
# ============================================================================
print("\n" + "=" * 70)
print("Summary: Power Analysis Best Practices")
print("=" * 70)

print("""
1. Plan ahead: Calculate sample size before collecting data
2. Use realistic effect sizes: Base on previous research or pilot studies
3. Aim for power ≥ 0.80: Standard threshold for adequate power
4. Consider practical constraints: Budget, time, participant availability
5. Report power: Include power analysis in your study design section

Common Mistakes to Avoid:
- Collecting data first, then checking power (too late!)
- Using unrealistic effect sizes (leads to underpowered studies)
- Ignoring power (may miss real effects)
- Using one-sided tests without justification
""")

print("=" * 70)

