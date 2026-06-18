"""Demo: causal inference methods.

Run with:  python examples/causal_inference_demo.py

Shows four quasi-experimental designs for estimating treatment effects when a
randomized experiment isn't possible — drawn from the Python for Business
Analytics book, Ch. 9.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

import real_simple_stats as rss
from real_simple_stats.plots import set_minimalist_style


# ---------------------------------------------------------------------------
# 1. Difference-in-Differences
#    Scenario: two stores, pre/post a marketing campaign
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Difference-in-Differences")
print("=" * 60)

rng = np.random.default_rng(42)
n = 400
post    = np.repeat([0, 0, 1, 1], n // 4)
treated = np.tile([0, 1], n // 2)
outcome = (
    100                           # baseline
    + 5  * post                   # time trend (affects both groups)
    + 8  * treated                # treated group is higher on average
    + 12 * post * treated         # the real treatment effect
    + rng.normal(0, 3, n)
)

r_did = rss.difference_in_differences(outcome, post, treated)
print(f"DiD estimate: {r_did['did_estimate']:.2f}  (true = 12.0)")
print(f"95% CI:       ({r_did['ci'][0]:.2f}, {r_did['ci'][1]:.2f})")
print(f"p-value:      {r_did['p_value']:.4f}")
print(f"Reject null:  {r_did['reject_null']}")
print()


# ---------------------------------------------------------------------------
# 2. Regression Discontinuity
#    Scenario: a scholarship program for students scoring ≥ 70
# ---------------------------------------------------------------------------

print("=" * 60)
print("2. Regression Discontinuity")
print("=" * 60)

n_rdd = 600
score = rng.uniform(40, 100, n_rdd)
cutoff = 70.0
gpa = (
    0.5 + 0.03 * (score - cutoff)
    + 0.8 * (score >= cutoff)     # scholarship effect on GPA
    + rng.normal(0, 0.4, n_rdd)
)

r_rdd = rss.regression_discontinuity(gpa, score, cutoff=cutoff)
print(f"RDD effect at cutoff: {r_rdd['effect']:.3f}  (true ≈ 0.8)")
print(f"95% CI:               ({r_rdd['ci'][0]:.3f}, {r_rdd['ci'][1]:.3f})")
print(f"p-value:              {r_rdd['p_value']:.4f}")
print()

# Plot the discontinuity
set_minimalist_style()
fig, ax = plt.subplots(figsize=(8, 4))
left  = score < cutoff
right = score >= cutoff
ax.scatter(score[left],  gpa[left],  alpha=0.3, s=10, color="#5E81AC", label="Control")
ax.scatter(score[right], gpa[right], alpha=0.3, s=10, color="#BF616A", label="Treated")
ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.5, label=f"Cutoff = {cutoff}")
ax.set_xlabel("Entrance score")
ax.set_ylabel("GPA")
ax.set_title(f"RDD: scholarship effect = {r_rdd['effect']:.2f} GPA points", loc="left")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("rdd_demo.png", dpi=150, bbox_inches="tight")
print("Saved rdd_demo.png")
print()


# ---------------------------------------------------------------------------
# 3. Synthetic Control
#    Scenario: a state implements a minimum-wage increase in 2010
# ---------------------------------------------------------------------------

print("=" * 60)
print("3. Synthetic Control")
print("=" * 60)

T, N, n_pre = 24, 15, 16
Y_donors = rng.normal(loc=50, scale=5, size=(T, N))
# Treated unit is a mix of donors plus a post-treatment bump
w_true = np.array([0.4, 0.3, 0.2, 0.1] + [0.0] * (N - 4))
y_base = Y_donors @ w_true + rng.normal(0, 0.5, T)
treatment_effect = np.concatenate([np.zeros(n_pre), np.full(T - n_pre, 3.5)])
y_treated = y_base + treatment_effect

r_sc = rss.synthetic_control(y_treated, Y_donors, n_pre=n_pre)
print(f"Average treatment effect (post): {r_sc['ate_post']:.2f}  (true = 3.5)")
print(f"Pre-treatment fit RMSE:          {r_sc['pre_fit_rmse']:.3f}")
print(f"Top 4 donor weights: {sorted(r_sc['weights'], reverse=True)[:4]}")
print()

set_minimalist_style()
fig2, ax2 = plt.subplots(figsize=(8, 4))
periods = np.arange(T)
ax2.plot(periods, y_treated, linewidth=2, label="Treated state", color="black")
ax2.plot(periods, r_sc["synthetic"], linewidth=2, linestyle="--",
         label="Synthetic control", color="#5E81AC")
ax2.axvline(n_pre - 0.5, color="red", linestyle=":", linewidth=1.5, label="Treatment start")
ax2.fill_between(
    periods[n_pre:],
    r_sc["synthetic"][n_pre:],
    y_treated[n_pre:],
    alpha=0.2, color="red", label=f"Gap (ATE = {r_sc['ate_post']:.1f})"
)
ax2.set_xlabel("Period")
ax2.set_ylabel("Outcome")
ax2.set_title("Synthetic control counterfactual", loc="left")
ax2.legend(frameon=False, fontsize=9)
fig2.tight_layout()
fig2.savefig("synthetic_control_demo.png", dpi=150, bbox_inches="tight")
print("Saved synthetic_control_demo.png")
print()


# ---------------------------------------------------------------------------
# 4. Panel Fixed Effects
#    Scenario: 50 retail stores observed over 12 months, estimating price
#    elasticity while absorbing each store's permanent sales level
# ---------------------------------------------------------------------------

print("=" * 60)
print("4. Panel Fixed Effects")
print("=" * 60)

n_stores, n_months = 50, 12
n_obs = n_stores * n_months
store_id = np.repeat(np.arange(n_stores), n_months)
store_fe = np.repeat(rng.normal(0, 10, n_stores), n_months)  # permanent store effects
price_change = rng.normal(0, 2, n_obs)
sales = -1.5 * price_change + store_fe + rng.normal(0, 1, n_obs)

r_fe = rss.panel_fixed_effects(sales, price_change, store_id)
print(f"Price elasticity (FE):  {r_fe['coefficients'][0]:.3f}  (true = -1.5)")
print(f"Standard error:         {r_fe['se'][0]:.3f}")
print(f"p-value:                {r_fe['p_values'][0]:.4f}")
print(f"n = {r_fe['n']}, entities = {r_fe['n_entities']}, df = {r_fe['df_residual']}")
