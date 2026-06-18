"""Demo: survival analysis.

Run with:  python examples/survival_demo.py

Estimates customer churn survival curves using Kaplan-Meier and compares
parametric models — drawn from the Python for Business Analytics book, Ch. 7.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.stats import weibull_min

import real_simple_stats as rss
from real_simple_stats.plots import plot_survival_curve


def main() -> None:
    rng = np.random.default_rng(7)
    n = 400

    # Simulate customer lifetimes: Weibull(shape=1.4, scale=18 months)
    # About 30% are still active when we observe them (censored)
    true_churn = weibull_min(c=1.4, scale=18).rvs(size=n, random_state=rng)
    observation_cutoff = rng.exponential(scale=30, size=n)
    durations = np.minimum(true_churn, observation_cutoff)
    event_observed = (true_churn <= observation_cutoff).astype(int)

    print(f"Customers: {n}")
    print(f"Churned (observed events): {event_observed.sum()}")
    print(f"Still active (censored):   {n - event_observed.sum()}")
    print()

    # Kaplan-Meier curve
    km = rss.kaplan_meier(durations, event_observed)
    print(f"Median churn time: {km['median_survival']:.1f} months  (true ≈ 15.7)")
    print(f"S(12 months) ≈ {km['survival_prob'][np.searchsorted(km['times'], 12)-1]:.3f}")
    print()

    # Compare parametric models
    comparisons = rss.compare_survival_models(durations, event_observed)
    print("Parametric model comparison (ranked by AIC):")
    print(f"{'Rank':<6} {'Distribution':<14} {'AIC':>10} {'BIC':>10}")
    print("-" * 44)
    for r in comparisons:
        print(f"{r['rank']:<6} {r['distribution']:<14} {r['aic']:>10.1f} {r['bic']:>10.1f}")
    print()
    print(f"Best fit: {comparisons[0]['distribution'].capitalize()}")
    print()

    # Plot KM + all parametric models
    fig, ax = plot_survival_curve(km, parametric_results=comparisons)
    ax.set_title("Customer churn survival curve", loc="left")
    fig.savefig("survival_demo.png", dpi=150, bbox_inches="tight")
    print("Saved survival_demo.png")


if __name__ == "__main__":
    main()
