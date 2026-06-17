"""Demo: self-explaining statistical results.

Run with:  python examples/explained_t_test_demo.py

Shows how a single test result can serve two audiences at once — a script
that wants numbers, and a human who wants to understand them.
"""

import matplotlib

matplotlib.use("Agg")  # so the demo runs headless in CI

import real_simple_stats as rss
from real_simple_stats.plots import plot_ci_coverage


def main() -> None:
    # A small sample of measurements; is the true mean different from 5.0?
    data = [5.2, 5.4, 5.1, 5.5, 5.3, 5.0, 5.6, 5.2, 5.4, 5.3]

    result = rss.one_sample_t_test_explained(data, mu=5.0)

    # 1) Use it as data — every headline number is an attribute.
    print(">>> Using the result as data")
    print(f"    t = {result.statistic:.3f}")
    print(f"    p = {result.p_value:.4f}")
    print(f"    95% CI = ({result.ci[0]:.3f}, {result.ci[1]:.3f})")
    print(f"    Cohen's d = {result.effect_size:.3f}")
    print()

    # 2) Or let it teach — the full narrative.
    print(">>> Letting the result explain itself")
    print(result)

    # 3) Or let it show you — the p-value as a shaded tail area.
    fig, _ = result.plot()
    fig.savefig("explained_p_value.png", dpi=150, bbox_inches="tight")
    print("Saved explained_p_value.png (the p-value as an area)")

    # 4) Bonus intuition: what '95% confidence' actually means, by simulation.
    fig2, _ = plot_ci_coverage(
        true_mean=5.0, true_sd=0.2, n=len(data), n_intervals=100, confidence=0.95
    )
    fig2.savefig("ci_coverage.png", dpi=150, bbox_inches="tight")
    print("Saved ci_coverage.png (95% of intervals should capture the true mean)")


if __name__ == "__main__":
    main()
