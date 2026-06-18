"""Plotting utilities for Real Simple Stats.

Uses PlotSmith when available (pip install real-simple-stats[plots]),
otherwise falls back to matplotlib.
"""

import numpy as np

try:
    from plotsmith import plot_histogram as _plot_histogram

    PLOTSMITH_AVAILABLE = True
except ImportError:
    PLOTSMITH_AVAILABLE = False

import matplotlib.pyplot as plt


def set_minimalist_style():
    """Apply a Tufte-inspired minimalist style to Matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.frameon": False,
            "axes.grid": False,
            "grid.color": "white",
        }
    )


def plot_norm_hist(
    data, mean, std, bins=30, show_pdf=True, show_lines=True, title=True
):
    """Plot a histogram of data with optional normal curve and markers.

    Uses PlotSmith when available (pip install real-simple-stats[plots]),
    otherwise matplotlib.
    """
    if PLOTSMITH_AVAILABLE:
        kwargs = {"bins": bins}
        if title:
            kwargs["title"] = f"Normal Distribution (μ = {mean:.2f}, σ = {std:.2f})"
        fig, ax = _plot_histogram(data, **kwargs)
        x_min, x_max = ax.get_xlim()
        x = np.linspace(x_min, x_max, 300)
        if show_pdf:
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
                -((x - mean) ** 2) / (2 * std**2)
            )
            ax.plot(x, y, color="black", linewidth=1.5)
        if show_lines:
            ax.axvline(mean - 2 * std, color="black", linestyle="--", linewidth=1)
            ax.axvline(mean + 2 * std, color="black", linestyle="--", linewidth=1)
        if not title:
            ax.set_title("")
        fig.savefig("norm_hist.png", bbox_inches="tight", dpi=300)
        plt.show()
        return

    set_minimalist_style()
    _, bins_edges, _ = plt.hist(
        data, bins=bins, density=True, alpha=0.5, edgecolor="black"
    )

    if show_pdf:
        x = np.linspace(min(bins_edges), max(bins_edges), 300)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std**2))
        plt.plot(x, y, color="black", linewidth=1.5)

    if show_lines:
        plt.axvline(mean - 2 * std, color="black", linestyle="--", linewidth=1)
        plt.axvline(mean + 2 * std, color="black", linestyle="--", linewidth=1)

    if title:
        plt.title(f"Normal Distribution (μ = {mean:.2f}, σ = {std:.2f})")

    plt.savefig("norm_hist.png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_box(data, showfliers=False):
    """Plot a horizontal boxplot without outliers."""
    set_minimalist_style()

    fig, ax = plt.subplots()
    ax.boxplot(
        data,
        vert=False,
        showfliers=showfliers,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )

    ax.set_title("Boxplot")

    plt.savefig("boxplot.png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_observed_vs_expected(observed, expected, title="Observed vs Expected"):
    """
    Minimalist bar plot comparing observed and expected frequencies.

    Args:
        observed: List of observed counts.
        expected: List of expected counts.
        title: Plot title.
    """
    set_minimalist_style()
    x = range(len(observed))
    width = 0.4
    plt.bar(
        [i - width / 2 for i in x],
        observed,
        width=width,
        label="Observed",
        color="black",
        alpha=0.7,
    )
    plt.bar(
        [i + width / 2 for i in x],
        expected,
        width=width,
        label="Expected",
        color="gray",
        alpha=0.5,
    )

    plt.xticks(x)
    plt.title(title)
    plt.legend()
    plt.savefig("observed_vs_expected.png", bbox_inches="tight", dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# Intuition plots — visualizations that teach a concept, not just display data.
# ---------------------------------------------------------------------------


def plot_p_value_area(
    t_stat,
    df,
    alternative="two-sided",
    alpha=0.05,
    ax=None,
):
    """Show a p-value for what it is: an area in the tail of a distribution.

    Draws the t-distribution under H₀, shades the tail region(s) beyond the
    observed statistic (that shaded area *is* the p-value), and marks the
    critical value(s) where the rejection region begins. This is the single
    picture that makes "p-value" click for most people.

    Parameters
    ----------
    t_stat : float
        The observed t statistic.
    df : int
        Degrees of freedom.
    alternative : {"two-sided", "greater", "less"}
        Test direction.
    alpha : float
        Significance level, used to draw the critical value(s).
    ax : matplotlib Axes, optional
        Draw onto an existing axis instead of creating a figure.

    Returns
    -------
    (fig, ax)
    """
    from scipy.stats import t as _t

    set_minimalist_style()
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    span = max(4.0, abs(t_stat) + 1.0)
    x = np.linspace(-span, span, 1000)
    y = _t.pdf(x, df)
    ax.plot(x, y, color="black", linewidth=1.2)

    shade = "#c0392b"
    if alternative == "two-sided":
        a = abs(t_stat)
        right = x >= a
        left = x <= -a
        ax.fill_between(x[right], y[right], color=shade, alpha=0.45)
        ax.fill_between(x[left], y[left], color=shade, alpha=0.45)
        crit = float(_t.ppf(1 - alpha / 2, df))
        for c in (crit, -crit):
            ax.axvline(c, color="gray", linestyle="--", linewidth=0.9)
        ax.axvline(t_stat, color=shade, linewidth=1.6)
        ax.axvline(-abs(t_stat), color=shade, linewidth=0.8, alpha=0.5)
    elif alternative == "greater":
        right = x >= t_stat
        ax.fill_between(x[right], y[right], color=shade, alpha=0.45)
        crit = float(_t.ppf(1 - alpha, df))
        ax.axvline(crit, color="gray", linestyle="--", linewidth=0.9)
        ax.axvline(t_stat, color=shade, linewidth=1.6)
    else:  # less
        left = x <= t_stat
        ax.fill_between(x[left], y[left], color=shade, alpha=0.45)
        crit = float(_t.ppf(alpha, df))
        ax.axvline(crit, color="gray", linestyle="--", linewidth=0.9)
        ax.axvline(t_stat, color=shade, linewidth=1.6)

    if alternative == "two-sided":
        p_value = 2 * float(_t.sf(abs(t_stat), df))
    elif alternative == "greater":
        p_value = float(_t.sf(t_stat, df))
    else:
        p_value = float(_t.cdf(t_stat, df))

    ax.annotate(
        f"t = {t_stat:.2f}",
        xy=(t_stat, _t.pdf(t_stat, df)),
        xytext=(t_stat, max(y) * 0.6),
        ha="center",
        color=shade,
        fontsize=11,
    )
    ax.set_title(
        f"The p-value is the shaded area  (p = {p_value:.4f})", loc="left"
    )
    ax.set_xlabel("t  (standard errors from the hypothesized mean)")
    ax.set_ylabel("density under H₀")
    ax.set_yticks([])
    if created:
        fig.tight_layout()
    return fig, ax


def plot_ci_coverage(
    true_mean=0.0,
    true_sd=1.0,
    n=30,
    n_intervals=100,
    confidence=0.95,
    seed=0,
    ax=None,
):
    """Show what "95% confidence" actually means, by simulation.

    Draws ``n_intervals`` fresh samples from a known population, builds a
    confidence interval from each, and plots them. Intervals that capture the
    true mean are black; the ones that miss are red. About ``confidence`` of
    them should contain the true value — that long-run capture rate, not any
    single interval, is what the confidence level refers to.

    Parameters
    ----------
    true_mean, true_sd : float
        The (known, simulated) population parameters.
    n : int
        Size of each sample.
    n_intervals : int
        How many samples/intervals to simulate.
    confidence : float
        Nominal confidence level, e.g. 0.95.
    seed : int
        RNG seed for reproducibility.
    ax : matplotlib Axes, optional

    Returns
    -------
    (fig, ax)
    """
    from scipy.stats import t as _t

    set_minimalist_style()
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 7))
    else:
        fig = ax.figure

    rng = np.random.default_rng(seed)
    df = n - 1
    crit = float(_t.ppf(1 - (1 - confidence) / 2, df))

    captured = 0
    miss_color, hit_color = "#c0392b", "black"
    for i in range(n_intervals):
        sample = rng.normal(true_mean, true_sd, n)
        m = float(np.mean(sample))
        se = float(np.std(sample, ddof=1)) / np.sqrt(n)
        lo, hi = m - crit * se, m + crit * se
        hit = lo <= true_mean <= hi
        captured += hit
        ax.plot(
            [lo, hi],
            [i, i],
            color=hit_color if hit else miss_color,
            linewidth=1.1 if hit else 1.6,
            alpha=0.8 if hit else 1.0,
        )
        ax.plot(m, i, "o", color=hit_color if hit else miss_color, markersize=2.5)

    ax.axvline(true_mean, color="#2c7fb8", linewidth=1.4)
    ax.text(
        true_mean,
        n_intervals * 1.01,
        "true mean",
        color="#2c7fb8",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    rate = captured / n_intervals
    ax.set_title(
        f"{int(confidence * 100)}% CIs over {n_intervals} samples — "
        f"{captured} captured the true mean ({rate:.0%})",
        loc="left",
        fontsize=11,
    )
    ax.set_xlabel("estimate of the mean")
    ax.set_yticks([])
    ax.set_ylabel(f"{n_intervals} repeated samples")
    if created:
        fig.tight_layout()
    return fig, ax


def plot_honest_vs_misleading(
    values,
    labels=None,
    title="Does your y-axis start at zero?",
    ax_pair=None,
):
    """Show the same bar chart with a truncated axis vs. an honest full axis.

    A truncated y-axis can make a tiny difference look dramatic.  This side-by-
    side comparison makes the distortion immediately obvious — a useful teaching
    tool for data visualization ethics.

    Args:
        values: Numeric values for each bar.
        labels: Category labels.  Defaults to A, B, C, …
        title: Overall figure title.
        ax_pair: Optional (ax_misleading, ax_honest) to draw onto.

    Returns:
        (fig, (ax_misleading, ax_honest))

    Example:
        >>> fig, axes = plot_honest_vs_misleading(
        ...     [98, 101, 99, 103], labels=["Q1", "Q2", "Q3", "Q4"]
        ... )
    """
    values = list(values)
    if labels is None:
        labels = [chr(65 + i) for i in range(len(values))]

    set_minimalist_style()

    if ax_pair is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        ax1, ax2 = ax_pair
        fig = ax1.figure

    color = "#5E81AC"
    span = max(values) - min(values)
    pad = span * 0.15 if span > 0 else 1.0

    # Misleading: truncated axis
    ax1.bar(labels, values, color=color, alpha=0.75)
    ax1.set_ylim(min(values) - pad, max(values) + pad)
    ax1.set_title("MISLEADING — truncated y-axis", color="#BF616A", fontsize=12)
    ax1.set_ylabel("Value")

    # Honest: axis starts at zero
    ax2.bar(labels, values, color=color, alpha=0.75)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.set_title("HONEST — y-axis starts at zero", color="#A3BE8C", fontsize=12)
    ax2.set_ylabel("Value")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_survival_curve(
    km_result,
    parametric_results=None,
    ax=None,
):
    """Plot a Kaplan-Meier curve with optional parametric overlays.

    Args:
        km_result: Dict from ``real_simple_stats.kaplan_meier``.
        parametric_results: Optional list of dicts from
            ``real_simple_stats.fit_parametric_survival`` or
            ``real_simple_stats.compare_survival_models``.  Each is plotted
            using its ``survival_fn``.
        ax: Existing Axes, or None to create one.

    Returns:
        (fig, ax)

    Example:
        >>> from real_simple_stats import kaplan_meier
        >>> r = kaplan_meier([2, 3, 5, 7, 4, 8, 10, 11], [1, 1, 1, 1, 1, 0, 1, 0])
        >>> fig, ax = plot_survival_curve(r)
    """
    set_minimalist_style()
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    t = km_result["times"]
    s = km_result["survival_prob"]
    ci_lo = km_result["ci_lower"]
    ci_hi = km_result["ci_upper"]

    ax.step(t, s, where="post", color="black", linewidth=2, label="Kaplan–Meier")
    ax.fill_between(
        t, ci_lo, ci_hi, step="post", alpha=0.15, color="black", label="95% CI"
    )

    if parametric_results:
        colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
        t_grid = np.linspace(0, float(t[-1]) * 1.05, 300)
        for i, pr in enumerate(parametric_results):
            s_fit = np.array([pr["survival_fn"](ti) for ti in t_grid])
            label = pr["distribution"].capitalize()
            if "rank" in pr:
                label += f" (AIC rank {pr['rank']})"
            ax.plot(
                t_grid, s_fit,
                linewidth=1.8,
                color=colors[i % len(colors)],
                linestyle="--",
                label=label,
            )

    median = km_result.get("median_survival")
    if median is not None:
        ax.axvline(median, color="gray", linestyle=":", linewidth=1.2)
        ax.annotate(
            f"median = {median:.1f}",
            xy=(median, 0.5),
            xytext=(median * 1.05, 0.55),
            fontsize=10,
            color="gray",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability  S(t)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    n_ev = km_result.get("n_events", "")
    n_ce = km_result.get("n_censored", "")
    ax.set_title(
        f"Survival curve  (events = {n_ev}, censored = {n_ce})", loc="left"
    )
    ax.legend(frameon=False)

    if created:
        fig.tight_layout()
    return fig, ax


def plot_variogram(
    variogram_result,
    fit_result=None,
    ax=None,
):
    """Plot an experimental variogram with an optional fitted model overlay.

    Args:
        variogram_result: Dict from ``real_simple_stats.compute_variogram``.
        fit_result: Optional dict from ``real_simple_stats.fit_variogram``.
            If provided, the fitted model curve is overlaid.
        ax: Existing Axes, or None to create a new figure.

    Returns:
        (fig, ax)

    Example:
        >>> from real_simple_stats import compute_variogram
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x, y = rng.uniform(0,100,60), rng.uniform(0,100,60)
        >>> v = np.sin(x/20) + rng.normal(0, 0.5, 60)
        >>> r = compute_variogram(x, y, v)
        >>> fig, ax = plot_variogram(r)
    """
    set_minimalist_style()
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    lags = variogram_result["lags"]
    gamma = variogram_result["gamma"]
    n_pairs = variogram_result.get("n_pairs")
    total_var = variogram_result.get("total_variance")

    sizes = (np.asarray(n_pairs) / max(np.asarray(n_pairs).max(), 1) * 120 + 20
             if n_pairs is not None else 60)
    ax.scatter(lags, gamma, s=sizes, color="black", alpha=0.7, zorder=3,
               label="Experimental")

    if total_var is not None:
        ax.axhline(total_var, color="#5E81AC", linestyle=":", linewidth=1.2,
                   label=f"Total variance = {total_var:.2f}")

    if fit_result is not None:
        h_grid = np.linspace(0, float(np.max(lags)) * 1.05, 200)
        g_fit = np.array([fit_result["model_fn"](h) for h in h_grid])
        label = (
            f"{fit_result['model'].capitalize()}  "
            f"(nugget={fit_result['nugget']:.2f}, "
            f"sill={fit_result['sill']:.2f}, "
            f"range={fit_result['range_param']:.1f})"
        )
        ax.plot(h_grid, g_fit, color="#BF616A", linewidth=2, label=label)

    ax.set_xlabel("Lag distance  h")
    ax.set_ylabel("Semivariance  γ(h)")
    ax.set_title("Variogram", loc="left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)

    if created:
        fig.tight_layout()
    return fig, ax


def plot_correlation_matrix(
    data,
    labels=None,
    title="Correlation Matrix",
    ax=None,
):
    """Plot a correlation matrix as an annotated heatmap (pure matplotlib).

    Args:
        data: 2-D array-like of shape (n_observations, n_variables), or a
            pre-computed (n_variables, n_variables) correlation matrix
            (values in [−1, 1]).
        labels: Variable names.  Defaults to "V1", "V2", …
        title: Plot title.
        ax: Existing Axes, or None to create one.

    Returns:
        (fig, ax)

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(100, 4))
        >>> fig, ax = plot_correlation_matrix(X, labels=["A","B","C","D"])
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2-D.")

    # If square with values in [-1, 1], treat as correlation matrix directly
    n, m = X.shape
    if n == m and np.all(np.abs(X) <= 1.0 + 1e-6):
        corr = X
    else:
        corr = np.corrcoef(X, rowvar=False)
        m = corr.shape[0]

    if labels is None:
        labels = [f"V{i+1}" for i in range(m)]

    set_minimalist_style()
    created = ax is None
    if created:
        size = max(4, m * 0.7)
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(m):
        for j in range(m):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(title, loc="left")

    if created:
        fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example usage

    mu, sigma = 50, 10
    data = np.random.normal(mu, sigma, 1000)

    plot_norm_hist(data, mu, sigma)
    plot_box(data)
