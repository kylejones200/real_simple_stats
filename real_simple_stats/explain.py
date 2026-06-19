"""Self-explaining statistical results — the reference template.

Most statistics libraries hand you a number and walk away. A beginner who runs
a t-test gets ``t`` and ``p`` and is left at the exact cliff where statistics
gets hard: *what does this mean, and what does it NOT mean?*

This module turns a test into a teacher. Every result carries the things a
learner actually needs:

    1. The QUESTION it answers
    2. The numeric RESULT (still fully accessible for code)
    3. The INTUITION — what the test is really doing
    4. The ASSUMPTIONS, and whether they hold for *this* data
    5. A plain-English INTERPRETATION
    6. A MISCONCEPTION GUARD — what the result does *not* mean
    ...plus concrete NEXT STEPS and an optional intuition PLOT.

``one_sample_t_test_explained`` is the canonical implementation. To make any
other test self-explaining, copy its shape: compute as you normally would,
then populate an :class:`ExplainedResult`. The narrative is the product; the
computation is the easy part. See ``docs/SELF_EXPLAINING_RESULTS.md``.

Example
-------
>>> import real_simple_stats as rss
>>> result = rss.one_sample_t_test_explained([5.1, 4.9, 5.3, 5.0, 5.2], mu=5.0)
>>> result.p_value            # use it as data
0.4...
>>> print(result)             # or let it teach
=== One-Sample t-Test ===
...
>>> result.plot()             # or let it show you the p-value
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from scipy.stats import t as t_dist

from . import assumptions as assume
from . import descriptive_statistics as desc

# Normalize the many names people use for a test direction into three canonical
# ones. Beginners type whatever their textbook used; we accept all of it.
_ALT_ALIASES = {
    "two-sided": "two-sided",
    "two_sided": "two-sided",
    "two-tailed": "two-sided",
    "two": "two-sided",
    "ne": "two-sided",
    "!=": "two-sided",
    "greater": "greater",
    "right": "greater",
    "right-tailed": "greater",
    "larger": "greater",
    "gt": "greater",
    ">": "greater",
    "less": "less",
    "left": "less",
    "left-tailed": "less",
    "smaller": "less",
    "lt": "less",
    "<": "less",
}

_RULE = "─" * 64


def _normalize_alternative(alternative: str) -> str:
    key = alternative.strip().lower()
    if key not in _ALT_ALIASES:
        raise ValueError(
            f"Unknown alternative {alternative!r}. "
            "Use 'two-sided', 'greater', or 'less'."
        )
    return _ALT_ALIASES[key]


def _wrap(text: str, indent: str = "  ") -> str:
    """Wrap prose to a comfortable width, preserving intentional line breaks."""
    out = []
    for line in text.strip().split("\n"):
        if not line.strip():
            out.append("")
            continue
        out.append(
            textwrap.fill(
                line.strip(),
                width=72,
                initial_indent=indent,
                subsequent_indent=indent,
            )
        )
    return "\n".join(out)


@dataclass
class ExplainedResult:
    """A statistical result that can explain itself.

    This is deliberately *generic*: it knows nothing about t-tests specifically.
    Any test can produce one. The headline numbers live in ``values`` and are
    also reachable as attributes (``result.p_value``), so the same object serves
    both a script that wants a float and a human who wants to understand it.

    Attributes
    ----------
    title:
        Short name of the test, e.g. ``"One-Sample t-Test"``.
    question:
        The real-world question the test answers, in one sentence.
    values:
        Headline numbers, name -> value. Exposed via attribute access too.
    decision:
        The verdict, e.g. ``"Reject H0"`` / ``"Fail to reject H0"`` (or None).
    intuition / interpretation:
        Prose. ``intuition`` explains the machinery; ``interpretation`` says
        what *this* result means in plain language.
    assumptions:
        Raw dict from the relevant assumptions checker, plus a one-line summary
        under the ``"summary"`` key.
    caveats:
        The misconception guard — what this result does NOT license you to say.
    next_steps:
        Concrete suggestions for what to do next.
    """

    title: str
    question: str
    values: dict[str, Any]
    intuition: str
    interpretation: str
    assumptions: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    decision: str | None = None
    _plot_fn: Callable[..., Any] | None = field(default=None, repr=False)

    # -- ergonomic numeric access: result.p_value, result.statistic, ... -------
    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for names not found normally, so this never
        # shadows real fields. Guard against the dataclass not being built yet.
        try:
            values = object.__getattribute__(self, "values")
        except AttributeError:  # pragma: no cover - during unpickling
            raise AttributeError(name) from None
        if name in values:
            return values[name]
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}. "
            f"Available numbers: {', '.join(values)}"
        )

    # -- rendering -------------------------------------------------------------
    def explain(self) -> str:
        """Return the full narrative as plain text (used by ``print``)."""
        lines: list[str] = [f"=== {self.title} ===", ""]

        lines.append("QUESTION")
        lines.append(_wrap(self.question))
        lines.append("")

        lines.append("RESULT")
        for key, val in self.values.items():
            lines.append(f"  {key:<22}{_fmt(val)}")
        if self.decision:
            lines.append(f"  {'decision':<22}{self.decision}")
        lines.append("")

        lines.append("WHAT THE TEST IS DOING")
        lines.append(_wrap(self.intuition))
        lines.append("")

        if self.assumptions:
            lines.append("ASSUMPTIONS")
            lines.append(_wrap(self.assumptions.get("summary", "—")))
            lines.append("")

        lines.append("INTERPRETATION")
        lines.append(_wrap(self.interpretation))
        lines.append("")

        if self.caveats:
            lines.append("WHAT THIS DOES *NOT* MEAN")
            for c in self.caveats:
                lines.append(_wrap(f"• {c}"))
            lines.append("")

        if self.next_steps:
            lines.append("NEXT STEPS")
            for s in self.next_steps:
                lines.append(_wrap(f"→ {s}"))
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def __str__(self) -> str:
        return self.explain()

    def _repr_markdown_(self) -> str:
        """Rich rendering inside Jupyter / IPython notebooks."""
        md: list[str] = [f"### {self.title}", ""]
        md.append(f"**Question.** {self.question}")
        md.append("")
        md.append("| quantity | value |")
        md.append("|---|---|")
        for key, val in self.values.items():
            md.append(f"| {key} | {_fmt(val)} |")
        if self.decision:
            md.append(f"| **decision** | **{self.decision}** |")
        md.append("")
        md.append(f"**What the test is doing.** {self.intuition.strip()}")
        md.append("")
        if self.assumptions:
            md.append(f"**Assumptions.** {self.assumptions.get('summary', '—')}")
            md.append("")
        md.append(f"**Interpretation.** {self.interpretation.strip()}")
        md.append("")
        if self.caveats:
            md.append("**What this does _not_ mean**")
            md.append("")
            for c in self.caveats:
                md.append(f"- {c}")
            md.append("")
        if self.next_steps:
            md.append("**Next steps**")
            md.append("")
            for s in self.next_steps:
                md.append(f"- {s}")
            md.append("")
        return "\n".join(md)

    def plot(self, **kwargs: Any) -> Any:
        """Draw the result's signature visualization, if it has one."""
        if self._plot_fn is None:
            raise NotImplementedError(
                f"{self.title} does not define a plot. "
                "Attach one by setting `_plot_fn` when you build the result."
            )
        return self._plot_fn(**kwargs)


def _fmt(val: Any) -> str:
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, float):
        if val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e6):
            return f"{val:.3e}"
        return f"{val:.4f}"
    if isinstance(val, tuple) and len(val) == 2:
        return f"({_fmt(val[0])}, {_fmt(val[1])})"
    return str(val)


def _fmt_p(p: float) -> str:
    """Format a p-value for prose, avoiding a misleading '0.0000'."""
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


def _np_array(obj: Any, dtype: Any = float) -> Any:
    import numpy as np
    return np.asarray(obj, dtype=dtype)


def _cohens_d_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def one_sample_t_test_explained(
    data: Sequence[float],
    mu: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    check_assumptions: bool = True,
) -> ExplainedResult:
    """Run a one-sample t-test that explains itself.

    Tests whether the mean of ``data`` differs from a hypothesized value ``mu``.
    Returns an :class:`ExplainedResult`: use it as data (``.p_value``,
    ``.statistic``, ``.ci``...) or let it teach (``print(result)``,
    ``result.plot()``).

    This function is the **reference template** for the self-explaining pattern.
    Notice the shape: a short computation block, then a much longer block that
    builds the narrative. That ratio is the point — the explanation is the
    feature. Clone this structure for any other test.

    Parameters
    ----------
    data:
        The sample.
    mu:
        The null-hypothesis mean (the value you're testing against).
    alpha:
        Significance threshold (default 0.05 — a convention, not a law).
    alternative:
        ``"two-sided"`` (default), ``"greater"``, or ``"less"``. Many aliases
        ("two-tailed", "right", "left"...) are accepted.
    check_assumptions:
        If True, run the t-test assumption checks and fold a summary in.

    Returns
    -------
    ExplainedResult
    """
    alt = _normalize_alternative(alternative)
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 observations for a one-sample t-test.")

    # --- compute (the easy part) ---------------------------------------------
    xbar = desc.mean(data)
    s = desc.sample_std_dev(data)
    df = n - 1
    se = s / math.sqrt(n)
    t_stat = (xbar - mu) / se

    if alt == "two-sided":
        p_value = 2 * float(t_dist.sf(abs(t_stat), df))
    elif alt == "greater":
        p_value = float(t_dist.sf(t_stat, df))
    else:  # less
        p_value = float(t_dist.cdf(t_stat, df))

    # Two-sided (1 - alpha) confidence interval for the mean — reported
    # regardless of `alt` because it's the most interpretable companion to p.
    t_crit = float(t_dist.ppf(1 - alpha / 2, df))
    margin = t_crit * se
    ci = (xbar - margin, xbar + margin)

    # Cohen's d for one sample: how many standard deviations the mean sits
    # from mu. This is the *size* of the effect, independent of sample size.
    d = (xbar - mu) / s if s > 0 else float("nan")

    reject = p_value < alpha
    decision = "Reject H₀" if reject else "Fail to reject H₀"

    values: dict[str, Any] = {
        "n": n,
        "sample mean": xbar,
        "hypothesized μ": mu,
        "t statistic": t_stat,
        "df": df,
        "p_value": p_value,
        "ci": ci,
        "effect_size": d,
        "statistic": t_stat,  # alias so result.statistic works generically
        "reject_null": reject,
    }

    # --- assumptions ----------------------------------------------------------
    assumptions_block: dict[str, Any] = {}
    if check_assumptions:
        raw = assume.check_t_test_assumptions(data, verbose=False)
        normality_ok = raw.get("normality_overall", None)
        if n < 30 and normality_ok is False:
            summary = (
                f"Small sample (n={n}) and the data look non-normal. The t-test "
                "leans on normality most heavily here — treat the p-value as "
                "approximate and consider the Wilcoxon signed-rank test or a "
                "bootstrap/permutation approach."
            )
        elif n < 30:
            summary = (
                f"Small sample (n={n}), but the data look roughly normal, so the "
                "t-test is reasonable. With small n the normality assumption "
                "still matters — eyeball a histogram or Q-Q plot to be sure."
            )
        else:
            summary = (
                f"Large sample (n={n}): the Central Limit Theorem makes the test "
                "robust to non-normality. Independence of observations is the "
                "assumption to worry about now — check how the data were collected."
            )
        assumptions_block = {"summary": summary, **raw}

    # --- narrative (the actual product) --------------------------------------
    dir_phrase = {
        "two-sided": f"differ from {mu}",
        "greater": f"exceed {mu}",
        "less": f"fall below {mu}",
    }[alt]
    question = (
        f"Is the true population mean different enough from {mu} that we "
        f"shouldn't chalk the gap up to random sampling? Specifically: does the "
        f"mean {dir_phrase}?"
    )

    intuition = (
        f"The t statistic is a signal-to-noise ratio. The signal is how far the "
        f"sample mean ({xbar:.4g}) sits from the hypothesized mean ({mu}); the "
        f"noise is the standard error ({se:.4g}), i.e. how much a sample mean "
        f"typically bounces around just from the luck of the draw. "
        f"Here t = {t_stat:.3f}, so the gap is about {abs(t_stat):.2f} standard "
        f"errors wide.\n"
        f"If H₀ were true, t would wander according to a t-distribution with "
        f"{df} degrees of freedom. The p-value ({_fmt_p(p_value)}) is the area in "
        f"that distribution's tail{'s' if alt == 'two-sided' else ''} beyond "
        f"your t — the chance of seeing a gap at least this extreme when nothing "
        f"is really going on. Call result.plot() to see that area."
    )

    sig_word = "is" if reject else "is not"
    interpretation = (
        f"At α = {alpha}, the result {sig_word} statistically significant "
        f"(p = {_fmt_p(p_value)}). We {decision.lower()}. "
        f"The sample mean is {xbar:.4g}, and a {int(round((1 - alpha) * 100))}% "
        f"confidence interval for the true mean runs from {ci[0]:.4g} to "
        f"{ci[1]:.4g}. The effect size (Cohen's d = {d:.3f}) is "
        f"{_cohens_d_magnitude(d)} — that's the practical magnitude of the "
        f"difference, which significance alone never tells you."
    )

    caveats = [
        f"p = {_fmt_p(p_value)} is NOT the probability that H₀ is true, nor the "
        "probability your result happened 'by chance'. It's the probability of "
        "data this extreme *assuming* H₀ is true.",
        "Statistical significance is not practical importance. A tiny, "
        "meaningless difference can be 'significant' with a big enough sample — "
        "look at the confidence interval and effect size to judge whether it "
        "matters.",
        f"α = {alpha} is a convention, not a bright line in nature. p = 0.049 "
        "and p = 0.051 are essentially the same evidence.",
    ]
    if not reject:
        caveats.append(
            "Failing to reject H₀ is NOT proof that H₀ is true. Absence of "
            "evidence isn't evidence of absence — a wide confidence interval "
            "here usually means the study was simply underpowered."
        )

    next_steps = [
        "Report the effect size and confidence interval alongside p — they "
        "answer 'how big?' where p only answers 'is it there?'.",
        "Call result.plot() to see the p-value as a shaded tail area.",
    ]
    if check_assumptions and n < 30:
        next_steps.append(
            "With this sample size, sanity-check normality (histogram or Q-Q "
            "plot). If it's badly skewed, try the Wilcoxon signed-rank test or "
            "a bootstrap confidence interval (see the resampling module)."
        )
    if not reject:
        next_steps.append(
            "If you expected an effect, run a power analysis to find the sample "
            "size you'd need to detect one (see the power_analysis module)."
        )

    # Bind the signature visualization (imported lazily so importing this
    # module never forces matplotlib to load).
    def _plot(**kwargs: Any) -> Any:
        from .plots import plot_p_value_area

        return plot_p_value_area(
            t_stat=t_stat, df=df, alternative=alt, alpha=alpha, **kwargs
        )

    return ExplainedResult(
        title="One-Sample t-Test",
        question=question,
        values=values,
        intuition=intuition,
        interpretation=interpretation,
        assumptions=assumptions_block,
        caveats=caveats,
        next_steps=next_steps,
        decision=decision,
        _plot_fn=_plot,
    )


# ---------------------------------------------------------------------------
# Private helpers shared across the explained functions below
# ---------------------------------------------------------------------------


def _eta2_label(e: float) -> str:
    if e < 0.01:
        return "negligible"
    if e < 0.06:
        return "small"
    if e < 0.14:
        return "medium"
    return "large"


def _cramers_v_label(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# One-Way ANOVA
# ---------------------------------------------------------------------------


def one_way_anova_explained(
    *groups: Sequence[float],
    alpha: float = 0.05,
) -> ExplainedResult:
    """One-way ANOVA that explains itself.

    Tests whether any of *k* groups have a different mean from the others.
    Returns an :class:`ExplainedResult` that reads as data (``.f_stat``,
    ``.p_value``, ``.eta_squared``) and as a teacher (``print(result)``,
    ``result.plot()``).

    Parameters
    ----------
    *groups:
        Two or more sequences of numeric observations.
    alpha:
        Significance threshold (default 0.05).
    """
    import numpy as np

    from .hypothesis_testing import one_way_anova

    r = one_way_anova(*groups, alpha=alpha)
    n_groups = r["n_groups"]
    n_total = r["n_total"]
    f_stat = r["f_stat"]
    p_value = r["p_value"]
    eta2 = r["eta_squared"]
    reject = r["reject_null"]
    means = r["group_means"]
    group_ns = r["group_ns"]

    decision = "Reject H₀" if reject else "Fail to reject H₀"
    means_str = ", ".join(
        f"Group {i + 1} (n={group_ns[i]}): {m:.3g}" for i, m in enumerate(means)
    )

    question = (
        f"Do any of these {n_groups} groups have a meaningfully different mean, "
        "or are the observed differences just random sampling variation?"
    )

    intuition = (
        f"ANOVA answers 'is the signal bigger than the noise?' by partitioning "
        f"total variance into two buckets. Between-group variance measures how "
        f"much the {n_groups} group means deviate from the grand mean — the "
        f"signal. Within-group variance measures how much individuals scatter "
        f"inside their own group — the noise. The F statistic is their ratio: "
        f"F = between / within. "
        f"Here F({r['df_between']}, {r['df_within']}) = {f_stat:.3f}, giving "
        f"p = {_fmt_p(p_value)}. "
        f"η² ({eta2:.3f}) is the fraction of total variance explained by group "
        f"membership — it answers 'how big?' where p only answers 'is it there?'"
    )

    if reject:
        interpretation = (
            f"At α = {alpha}, the omnibus F-test is significant (p = {_fmt_p(p_value)}). "
            f"We reject the null that all {n_groups} means are equal. "
            f"Group means: {means_str}. "
            f"η² = {eta2:.3f} ({_eta2_label(eta2)} effect): group membership "
            f"explains {eta2 * 100:.1f}% of the outcome variance."
        )
    else:
        interpretation = (
            f"At α = {alpha}, the result is not significant (p = {_fmt_p(p_value)}). "
            f"Insufficient evidence that any group means differ. "
            f"Group means: {means_str}. η² = {eta2:.3f} ({_eta2_label(eta2)} effect)."
        )

    assumptions: dict[str, Any] = {
        "summary": (
            f"ANOVA requires (1) independent observations, (2) approximately "
            f"normal distributions within each group — the CLT provides some "
            f"robustness with n={n_total} total — and (3) roughly equal variances "
            f"across groups (homoscedasticity). Unequal variances inflate the "
            f"Type I error rate; Welch's ANOVA is more robust when variances differ."
        )
    }

    caveats = [
        f"A significant F only says *some* group differs — not which ones. "
        "You need post-hoc tests to identify the specific contrasts.",
        f"η² benchmarks (Cohen 1988): 0.01 small, 0.06 medium, 0.14 large. "
        f"Your η² = {eta2:.3f} is {_eta2_label(eta2)}. "
        "With large samples even trivial differences reach significance.",
        "ANOVA compares means only. Two groups can have equal means but very "
        "different spreads or shapes — always look at the distributions.",
    ]
    if not reject:
        caveats.append(
            "Failing to reject H₀ is NOT proof that all means are equal. "
            "Small group sizes reduce power — run a power analysis before concluding."
        )

    next_steps_list = []
    if reject:
        next_steps_list.append(
            "Run post-hoc pairwise tests (Tukey's HSD or Bonferroni correction) "
            "to find which specific group pairs differ."
        )
    next_steps_list.extend([
        "Verify equal variance: the largest group SD should not exceed ~2× the smallest.",
        "Call result.plot() to see box plots of each group.",
    ])
    if not reject:
        next_steps_list.append(
            "Run a power analysis to determine the sample size needed to detect "
            "your hypothesized effect size."
        )

    _arrays = [_np_array(g) for g in groups]
    _grand_mean = float(sum(a.sum() for a in _arrays) / n_total)
    _means = means
    _group_ns = group_ns
    _n_groups = n_groups
    _f_stat = f_stat
    _p_str = _fmt_p(p_value)

    def _plot(**kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        from .plots import set_minimalist_style

        set_minimalist_style()
        ax = kwargs.get("ax")
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(5, _n_groups * 1.5), 4))
        else:
            fig = ax.figure
        labels = [f"Group {i + 1}\n(n={_group_ns[i]})" for i in range(_n_groups)]
        bp = ax.boxplot(
            [a.tolist() for a in _arrays],
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="#e8e8e8", color="#444444"),
            medianprops=dict(color="#222222", linewidth=2),
            whiskerprops=dict(color="#444444"),
            capprops=dict(color="#444444"),
            flierprops=dict(marker="o", color="#444444", alpha=0.5, markersize=4),
        )
        ax.axhline(_grand_mean, ls="--", lw=1.2, color="#888888", label="grand mean")
        for i, m in enumerate(_means):
            ax.plot(i + 1, m, "D", color="#222222", zorder=5, markersize=6)
        ax.set_title(
            f"One-Way ANOVA — F={_f_stat:.2f}, p={_p_str}",
            fontsize=11,
        )
        ax.set_ylabel("Value")
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        return fig

    return ExplainedResult(
        title="One-Way ANOVA",
        question=question,
        values={
            "f_stat": f_stat,
            "p_value": p_value,
            "df_between": r["df_between"],
            "df_within": r["df_within"],
            "eta_squared": eta2,
            "reject_null": reject,
            "n_groups": n_groups,
            "n_total": n_total,
        },
        intuition=intuition,
        interpretation=interpretation,
        assumptions=assumptions,
        caveats=caveats,
        next_steps=next_steps_list,
        decision=decision,
        _plot_fn=_plot,
    )


# ---------------------------------------------------------------------------
# Chi-Square Test of Independence
# ---------------------------------------------------------------------------


def chi_square_independence_explained(
    observed: Sequence[Sequence[int]],
    alpha: float = 0.05,
) -> ExplainedResult:
    """Chi-square independence test that explains itself.

    Tests whether two categorical variables are associated or independent.
    Returns an :class:`ExplainedResult` with the test statistics and a
    plain-English narrative.

    Parameters
    ----------
    observed:
        2-D contingency table (rows × columns) of counts.
    alpha:
        Significance threshold (default 0.05).
    """
    import numpy as np

    from .hypothesis_testing import chi_square_independence

    r = chi_square_independence(observed, alpha=alpha)
    chi2 = r["chi2"]
    p_value = r["p_value"]
    dof = r["dof"]
    cramers_v = r["cramers_v"]
    reject = r["reject_null"]
    low_cells = r["low_expected_cells"]
    expected = r["expected"]

    obs = _np_array(observed)
    n = float(obs.sum())
    n_rows, n_cols = obs.shape

    decision = "Reject H₀ (variables are associated)" if reject else "Fail to reject H₀ (no detected association)"

    question = (
        f"Are these two categorical variables ({n_rows} rows × {n_cols} columns) "
        "associated with each other, or are they statistically independent?"
    )

    intuition = (
        "Under independence, the expected count in each cell is simply "
        "(row total × column total) / grand total. The χ² statistic sums "
        "the squared deviations of the observed counts from those expected "
        "values, scaled by the expected: χ² = Σ (O−E)²/E. "
        f"Large χ² means the data fit independence poorly. "
        f"Here χ²({dof}) = {chi2:.3f}, p = {_fmt_p(p_value)}. "
        f"Cramér's V ({cramers_v:.3f}) scales χ² to [0, 1] so you can judge "
        "effect size regardless of table dimensions or sample size."
    )

    v_label = _cramers_v_label(cramers_v)
    if reject:
        interpretation = (
            f"At α = {alpha}, the test is significant (p = {_fmt_p(p_value)}). "
            f"The two variables are associated. "
            f"Cramér's V = {cramers_v:.3f} ({v_label} association). "
            f"The observed table has {n:.0f} total observations across "
            f"{n_rows}×{n_cols} cells."
        )
        if low_cells > 0:
            interpretation += (
                f" Note: {low_cells} cell(s) have expected count < 5, which "
                "can inflate χ² — treat this result with extra caution."
            )
    else:
        interpretation = (
            f"At α = {alpha}, the result is not significant (p = {_fmt_p(p_value)}). "
            f"No evidence of association detected. "
            f"Cramér's V = {cramers_v:.3f} ({v_label}). "
            f"The observed table has {n:.0f} total observations."
        )

    assumptions: dict[str, Any] = {
        "summary": (
            "Chi-square requires (1) independent observations, (2) expected "
            f"cell counts ≥ 5 in at least 80% of cells — currently "
            f"{low_cells} cell(s) fall below this threshold"
            + (" (consider Fisher's exact test or collapsing categories)." if low_cells > 0 else ".") +
            " The test is sensitive to sample size: large n inflates χ² even "
            "for trivial associations. Always report Cramér's V alongside p."
        )
    }

    caveats = [
        "Association is NOT causation. A significant χ² only tells you the "
        "variables co-vary — it says nothing about which causes which.",
        f"Cramér's V benchmarks (approx. for 2×2): < 0.1 negligible, "
        "0.1–0.3 small, 0.3–0.5 medium, > 0.5 large. "
        f"Your V = {cramers_v:.3f} is {v_label}.",
        "χ² depends on sample size: adding more data to the same proportions "
        "always increases χ² and decreases p. A near-zero V with p < 0.05 "
        "usually means you have a large but unimportant association.",
    ]
    if low_cells > 0:
        caveats.append(
            f"{low_cells} expected cell count(s) are below 5. The χ² "
            "approximation may be invalid here — consider Fisher's exact test."
        )
    if not reject:
        caveats.append(
            "Failing to reject independence is NOT proof the variables are "
            "unrelated. A small sample may simply lack power."
        )

    next_steps_list = [
        "Report Cramér's V as the effect size measure alongside p.",
    ]
    if low_cells > 0:
        next_steps_list.append(
            "Some expected counts are < 5. Use Fisher's exact test or collapse "
            "small categories before drawing conclusions."
        )
    if reject:
        next_steps_list.append(
            "Examine which specific cells deviate most from expected counts "
            "(standardized residuals = (O−E)/√E) to understand the pattern."
        )
    next_steps_list.append("Call result.plot() to compare observed vs. expected counts.")

    _obs = obs.copy()
    _exp = expected.copy()
    _n_rows, _n_cols = n_rows, n_cols
    _chi2 = chi2
    _p_str = _fmt_p(p_value)

    def _plot(**kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        from .plots import set_minimalist_style

        set_minimalist_style()
        ax = kwargs.get("ax")
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, _n_cols * 1.5), 4))
        else:
            fig = ax.figure

        x = _np_array(list(range(_n_rows * _n_cols)))
        width = 0.4
        pos = _np_array(list(range(_n_rows * _n_cols)), dtype=float)
        obs_flat = _obs.flatten()
        exp_flat = _exp.flatten()
        ax.bar(pos - width / 2, obs_flat, width=width, label="Observed", color="#444444")
        ax.bar(pos + width / 2, exp_flat, width=width, label="Expected", color="#aaaaaa")
        cell_labels = [
            f"r{r}c{c}" for r in range(1, _n_rows + 1) for c in range(1, _n_cols + 1)
        ]
        ax.set_xticks(pos)
        ax.set_xticklabels(cell_labels, fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(f"Chi-Square Independence — χ²={_chi2:.2f}, p={_p_str}")
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        return fig

    return ExplainedResult(
        title="Chi-Square Test of Independence",
        question=question,
        values={
            "chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "cramers_v": cramers_v,
            "reject_null": reject,
            "low_expected_cells": low_cells,
        },
        intuition=intuition,
        interpretation=interpretation,
        assumptions=assumptions,
        caveats=caveats,
        next_steps=next_steps_list,
        decision=decision,
        _plot_fn=_plot,
    )


# ---------------------------------------------------------------------------
# Difference-in-Differences
# ---------------------------------------------------------------------------


def difference_in_differences_explained(
    outcome: Sequence[float],
    post: Sequence[int],
    treated: Sequence[int],
    alpha: float = 0.05,
) -> ExplainedResult:
    """Difference-in-differences estimator that explains itself.

    Compares the change in the treated group to the change in the control
    group across a pre/post boundary, attributing the extra change to the
    treatment.

    Parameters
    ----------
    outcome:
        Outcome variable, length n.
    post:
        Binary — 1 for post-treatment period, 0 for pre. Length n.
    treated:
        Binary — 1 for treatment group, 0 for control. Length n.
    alpha:
        Significance threshold (default 0.05).
    """
    import numpy as np

    from .causal_inference import difference_in_differences

    r = difference_in_differences(outcome, post, treated, alpha=alpha)
    did = r["did_estimate"]
    se = r["se"]
    t_stat = r["t_stat"]
    p_value = r["p_value"]
    ci = r["ci"]
    reject = r["reject_null"]

    y = _np_array(outcome)
    post_ = _np_array(post)
    treated_ = _np_array(treated)

    def _cell_mean(p_val: int, t_val: int) -> float:
        mask = (post_ == p_val) & (treated_ == t_val)
        return float(y[mask].mean()) if mask.any() else float("nan")

    ctrl_pre = _cell_mean(0, 0)
    ctrl_post = _cell_mean(1, 0)
    trt_pre = _cell_mean(0, 1)
    trt_post = _cell_mean(1, 1)

    naive_change_ctrl = ctrl_post - ctrl_pre
    naive_change_trt = trt_post - trt_pre

    decision = "Reject H₀ (treatment effect detected)" if reject else "Fail to reject H₀ (no detected effect)"

    question = (
        "Did the treatment cause an effect beyond the time trend that would "
        "have happened anyway — i.e., beyond what the control group experienced?"
    )

    intuition = (
        "DiD is a before/after comparison with a safety net. Instead of just "
        "measuring how much the treated group changed, it subtracts the change "
        "the control group experienced over the same period. That subtraction "
        "removes the part of the trend due to time alone (economy, seasonality, "
        "regression to the mean), leaving only the part attributable to treatment. "
        f"Control group changed {naive_change_ctrl:+.3g}; "
        f"treated group changed {naive_change_trt:+.3g}. "
        f"DiD estimate = {naive_change_trt:.3g} − {naive_change_ctrl:.3g} "
        f"= {did:+.4g}. "
        f"The OLS interaction term β₃ gives t = {t_stat:.3f}, p = {_fmt_p(p_value)}."
    )

    dir_word = "increased" if did > 0 else "decreased"
    if reject:
        interpretation = (
            f"At α = {alpha}, the treatment effect is significant (p = {_fmt_p(p_value)}). "
            f"The treatment is associated with a DiD-adjusted {dir_word} of "
            f"{abs(did):.4g} units (95% CI: {ci[0]:.4g} to {ci[1]:.4g}). "
            f"This is the estimated causal effect of the treatment, "
            f"conditional on the parallel trends assumption holding."
        )
    else:
        interpretation = (
            f"At α = {alpha}, the result is not significant (p = {_fmt_p(p_value)}). "
            f"No reliable treatment effect detected (DiD = {did:+.4g}, "
            f"95% CI: {ci[0]:.4g} to {ci[1]:.4g}). "
            "This could mean the treatment had no effect, or that the design "
            "lacked statistical power."
        )

    assumptions: dict[str, Any] = {
        "summary": (
            "DiD relies critically on the *parallel trends assumption*: absent "
            "treatment, the treated and control groups would have changed at the "
            "same rate. This cannot be directly tested in the post-period, but "
            "you can test it in pre-treatment periods — if the two groups trended "
            "together historically, the assumption is more credible. DiD also "
            "requires the composition of each group to remain stable across periods "
            "and that the treatment did not affect who is in the control group "
            "(no spillovers, no general-equilibrium effects)."
        )
    }

    caveats = [
        "The DiD estimate is only causal if the parallel trends assumption holds. "
        "If the treated group was already trending differently before treatment "
        "began, the estimate is biased.",
        "The confidence interval captures statistical uncertainty about β₃ — it "
        "does NOT capture violations of parallel trends, which are the bigger "
        "threat to validity.",
        f"DiD identifies the *Average Treatment Effect on the Treated* (ATT). "
        "It does not tell you what would happen if a different population were "
        "treated, or what the dose-response looks like.",
    ]

    next_steps_list = [
        "Test the parallel trends assumption using pre-treatment data: check "
        "whether the treated and control groups trended similarly before treatment.",
        "Call result.plot() to see the classic 2×2 DiD diagram.",
        "If your data have multiple time periods, consider a two-way "
        "fixed-effects (TWFE) panel model for more credible inference.",
    ]
    if not reject:
        next_steps_list.append(
            "If you expected an effect, check whether the bandwidth around "
            "the treatment date is wide enough and whether the control group "
            "is a credible counterfactual."
        )

    _ctrl_pre, _ctrl_post = ctrl_pre, ctrl_post
    _trt_pre, _trt_post = trt_pre, trt_post
    _did = did
    _p_str = _fmt_p(p_value)

    def _plot(**kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        from .plots import set_minimalist_style

        set_minimalist_style()
        ax = kwargs.get("ax")
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        ax.plot([0, 1], [_ctrl_pre, _ctrl_post], "o-", color="#888888",
                linewidth=2, markersize=8, label="Control")
        ax.plot([0, 1], [_trt_pre, _trt_post], "o-", color="#222222",
                linewidth=2, markersize=8, label="Treated")
        # Counterfactual line: treated pre + control change
        cf_post = _trt_pre + (_ctrl_post - _ctrl_pre)
        ax.plot([0, 1], [_trt_pre, cf_post], "--", color="#888888",
                linewidth=1.5, label="Counterfactual")
        ax.annotate(
            f"DiD = {_did:+.3g}",
            xy=(1, (_trt_post + cf_post) / 2),
            xytext=(0.75, (_trt_post + cf_post) / 2),
            fontsize=9,
            ha="right",
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"])
        ax.set_ylabel("Mean outcome")
        ax.set_title(f"Difference-in-Differences (p={_p_str})")
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        return fig

    return ExplainedResult(
        title="Difference-in-Differences",
        question=question,
        values={
            "did_estimate": did,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci": ci,
            "reject_null": reject,
            "r_squared": r["r_squared"],
            "n": r["n"],
        },
        intuition=intuition,
        interpretation=interpretation,
        assumptions=assumptions,
        caveats=caveats,
        next_steps=next_steps_list,
        decision=decision,
        _plot_fn=_plot,
    )


# ---------------------------------------------------------------------------
# Kaplan-Meier Survival Curve
# ---------------------------------------------------------------------------


def kaplan_meier_explained(
    durations: Sequence[float],
    event_observed: Sequence[int],
    alpha: float = 0.05,
) -> ExplainedResult:
    """Kaplan-Meier survival estimate that explains itself.

    Estimates the probability that a subject survives (or avoids an event)
    past each time point. Correct for right-censored data.

    Parameters
    ----------
    durations:
        Observed time to event or censoring.
    event_observed:
        1 if the event occurred, 0 if right-censored.
    alpha:
        Significance level for Greenwood confidence bands (default 0.05).
    """
    from .survival import kaplan_meier

    r = kaplan_meier(durations, event_observed, alpha=alpha)
    n_events = r["n_events"]
    n_censored = r["n_censored"]
    median = r["median_survival"]
    n = n_events + n_censored

    pct_censored = 100.0 * n_censored / n if n > 0 else 0.0
    final_surv = float(r["survival_prob"][-1])

    median_str = f"{median:.3g}" if median is not None else "not reached (S(t) never crossed 0.5)"

    question = (
        "How does the probability of survival (or of not yet experiencing "
        "the event) evolve over time, accounting for subjects we lost track of?"
    )

    intuition = (
        "At each observed event time tᵢ, the Kaplan-Meier estimator computes "
        "the fraction of at-risk subjects who experienced the event: dᵢ/nᵢ. "
        "It multiplies these conditional hazards together to build a survival "
        "curve: S(tᵢ) = S(tᵢ₋₁) × (1 − dᵢ/nᵢ). "
        "Censored observations — subjects who left the study before the event "
        "occurred — are removed from the risk set at their last known time "
        "without biasing the curve. This is the key advantage over simply "
        f"ignoring them. Here {n_censored} of {n} subjects ({pct_censored:.1f}%) "
        "are censored. Greenwood's formula gives pointwise confidence bands."
    )

    interpretation = (
        f"The study followed {n} subjects; {n_events} experienced the event "
        f"and {n_censored} were censored ({pct_censored:.1f}%). "
        f"The estimated median survival time is {median_str}. "
        f"By the end of the observation window, the estimated survival "
        f"probability is {final_surv:.3f} ({final_surv * 100:.1f}%). "
        "The step-function survival curve drops only at observed event times; "
        "the shaded band shows the pointwise Greenwood confidence interval."
    )

    assumptions: dict[str, Any] = {
        "summary": (
            "KM requires (1) non-informative censoring: subjects who leave the "
            "study must be no more or less likely to experience the event than "
            "those who remain — if patients who are sicker drop out, the estimate "
            "is biased. (2) Independent censoring times from event times. "
            "(3) Representativeness of the observed sample. The estimate is "
            "correct for the observation window; extrapolation beyond the last "
            "observed event time is unreliable."
        )
    }

    caveats = [
        "The KM curve is descriptive — it does not tell you *why* survival "
        "differs across groups or what predicts time-to-event.",
        "Confidence bands are pointwise (each time's interval is separately "
        "valid at 1−α). They are NOT simultaneous bands across all time points.",
        "If censoring is informative (sicker subjects drop out), KM is biased "
        "toward optimism — this is often the most important assumption to check.",
        "The curve is not defined beyond the last observed event; survival "
        "after that point is unknown, not zero.",
    ]

    next_steps_list = [
        "Call result.plot() to visualize the step-function curve with confidence bands.",
        "Fit a parametric model (Weibull, log-normal) to smooth the curve and "
        "extrapolate beyond the observation window — see compare_survival_models().",
        "If you have multiple groups, use a log-rank test to compare their "
        "survival curves.",
        "To understand which covariates predict survival time, fit a Cox "
        "proportional-hazards model.",
    ]

    _km_result = r

    def _plot(**kwargs: Any) -> Any:
        from .plots import plot_survival_curve

        return plot_survival_curve(_km_result, ax=kwargs.get("ax"))

    return ExplainedResult(
        title="Kaplan-Meier Survival Estimate",
        question=question,
        values={
            "n": n,
            "n_events": n_events,
            "n_censored": n_censored,
            "median_survival": median if median is not None else float("nan"),
            "final_survival_prob": final_surv,
        },
        intuition=intuition,
        interpretation=interpretation,
        assumptions=assumptions,
        caveats=caveats,
        next_steps=next_steps_list,
        decision=None,
        _plot_fn=_plot,
    )


# ---------------------------------------------------------------------------
# Moran's I — Spatial Autocorrelation
# ---------------------------------------------------------------------------


def morans_i_explained(
    x: Sequence[float],
    y: Sequence[float],
    values: Sequence[float],
    distance_threshold: float | None = None,
) -> ExplainedResult:
    """Moran's I spatial autocorrelation test that explains itself.

    Tests whether similar values cluster together in space or are randomly
    distributed. Returns an :class:`ExplainedResult` with the statistic,
    z-score, p-value, and a plain-English narrative.

    Parameters
    ----------
    x, y:
        Coordinates of each observation.
    values:
        Attribute values at each location.
    distance_threshold:
        If given, only pairs within this distance are neighbours.
        None uses all pairs.
    """
    import numpy as np

    from .spatial_stats import morans_i

    r = morans_i(x, y, values, distance_threshold=distance_threshold)
    I = r["moran_i"]
    E_I = r["expected_i"]
    z = r["z_score"]
    p = r["p_value"]
    n = r["n"]
    interp = r["interpretation"]

    significant = p < 0.05

    question = (
        f"Are similar values spatially clustered across these {n} locations, "
        "or are they distributed as randomly as if the map were shuffled?"
    )

    thresh_str = (
        f"within {distance_threshold} units" if distance_threshold is not None
        else "all pairs (fully global)"
    )
    intuition = (
        f"Moran's I is a spatial weighted correlation coefficient. For each "
        f"location, it compares its value to the average of its neighbours "
        f"({thresh_str}). Positive I means high-value locations tend to sit "
        "next to other high-value locations (and low next to low) — clustering. "
        "Negative I means high-value locations are surrounded by low-value ones "
        "— a checkerboard dispersion pattern. "
        f"Under the null hypothesis of spatial randomness, E[I] = −1/(n−1) = "
        f"{E_I:.4f} (approximately 0). "
        f"Here I = {I:.4f}, z = {z:.3f}, p = {_fmt_p(p)}."
    )

    if significant:
        if I > E_I:
            interpretation = (
                f"p = {_fmt_p(p)}: the spatial pattern is significantly clustered "
                f"(I = {I:.4f} > E[I] = {E_I:.4f}). "
                "Similar values are closer to each other than chance would predict. "
                f"{interp}"
            )
        else:
            interpretation = (
                f"p = {_fmt_p(p)}: the spatial pattern shows significant dispersion "
                f"(I = {I:.4f} < E[I] = {E_I:.4f}). "
                "Dissimilar values tend to be neighbours more than chance predicts. "
                f"{interp}"
            )
    else:
        interpretation = (
            f"p = {_fmt_p(p)}: no significant departure from spatial randomness "
            f"(I = {I:.4f}, E[I] = {E_I:.4f}). "
            "The data are consistent with values being randomly distributed "
            "across space. This does NOT mean no spatial structure exists — "
            "global Moran's I may miss local clusters."
        )

    assumptions: dict[str, Any] = {
        "summary": (
            "Moran's I under the normality assumption assumes (1) a correctly "
            "specified spatial weight matrix — the choice of distance threshold "
            "or neighbourhood definition strongly influences I, (2) stationarity: "
            "the spatial pattern should be the same everywhere in the study area, "
            "and (3) approximately normal values or large enough n for the "
            "z-score approximation. Binary or highly skewed values need a "
            "permutation-based p-value instead."
        )
    }

    caveats = [
        "Global Moran's I summarises the entire map in one number. Local "
        "clusters can cancel out (high-high in the north, low-low in the south "
        "→ global I ≈ 0). Use Local Indicators of Spatial Association (LISA) "
        "to find *where* clusters are.",
        "The result is sensitive to the weight matrix. A tight distance threshold "
        "may show clustering that disappears with a wider one. Always report "
        "your choice of threshold.",
        "Moran's I measures spatial autocorrelation in the *marginal* values, "
        "not the residuals. If a spatial trend (gradient) exists, remove it "
        "first (detrend or spatial regression) before interpreting I.",
    ]

    next_steps_list = [
        "Call result.plot() to see the spatial distribution of values.",
        "Compute a variogram to characterise how spatial autocorrelation "
        "decays with distance — see compute_variogram() and fit_variogram().",
    ]
    if significant and I > E_I:
        next_steps_list.append(
            "Investigate local cluster structure with LISA statistics "
            "(Anselin's local Moran's I) to find hotspot and coldspot regions."
        )
    next_steps_list.append(
        "If values are non-normal or n is small, validate with a permutation "
        "test: shuffle values across locations many times and compare the "
        "observed I to the permutation distribution."
    )

    _x = _np_array(x)
    _y_coord = _np_array(y)
    _vals = _np_array(values)
    _I = I
    _p_str = _fmt_p(p)

    def _plot(**kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        from .plots import set_minimalist_style

        set_minimalist_style()
        ax = kwargs.get("ax")
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure
        sc = ax.scatter(
            _x, _y_coord, c=_vals, cmap="RdYlBu_r", s=60, edgecolors="#444444",
            linewidths=0.4, alpha=0.85
        )
        plt.colorbar(sc, ax=ax, label="Value", shrink=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Spatial Values — Moran's I = {_I:.3f}, p = {_p_str}")
        plt.tight_layout()
        return fig

    return ExplainedResult(
        title="Moran's I — Global Spatial Autocorrelation",
        question=question,
        values={
            "moran_i": I,
            "expected_i": E_I,
            "z_score": z,
            "p_value": p,
            "n": n,
        },
        intuition=intuition,
        interpretation=interpretation,
        assumptions=assumptions,
        caveats=caveats,
        next_steps=next_steps_list,
        decision=None,
        _plot_fn=_plot,
    )


# ---------------------------------------------------------------------------
# Detect Change Points
# ---------------------------------------------------------------------------


def detect_change_points_explained(
    data: Sequence[float],
    n_breaks: int = 1,
    min_size: int = 5,
) -> ExplainedResult:
    """Change point detection that explains itself.

    Finds positions where the series mean shifts most, using binary
    segmentation. Returns an :class:`ExplainedResult` with the break
    locations and a narrative.

    Parameters
    ----------
    data:
        Time series values.
    n_breaks:
        Number of change points to find (default 1).
    min_size:
        Minimum segment length on either side of a break (default 5).
    """
    import numpy as np

    from .time_series import detect_change_points

    r = detect_change_points(data, n_breaks=n_breaks, min_size=min_size)
    cps = r["change_points"]
    seg_means = r["segment_means"]
    rss_red = r["rss_reduction"]
    found = len(cps)

    x = _np_array(data)
    n = len(x)

    question = (
        f"Where in this {n}-point time series does the mean shift abruptly, "
        f"suggesting a structural change?"
    )

    intuition = (
        "Binary segmentation works like a greedy recursive splitter. It finds "
        "the single position that most reduces the total within-segment variance "
        "if we treat each side as its own constant-mean process. It repeats "
        f"this up to {n_breaks} time(s), each time picking the best remaining "
        "split among all current segments. A large variance reduction (rss_reduction) "
        "means the breaks correspond to genuine mean shifts, not just noise fluctuations. "
        f"Here rss_reduction = {rss_red:.4g}."
    )

    if found == 0:
        interpretation = (
            "No change points were found. Every candidate split produced less "
            "variance reduction than the min_size constraint allowed, suggesting "
            "the series is consistent with a single constant-mean process "
            "(or the segments are too short to detect a break)."
        )
        decision_str = "No breaks detected"
    else:
        breaks_str = ", ".join(str(cp) for cp in cps)
        means_str = " → ".join(f"{m:.3g}" for m in seg_means)
        interpretation = (
            f"{found} change point(s) detected at index/indices [{breaks_str}]. "
            f"Segment means: {means_str}. "
            f"The break{'s split' if found > 1 else ' splits'} the series into "
            f"{found + 1} segments. "
            f"Total variance reduction from the breaks: {rss_red:.4g}."
        )
        decision_str = f"{found} break(s) found at {cps}"

    assumptions: dict[str, Any] = {
        "summary": (
            "Binary segmentation assumes change points manifest as *mean shifts* "
            "in an otherwise stationary process. It does not detect changes in "
            "variance, trend slope, or distributional shape. The min_size "
            f"parameter (currently {min_size}) enforces a minimum segment length "
            "that prevents detecting noise as a break but may miss genuine "
            "breaks close to the start or end of the series."
        )
    }

    caveats = [
        "Binary segmentation is greedy — it finds the globally optimal first "
        "break, then optimal conditional breaks, but not the globally optimal "
        "set of all breaks simultaneously. For precision, consider PELT or "
        "dynamic programming approaches (ruptures library).",
        "Change points in means only: if your series has trend or changing "
        "variance, detrend first or the algorithm may misplace breaks.",
        f"rss_reduction = {rss_red:.4g} is not on a standard scale. Compare "
        "it to the total series variance to judge whether the detected breaks "
        "are practically meaningful.",
    ]

    next_steps_list = [
        "Call result.plot() to see the series with detected breaks and segment means.",
        "Inspect whether detected break points align with known events (policy "
        "changes, external shocks, data collection changes).",
        "If you expected more breaks than found, reduce min_size carefully — "
        "but watch for false positives.",
        "Validate with a held-out portion of the series or domain knowledge "
        "before acting on detected breaks.",
    ]

    _x = x
    _cps = cps
    _seg_means = seg_means
    _n = n
    _rss = rss_red

    def _plot(**kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        from .plots import set_minimalist_style

        set_minimalist_style()
        ax = kwargs.get("ax")
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.figure

        t = list(range(_n))
        ax.plot(t, _x, color="#444444", linewidth=1.2, label="Series")

        boundaries = [0] + list(_cps) + [_n]
        colors_seg = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#e67e22"]
        for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            color = colors_seg[i % len(colors_seg)]
            ax.hlines(
                _seg_means[i], lo, hi - 1,
                colors=color, linewidths=2.5, label=f"Seg {i+1} mean={_seg_means[i]:.3g}"
            )
        for cp in _cps:
            ax.axvline(cp, color="#888888", ls="--", lw=1.2)

        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(
            f"Change Points — {len(_cps)} break(s) detected, "
            f"rss_reduction = {_rss:.3g}"
        )
        ax.legend(frameon=False, fontsize=8, loc="best")
        plt.tight_layout()
        return fig

    return ExplainedResult(
        title="Change Point Detection (Binary Segmentation)",
        question=question,
        values={
            "change_points": cps,
            "n_breaks_found": found,
            "n_breaks_requested": n_breaks,
            "rss_reduction": rss_red,
            "segment_means": seg_means,
            "n": n,
        },
        intuition=intuition,
        interpretation=interpretation,
        assumptions=assumptions,
        caveats=caveats,
        next_steps=next_steps_list,
        decision=decision_str,
        _plot_fn=_plot,
    )


__all__ = [
    "ExplainedResult",
    "one_sample_t_test_explained",
    "one_way_anova_explained",
    "chi_square_independence_explained",
    "difference_in_differences_explained",
    "kaplan_meier_explained",
    "morans_i_explained",
    "detect_change_points_explained",
]
