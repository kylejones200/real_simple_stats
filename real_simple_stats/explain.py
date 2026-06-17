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


__all__ = ["ExplainedResult", "one_sample_t_test_explained"]
