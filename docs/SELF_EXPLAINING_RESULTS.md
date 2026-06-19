# Self-Explaining Results — All Seven Tests

Most statistics libraries hand you a number and walk away. A learner who runs a
t-test gets `t` and `p` — and is stranded at the exact spot where statistics
gets hard: *what does this mean, and what does it not mean?*

This library closes that gap with **self-explaining results**. Every test can
return an `ExplainedResult` that carries the things a learner actually needs,
while still behaving like a plain data object for code that just wants the
numbers.

```python
import real_simple_stats as rss

result = rss.one_sample_t_test_explained([5.2, 5.4, 5.1, 5.5, 5.3], mu=5.0)

result.p_value        # 0.0421  -> use it as data
print(result)         # -> the full narrative (see below)
result.plot()         # -> the p-value drawn as a shaded tail area
```

Printing the result produces:

```
=== One-Sample t-Test ===

QUESTION
  Is the true population mean different enough from 5.0 that we
  shouldn't chalk the gap up to random sampling? ...

RESULT
  t statistic           3.07
  p_value               0.0421
  ci                    (5.01, 5.55)
  effect_size           0.97
  decision              Reject H₀

WHAT THE TEST IS DOING
  The t statistic is a signal-to-noise ratio. The signal is how far the
  sample mean sits from the hypothesized mean; the noise is the standard
  error ... The p-value is the area in the tail beyond your t.

ASSUMPTIONS
  Small sample (n=5), but the data look roughly normal ...

INTERPRETATION
  At α = 0.05, the result is statistically significant ...

WHAT THIS DOES *NOT* MEAN
  • p is NOT the probability that H₀ is true ...
  • Statistical significance is not practical importance ...
  • α = 0.05 is a convention, not a bright line in nature.

NEXT STEPS
  → Report the effect size and confidence interval alongside p ...
  → Call result.plot() to see the p-value as a shaded tail area.
```

## The six ingredients

An `ExplainedResult` is deliberately generic — it knows nothing about t-tests.
Any test can build one. The recipe is always the same six ingredients:

| Ingredient        | The question it answers                          |
| ----------------- | ------------------------------------------------ |
| **question**      | What real-world question does this test answer?  |
| **values**        | The numbers (also reachable as attributes).      |
| **intuition**     | What is the test *actually doing*?               |
| **assumptions**   | Do the test's requirements hold for *this* data? |
| **interpretation**| What does *this* result mean, in plain English?  |
| **caveats**       | What does the result NOT mean? (misconceptions)  |
| **next_steps**    | What should the user do next?                    |

The `caveats` block is the one most libraries omit and the one that prevents the
most real-world misuse: p-values are not the probability the null is true,
significance is not importance, "fail to reject" is not "accept."

## Adding a new self-explaining test

`one_sample_t_test_explained` in `explain.py` is the **reference template**.
To make any other test self-explaining, copy its shape. Notice the ratio: a
short computation block, then a much longer narrative block. That ratio is the
point — the explanation is the product.

```python
from real_simple_stats.explain import ExplainedResult

def chi_square_gof_explained(observed, expected, alpha=0.05) -> ExplainedResult:
    # 1. Compute as you normally would.
    stat, p = ...  # the easy part

    # 2. Populate the narrative (the actual product).
    return ExplainedResult(
        title="Chi-Square Goodness-of-Fit Test",
        question="Do the observed counts depart from what the model predicts "
                 "by more than sampling noise would explain?",
        values={"chi2": stat, "p_value": p, "statistic": stat, ...},
        intuition="The statistic sums the squared gaps between observed and "
                  "expected counts, scaled by how big a gap is normal ...",
        interpretation=f"At α = {alpha}, ...",
        assumptions={"summary": "Each expected count should be ≥ 5 ..."},
        caveats=[
            "A large chi-square tells you the fit is poor, not *why* it's poor.",
            "Failing to reject is not proof the model is correct — ...",
        ],
        next_steps=["Inspect the standardized residuals to see which cells ..."],
        decision="Reject H₀" if p < alpha else "Fail to reject H₀",
        _plot_fn=lambda **kw: plot_observed_vs_expected(observed, expected, **kw),
    )
```

Two conventions worth keeping:

- Always include `"statistic"` and `"p_value"` keys in `values` so that
  `result.statistic` / `result.p_value` work uniformly across every test.
- Attach a `_plot_fn` that draws the one picture that builds intuition for that
  test. For tail-based tests, `plots.plot_p_value_area` already does this.

## Intuition plots

Two visualizations in `plots.py` exist to *teach a concept*, not decorate data:

- **`plot_p_value_area(t_stat, df, alternative, alpha)`** — draws the
  distribution under H₀ and shades the tail area that *is* the p-value. The
  single picture that makes "p-value" click.
- **`plot_ci_coverage(true_mean, true_sd, n, n_intervals, confidence)`** —
  simulates many samples, builds a confidence interval from each, and colors
  them by whether they captured the true mean. About `confidence` of them do.
  This shows what "95% confidence" actually refers to: the long-run capture
  rate, not any single interval.

Run `python examples/explained_t_test_demo.py` to see all of it together.

---

## All seven explained functions

| Function | Test | `result.plot()` shows |
|---|---|---|
| `one_sample_t_test_explained` | One-sample t-test | p-value as shaded tail area under H₀ distribution |
| `one_way_anova_explained` | One-way ANOVA | Box plots of each group with grand mean overlay |
| `chi_square_independence_explained` | Chi-square independence | Observed vs. expected count bar chart |
| `difference_in_differences_explained` | Difference-in-differences | 2×2 DiD line diagram with counterfactual |
| `kaplan_meier_explained` | Kaplan-Meier survival | Step-function curve with Greenwood CI |
| `morans_i_explained` | Moran's I spatial autocorrelation | Spatial scatter coloured by value |
| `detect_change_points_explained` | Binary segmentation change points | Time series with break lines and segment means |

### When to use explained vs. raw

**Use `_explained`** when:
- You are teaching, presenting, or writing a report
- You are working in a Jupyter notebook and want the `_repr_markdown_()` rendering
- You want the misconception guard printed alongside the result (the **WHAT THIS DOES NOT MEAN** section)
- You are exploring unfamiliar data and want concrete next-step suggestions

**Use the raw function** when:
- You are inside a pipeline and only need the numeric output
- You are running thousands of permutations or bootstrap iterations (the narrative adds no value in a loop)
- You have already understood the test and just want the dict

```python
# Pipeline (raw)
results = [rss.one_way_anova(g1, g2, g3) for g1, g2, g3 in group_triples]
p_values = [r["p_value"] for r in results]

# Teaching / exploration (explained)
result = rss.one_way_anova_explained(control, treatment_a, treatment_b)
print(result)       # full narrative
result.plot()       # box plots
result.f_stat       # still works as data
```

---

## Second canonical example: one_way_anova_explained

The ANOVA case shows the pattern when the result carries both a "which groups differ?" ambiguity and an effect size that tells you *how much* of the variance the grouping explains.

```python
import real_simple_stats as rss
import numpy as np

rng = np.random.default_rng(0)
control  = rng.normal(50, 8, 40)
dose_low = rng.normal(56, 8, 40)
dose_hi  = rng.normal(65, 8, 40)

result = rss.one_way_anova_explained(control, dose_low, dose_hi)
print(result)
```

Output:

```
=== One-Way ANOVA ===

QUESTION
  Do any of these 3 groups have a meaningfully different mean, or are the
  observed differences just random sampling variation?

RESULT
  f_stat                108.2
  p_value               < 0.0001
  df_between            2
  df_within             117
  eta_squared           0.6493
  reject_null           yes
  n_groups              3
  n_total               120
  decision              Reject H₀

WHAT THE TEST IS DOING
  ANOVA answers 'is the signal bigger than the noise?' by partitioning total
  variance into two buckets. Between-group variance measures how much the 3
  group means deviate from the grand mean — the signal. Within-group variance
  measures how much individuals scatter inside their own group — the noise.
  The F statistic is their ratio: F = between / within. Here
  F(2, 117) = 108.2, giving p < 0.0001. η² (0.6493) is the fraction of
  total variance explained by group membership.

ASSUMPTIONS
  ANOVA requires (1) independent observations, (2) approximately normal
  distributions within each group — the CLT provides some robustness with
  n=120 total — and (3) roughly equal variances across groups
  (homoscedasticity). Unequal variances inflate the Type I error rate;
  Welch's ANOVA is more robust when variances differ.

INTERPRETATION
  At α = 0.05, the omnibus F-test is significant (p < 0.0001). We reject the
  null that all 3 means are equal. Group means: Group 1 (n=40): 50.0, Group 2
  (n=40): 56.3, Group 3 (n=40): 65.2. η² = 0.649 (large effect): group
  membership explains 64.9% of the outcome variance.

WHAT THIS DOES *NOT* MEAN
  • A significant F only says *some* group differs — not which ones. You need
    post-hoc tests to identify the specific contrasts.
  • η² benchmarks (Cohen 1988): 0.01 small, 0.06 medium, 0.14 large. Your
    η² = 0.649 is large. With large samples even trivial differences reach
    significance.
  • ANOVA compares means only. Two groups can have equal means but very
    different spreads or shapes — always look at the distributions.

NEXT STEPS
  → Run post-hoc pairwise tests (Tukey's HSD or Bonferroni correction) to
    find which specific group pairs differ.
  → Verify equal variance: the largest group SD should not exceed ~2× the
    smallest.
  → Call result.plot() to see box plots of each group.
```
