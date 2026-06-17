# Self-Explaining Results

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
