# Which Test Should I Use?

Start with your question. Follow the branches. Each leaf gives you the exact `rss` function to call and a one-paragraph rationale.

---

## Branch 1: What kind of outcome do you have?

```
What is your outcome variable?
│
├── Continuous (heights, revenues, test scores, time) ──────────────────────► Branch 2
│
├── Categorical (yes/no, grades, product types) ─────────────────────────────► Branch 3
│
├── Time-to-event (days until churn, failure, recovery) ────────────────────► Branch 4
│
├── Spatial (measurements at geographic locations) ──────────────────────────► Branch 5
│
└── Sequential / time series (measurements in order over time) ───────────────► Branch 6
```

---

## Branch 2: Continuous outcome

```
Continuous outcome
│
├── Are you describing, or testing? ──────────────────────────────────────────► Describe
│   rss.mean(), rss.five_number_summary(), rss.rolling_statistics()
│
├── Are you comparing groups?
│   │
│   ├── One group vs. a known reference value ────────────────────────────────► One-sample t-test
│   │   rss.one_sample_t_test_explained(data, mu=reference_value)
│   │   WHY: Tests whether the sample mean differs from a fixed value.
│   │   Always pair with Cohen's d (rss.cohens_d) for effect size.
│   │
│   ├── Two independent groups ────────────────────────────────────────────────► Two-sample t-test
│   │   rss.two_sample_t_test(g1, g2)
│   │   WHY: Compares means between two groups using pooled or Welch SE.
│   │   Use equal_var=False (Welch) if group SDs differ substantially.
│   │
│   ├── Two paired measurements (before/after on same unit) ────────────────────► Paired t-test
│   │   rss.paired_t_test(before, after)
│   │   WHY: Pairing removes between-subject noise, increasing power.
│   │
│   ├── Three or more groups ─────────────────────────────────────────────────► One-way ANOVA
│   │   rss.one_way_anova_explained(g1, g2, g3)
│   │   WHY: Tests omnibus H₀ that all k means are equal. Significant result
│   │   only tells you *some* group differs — follow up with post-hoc tests.
│   │   Reports η² effect size (fraction of variance explained by groups).
│   │
│   └── Non-normal data or ordinal scale?
│       ├── Two groups ──────────────────────────────────────────────────────► Mann-Whitney U
│       │   rss.mann_whitney_u(g1, g2)
│       └── Paired ───────────────────────────────────────────────────────────► Wilcoxon signed-rank
│           rss.wilcoxon_signed_rank(before, after)
│
└── Did the treatment CAUSE the change? (causal question) ─────────────────────► Branch 7
```

---

## Branch 3: Categorical outcome

```
Categorical outcome
│
├── Does my data match an expected distribution? ─────────────────────────────► Chi-square goodness-of-fit
│   rss.chi_square_statistic(observed, expected)
│   WHY: Compares observed counts to theoretical proportions.
│
└── Are two categorical variables associated? ────────────────────────────────► Chi-square independence
    rss.chi_square_independence_explained(contingency_table)
    WHY: Tests H₀ that row and column variables are independent.
    Reports Cramér's V for effect size — far more useful than p alone.
    Rule of thumb: expected cell counts should all be ≥ 5.
```

---

## Branch 4: Time-to-event outcome

```
Time-to-event (days to churn, failure, conversion, recovery)
│
├── First step: non-parametric description ──────────────────────────────────► Kaplan-Meier
│   rss.kaplan_meier_explained(durations, event_observed)
│   WHY: Makes no distributional assumptions. Correct for right-censored data.
│   The only correct way to compute median survival time with censored data.
│   result.plot() shows the step-function curve with confidence bands.
│
├── I need to extrapolate or fit a smooth curve ──────────────────────────────► Parametric survival
│   rss.compare_survival_models(durations, event_observed)
│   WHY: Fits four distributions (Exponential, Weibull, Lognormal, Log-logistic)
│   via MLE and ranks by AIC. Use the best-fitting model's survival_fn to
│   predict P(survive past time t) for any t, including future time points.
│
└── I need to compare two groups' survival curves ────────────────────────────► Log-rank test
    Not yet in rss — use scipy.stats.ttest_ind on survival times as proxy,
    or compute Kaplan-Meier for each group and inspect visually.
```

---

## Branch 5: Spatial outcome

```
Spatial data (measurements at geographic/spatial coordinates)
│
├── Do similar values cluster together in space? ─────────────────────────────► Moran's I
│   rss.morans_i_explained(x, y, values, distance_threshold=d)
│   WHY: Global autocorrelation measure. I ≈ +1 means clustering;
│   I ≈ 0 means spatial randomness; I ≈ -1 means dispersion.
│   Key choice: distance_threshold defines "neighbours" — vary it and check
│   sensitivity. result.plot() shows spatial scatter coloured by value.
│
└── How does spatial autocorrelation decay with distance? ───────────────────► Variogram
    rss.compute_variogram(x, y, values)           # experimental semivariance
    rss.fit_variogram(lags, gamma, model="spherical")  # fitted model
    WHY: The range parameter tells you the distance beyond which points are
    spatially uncorrelated. The nugget captures measurement error. The sill
    equals total variance. Essential for kriging interpolation.
    Compare spherical/exponential/Gaussian by RMSE to choose the best model.
```

---

## Branch 6: Time series outcome

```
Ordered time series
│
├── I want to smooth or forecast ────────────────────────────────────────────► Exponential smoothing
│   rss.exponential_smoothing(data, alpha=0.3)          # level only
│   rss.double_exponential_smoothing(data, alpha, beta)  # level + trend
│   WHY: SES handles series with no trend; Holt's DES handles linear trends.
│   Small α → smooth but sluggish; large α → responsive but noisy.
│
├── I want rolling summaries ────────────────────────────────────────────────► Rolling statistics
│   rss.rolling_statistics(data, window=7)
│   Returns: mean, std, min, max, expanding_mean
│
├── I want to know when the mean shifted ────────────────────────────────────► Change point detection
│   rss.detect_change_points_explained(data, n_breaks=2)
│   WHY: Binary segmentation greedily finds the splits that most reduce
│   within-segment variance. result.plot() overlays segment means.
│   Always validate detected breaks against known events.
│
├── Is my series autocorrelated? ────────────────────────────────────────────► ACF / PACF
│   rss.autocorrelation(data, max_lag=20)
│   rss.partial_autocorrelation(data, max_lag=20)
│
├── Does my series have a linear trend? ────────────────────────────────────► Linear trend
│   rss.linear_trend(data)       # (slope, intercept, r²)
│   rss.detrend(data)            # remove the trend
│
└── How accurate is my forecast? ────────────────────────────────────────────► MASE
    rss.mean_absolute_scaled_error(actual, forecast)
    WHY: Scale-independent. MASE < 1 means your model beats a naïve
    lag-1 forecast; MASE > 1 means it doesn't.
```

---

## Branch 7: Causal question

You have a continuous or binary outcome and want to know whether something *caused* a change — not just whether groups differ.

```
Causal question: "Did X cause Y?"
│
├── Do you have a pre/post period AND a control group?
│   └── YES ─────────────────────────────────────────────────────────────────► Difference-in-Differences
│       rss.difference_in_differences_explained(outcome, post, treated)
│       WHY: Compares the change in treated units to the change in control
│       units. The difference of those differences is the causal effect,
│       conditional on the parallel trends assumption. result.plot() shows
│       the 2×2 DiD diagram with the counterfactual line.
│       KEY CHECK: were the groups trending similarly before treatment?
│
├── Was treatment assigned by crossing a numerical threshold?
│   └── YES ─────────────────────────────────────────────────────────────────► Regression Discontinuity
│       rss.regression_discontinuity(outcome, running_var, cutoff=65)
│       WHY: Near the cutoff, units just above and just below are otherwise
│       similar — the only difference is which side of the line they fall on.
│       Estimates a Local Average Treatment Effect (LATE) at the cutoff.
│       KEY CHECK: Does the density of the running variable have a discontinuity
│       at the cutoff? If so, sorting/manipulation may have occurred.
│
├── No valid control group — but you have multiple potential "donor" units?
│   └── YES ─────────────────────────────────────────────────────────────────► Synthetic Control
│       rss.synthetic_control(y_treated, Y_controls, n_pre=20)
│       WHY: Builds a weighted combination of control units that matches the
│       treated unit pre-treatment. The post-treatment divergence is the effect.
│       KEY CHECK: pre_fit_mse should be small — poor pre-treatment fit
│       undermines the counterfactual.
│
├── Repeated observations per entity over time?
│   └── YES ─────────────────────────────────────────────────────────────────► Panel Fixed Effects
│       rss.panel_fixed_effects(outcome, predictors, entity)
│       WHY: Within-entity demeaning removes all time-invariant confounders
│       (entity fixed effects) without explicitly modelling them.
│       KEY CHECK: treatment must be uncorrelated with time-varying
│       unobserved factors — the hardest assumption to verify.
│
└── None of the above (pure observational data, no design)
    └── No clean causal identification available from this library.
        Consider propensity score matching (scikit-learn) or
        instrumental variables (statsmodels).
```

---

## One-paragraph method summaries

**One-sample t-test**: Tests whether your sample mean differs from a fixed reference value. The t-statistic is a signal-to-noise ratio: gap / standard error. Use when n ≥ 2 and data are roughly continuous. Always report Cohen's d alongside p — a tiny d with p < 0.05 just means your sample was large enough to detect a trivial gap.

**Two-sample t-test**: The workhorse for A/B comparisons. Use Welch (equal_var=False) by default — it's robust to unequal variances and only slightly less powerful when variances are equal.

**One-way ANOVA**: Generalises the two-sample t-test to k groups. Significant F only tells you *some* group differs — run Tukey's HSD or Bonferroni post-hoc tests to find which ones. Report η² (effect size) not just p.

**Chi-square independence**: Tests whether two categorical variables are associated. Cramér's V (0 to 1) measures how strong the association is — report it alongside p, because large n makes even trivial associations "significant."

**Kaplan-Meier**: The correct way to describe time-to-event data with right censoring. Never compute median survival time by ignoring censored observations — that's survivor bias.

**Difference-in-differences**: Identifies causal effects by subtracting the control group's time trend from the treated group's change. Only valid if the two groups trended in parallel before treatment — test this with pre-period data.

**Regression discontinuity**: Exploits a threshold cutoff to create local randomisation. Units just below and just above the cutoff are approximately exchangeable. Estimates a LATE at the cutoff, not an ATE across the full population.

**Synthetic control**: Builds a weighted counterfactual from donor units. Preferred over DiD when there is only one treated unit with a long pre-treatment history (e.g. one state, one company).

**Moran's I**: Spatial generalisation of the correlation coefficient. The distance threshold is the most consequential analytic choice — always report it and check sensitivity.

**Change point detection**: Binary segmentation is fast but greedy — it finds the globally optimal first break, then optimal conditional breaks, not the jointly optimal set. Use min_size to avoid detecting noise as a break.

---

## See also

- [CAUSAL_INFERENCE_GUIDE.md](CAUSAL_INFERENCE_GUIDE.md) — deep dive on the four causal methods
- [SURVIVAL_ANALYSIS_GUIDE.md](SURVIVAL_ANALYSIS_GUIDE.md) — censoring, KM, parametric models
- [SPATIAL_STATS_GUIDE.md](SPATIAL_STATS_GUIDE.md) — Moran's I, variograms, model selection
- [MATHEMATICAL_FORMULAS.md](MATHEMATICAL_FORMULAS.md) — the math behind every test
- [FAQ.md](FAQ.md) — common questions by module
