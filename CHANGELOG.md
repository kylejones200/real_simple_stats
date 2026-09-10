# Changelog

All notable changes to Real Simple Stats will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-09-09

Rewrite of the entire numeric backend in Rust. The public API is unchanged in
shape — same function names, same arguments, same dictionary keys — but the
library now has **no runtime dependencies at all**, and several functions
return lists where they previously returned NumPy arrays. Read the Removed and
Changed sections before upgrading.

### Added

- **Native Rust core**, shipped inside the wheel as `real_simple_stats._rss`.
  Two crates: `rss-core` (pure Rust numerics, no Python linkage) and `rss-py`
  (PyO3 bindings). Covers special functions, 17 distributions, noncentral t and
  F, Shapiro-Wilk, descriptive statistics, dense linear algebra, three
  optimizers, maximum-likelihood survival fitting, resampling, Monte Carlo path
  simulation, and spatial statistics.
- **Zero-copy input.** Anything exposing the buffer protocol with contiguous
  float64 data — a NumPy array, `array.array('d')`, a `memoryview` — is read in
  place and never copied. Lists still work and take a direct unboxing path.
- **`real_simple_stats.Rng`**, a seeded PCG64 generator, replacing the NumPy
  generator the library used internally.
- **Resampling by statistic name.** `bootstrap(data, "mean")`,
  `permutation_test(a, b, "mean")` and friends keep the whole resample inside
  Rust and run it across every core. `"mean"`, `"median"`, `"std"`, `"var"`,
  `"min"`, `"max"` and `"sum"` take that path; any other callable still works
  and runs in Python.
- **`pandas` extra**, alongside the existing `plots` extra.
- **`posterior_predictive(..., random_seed=...)`** for reproducible draws.
- 44 Rust unit tests, plus a Python parity suite gating the special functions
  against mpmath at 60 digits and the rest against SciPy.
- **Fourteen functions the README had always documented but that never
  existed.** They are thin wrappers over kernels the Rust backend already
  provides, and each is pinned against SciPy where SciPy has an equivalent:
  `skewness`, `kurtosis`, `detect_outliers_iqr`, `one_sample_t_test`,
  `two_sample_t_test`, `paired_t_test`, `z_test`, `one_proportion_z_test`,
  `mann_whitney_u`, `wilcoxon_signed_rank`, `spearman_correlation`,
  `calculate_residuals`, `simple_probability`, and `binomial_cdf`.

  Note that `wilcoxon_signed_rank` applies the continuity correction by
  default, matching this library's `mann_whitney_u` and the textbook
  treatment. `scipy.stats.wilcoxon` defaults the other way, so pass
  `correction=False` to reproduce SciPy's default output exactly.

### Changed

- **Performance.** Measured on an M-series Mac against the pure-Python 0.4.x
  implementations, one million values: `sample_std_dev` 54.6 ms → 0.15 ms on a
  buffer (372x) or 3.7 ms on a list (15x); `five_number_summary` 165.9 ms →
  3.8 ms; `median` 136.1 ms → 3.9 ms. Bootstrap over 10,000 iterations
  261.6 ms → 2.1 ms; a 10,000-permutation test 738.7 ms → 5.9 ms.
- **Import time: 470 ms → 13 ms**, since SciPy is no longer imported and
  `__version__` resolves lazily.
- **Accuracy.** Measured against mpmath rather than SciPy, the normal CDF and
  survival function are roughly 50x more accurate than SciPy's own
  (2.5e-15 relative error against 1.3e-13); SciPy is not a valid oracle at that
  precision. Everything else agrees with SciPy to about 1e-12 or better.
- **Reproducibility under parallelism.** A given `random_seed` produces
  identical results regardless of how many cores the work is spread across,
  because each iteration draws from its own derived stream rather than a shared
  generator. Verified bit-identical at 1, 2 and 8 threads.
- **Build backend** is now maturin rather than setuptools. Installing from an
  sdist needs a Rust toolchain; the published wheels do not.
- `real_simple_stats.mean`, `.median` and `.mode` now resolve to the documented,
  validated implementations in `descriptive_statistics`. `pre_statistics`
  declared no `__all__` and is star-imported later, so its deliberately
  elementary teaching versions had been shadowing them — `mean([])` raised
  `ZeroDivisionError` instead of a clear `ValueError`. The teaching versions
  remain available as `real_simple_stats.pre_statistics.mean`.

### Removed

- **NumPy, SciPy and matplotlib as runtime dependencies.** The dependency list
  is now empty. matplotlib moved to the `plots` extra; importing
  `real_simple_stats.plots` without it now raises an `ImportError` naming the
  extra, rather than a bare "No module named 'numpy'".
- The unreachable Numba JIT paths in `resampling` and `monte_carlo`. Numba was
  never a declared dependency, so every default install had been running the
  pure-Python fallback.
- The `sys.path` manipulation and pytest plugin-autoload disable in
  `__init__.py`, both of which existed only to work around incompatible
  system-wide NumPy/SciPy binaries.

### Breaking

- **Arrays became lists.** These previously returned `numpy.ndarray` and now
  return plain lists (or lists of lists): `chi_square_independence()["expected"]`,
  `geometric_brownian_motion()` (`paths`, `mean_path`, `times`, `final_values`),
  `kaplan_meier()` (`times`, `survival_prob`, `ci_lower`, `ci_upper`),
  `compute_variogram()` (`lags`, `gamma`, `n_pairs`), `synthetic_control()`
  (`weights`, `synthetic`, `gap`), `encode_transactions()`, and the
  `multivariate` results (`pca`, `factor_analysis`, `canonical_correlation`).
  Code calling `.shape`, `.sum()` or boolean-mask indexing on these needs
  updating; `len(x)`, iteration and indexing are unaffected.
- **`monte_carlo_integration` and `monte_carlo_probability` call `func` and
  `condition` once per sample** — with a float in one dimension, or a tuple of
  floats in several — instead of passing a whole array. Scalar-style lambdas
  such as `lambda x: x**2` and `lambda xy: xy[0]**2 + xy[1]**2 <= 1` are
  unaffected; genuinely vectorised callables need rewriting.
- **Descriptive results are uniformly `float`.** Integer input previously
  produced integer output for `min`, `median` and `max`.
- **Random sequences differ for a given seed**, because the generator changed
  from NumPy's to PCG64. Seeded runs remain exactly reproducible; they are
  simply a different sequence than in 0.4.x.
- **Python 3.12+ and a supported platform are required.** Wheels cover Linux
  (x86_64, aarch64), macOS (Apple silicon and Intel) and Windows (x64); they are
  built against the CPython limited API, so one wheel per platform serves 3.12,
  3.13, 3.14 and later.

### Fixed

- **Twenty-one of the 109 `rss.*` references in the README did not exist.**
  Seven were renames of functions that do (`iqr` ->
  `interquartile_range`, `r_squared` -> `coefficient_of_determination`,
  `predict` -> `regression_equation`, and so on); the other fourteen are
  listed under Added above. Also corrected in the README: `linear_regression`
  unpacked as two values when it returns five, wrong keyword names for
  `bayes_theorem` and `z_score`, three dictionary keys that do not exist
  (`p_values`, `fold_scores`, `components`), and `antecedent`/`consequent`,
  which are `antecedents`/`consequents` and are frozensets.
- **Seven documented outputs that were simply wrong**, each verified by hand
  or against SciPy: `difference_in_differences` claimed a DiD estimate of 5.0
  where the correct answer is 9.5; `chi_square_independence` claimed
  `reject_null` was True for a table whose Yates-corrected p is 0.0562;
  `empirical_bayes_estimate` checked for a key it does not return;
  `gamma_poisson_update` referenced an undefined name; `probability_tree`
  claimed 0.5 for a tree summing to 0.475; plus two formatting mismatches.
  Every doctest in the package now passes.
- `median`'s doctest, which was failing at 0.4.1.
- Two doctests in `survival`, likewise failing at 0.4.1.
- `pearson_correlation` now returns exactly 1.0 for a perfectly correlated
  pair. It computed `sxy / (sqrt(sxx) * sqrt(syy))`, which cannot, because
  squaring a rounded square root does not recover the product.

## [0.4.1] - 2026-07-16

### Fixed

- **Packaging**: explicit setuptools package discovery (`include = ["real_simple_stats*"]`) — the top-level `app/` directory (React web app) broke flat-layout auto-discovery, making the package unbuildable and failing CI since 2026-06-19. `app/` is excluded from wheels and sdists.
- **Publish workflow**: repaired the never-exercised PyPI pipeline — deprecated `actions/upload-artifact@v3`/`download-artifact@v3` bumped to v4, and the quality gate now matches `ci.yml` policy (Python 3.12 per `.python-version`, blocking pytest, non-blocking ruff/mypy; the old matrix installed on Python 3.8–3.11, which `requires-python >=3.12` made impossible).

### Added

- **Self-explaining results for six new tests** — each wraps the underlying function and returns an `ExplainedResult` carrying an intuition section, plain-English interpretation, assumption check, misconception guard (caveats), and concrete next steps. All six attach a signature visualization via `result.plot()`:
  - `one_way_anova_explained` — box plots of each group + η² narrative
  - `chi_square_independence_explained` — observed vs. expected bar chart + Cramér's V
  - `difference_in_differences_explained` — 2×2 DiD diagram with counterfactual line
  - `kaplan_meier_explained` — step-function survival curve with Greenwood CI
  - `morans_i_explained` — spatial scatter coloured by value
  - `detect_change_points_explained` — time series with break lines and segment means

- 60 new tests for the explained wrappers (structural, numeric, plot-output checks).

- `docs/WHICH_TEST.md` — statistical decision guide mapping problem type to the right `rss` function.
- `docs/CAUSAL_INFERENCE_GUIDE.md` — deep dive on DiD, RDD, synthetic control, panel FE.
- `docs/SURVIVAL_ANALYSIS_GUIDE.md` — censoring, KM vs. parametric, AIC model selection.
- `docs/SPATIAL_STATS_GUIDE.md` — Moran's I, variogram (sill/range/nugget), model families.

### Changed

- README.md rewritten to lead with the ExplainedResult feature and cover all modules.
- `docs/WHAT_CAN_STATISTICS_DO.md` rewritten — removed outdated "limitations" section that incorrectly said the library lacked causal inference and spatial statistics.
- `docs/FAQ.md` extended with four new sections (causal inference, survival, market basket, spatial stats).
- `docs/MATHEMATICAL_FORMULAS.md` extended with formulas for DiD, KM, market basket, Moran's I, and variograms.
- `QUICK_REFERENCE.md` rewritten to cover all current modules.

### Stats

- 763 tests total, all passing.

---

## [0.4.0] - 2026-06-16

### Added

- **Causal inference module** (`causal_inference.py`) — four quasi-experimental estimators:
  - `difference_in_differences` — OLS with post×treated interaction (β₃ = DiD estimator)
  - `regression_discontinuity` — local polynomial estimation at a threshold cutoff
  - `synthetic_control` — SLSQP optimisation of non-negative donor weights summing to 1
  - `panel_fixed_effects` — within-entity demeaning (equivalent to entity fixed effects)

- **Survival analysis module** (`survival.py`) — three functions:
  - `kaplan_meier` — non-parametric step-function S(t) with Greenwood confidence intervals; handles right-censored data
  - `fit_parametric_survival` — MLE fit for Exponential, Weibull, Lognormal, Log-logistic; returns `survival_fn` callable
  - `compare_survival_models` — fits all four distributions and returns AIC-ranked list

- **Market basket analysis module** (`market_basket.py`) — three functions:
  - `encode_transactions` — convert list-of-lists to binary transaction matrix
  - `frequent_itemsets` — Apriori itemset mining (support threshold, max_length)
  - `association_rules` — confidence + lift rules from frequent itemsets

- **Spatial statistics module** (`spatial_stats.py`) — six functions:
  - `morans_i` — global spatial autocorrelation with z-score and p-value under normality
  - `compute_variogram` — experimental semivariance by lag distance bins
  - `fit_variogram` — scipy `curve_fit` for spherical, exponential, Gaussian models
  - `variogram_spherical`, `variogram_exponential`, `variogram_gaussian` — model callables

- **Time series additions** (`time_series.py`):
  - `mean_absolute_scaled_error` — MASE: scale-independent forecast accuracy
  - `exponential_smoothing` — simple SES (level only), α ∈ (0, 1]
  - `double_exponential_smoothing` — Holt's method (level + trend)
  - `rolling_statistics` — rolling mean, std, min, max, expanding mean
  - `detect_change_points` — binary segmentation; returns break indices + segment means

- **Hypothesis testing additions** (`hypothesis_testing.py`):
  - `one_way_anova` — one-way ANOVA with η² effect size
  - `chi_square_independence` — chi-square test with Cramér's V

- Visualisations in `plots.py`: `plot_survival_curve`, `plot_variogram`, `plot_correlation_matrix`.

- 30 new tests for causal inference, 22 for survival, 23 for market basket, 31 for spatial stats, 37 for time-series additions, 34 for hypothesis-testing additions.

- Example scripts: `examples/causal_inference_demo.py`, `examples/survival_demo.py`, `examples/market_basket_demo.py`.

### Stats

- 703 tests total at end of v0.4.0 (prior to explained-wrappers work in v0.4.1).

---

## [0.3.2] - 2026-03-21

### Fixed

- CLI tests were failing in CI because `descriptive_stats_command` and other commands use `logger.info()` but logging was only configured in `main()`. Tests that call commands directly now configure logging with the captured stdout.

### Changed

- Removed emojis project-wide from docs, examples, and scripts
- Interactive examples documentation now recommends Chart.js and Observable for web visualizations instead of Streamlit

---

## [0.3.1] - 2025-11-17

### Fixed

Fixed several critical bugs that were causing crashes and incorrect results:

- Resolved segmentation fault when running tests. The issue was caused by incompatible system-wide NumPy/SciPy installations. Package initialization now prefers the bundled virtualenv dependencies.

- `five_number_summary()` was crashing on small datasets. It now handles n=1, n=2, and n=3 cases properly instead of trying to calculate quartiles from empty halves.

- One-sided power analysis was ignoring the sign of effect sizes. Negative effects with "less-than" alternatives were returning the same power as positive effects. The calculations now correctly handle directionality.

### Added

- Normal distribution functions are now fully implemented. The CLI previously lacked normal distribution support; `normal_pdf()` and `normal_cdf()` are now available both in Python and via the command line.

- CLI argument validation across all subcommands. Missing or invalid arguments now show helpful error messages instead of cryptic Python exceptions. For example, running `rss-calc prob --type binomial --n 10` without `--k` and `--p` will tell you exactly what's missing.

- Input validation for probability helper functions. Functions like `joint_probability()`, `bayes_theorem()`, and `expected_value()` now check that probabilities are in valid ranges and raise clear errors instead of producing `nan` or cryptic `math.comb` errors.

- `stratified_split()` now ensures minority classes get at least one test sample when possible, preventing rare classes from being completely excluded from test sets.

### Changed

- Package initialization was tweaked to avoid dependency conflicts. The `__init__.py` now sets up the environment before any imports happen.

- Error messages are more helpful. Instead of "TypeError: '<' not supported between instances of 'NoneType' and 'int'", you'll see "Error: --n (number of trials) is required for binomial distribution".

- Added 10 new tests covering edge cases and validation (511 tests total, all passing).

---

## [0.3.0] - 2025-01-05

### Added - Major Feature Release

#### New Statistical Modules (6 modules, 45+ functions)
- **Time Series Analysis** (`time_series.py`)
  - Moving averages (simple, exponential, weighted)
  - Autocorrelation and partial autocorrelation
  - Linear trend analysis and detrending
  - Seasonal decomposition
  - Differencing operations

- **Multivariate Analysis** (`multivariate.py`)
  - Multiple linear regression with diagnostics
  - Principal Component Analysis (PCA)
  - Factor analysis
  - Canonical Correlation Analysis
  - Mahalanobis distance

- **Bayesian Statistics** (`bayesian_stats.py`)
  - Conjugate prior updates (Beta-Binomial, Normal-Normal, Gamma-Poisson)
  - Credible intervals and HDI
  - Bayes factors
  - Posterior predictive distributions
  - Empirical Bayes estimation

- **Resampling Methods** (`resampling.py`)
  - Bootstrap with confidence intervals
  - Bootstrap hypothesis testing
  - Permutation tests
  - Jackknife estimation
  - K-fold cross-validation
  - Stratified train-test splitting

- **Effect Sizes** (`effect_sizes.py`)
  - Cohen's d, Hedges' g, Glass's delta
  - Eta-squared, partial eta-squared, omega-squared
  - Cramér's V and phi coefficient
  - Odds ratios and relative risk
  - Cohen's h for proportions
  - Effect size interpretation

- **Power Analysis** (`power_analysis.py`)
  - Power analysis for t-tests
  - Power analysis for proportion tests
  - Power analysis for ANOVA
  - Power analysis for correlation tests
  - Minimum detectable effect calculations
  - Sample size summaries

#### Comprehensive Test Suite
- **Test coverage increased from 47% to 86%**
- Added 346 new tests (460 total)
- 10 new test files covering all modules
- Parametrized tests for edge cases
- Mocked external dependencies
- Integration tests for workflows

#### Documentation Improvements
- **API Comparison Guide** (`docs/API_COMPARISON.md`)
  - Function comparison tables with NumPy, SciPy, pandas, statsmodels
  - Quick lookup by use case
  - 40+ comparison tables

- **Mathematical Formulas** (`docs/MATHEMATICAL_FORMULAS.md`)
  - 60+ LaTeX formulas for all functions
  - Complete mathematical reference
  - Parameter definitions and interpretations

- **Interactive Examples** (`docs/INTERACTIVE_EXAMPLES.md`)
  - Google Colab and Binder integration
  - 8 comprehensive tutorial notebooks
  - Interactive widgets and visualizations
  - Browser-based learning (no installation required)

- **FAQ** (`docs/FAQ.md`)
  - 50+ common questions answered
  - Installation, usage, and troubleshooting
  - Best practices and tips

- **Troubleshooting Guide** (`docs/TROUBLESHOOTING.md`)
  - 30+ common errors with solutions
  - Debugging strategies
  - Performance optimization tips

- **Migration Guide** (`docs/MIGRATION_GUIDE.md`)
  - From R, SciPy, statsmodels, SPSS, Excel
  - Side-by-side code comparisons
  - Function translation tables
  - Complete workflow examples

#### Release Documentation
- `RELEASE_NOTES_v0.3.0.md` - Detailed release notes
- `ADVANCED_FEATURES_SUMMARY.md` - Feature guide
- `QUICK_REFERENCE.md` - Quick reference for all functions
- `TEST_COVERAGE_REPORT.md` - Coverage analysis

### Changed
- Updated `__init__.py` to export all new functions
- Enhanced CLI with renamed `hypothesis_test_command()` (was `test_command()`)
- Improved docstrings with mathematical notation
- Updated Sphinx documentation build

### Fixed
- CLI function naming conflict with pytest
- CLT probability test boundary conditions
- Various edge cases in statistical functions

### Technical Details
- **Lines of Code**: +2,562 lines (new modules)
- **Documentation**: +3,600 lines (new docs)
- **Tests**: +346 tests
- **Coverage**: 47% → 86%
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Dependencies**: numpy>=1.20.0, scipy>=1.7.0, matplotlib>=3.3.0

---

## [0.2.0] - 2024

### Added
- Enhanced plotting capabilities
- CLI improvements
- Additional statistical functions

---

## [0.1.1] - 2024

### Fixed
- Bug fixes and improvements
- Documentation updates

---

## [0.1.0] - 2024

### Added
- Initial release
- Basic descriptive statistics
- Probability distributions
- Hypothesis testing
- Linear regression
- Chi-square tests
- Confidence intervals
- Basic plotting

---

## Links
- [PyPI](https://pypi.org/project/real-simple-stats/)
- [GitHub](https://github.com/kylejones200/real_simple_stats)
- [Documentation](https://real-simple-stats.readthedocs.io/)
