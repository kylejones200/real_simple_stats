# Changelog

All notable changes to Real Simple Stats will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
