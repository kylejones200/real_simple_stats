# Changelog

All notable changes to Real Simple Stats will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
