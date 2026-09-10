Changelog
=========

All notable changes to Real Simple Stats are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.2.0] - 2025-07-26
--------------------

This is a major quality improvement release focusing on code standards, testing, and developer experience.

Added
~~~~~

* **Comprehensive Type Hints**: Added detailed type annotations to all functions
* **Enhanced Documentation**: Google-style docstrings with examples and mathematical explanations
* **Testing Infrastructure**: 35 comprehensive unit tests with pytest framework
* **Command Line Interface**: ``rss-calc`` CLI tool for quick statistical calculations
* **Development Tools**:

  * Black code formatting
  * Flake8 linting
  * MyPy type checking
  * Pre-commit hooks
  * Makefile for development tasks

* **Quality Assurance**:

  * Automated code formatting
  * Comprehensive error handling
  * Input validation with meaningful error messages
  * Coverage reporting

* **Documentation**:

  * Professional Sphinx documentation with RTD theme
  * Installation guide
  * Quick start guide
  * CLI reference
  * Contributing guidelines
  * Code quality standards
  * Jupyter notebook tutorial

* **Package Improvements**:

  * Enhanced ``pyproject.toml`` with tool configurations
  * Development dependencies and optional extras
  * Project metadata and classifiers
  * ``.gitignore`` for Python projects

Changed
~~~~~~~

* **Version**: Bumped from 0.1.1 to 0.2.0
* **Code Quality**: All code now follows Black formatting standards
* **Error Handling**: Improved error messages and input validation
* **Documentation**: Completely rewritten README with badges and comprehensive information
* **Package Structure**: Better organization and modern Python packaging practices

Fixed
~~~~~

* **Coefficient of Variation**: Fixed calculation bug for edge cases
* **Empty Input Handling**: Proper error handling for empty datasets
* **Division by Zero**: Added checks for zero denominators
* **Import Issues**: Cleaned up unused imports and circular dependencies
* **Type Safety**: Fixed type inconsistencies and added proper annotations

Removed
~~~~~~~

* **Unused Imports**: Cleaned up all unused import statements
* **Dead Code**: Removed commented-out and unreachable code
* **Redundant Functions**: Consolidated duplicate functionality

Security
~~~~~~~~

* **Input Validation**: Added comprehensive input validation to prevent errors
* **Type Safety**: Static type checking helps prevent runtime errors

[0.1.1] - 2024-XX-XX
--------------------

Initial release with basic statistical functionality.

Added
~~~~~

* Basic descriptive statistics (mean, median, mode, variance, standard deviation)
* Probability utilities (simple, joint, conditional probability)
* Hypothesis testing functions (t-tests, F-tests, critical values)
* Probability distributions (normal, binomial, Poisson)
* Linear regression utilities
* Chi-square test functions
* Confidence interval calculations
* Basic plotting capabilities
* Statistical glossary
* Sphinx documentation setup
* PyPI package distribution

[0.3.1] - 2025-01-XX
--------------------

Fixed
~~~~~

Fixed several critical bugs that were causing crashes and incorrect results:

* Resolved segmentation fault when running tests. The issue was caused by incompatible system-wide NumPy/SciPy installations. Package initialization now prefers the bundled virtualenv dependencies.

* ``five_number_summary()`` was crashing on small datasets. It now handles n=1, n=2, and n=3 cases properly instead of trying to calculate quartiles from empty halves.

* One-sided power analysis was ignoring the sign of effect sizes. Negative effects with "less-than" alternatives were returning the same power as positive effects. The calculations now correctly handle directionality.

Added
~~~~~

* Normal distribution functions are now fully implemented. The CLI previously lacked normal distribution support; ``normal_pdf()`` and ``normal_cdf()`` are now available both in Python and via the command line.

* CLI argument validation across all subcommands. Missing or invalid arguments now show helpful error messages instead of cryptic Python exceptions.

* Input validation for probability helper functions. Functions like ``joint_probability()``, ``bayes_theorem()``, and ``expected_value()`` now check that probabilities are in valid ranges and raise clear errors.

* ``stratified_split()`` now ensures minority classes get at least one test sample when possible, preventing rare classes from being completely excluded from test sets.

Changed
~~~~~~~

* Package initialization was tweaked to avoid dependency conflicts. The ``__init__.py`` now sets up the environment before any imports happen.

* Error messages are more helpful. Instead of cryptic Python exceptions, you'll see clear messages like "Error: --n (number of trials) is required for binomial distribution".

* Added 10 new tests covering edge cases and validation (511 tests total, all passing).

[0.3.2] - 2026-03-21
--------------------

Fixed
~~~~~

* CLI tests were failing in CI because commands use ``logger.info()`` but logging was only configured in ``main()``. Tests that call commands directly now configure logging with the captured stdout.

Changed
~~~~~~~

* Removed emojis project-wide from docs, examples, and scripts
* Interactive examples documentation now recommends Chart.js and Observable for web visualizations instead of Streamlit

[0.5.0] - 2026-09-09
--------------------

The numeric backend is now Rust, and the library has **no runtime dependencies
at all**. The public API keeps its shape -- same names, same arguments, same
dictionary keys -- but several functions return lists where they previously
returned NumPy arrays. Read *Breaking* before upgrading.

Added
~~~~~

* **Native Rust core**, shipped inside the wheel as ``real_simple_stats._rss``.
  Special functions, 17 distributions, noncentral t and F, Shapiro-Wilk,
  descriptive statistics, dense linear algebra, three optimizers,
  maximum-likelihood survival fitting, resampling, Monte Carlo simulation and
  spatial statistics.
* **Zero-copy input.** Anything exposing the buffer protocol with contiguous
  float64 data -- a NumPy array, ``array.array('d')``, a ``memoryview`` -- is
  read in place rather than copied. Lists still work.
* **``real_simple_stats.Rng``**, a seeded PCG64 generator.
* **Resampling by statistic name**: ``bootstrap(data, "mean")`` and friends keep
  the whole resample inside Rust and run it across every core.
* **Fourteen functions this documentation had always described but that did not
  exist**: ``skewness``, ``kurtosis``, ``detect_outliers_iqr``,
  ``one_sample_t_test``, ``two_sample_t_test``, ``paired_t_test``, ``z_test``,
  ``one_proportion_z_test``, ``mann_whitney_u``, ``wilcoxon_signed_rank``,
  ``spearman_correlation``, ``calculate_residuals``, ``simple_probability`` and
  ``binomial_cdf``.
* ``pandas`` extra, alongside the existing ``plots`` extra.

Changed
~~~~~~~

* **Performance** (one million values, against the previous pure-Python
  implementations): ``sample_std_dev`` 52.8 ms to 0.12 ms on a buffer;
  ``five_number_summary`` 164.9 ms to 3.8 ms; a 10,000-iteration bootstrap
  265.0 ms to 1.9 ms; a 10,000-permutation test 739.9 ms to 5.4 ms. Each is also
  two to four times faster than the NumPy equivalent.
* **Import time: 470 ms to 13 ms.**
* **Accuracy.** The special functions are gated against mpmath at 60 decimal
  digits rather than SciPy, because SciPy's own ``erfc``/``ndtr`` carry about
  1e-13 relative error in the tails. Measured against true values, the normal
  CDF and survival function here are roughly 50x more accurate than SciPy's.
* **Reproducibility under parallelism.** A given ``random_seed`` produces
  identical results regardless of core count, because each iteration draws from
  its own derived stream rather than a shared generator.
* Build backend is maturin rather than setuptools. Installing from an sdist
  needs a Rust toolchain; the published wheels do not.
* ``mean``, ``median`` and ``mode`` now resolve to the documented, validated
  implementations. ``pre_statistics`` declared no ``__all__`` and is
  star-imported later, so its deliberately elementary teaching versions had been
  shadowing them.

Removed
~~~~~~~

* **NumPy, SciPy and matplotlib as runtime dependencies.** matplotlib moved to
  the ``plots`` extra.
* The unreachable Numba code paths in ``resampling`` and ``monte_carlo``. Numba
  was never a declared dependency, so every default install had been running the
  pure-Python fallback.

Breaking
~~~~~~~~

* These now return plain lists rather than ``numpy.ndarray``:
  ``chi_square_independence()["expected"]``, ``geometric_brownian_motion``,
  ``kaplan_meier``, ``compute_variogram``, ``synthetic_control``,
  ``encode_transactions``, and the ``multivariate`` results.
* ``monte_carlo_integration`` and ``monte_carlo_probability`` call ``func`` and
  ``condition`` once per sample -- a float in one dimension, a tuple in several
  -- instead of passing a whole array. Scalar-style lambdas are unaffected.
* Descriptive results are uniformly ``float``; integer input previously produced
  integer output for ``min``, ``median`` and ``max``.
* Random sequences differ for a given seed, because the generator changed to
  PCG64. Seeded runs remain exactly reproducible.
* Python 3.12 or later is required.

Fixed
~~~~~

* Twenty-one of the 109 ``rss.*`` references in the README did not exist, and
  seven documented outputs were simply wrong -- among them a
  difference-in-differences estimate given as 5.0 where the correct answer is
  9.5. Every doctest in the package now passes.
* ``pearson_correlation`` returns exactly 1.0 for a perfectly correlated pair.

[Unreleased]
------------

(No unreleased changes)

Migration Guide
---------------

Upgrading from 0.1.1 to 0.2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Breaking Changes**: None - this is a backward-compatible release.

**Recommended Actions**:

1. **Update your installation**::

    pip install --upgrade real-simple-stats

2. **Try the new CLI**::

    rss-calc --help

3. **Check the new documentation** for enhanced examples and tutorials

4. **Consider using type hints** in your code for better IDE support

**Deprecated Features**: None in this release.

**New Opportunities**:

* Use the CLI for quick calculations
* Leverage improved error messages for debugging
* Benefit from comprehensive type hints in your IDE
* Contribute to the project using our development tools

Version Support
---------------

**Supported Versions**:

* **0.2.x**: Active development, bug fixes, and new features
* **0.1.x**: Security fixes only (until 0.3.0 release)

**Python Version Support**:

* **Python 3.7+**: Fully supported
* **Python 3.6**: No longer supported (use version 0.1.x)

**Dependency Updates**:

* **NumPy**: 1.19.0+ (was 1.18.0+)
* **SciPy**: 1.5.0+ (was 1.4.0+)
* **Matplotlib**: 3.3.0+ (was 3.1.0+)

Contributing to Changelog
-------------------------

When contributing to the project:

1. **Add entries** to the [Unreleased] section
2. **Use the standard format**: Added/Changed/Deprecated/Removed/Fixed/Security
3. **Be descriptive**: Explain what changed and why
4. **Link to issues**: Reference GitHub issues when applicable
5. **Credit contributors**: Acknowledge community contributions

For more details, see our :doc:`contributing` guide.
