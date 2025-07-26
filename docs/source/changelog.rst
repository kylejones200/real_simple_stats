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

[Unreleased]
-----------

Features planned for future releases:

* **Enhanced CLI**: More statistical tests and interactive mode
* **Advanced Statistics**: ANOVA, non-parametric tests, effect sizes
* **Performance**: Vectorized operations and optimization
* **Visualization**: Enhanced plotting with seaborn integration
* **Data Support**: pandas DataFrame integration
* **Web Interface**: Optional web dashboard for statistical analysis

Migration Guide
--------------

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
--------------

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
