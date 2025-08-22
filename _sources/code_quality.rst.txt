Code Quality Standards
======================

Real Simple Stats maintains high code quality standards to ensure reliability, maintainability, and ease of use. This document outlines our quality practices and tools.

Quality Metrics
--------------

Current Status
~~~~~~~~~~~~~

.. list-table:: Quality Metrics
   :header-rows: 1
   :widths: 30 20 20 30

   * - Metric
     - Current
     - Target
     - Status
   * - Test Coverage
     - 41%
     - 80%+
     - 🟡 Improving
   * - Type Coverage
     - 95%
     - 100%
     - 🟢 Excellent
   * - Linting Issues
     - 0
     - 0
     - 🟢 Clean
   * - Documentation
     - 90%
     - 95%
     - 🟢 Good

Tools and Standards
------------------

Code Formatting
~~~~~~~~~~~~~~

**Black** - Automatic code formatting

* Line length: 88 characters
* Consistent style across entire codebase
* Integrated with pre-commit hooks

Configuration in ``pyproject.toml``::

    [tool.black]
    line-length = 88
    target-version = ['py37']
    include = '\.pyi?$'

Usage::

    make format        # Format all code
    make format-check  # Check formatting without changes

Linting
~~~~~~

**Flake8** - Code style and error checking

* Enforces PEP 8 style guide
* Catches common errors and code smells
* Custom configuration for compatibility with Black

Configuration in ``.flake8``::

    [flake8]
    max-line-length = 88
    extend-ignore = E203, W503, E501
    exclude = .git, __pycache__, .pytest_cache, venv, build, dist

Usage::

    make lint  # Run linting checks

Type Checking
~~~~~~~~~~~~

**MyPy** - Static type checking

* Comprehensive type hints required
* Strict type checking enabled
* Integration with popular libraries

Configuration in ``mypy.ini``::

    [mypy]
    python_version = 3.7
    warn_return_any = True
    warn_unused_configs = True
    disallow_untyped_defs = True
    disallow_incomplete_defs = True
    check_untyped_defs = True
    disallow_untyped_decorators = True

Usage::

    make type-check  # Run type checking

Testing
~~~~~~

**Pytest** - Testing framework

* Comprehensive test suite with 35+ tests
* Coverage reporting with pytest-cov
* Parameterized tests for multiple scenarios

Configuration in ``pyproject.toml``::

    [tool.pytest.ini_options]
    testpaths = ["tests"]
    python_files = ["test_*.py"]
    python_classes = ["Test*"]
    python_functions = ["test_*"]
    addopts = "--strict-markers --strict-config"

Usage::

    make test      # Run all tests
    make test-cov  # Run tests with coverage report

Development Workflow
-------------------

Pre-commit Hooks
~~~~~~~~~~~~~~~

Automatic quality checks before each commit:

.. code-block:: yaml

    repos:
      - repo: https://github.com/pre-commit/pre-commit-hooks
        rev: v4.4.0
        hooks:
          - id: trailing-whitespace
          - id: end-of-file-fixer
          - id: check-yaml
          - id: debug-statements

      - repo: https://github.com/psf/black
        rev: 23.1.0
        hooks:
          - id: black

      - repo: https://github.com/pycqa/flake8
        rev: 6.0.0
        hooks:
          - id: flake8

      - repo: https://github.com/pre-commit/mirrors-mypy
        rev: v1.0.1
        hooks:
          - id: mypy

Installation::

    make pre-commit-install

Makefile Commands
~~~~~~~~~~~~~~~~

Convenient commands for development tasks:

.. code-block:: makefile

    # Quality checks
    quality: format-check lint type-check test

    # Individual tools
    format: black real_simple_stats/ tests/
    lint: flake8 real_simple_stats/ tests/
    type-check: mypy real_simple_stats/
    test: pytest tests/ -v
    test-cov: pytest tests/ --cov=real_simple_stats --cov-report=html

Usage::

    make quality  # Run all quality checks
    make help     # Show all available commands

Code Standards
-------------

Type Hints
~~~~~~~~~~

All functions must have comprehensive type annotations:

.. code-block:: python

    from typing import List, Union, Optional, Tuple

    def calculate_statistics(
        values: List[Union[int, float]],
        include_mode: bool = True
    ) -> Tuple[float, float, Optional[Union[int, float]]]:
        """Calculate basic statistics for a dataset.

        Args:
            values: List of numeric values
            include_mode: Whether to calculate mode

        Returns:
            Tuple of (mean, std_dev, mode)
        """

Docstrings
~~~~~~~~~

Google-style docstrings with comprehensive information:

.. code-block:: python

    def standard_deviation(values: List[float]) -> float:
        """Calculate the population standard deviation.

        The standard deviation measures the amount of variation or
        dispersion of a set of values. A low standard deviation indicates
        that the values tend to be close to the mean, while a high
        standard deviation indicates that the values are spread out
        over a wider range.

        Formula: σ = √(Σ(xi - μ)² / N)

        Args:
            values: List of numeric values. Must contain at least one value.

        Returns:
            The population standard deviation as a float.

        Raises:
            ValueError: If the input list is empty.
            TypeError: If values contains non-numeric types.

        Example:
            >>> standard_deviation([2, 4, 4, 4, 5, 5, 7, 9])
            2.0

        Note:
            This calculates the population standard deviation (divides by N).
            For sample standard deviation, use sample_standard_deviation().
        """

Error Handling
~~~~~~~~~~~~~

Comprehensive input validation and meaningful error messages:

.. code-block:: python

    def coefficient_of_variation(values: List[float]) -> float:
        """Calculate coefficient of variation (CV)."""
        if not values:
            raise ValueError("Cannot calculate CV for empty dataset")

        if not all(isinstance(x, (int, float)) for x in values):
            raise TypeError("All values must be numeric (int or float)")

        mean_val = mean(values)
        if mean_val == 0:
            raise ValueError("Cannot calculate CV when mean is zero")

        std_val = standard_deviation(values)
        return (std_val / abs(mean_val)) * 100

Testing Standards
----------------

Test Coverage
~~~~~~~~~~~~

We aim for high test coverage with meaningful tests:

.. code-block:: python

    class TestDescriptiveStatistics:
        """Test suite for descriptive statistics functions."""

        def test_mean_normal_case(self):
            """Test mean calculation with normal input."""
            assert mean([1, 2, 3, 4, 5]) == 3.0

        def test_mean_single_value(self):
            """Test mean with single value."""
            assert mean([42]) == 42.0

        def test_mean_empty_list(self):
            """Test mean raises error for empty list."""
            with pytest.raises(ValueError, match="empty"):
                mean([])

        @pytest.mark.parametrize("values,expected", [
            ([1, 1, 1], 1.0),
            ([0, 0, 0], 0.0),
            ([-1, -2, -3], -2.0),
        ])
        def test_mean_edge_cases(self, values, expected):
            """Test mean with various edge cases."""
            assert mean(values) == expected

Test Organization
~~~~~~~~~~~~~~~~

* **Descriptive names**: Test names clearly describe what is being tested
* **Arrange-Act-Assert**: Clear test structure
* **Edge cases**: Test boundary conditions and error states
* **Parameterized tests**: Test multiple scenarios efficiently

Continuous Integration
---------------------

GitHub Actions
~~~~~~~~~~~~~

Automated quality checks on every pull request:

.. code-block:: yaml

    name: Quality Checks
    on: [push, pull_request]

    jobs:
      test:
        runs-on: ubuntu-latest
        strategy:
          matrix:
            python-version: [3.7, 3.8, 3.9, "3.10", "3.11"]

        steps:
        - uses: actions/checkout@v3
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: ${{ matrix.python-version }}
        - name: Install dependencies
          run: |
            pip install -e ".[dev]"
        - name: Run quality checks
          run: |
            make quality

Quality Gates
~~~~~~~~~~~~

Pull requests must pass all quality checks:

* ✅ All tests pass
* ✅ No linting errors
* ✅ Type checking passes
* ✅ Code is properly formatted
* ✅ Documentation is updated

Monitoring and Reporting
-----------------------

Coverage Reports
~~~~~~~~~~~~~~~

HTML coverage reports generated automatically::

    make test-cov
    open htmlcov/index.html

Coverage badges in README show current status.

Quality Metrics
~~~~~~~~~~~~~~

Regular monitoring of:

* Test coverage percentage
* Number of linting issues
* Type checking errors
* Documentation coverage
* Code complexity metrics

Future Improvements
------------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~

1. **Increase Test Coverage** to 80%+
2. **Add Performance Benchmarks**
3. **Implement Security Scanning**
4. **Add Complexity Analysis**
5. **Enhance Documentation Coverage**

Tools Under Consideration
~~~~~~~~~~~~~~~~~~~~~~~

* **Bandit** - Security linting
* **Radon** - Code complexity analysis
* **Safety** - Dependency vulnerability checking
* **Sphinx** - Enhanced documentation generation

Best Practices Summary
---------------------

For Contributors
~~~~~~~~~~~~~~~

1. **Run quality checks** before committing: ``make quality``
2. **Write comprehensive tests** for new functionality
3. **Add type hints** to all new functions
4. **Document thoroughly** with examples
5. **Follow existing patterns** in the codebase

For Maintainers
~~~~~~~~~~~~~~

1. **Review quality metrics** regularly
2. **Update tools and dependencies** periodically
3. **Monitor test coverage** trends
4. **Ensure CI/CD pipelines** are working
5. **Document quality standards** clearly

The quality standards ensure Real Simple Stats remains reliable, maintainable, and easy to contribute to. These practices help us deliver a professional-grade statistical library that users can trust.
