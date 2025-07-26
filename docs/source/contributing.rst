Contributing to Real Simple Stats
==================================

We welcome contributions to Real Simple Stats! This guide will help you get started with contributing to the project.

Getting Started
--------------

Development Setup
~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork** locally::

    git clone https://github.com/yourusername/real_simple_stats.git
    cd real_simple_stats

3. **Set up development environment**::

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install development dependencies
    make install-dev
    # Or manually: pip install -e ".[dev]"

4. **Install pre-commit hooks**::

    make pre-commit-install
    # Or manually: pre-commit install

5. **Verify setup**::

    make test
    make lint
    make type-check

Development Workflow
~~~~~~~~~~~~~~~~~~

1. **Create a feature branch**::

    git checkout -b feature/your-feature-name

2. **Make your changes** following our coding standards
3. **Run quality checks**::

    make quality  # Runs tests, linting, formatting, and type checking

4. **Commit your changes**::

    git add .
    git commit -m "Add your descriptive commit message"

5. **Push and create pull request**::

    git push origin feature/your-feature-name

Code Quality Standards
---------------------

We maintain high code quality standards. All contributions must meet these requirements:

Code Style
~~~~~~~~~

* **Formatting**: Code is automatically formatted with Black (88 character line length)
* **Linting**: Must pass Flake8 linting with our configuration
* **Type Hints**: All functions must have comprehensive type annotations
* **Docstrings**: All public functions must have Google-style docstrings

Example of properly formatted function::

    def calculate_mean(values: List[float]) -> float:
        """Calculate the arithmetic mean of a list of values.
        
        Args:
            values: List of numeric values to calculate mean for.
                   Must contain at least one value.
        
        Returns:
            The arithmetic mean of the input values.
            
        Raises:
            ValueError: If the input list is empty.
            
        Example:
            >>> calculate_mean([1, 2, 3, 4, 5])
            3.0
        """
        if not values:
            raise ValueError("Cannot calculate mean of empty list")
        return sum(values) / len(values)

Testing Requirements
~~~~~~~~~~~~~~~~~~

* **Test Coverage**: New code should maintain or improve test coverage
* **Test Types**: Include unit tests for all new functions
* **Edge Cases**: Test error conditions and edge cases
* **Documentation**: Test examples in docstrings should work

Example test structure::

    def test_calculate_mean():
        """Test mean calculation with various inputs."""
        # Test normal case
        assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
        
        # Test edge cases
        assert calculate_mean([5]) == 5.0
        assert calculate_mean([1.5, 2.5]) == 2.0
        
        # Test error conditions
        with pytest.raises(ValueError):
            calculate_mean([])

Quality Checks
~~~~~~~~~~~~~

Before submitting, ensure all quality checks pass::

    make format-check  # Check code formatting
    make lint         # Check code style
    make type-check   # Check type annotations
    make test         # Run all tests
    make test-cov     # Run tests with coverage report

Or run everything at once::

    make quality

Types of Contributions
---------------------

Bug Reports
~~~~~~~~~~

When reporting bugs, please include:

* **Clear description** of the issue
* **Steps to reproduce** the problem
* **Expected vs actual behavior**
* **Environment details** (Python version, OS, package version)
* **Minimal code example** that demonstrates the issue

Feature Requests
~~~~~~~~~~~~~~~

For new features, please:

* **Check existing issues** to avoid duplicates
* **Describe the use case** and why it's needed
* **Provide examples** of how it would be used
* **Consider implementation complexity**

Code Contributions
~~~~~~~~~~~~~~~~~

We welcome various types of code contributions:

**New Statistical Functions**
    * Implement additional statistical tests
    * Add new probability distributions
    * Extend descriptive statistics

**Performance Improvements**
    * Optimize existing algorithms
    * Add vectorized operations
    * Improve memory efficiency

**Documentation**
    * Improve existing documentation
    * Add examples and tutorials
    * Fix typos and clarify explanations

**Testing**
    * Increase test coverage
    * Add integration tests
    * Improve test quality

**Infrastructure**
    * Improve build processes
    * Enhance CI/CD pipelines
    * Update development tools

Coding Guidelines
----------------

Function Design
~~~~~~~~~~~~~~

* **Single Responsibility**: Each function should do one thing well
* **Clear Naming**: Use descriptive names that explain what the function does
* **Input Validation**: Validate inputs and provide clear error messages
* **Educational Value**: Include mathematical explanations in docstrings

Statistical Accuracy
~~~~~~~~~~~~~~~~~~~

* **Verify Formulas**: Ensure statistical formulas are mathematically correct
* **Test Against Known Values**: Compare results with established statistical software
* **Handle Edge Cases**: Consider what happens with small samples, extreme values, etc.
* **Document Assumptions**: Clearly state any assumptions made by the function

Error Handling
~~~~~~~~~~~~~

* **Meaningful Messages**: Error messages should help users understand what went wrong
* **Appropriate Exceptions**: Use standard Python exceptions (ValueError, TypeError, etc.)
* **Input Validation**: Check inputs early and provide clear feedback

Example::

    if not isinstance(values, (list, tuple, np.ndarray)):
        raise TypeError("Values must be a list, tuple, or numpy array")
    
    if len(values) == 0:
        raise ValueError("Cannot calculate statistics for empty dataset")
    
    if not all(isinstance(x, (int, float)) for x in values):
        raise ValueError("All values must be numeric (int or float)")

Documentation Standards
----------------------

Docstring Format
~~~~~~~~~~~~~~~

We use Google-style docstrings::

    def function_name(param1: Type1, param2: Type2) -> ReturnType:
        """Brief description of what the function does.
        
        Longer description if needed, explaining the mathematical
        background or implementation details.
        
        Args:
            param1: Description of first parameter.
            param2: Description of second parameter.
            
        Returns:
            Description of return value.
            
        Raises:
            ExceptionType: Description of when this exception is raised.
            
        Example:
            >>> function_name(arg1, arg2)
            expected_output
            
        Note:
            Any additional notes about usage or mathematical background.
        """

Code Comments
~~~~~~~~~~~~

* **Explain Why**: Comments should explain why something is done, not what is done
* **Mathematical Context**: Explain statistical concepts and formulas
* **Complex Logic**: Break down complex calculations with comments

Release Process
--------------

Version Numbers
~~~~~~~~~~~~~~

We follow semantic versioning (MAJOR.MINOR.PATCH):

* **MAJOR**: Breaking changes to the API
* **MINOR**: New features, backward compatible
* **PATCH**: Bug fixes, backward compatible

Changelog
~~~~~~~~

All changes are documented in the changelog with:

* **Added**: New features
* **Changed**: Changes in existing functionality  
* **Deprecated**: Soon-to-be removed features
* **Removed**: Removed features
* **Fixed**: Bug fixes
* **Security**: Security improvements

Getting Help
-----------

If you need help with contributing:

* **Check Documentation**: Read through this guide and the API documentation
* **Ask Questions**: Open a GitHub issue with the "question" label
* **Join Discussions**: Participate in GitHub discussions
* **Review Examples**: Look at existing code for patterns and style

Communication
------------

* **Be Respectful**: Follow our code of conduct
* **Be Patient**: Maintainers review contributions in their spare time
* **Be Descriptive**: Provide clear descriptions in issues and pull requests
* **Be Collaborative**: We're all working together to improve the project

Recognition
----------

Contributors are recognized in:

* **README**: Major contributors listed
* **Changelog**: Contributors credited for their changes
* **Documentation**: Authors acknowledged in relevant sections

Thank you for contributing to Real Simple Stats! Your efforts help make statistical analysis more accessible to everyone.
