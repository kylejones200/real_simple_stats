Command Line Interface Reference
=================================

Real Simple Stats includes a command-line tool for quick calculations without writing Python code.

Installation and Setup
----------------------

The CLI is automatically installed with the package::

    pip install real-simple-stats

Verify installation::

    rss-calc --help

Basic Usage
----------

The CLI uses subcommands for different types of operations::

    rss-calc <subcommand> [options]

Available subcommands:

* ``stats`` - Descriptive statistics calculations
* ``probability`` - Probability calculations
* ``hypothesis`` - Hypothesis testing
* ``glossary`` - Statistical term lookup

Global Options
-------------

.. option:: --help, -h

   Show help message and exit

.. option:: --version

   Show version information

Statistics Commands
------------------

Calculate descriptive statistics for datasets.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

    rss-calc stats --data "1,2,3,4,5" --stat mean

Options
~~~~~~

.. option:: --data DATA

   Comma-separated list of numeric values (required)

.. option:: --stat STATISTIC

   Statistic to calculate. Options:

   * ``mean`` - Arithmetic mean
   * ``median`` - Middle value
   * ``mode`` - Most frequent value
   * ``variance`` - Population variance
   * ``std`` - Standard deviation
   * ``cv`` - Coefficient of variation
   * ``all`` - All available statistics

Examples
~~~~~~~

Calculate mean::

    rss-calc stats --data "10,20,30,40,50" --stat mean
    # Output: Mean: 30.0

Calculate all statistics::

    rss-calc stats --data "1,2,2,3,4,5" --stat all
    # Output:
    # Mean: 2.83
    # Median: 2.5
    # Mode: 2
    # Variance: 2.47
    # Standard Deviation: 1.57
    # Coefficient of Variation: 55.56%

Probability Commands
-------------------

Perform probability calculations and work with distributions.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

    rss-calc probability --type binomial --n 10 --k 3 --p 0.5

Options
~~~~~~

.. option:: --type TYPE

   Type of probability calculation:

   * ``binomial`` - Binomial probability
   * ``normal`` - Normal distribution (PDF or CDF)
   * ``bayes`` - Bayes' theorem

Normal Distribution Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --x X

   Value at which to evaluate the PDF or CDF (required for normal)

.. option:: --mean MEAN

   Mean of the normal distribution (default: 0.0)

.. option:: --std STD

   Standard deviation of the normal distribution (default: 1.0, must be positive)

.. option:: --cdf

   Calculate cumulative distribution function (CDF) instead of PDF

Binomial Distribution Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --n N

   Number of trials (required for binomial, must be non-negative)

.. option:: --k K

   Number of successes (required for binomial, must be between 0 and n)

.. option:: --p P

   Probability of success (required for binomial, must be between 0 and 1)

Bayes' Theorem Options
~~~~~~~~~~~~~~~~~~~~~

.. option:: --p_b_given_a P_B_GIVEN_A

   Conditional probability P(B|A) (required, must be between 0 and 1)

.. option:: --p_a P_A

   Prior probability P(A) (required, must be between 0 and 1)

.. option:: --p_b P_B

   Prior probability P(B) (required, must be between 0 and 1, cannot be zero)

Combination/Permutation Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --n N

   Total number of items

.. option:: --k K

   Number of items to choose/arrange

Simple Probability Options
~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --favorable F

   Number of favorable outcomes

.. option:: --total T

   Total number of possible outcomes

Examples
~~~~~~~

Normal distribution PDF::

    rss-calc prob --type normal --x 0 --mean 0 --std 1
    # Output: PDF(X = 0.0) = 0.398942

Normal distribution CDF::

    rss-calc prob --type normal --x 1.96 --mean 0 --std 1 --cdf
    # Output: P(X ≤ 1.96) = 0.975002

Binomial probability::

    rss-calc prob --type binomial --n 10 --k 3 --p 0.5
    # Output: P(X = 3) = 0.117188

Bayes' theorem::

    rss-calc prob --type bayes --p_b_given_a 0.9 --p_a 0.01 --p_b 0.05
    # Output: P(A|B) = 0.180000

Hypothesis Testing Commands
--------------------------

Perform statistical hypothesis tests.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

    rss-calc hypothesis --test t-test --data "1,2,3,4,5" --mu 3.0

Options
~~~~~~

.. option:: --test TEST

   Type of hypothesis test:

   * ``t-test`` - One-sample t-test

.. option:: --data DATA

   Comma-separated sample data (required)

.. option:: --mu MU

   Null hypothesis mean (required for t-test)

.. option:: --alpha ALPHA

   Significance level (default: 0.05)

Examples
~~~~~~~

One-sample t-test::

    rss-calc hypothesis --test t-test --data "23,25,27,24,26" --mu 24.0 --alpha 0.05
    # Output:
    # One-sample t-test:
    # Sample data: [23.0, 25.0, 27.0, 24.0, 26.0]
    # Null hypothesis mean: 24.0
    # Significance level: α = 0.05

Glossary Commands
----------------

Look up definitions of statistical terms.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

    rss-calc glossary --term "standard deviation"

Options
~~~~~~

.. option:: --term TERM

   Statistical term to look up (required)

.. option:: --list

   List all available terms

Examples
~~~~~~~

Look up a term::

    rss-calc glossary --term "p-value"
    # Output: [Definition of p-value]

List all terms::

    rss-calc glossary --list
    # Output: [List of all available terms]

Advanced Usage
-------------

Piping and Redirection
~~~~~~~~~~~~~~~~~~~~~

Save results to file::

    rss-calc stats --data "1,2,3,4,5" --stat all > results.txt

Use with other commands::

    echo "10,20,30,40,50" | rss-calc stats --stat mean

Batch Processing
~~~~~~~~~~~~~~~

Process multiple datasets::

    #!/bin/bash
    datasets=("1,2,3,4,5" "10,20,30" "100,200,300,400")

    for data in "${datasets[@]}"; do
        echo "Dataset: $data"
        rss-calc stats --data "$data" --stat mean
        echo "---"
    done

Integration with Scripts
~~~~~~~~~~~~~~~~~~~~~~

Use in Python scripts::

    import subprocess

    result = subprocess.run([
        'rss-calc', 'stats',
        '--data', '1,2,3,4,5',
        '--stat', 'mean'
    ], capture_output=True, text=True)

    print(result.stdout)

Error Handling
-------------

Common Errors and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Command not found: rss-calc**
    * Ensure the package is installed: ``pip install real-simple-stats``
    * Check if it's in your PATH
    * Try: ``python -m real_simple_stats.cli --help``

**Invalid data format**
    * Use comma-separated values without spaces: ``"1,2,3,4,5"``
    * Ensure all values are numeric
    * Quote the data string to prevent shell interpretation

**Missing required arguments**
    * Check the help for required options: ``rss-calc <subcommand> --help``
    * The CLI will tell you exactly which arguments are missing

**Invalid argument values**
    * Binomial: ``--n`` must be non-negative, ``--k`` must be between 0 and n, ``--p`` must be between 0 and 1
    * Normal: ``--std`` must be positive
    * Bayes: All probabilities must be between 0 and 1, and ``--p_b`` cannot be zero
    * You'll get a specific error message explaining what's wrong

**Invalid statistic type**
    * Use ``--stat all`` to see available options
    * Check spelling of statistic names

Tips and Best Practices
-----------------------

1. **Quote your data**: Always quote comma-separated data to prevent shell issues
2. **Use meaningful filenames**: When redirecting output, use descriptive names
3. **Check help first**: Use ``--help`` with any command to see available options
4. **Validate your data**: Ensure your input data makes sense for the calculation
5. **Use appropriate precision**: Consider rounding results for readability

Output Formats
-------------

The CLI provides human-readable output by default. Future versions may include:

* JSON output for programmatic use
* CSV format for spreadsheet import
* Formatted tables for complex results

Getting More Help
----------------

* Use ``--help`` with any command for detailed usage
* Check the main documentation for Python API details
* Report CLI bugs on `GitHub Issues <https://github.com/kylejones200/real_simple_stats/issues>`_
* Request new CLI features through GitHub

The CLI is straightforward to use. Start with simple commands and explore more features as you need them.
