Quick Start Guide
=================

This guide will get you up and running with Real Simple Stats in just a few minutes.

Basic Usage
----------

Import and Calculate Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from real_simple_stats import descriptive_statistics as desc
    
    # Sample data
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Calculate basic statistics
    mean_val = desc.mean(data)
    median_val = desc.median(data)
    std_dev = desc.standard_deviation(data)
    variance = desc.variance(data)
    
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Variance: {variance:.2f}")

Output::

    Mean: 5.5
    Median: 5.5
    Standard Deviation: 3.03
    Variance: 9.17

Probability Calculations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from real_simple_stats import probability_utils as prob
    
    # Simple probability
    favorable = 3
    total = 10
    simple_prob = prob.simple_probability(favorable, total)
    print(f"Simple probability: {simple_prob}")
    
    # Combinations and permutations
    n, k = 10, 3
    combinations = prob.combinations(n, k)
    permutations = prob.permutations(n, k)
    
    print(f"Combinations C({n},{k}): {combinations}")
    print(f"Permutations P({n},{k}): {permutations}")

Output::

    Simple probability: 0.3
    Combinations C(10,3): 120
    Permutations P(10,3): 720

Hypothesis Testing
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from real_simple_stats import hypothesis_testing as ht
    
    # Sample data for t-test
    sample_data = [23, 25, 27, 24, 26, 28, 22, 29, 25, 27]
    
    # Calculate t-score
    sample_mean = 25.6
    population_mean = 24.0
    sample_std = 2.1
    n = len(sample_data)
    
    t_score = ht.t_score(sample_mean, population_mean, sample_std, n)
    print(f"T-score: {t_score:.3f}")
    
    # Get critical value
    alpha = 0.05
    df = n - 1
    critical_val = ht.critical_value_t(alpha, df)
    print(f"Critical value (α=0.05, df={df}): {critical_val:.3f}")

Working with Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from real_simple_stats import binomial_distributions as binom
    
    # Binomial probability
    n_trials = 10
    k_successes = 3
    p_success = 0.5
    
    prob_exact = binom.binomial_probability(n_trials, k_successes, p_success)
    print(f"P(X = {k_successes}): {prob_exact:.4f}")
    
    # Expected value and variance
    expected = binom.binomial_expected_value(n_trials, p_success)
    variance = binom.binomial_variance(n_trials, p_success)
    
    print(f"Expected value: {expected}")
    print(f"Variance: {variance}")

Command Line Interface
---------------------

Real Simple Stats includes a powerful CLI for quick calculations:

Basic Statistics
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Calculate mean
    rss-calc stats --data "1,2,3,4,5" --stat mean
    
    # Calculate multiple statistics
    rss-calc stats --data "10,20,30,40,50" --stat all

Probability Calculations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Binomial probability
    rss-calc probability --type binomial --n 10 --k 3 --p 0.5
    
    # Combinations
    rss-calc probability --type combination --n 10 --k 3

Glossary Lookup
~~~~~~~~~~~~~~

.. code-block:: bash

    # Look up statistical terms
    rss-calc glossary --term "standard deviation"
    rss-calc glossary --term "p-value"

Common Workflows
---------------

Analyzing a Dataset
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from real_simple_stats import descriptive_statistics as desc
    import real_simple_stats.plots as plots
    
    # Your dataset
    scores = [78, 85, 92, 88, 76, 89, 94, 82, 87, 91, 79, 86]
    
    # Comprehensive analysis
    print("Dataset Analysis")
    print("=" * 20)
    print(f"Sample size: {len(scores)}")
    print(f"Mean: {desc.mean(scores):.2f}")
    print(f"Median: {desc.median(scores):.2f}")
    print(f"Mode: {desc.mode(scores)}")
    print(f"Range: {max(scores) - min(scores)}")
    print(f"Standard deviation: {desc.standard_deviation(scores):.2f}")
    print(f"Coefficient of variation: {desc.coefficient_of_variation(scores):.2f}%")

Comparing Two Groups
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from real_simple_stats import descriptive_statistics as desc
    from real_simple_stats import hypothesis_testing as ht
    
    # Two groups of data
    group_a = [23, 25, 27, 24, 26, 28, 22, 29]
    group_b = [30, 32, 28, 31, 33, 29, 35, 30]
    
    # Compare means
    mean_a = desc.mean(group_a)
    mean_b = desc.mean(group_b)
    
    print(f"Group A mean: {mean_a:.2f}")
    print(f"Group B mean: {mean_b:.2f}")
    print(f"Difference: {mean_b - mean_a:.2f}")
    
    # Calculate effect size (if available)
    std_a = desc.standard_deviation(group_a)
    std_b = desc.standard_deviation(group_b)
    
    print(f"Group A std: {std_a:.2f}")
    print(f"Group B std: {std_b:.2f}")

Next Steps
---------

Now that you've learned the basics:

1. **Explore the API Reference** - Learn about all available functions
2. **Check out Tutorials** - Work through detailed examples
3. **Try the CLI** - Use the command-line interface for quick calculations
4. **Read the Examples** - See real-world applications

Common Patterns
--------------

Error Handling
~~~~~~~~~~~~~

.. code-block:: python

    from real_simple_stats import descriptive_statistics as desc
    
    try:
        result = desc.mean([])  # Empty list
    except ValueError as e:
        print(f"Error: {e}")
        
    try:
        result = desc.standard_deviation([5])  # Single value
    except ValueError as e:
        print(f"Error: {e}")

Working with Different Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Real Simple Stats works with various numeric types
    integers = [1, 2, 3, 4, 5]
    floats = [1.5, 2.7, 3.2, 4.8, 5.1]
    mixed = [1, 2.5, 3, 4.7, 5]
    
    # All work the same way
    print(f"Integer mean: {desc.mean(integers)}")
    print(f"Float mean: {desc.mean(floats):.2f}")
    print(f"Mixed mean: {desc.mean(mixed):.2f}")

Getting Help
-----------

* **Documentation**: You're reading it! Check the API reference for detailed function documentation.
* **Examples**: See the ``examples/`` directory in the repository.
* **Issues**: Report bugs or request features on `GitHub <https://github.com/kylejones200/real_simple_stats/issues>`_.
* **Community**: Join discussions and ask questions.

Ready to dive deeper? Check out the :doc:`tutorials` section for more comprehensive examples!
