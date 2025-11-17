Descriptive Statistics
=====================

The ``descriptive_statistics`` module provides functions for calculating basic statistical measures that describe the central tendency, variability, and distribution of datasets.

.. automodule:: real_simple_stats.descriptive_statistics
   :members:
   :undoc-members:
   :show-inheritance:

Functions Overview
-----------------

Central Tendency
~~~~~~~~~~~~~~~

.. autofunction:: real_simple_stats.descriptive_statistics.mean
.. autofunction:: real_simple_stats.descriptive_statistics.median
.. autofunction:: real_simple_stats.descriptive_statistics.mode

Variability
~~~~~~~~~~

.. autofunction:: real_simple_stats.descriptive_statistics.variance
.. autofunction:: real_simple_stats.descriptive_statistics.standard_deviation
.. autofunction:: real_simple_stats.descriptive_statistics.sample_variance
.. autofunction:: real_simple_stats.descriptive_statistics.sample_standard_deviation
.. autofunction:: real_simple_stats.descriptive_statistics.coefficient_of_variation

Usage Examples
-------------

Basic Statistics
~~~~~~~~~~~~~~~

Calculate common descriptive statistics for a dataset:

.. code-block:: python

    from real_simple_stats import descriptive_statistics as desc

    # Sample dataset
    data = [12, 15, 18, 20, 22, 25, 28, 30, 32, 35]

    # Central tendency
    mean_val = desc.mean(data)
    median_val = desc.median(data)
    mode_val = desc.mode(data)

    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Mode: {mode_val}")

    # Variability
    variance_val = desc.variance(data)
    std_dev = desc.standard_deviation(data)
    cv = desc.coefficient_of_variation(data)

    print(f"Variance: {variance_val:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Coefficient of Variation: {cv:.2f}%")

Population vs Sample Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understanding the difference between population and sample statistics:

.. code-block:: python

    # Same dataset, different calculations
    sample_data = [85, 90, 78, 92, 88, 76, 95, 82, 89, 91]

    # Population statistics (when you have the entire population)
    pop_variance = desc.variance(sample_data)
    pop_std = desc.standard_deviation(sample_data)

    # Sample statistics (when you have a sample from a larger population)
    sample_variance = desc.sample_variance(sample_data)
    sample_std = desc.sample_standard_deviation(sample_data)

    print("Population Statistics:")
    print(f"  Variance: {pop_variance:.2f}")
    print(f"  Standard Deviation: {pop_std:.2f}")

    print("Sample Statistics:")
    print(f"  Variance: {sample_variance:.2f}")
    print(f"  Standard Deviation: {sample_std:.2f}")

Error Handling
~~~~~~~~~~~~~

The functions include comprehensive error handling:

.. code-block:: python

    import real_simple_stats.descriptive_statistics as desc

    # Empty dataset
    try:
        result = desc.mean([])
    except ValueError as e:
        print(f"Error: {e}")

    # Single value for sample statistics
    try:
        result = desc.sample_variance([42])
    except ValueError as e:
        print(f"Error: {e}")

    # Five-number summary works with small datasets too
    summary_single = desc.five_number_summary([5])
    # For a single value, all stats equal that value
    
    summary_two = desc.five_number_summary([1, 2])
    # With two values, Q1=min and Q3=max

    # Non-numeric data
    try:
        result = desc.mean([1, 2, "three", 4])
    except TypeError as e:
        print(f"Error: {e}")

Mathematical Background
----------------------

Mean (Arithmetic Average)
~~~~~~~~~~~~~~~~~~~~~~~~

The arithmetic mean is the sum of all values divided by the number of values:

.. math::

    \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i

Where:
- :math:`\bar{x}` is the sample mean
- :math:`n` is the number of observations
- :math:`x_i` is the i-th observation

Median
~~~~~

The median is the middle value when data is arranged in ascending order:

- For odd n: median = middle value
- For even n: median = average of two middle values

Variance
~~~~~~~

**Population Variance:**

.. math::

    \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2

**Sample Variance:**

.. math::

    s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2

Standard Deviation
~~~~~~~~~~~~~~~~~

The standard deviation is the square root of the variance:

- Population: :math:`\sigma = \sqrt{\sigma^2}`
- Sample: :math:`s = \sqrt{s^2}`

Coefficient of Variation
~~~~~~~~~~~~~~~~~~~~~~~

The coefficient of variation expresses the standard deviation as a percentage of the mean:

.. math::

    CV = \frac{\sigma}{|\mu|} \times 100\%

This allows comparison of variability between datasets with different units or scales.

See Also
--------

* :doc:`probability_utils` - For probability calculations
* :doc:`hypothesis_testing` - For statistical testing
* :doc:`../tutorials/basic_statistics` - Tutorial on descriptive statistics
