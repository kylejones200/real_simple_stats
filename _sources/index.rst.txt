Real Simple Stats Documentation
==============================

**Real Simple Stats** is a comprehensive Python library for statistical analysis and education.
It provides easy-to-use functions for descriptive statistics, probability calculations,
hypothesis testing, and data visualization.

.. image:: https://img.shields.io/pypi/v/real-simple-stats.svg
   :target: https://pypi.org/project/real-simple-stats/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/real-simple-stats.svg
   :target: https://pypi.org/project/real-simple-stats/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/kylejones200/real_simple_stats.svg
   :target: https://github.com/kylejones200/real_simple_stats/blob/main/LICENSE
   :alt: License

Key Features
-----------

* **Descriptive Statistics**: Mean, median, mode, variance, standard deviation, and more
* **Probability Utilities**: Simple, joint, conditional probability calculations
* **Hypothesis Testing**: t-tests, F-tests, chi-square tests with p-values
* **Probability Distributions**: Normal, binomial, Poisson distributions
* **Linear Regression**: Simple and multiple regression analysis
* **Data Visualization**: Statistical plots and charts
* **Command Line Interface**: Easy-to-use CLI for quick calculations
* **Educational Focus**: Clear explanations and examples for learning

Quick Start
----------

Installation::

    pip install real-simple-stats

Basic usage::

    from real_simple_stats import descriptive_statistics as desc

    data = [1, 2, 3, 4, 5]
    mean = desc.mean(data)
    std_dev = desc.standard_deviation(data)

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")

Command line usage::

    rss-calc stats --data "1,2,3,4,5" --stat mean
    rss-calc probability --type binomial --n 10 --k 3 --p 0.5

Documentation Contents
=====================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials
   cli_reference

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/descriptive_statistics
   api/probability_utils
   api/hypothesis_testing
   api/distributions
   api/regression
   api/plotting
   modules

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   code_quality
   changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
