Installation Guide
==================

Requirements
-----------

Real Simple Stats requires Python 3.7 or later and the following dependencies:

* **numpy** >= 1.19.0 - For numerical computations
* **scipy** >= 1.5.0 - For statistical functions
* **matplotlib** >= 3.3.0 - For data visualization

Installation Methods
-------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install Real Simple Stats is using pip::

    pip install real-simple-stats

This will automatically install all required dependencies.

From Source
~~~~~~~~~~

To install the latest development version from GitHub::

    git clone https://github.com/kylejones200/real_simple_stats.git
    cd real_simple_stats
    pip install -e .

For Development
~~~~~~~~~~~~~~

If you want to contribute to the project, install with development dependencies::

    git clone https://github.com/kylejones200/real_simple_stats.git
    cd real_simple_stats
    pip install -e ".[dev]"

This includes testing, linting, and documentation tools.

Using Conda
~~~~~~~~~~

Real Simple Stats can also be installed using conda::

    conda install -c conda-forge real-simple-stats

Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's recommended to install Real Simple Stats in a virtual environment::

    # Create virtual environment
    python -m venv rss_env
    
    # Activate virtual environment
    # On Windows:
    rss_env\Scripts\activate
    # On macOS/Linux:
    source rss_env/bin/activate
    
    # Install Real Simple Stats
    pip install real-simple-stats

Verification
-----------

To verify your installation, run::

    python -c "import real_simple_stats; print(real_simple_stats.__version__)"

You should see the version number printed.

You can also test the command-line interface::

    rss-calc --help

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**ImportError: No module named 'real_simple_stats'**
    Make sure you've activated the correct virtual environment and installed the package.

**Permission denied errors**
    Try installing with the ``--user`` flag: ``pip install --user real-simple-stats``

**Dependency conflicts**
    Create a fresh virtual environment and install there.

**Command 'rss-calc' not found**
    The CLI might not be in your PATH. Try: ``python -m real_simple_stats.cli --help``

Getting Help
~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/kylejones200/real_simple_stats/issues>`_
2. Create a new issue with details about your environment and the error
3. Join our community discussions

Upgrading
--------

To upgrade to the latest version::

    pip install --upgrade real-simple-stats

To upgrade from source::

    cd real_simple_stats
    git pull origin main
    pip install -e .

Uninstallation
-------------

To uninstall Real Simple Stats::

    pip uninstall real-simple-stats

This will remove the package but keep any data files you've created.
