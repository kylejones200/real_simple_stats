Installation Guide
==================

Requirements
------------

Real Simple Stats requires **Python 3.12 or later**. That is the whole list.

Since 0.5.0 every numeric routine runs in a compiled Rust extension shipped
inside the wheel, so the library has no runtime dependencies -- no NumPy, no
SciPy -- and nothing to conflict with anything else in your environment.

Two optional extras are available:

* ``plots`` - matplotlib (and NumPy, which it requires), for the ``.plot()``
  methods on self-explaining results
* ``pandas`` - the pandas interoperability helpers

Wheels are published for Linux (x86_64 and aarch64), macOS (Apple silicon and
Intel), and Windows (x64). They are built against the CPython limited API, so
one wheel per platform covers Python 3.12, 3.13, 3.14 and later.

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install Real Simple Stats is using pip::

    pip install real-simple-stats

There are no dependencies to resolve, so this is a single wheel download.

To include the optional extras::

    pip install "real-simple-stats[plots]"
    pip install "real-simple-stats[plots,pandas]"

From Source
~~~~~~~~~~~

Building from source compiles the Rust extension, so it needs a Rust
toolchain. Install one from `rustup.rs <https://rustup.rs>`_ first::

    git clone https://github.com/kylejones200/real_simple_stats.git
    cd real_simple_stats
    pip install -e .

The first build takes a couple of minutes. Installing the published wheel
instead requires no toolchain.

For Development
~~~~~~~~~~~~~~~

If you want to contribute to the project, install with development dependencies::

    git clone https://github.com/kylejones200/real_simple_stats.git
    cd real_simple_stats
    pip install -e ".[dev]"

This includes testing, linting, and documentation tools.

Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
------------

To verify your installation, run::

    python -c "import real_simple_stats; print(real_simple_stats.__version__)"

You should see the version number printed.

You can also test the command-line interface::

    rss-calc --help

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'real_simple_stats'**
    Make sure you've activated the correct virtual environment and installed the package.

**Permission denied errors**
    Try installing with the ``--user`` flag: ``pip install --user real-simple-stats``

**Dependency conflicts**
    Real Simple Stats declares no runtime dependencies, so it cannot itself
    conflict with anything. If pip reports a conflict, it comes from another
    package in the environment.

**ImportError mentioning matplotlib when calling .plot()**
    Plotting is optional. Install it with ``pip install
    "real-simple-stats[plots]"``.

**"cargo: command not found" when installing from source**
    Building from source needs a Rust toolchain; see `rustup.rs
    <https://rustup.rs>`_. Installing the published wheel avoids this.

**Command 'rss-calc' not found**
    The CLI might not be in your PATH. Try: ``python -m real_simple_stats.cli --help``

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/kylejones200/real_simple_stats/issues>`_
2. Create a new issue with details about your environment and the error
3. Join our community discussions

Upgrading
---------

To upgrade to the latest version::

    pip install --upgrade real-simple-stats

To upgrade from source (needs the Rust toolchain)::

    cd real_simple_stats
    git pull origin main
    pip install -e .

Uninstallation
--------------

To uninstall Real Simple Stats::

    pip uninstall real-simple-stats

This will remove the package but keep any data files you've created.
