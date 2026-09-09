"""Real Simple Stats - an educational statistics library with a Rust backend.

Every numeric routine runs in the bundled native extension
(``real_simple_stats._rss``); the library has no runtime dependencies. Plotting
is optional and pulls in matplotlib via the ``plots`` extra.

Inputs may be lists, tuples, or anything exposing the buffer protocol. Passing
a contiguous float64 buffer (a NumPy array, ``array.array('d')``, a
``memoryview``) lets the backend read it without copying, which is where the
large speedups come from.
"""

from __future__ import annotations


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` lazily (PEP 562).

    Reading it eagerly costs ~13 ms, because importlib.metadata drags in
    pathlib and the email package. That was lost in the noise when SciPy
    dominated import time; now that the whole import is ~10 ms, it is most of
    it. Users still just write ``real_simple_stats.__version__``.
    """
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("real-simple-stats")
        except PackageNotFoundError:
            # Running from a source tree with no installed metadata.
            return "0.0.0"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# The native random generator (PCG64), used wherever the library samples.
from ._rss import Rng  # noqa: E402, F401

# Top-level exports for convenience
from .assumptions import *  # noqa: F403, F401, E402
from .bayesian_stats import *  # noqa: F403, F401
from .binomial_distributions import *  # noqa: F403, F401

# Methods from Python for Business Analytics
from .causal_inference import *  # noqa: F403, F401
from .chi_square_utils import *  # noqa: F403, F401
from .descriptive_statistics import *  # noqa: F403, F401
from .effect_sizes import *  # noqa: F403, F401
from .explain import *  # noqa: F403, F401
from .glossary import GLOSSARY, lookup  # noqa: F401
from .hypothesis_testing import *  # noqa: F403, F401
from .linear_regression_utils import *  # noqa: F403, F401
from .market_basket import *  # noqa: F403, F401
from .monte_carlo import *  # noqa: F403, F401
from .multivariate import *  # noqa: F403, F401
from .normal_distributions import *  # noqa: F403, F401
from .power_analysis import *  # noqa: F403, F401
from .pre_statistics import *  # noqa: F403, F401
from .probability_distributions import *  # noqa: F403, F401
from .probability_utils import *  # noqa: F403, F401
from .resampling import *  # noqa: F403, F401
from .sampling_and_intervals import *  # noqa: F403, F401
from .spatial_stats import *  # noqa: F403, F401
from .survival import *  # noqa: F403, F401

# Advanced statistical methods (new in v0.3.0)
from .time_series import *  # noqa: F403, F401
from .verbose_stats import *  # noqa: F403, F401
