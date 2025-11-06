# Expose package version
try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except Exception:  # pragma: no cover - fallback if needed
    # Fallback for environments with importlib_metadata backport
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("real-simple-stats")
except PackageNotFoundError:  # When running from source without installed metadata
    __version__ = "0.0.0"

# Top-level exports for convenience
from .pre_statistics import *  # noqa: F403, F401
from .descriptive_statistics import *  # noqa: F403, F401
from .probability_utils import *  # noqa: F403, F401
from .binomial_distributions import *  # noqa: F403, F401
from .normal_distributions import *  # noqa: F403, F401
from .sampling_and_intervals import *  # noqa: F403, F401
from .hypothesis_testing import *  # noqa: F403, F401
from .linear_regression_utils import *  # noqa: F403, F401
from .chi_square_utils import *  # noqa: F403, F401
from .probability_distributions import *  # noqa: F403, F401
from .glossary import GLOSSARY, lookup

# Advanced statistical methods (new in v0.3.0)
from .time_series import *  # noqa: F403, F401
from .multivariate import *  # noqa: F403, F401
from .bayesian_stats import *  # noqa: F403, F401
from .resampling import *  # noqa: F403, F401
from .effect_sizes import *  # noqa: F403, F401
from .power_analysis import *  # noqa: F403, F401
from .monte_carlo import *  # noqa: F403, F401
