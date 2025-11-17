import os
import sys
from pathlib import Path

# Ensure the project’s bundled virtualenv (if present) takes precedence over any
# globally installed packages. This prevents crashes caused by incompatible
# system-wide NumPy/SciPy builds when running the test suite.
_VENV_SITE = Path(__file__).resolve().parents[1] / "venv" / "lib"
if _VENV_SITE.exists():
    for candidate in sorted(_VENV_SITE.glob("python*/site-packages")):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

# Disable autoloading of externally installed pytest plugins unless the user
# explicitly opts back in. Some environments have globally installed plugins
# that pull in native extensions (e.g. pyarrow) compiled against incompatible
# BLAS stacks, which previously triggered segmentation faults during import.
if "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in os.environ:
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

# Expose package version
try:
    from importlib.metadata import PackageNotFoundError, version  # Python 3.8+
except Exception:  # pragma: no cover - fallback if needed
    # Fallback for environments with importlib_metadata backport
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("real-simple-stats")
except PackageNotFoundError:  # When running from source without installed metadata
    __version__ = "0.0.0"

# Top-level exports for convenience
from .assumptions import *  # noqa: F403, F401, E402
from .bayesian_stats import *  # noqa: F403, F401
from .binomial_distributions import *  # noqa: F403, F401
from .chi_square_utils import *  # noqa: F403, F401
from .descriptive_statistics import *  # noqa: F403, F401
from .effect_sizes import *  # noqa: F403, F401
from .glossary import GLOSSARY, lookup  # noqa: F401
from .hypothesis_testing import *  # noqa: F403, F401
from .linear_regression_utils import *  # noqa: F403, F401
from .monte_carlo import *  # noqa: F403, F401
from .multivariate import *  # noqa: F403, F401
from .normal_distributions import *  # noqa: F403, F401
from .power_analysis import *  # noqa: F403, F401
from .pre_statistics import *  # noqa: F403, F401
from .probability_distributions import *  # noqa: F403, F401
from .probability_utils import *  # noqa: F403, F401
from .resampling import *  # noqa: F403, F401
from .sampling_and_intervals import *  # noqa: F403, F401

# Advanced statistical methods (new in v0.3.0)
from .time_series import *  # noqa: F403, F401
from .verbose_stats import *  # noqa: F403, F401
