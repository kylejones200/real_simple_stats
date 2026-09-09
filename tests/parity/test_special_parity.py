"""Ground-truth accuracy gate for the Rust special-function layer.

The oracle is mpmath at 60 decimal digits, NOT SciPy. This matters: SciPy's
own erfc/ndtr carry ~1e-13 relative error in the tails, so validating against
SciPy would cap our accuracy at SciPy's and would flag genuine improvements as
regressions. Measured against true values, this layer is ~50x more accurate
than SciPy for the normal CDF/SF.

Points where the true value underflows float64 are skipped: both any correct
implementation and SciPy return 0.0 there, so a relative metric is undefined.
"""

import numpy as np
import pytest

mpmath = pytest.importorskip("mpmath")
from mpmath import mp, mpf  # noqa: E402

from real_simple_stats import _rss  # noqa: E402

mp.dps = 60
UNDERFLOW = mpf("1e-300")


def max_rel_err(ours, truth, points, absolute=False):
    """Worst relative error of `ours` against `truth` over `points`.

    `absolute=True` floors the denominator at 1, which is the correct metric for
    log-scale outputs (ln_gamma has zeros at x=1 and x=2, and it is consumed via
    exp() where absolute log error becomes relative error).
    """
    worst, worst_at = 0.0, None
    for pt in points:
        args = pt if isinstance(pt, tuple) else (pt,)
        t = truth(*args)
        if t == 0 or not mpmath.isfinite(t) or abs(t) < UNDERFLOW:
            continue
        denom = max(abs(t), mpf(1)) if absolute else abs(t)
        err = float(abs(mpf(ours(*args)) - t) / denom)
        if err > worst:
            worst, worst_at = err, pt
    return worst, worst_at


def test_erf():
    pts = list(np.linspace(-6, 6, 121)) + [-30.0, -10.0, 10.0, 30.0]
    err, at = max_rel_err(_rss.erf, lambda x: mpmath.erf(mpf(x)), pts)
    assert err < 1e-14, f"erf rel err {err:.3e} at {at}"


def test_erfc():
    pts = list(np.linspace(-4, 28, 161))
    err, at = max_rel_err(_rss.erfc, lambda x: mpmath.erfc(mpf(x)), pts)
    assert err < 1e-14, f"erfc rel err {err:.3e} at {at}"


def test_norm_cdf_and_sf():
    err, at = max_rel_err(
        _rss.norm_cdf, lambda x: mpmath.ncdf(mpf(x)), list(np.linspace(-30, 8, 191))
    )
    assert err < 1e-14, f"norm_cdf rel err {err:.3e} at {at}"
    err, at = max_rel_err(
        _rss.norm_sf, lambda x: mpmath.ncdf(-mpf(x)), list(np.linspace(-8, 30, 191))
    )
    assert err < 1e-14, f"norm_sf rel err {err:.3e} at {at}"


def test_norm_ppf():
    pts = list(np.linspace(1e-10, 1 - 1e-10, 200)) + [1e-12, 1e-8, 0.5]
    err, at = max_rel_err(
        _rss.norm_ppf, lambda p: mpmath.sqrt(2) * mpmath.erfinv(2 * mpf(p) - 1), pts
    )
    assert err < 1e-14, f"norm_ppf rel err {err:.3e} at {at}"


def test_ln_gamma():
    pts = list(np.linspace(0.02, 200, 200))
    err, at = max_rel_err(
        _rss.ln_gamma, lambda x: mpmath.loggamma(mpf(x)), pts, absolute=True
    )
    assert err < 1e-14, f"ln_gamma err {err:.3e} at {at}"


def test_incomplete_gamma():
    pts = [(a, x) for a in (0.1, 0.5, 1, 2, 5, 20, 100) for x in (1e-6, 0.1, 1, 3, 10, 50, 200)]
    err, at = max_rel_err(
        _rss.gammainc_p,
        lambda a, x: mpmath.gammainc(mpf(a), 0, mpf(x), regularized=True),
        pts,
    )
    assert err < 1e-12, f"gammainc_p rel err {err:.3e} at {at}"
    err, at = max_rel_err(
        _rss.gammainc_q,
        lambda a, x: mpmath.gammainc(mpf(a), mpf(x), mpmath.inf, regularized=True),
        pts,
    )
    assert err < 1e-12, f"gammainc_q rel err {err:.3e} at {at}"


def test_incomplete_beta():
    pts = [
        (a, b, x)
        for a in (0.3, 1, 2, 5, 30)
        for b in (0.3, 1, 2, 5, 30)
        for x in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    ]
    err, at = max_rel_err(
        _rss.betainc,
        lambda a, b, x: mpmath.betainc(mpf(a), mpf(b), 0, mpf(x), regularized=True),
        pts,
    )
    assert err < 1e-12, f"betainc rel err {err:.3e} at {at}"


@pytest.mark.parametrize(
    "a,b,p", [(0.5, 0.5, 0.01), (1, 50, 1 - 1e-8), (2, 5, 1e-8), (5, 50, 0.99), (30, 2, 0.5)]
)
def test_betaincinv_roundtrip(a, b, p):
    """The inverse must invert: I_x(a,b) at the returned x recovers p."""
    x = _rss.betaincinv(a, b, p)
    back = float(mpmath.betainc(mpf(a), mpf(b), 0, mpf(x), regularized=True))
    assert abs(back - p) <= 1e-13 * max(p, 1e-8)


@pytest.mark.parametrize("a,p", [(0.5, 1 - 1e-8), (2, 1e-8), (5, 0.5), (500, 0.99)])
def test_gammaincinv_roundtrip(a, p):
    x = _rss.gammaincinv(a, p)
    back = float(mpmath.gammainc(mpf(a), 0, mpf(x), regularized=True))
    assert abs(back - p) <= 1e-13 * max(p, 1e-8)
