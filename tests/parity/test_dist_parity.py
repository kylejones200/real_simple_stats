"""Parity gate for the Rust distribution layer, against SciPy.

SciPy is an adequate oracle here: these are evaluated in the bulk, where its
own error is ~1e-15. (The special-function layer underneath is gated against
mpmath instead -- see test_special_parity.py for why.)
"""

import numpy as np
import pytest

st = pytest.importorskip("scipy.stats")

from real_simple_stats import _rss  # noqa: E402


def worst_rel(ours, ref, points):
    worst, at = 0.0, None
    for pt in points:
        r = float(ref(*pt))
        if not np.isfinite(r) or r == 0.0:
            continue
        err = abs(float(ours(*pt)) - r) / abs(r)
        if err > worst:
            worst, at = err, pt
    return worst, at


DFS = [1, 2, 3, 5, 10, 30, 100, 1000]
XS = [-5, -2, -0.5, 0, 0.5, 1, 2, 3, 5]


@pytest.mark.parametrize(
    "name,ours,ref,points",
    [
        ("t.cdf", _rss.t_cdf, lambda x, d: st.t.cdf(x, d), [(x, d) for x in XS for d in DFS]),
        ("t.sf", _rss.t_sf, lambda x, d: st.t.sf(x, d), [(x, d) for x in XS for d in DFS]),
        ("t.pdf", _rss.t_pdf, lambda x, d: st.t.pdf(x, d), [(x, d) for x in XS for d in DFS]),
        ("t.ppf", _rss.t_ppf, lambda p, d: st.t.ppf(p, d),
         [(p, d) for p in (1e-6, .001, .01, .05, .25, .5, .75, .95, .99) for d in DFS]),
        ("chi2.cdf", _rss.chi2_cdf, lambda x, d: st.chi2.cdf(x, d),
         [(x, d) for x in (0.1, 1, 5, 20, 100) for d in DFS]),
        ("chi2.sf", _rss.chi2_sf, lambda x, d: st.chi2.sf(x, d),
         [(x, d) for x in (0.1, 1, 5, 20, 100) for d in DFS]),
        ("chi2.ppf", _rss.chi2_ppf, lambda p, d: st.chi2.ppf(p, d),
         [(p, d) for p in (.001, .05, .5, .95, .999) for d in DFS]),
        ("chi2.isf", _rss.chi2_isf, lambda q, d: st.chi2.isf(q, d),
         [(q, d) for q in (.001, .05, .5, .95, .999) for d in DFS]),
        ("f.cdf", _rss.f_cdf, lambda x, a, b: st.f.cdf(x, a, b),
         [(x, a, b) for x in (0.1, 0.5, 1, 2, 5, 20) for a in (1, 2, 5, 20) for b in (1, 2, 5, 20, 100)]),
        ("f.sf", _rss.f_sf, lambda x, a, b: st.f.sf(x, a, b),
         [(x, a, b) for x in (0.1, 0.5, 1, 2, 5, 20) for a in (1, 2, 5, 20) for b in (1, 2, 5, 20, 100)]),
        ("f.ppf", _rss.f_ppf, lambda p, a, b: st.f.ppf(p, a, b),
         [(p, a, b) for p in (.01, .05, .5, .95, .99) for a in (1, 2, 5, 20) for b in (1, 5, 20, 100)]),
        ("gamma.cdf", _rss.gamma_cdf, lambda x, a, s: st.gamma.cdf(x, a, scale=s),
         [(x, a, s) for x in (.1, 1, 5, 20) for a in (.5, 1, 3, 10) for s in (.5, 1, 3)]),
        ("beta.cdf", _rss.beta_cdf, lambda x, a, b: st.beta.cdf(x, a, b),
         [(x, a, b) for x in (.05, .25, .5, .75, .95) for a in (.5, 1, 3, 10) for b in (.5, 1, 3, 10)]),
        ("beta.sf", _rss.beta_sf, lambda x, a, b: st.beta.sf(x, a, b),
         [(x, a, b) for x in (.05, .25, .5, .75, .95) for a in (.5, 1, 3, 10) for b in (.5, 1, 3, 10)]),
        ("expon.cdf", _rss.expon_cdf, lambda x, s: st.expon.cdf(x, scale=s),
         [(x, s) for x in (.01, .5, 2, 10) for s in (.5, 1, 4)]),
        ("lognorm.cdf", _rss.lognorm_cdf, lambda x, s, sc: st.lognorm.cdf(x, s, scale=sc),
         [(x, s, sc) for x in (.1, 1, 5) for s in (.5, 1, 2) for sc in (1, 3)]),
        ("weibull.cdf", _rss.weibull_cdf, lambda x, c, sc: st.weibull_min.cdf(x, c, scale=sc),
         [(x, c, sc) for x in (.1, 1, 5) for c in (.5, 1, 2, 5) for sc in (1, 3)]),
        ("fisk.cdf", _rss.fisk_cdf, lambda x, c, sc: st.fisk.cdf(x, c, scale=sc),
         [(x, c, sc) for x in (.1, 1, 5) for c in (.5, 1, 2, 5) for sc in (1, 3)]),
        ("binom.cdf", _rss.binom_cdf, lambda k, n, p: st.binom.cdf(k, int(n), p),
         [(k, n, p) for n in (5, 20, 100) for k in (0, 1, 3, 10, 19) for p in (.1, .5, .9) if k <= n]),
        ("binom.pmf", _rss.binom_pmf, lambda k, n, p: st.binom.pmf(k, int(n), p),
         [(k, n, p) for n in (5, 20, 100) for k in (0, 1, 3, 10, 19) for p in (.1, .5, .9) if k <= n]),
        ("poisson.cdf", _rss.poisson_cdf, lambda k, m: st.poisson.cdf(k, m),
         [(k, m) for k in (0, 1, 3, 10, 30) for m in (.5, 2, 10, 50)]),
        ("geom.cdf", _rss.geom_cdf, lambda k, p: st.geom.cdf(k, p),
         [(k, p) for k in (1, 2, 5, 20) for p in (.05, .3, .8)]),
        ("nbinom.cdf", _rss.nbinom_cdf, lambda k, n, p: st.nbinom.cdf(k, n, p),
         [(k, n, p) for k in (0, 1, 5, 20) for n in (1, 3, 10) for p in (.2, .5, .8)]),
    ],
)
def test_distribution_parity(name, ours, ref, points):
    err, at = worst_rel(ours, ref, points)
    assert err < 1e-11, f"{name} rel err {err:.3e} at {at}"


def test_ncf_cdf():
    """Noncentral F -- one of the two functions that made dropping SciPy a risk."""
    pts = [
        (f, a, b, nc)
        for f in (0.5, 1, 2, 5, 10)
        for a in (1, 3, 10)
        for b in (5, 20, 100)
        for nc in (0.5, 2, 10, 40)
    ]
    err, at = worst_rel(_rss.ncf_cdf, lambda f, a, b, nc: st.ncf.cdf(f, a, b, nc), pts)
    assert err < 1e-11, f"ncf.cdf rel err {err:.3e} at {at}"


def test_nct_cdf():
    """Noncentral t (AS 243).

    Guarantee, stated as measured rather than as hoped for:

        |ours - scipy| <= 1e-14 + 1e-10 * |scipy|

    i.e. absolute error below 1e-14, degrading to a pure relative bound only
    once the probability is large enough for that to be the binding term
    (around 1e-4 and up).

    For t < 0 the series is evaluated by the reflection 1 - F(-t, -delta), and
    that subtraction is capped at the float64 cancellation floor (~2e-16
    absolute). Once the result itself is ~1e-12 that floor dominates the
    relative metric. It does not matter in practice: nct backs power analysis,
    and a power of 1e-12 is reported as "no power", not to 12 digits.
    """
    pts = [
        (t, df, d)
        for t in (-3, -1, 0, 1, 2, 4, 8)
        for df in (1, 3, 10, 30, 100)
        for d in (0, 0.5, 1, 2, 5)
    ]
    for pt in pts:
        r = float(st.nct.cdf(*pt))
        o = float(_rss.nct_cdf(*pt))
        assert abs(o - r) <= 1e-14 + 1e-10 * abs(r), (
            f"nct.cdf{pt}: ours={o:.6e} scipy={r:.6e} diff={abs(o - r):.3e}"
        )


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 11, 12, 20, 50, 200, 1000])
@pytest.mark.parametrize("kind", ["normal", "uniform", "expon", "t3"])
def test_shapiro_wilk(n, kind):
    rng = np.random.default_rng(abs(hash((n, kind))) % (2**32))
    d = {
        "normal": lambda: rng.normal(0, 1, n),
        "uniform": lambda: rng.uniform(0, 1, n),
        "expon": lambda: rng.exponential(1, n),
        "t3": lambda: rng.standard_t(3, n),
    }[kind]()
    w, p = _rss.shapiro_wilk(list(d))
    rw, rp = st.shapiro(d)
    assert abs(w - rw) / abs(rw) < 1e-12, f"W {w} vs {rw}"
    if rp > 1e-10:
        assert abs(p - rp) / abs(rp) < 1e-9, f"p {p} vs {rp}"
