"""Parity gate for the Rust linear algebra, regression, and optimizer layers.

Reference implementations are NumPy/SciPy, which are reliable oracles for these
(well-conditioned dense problems, no extreme tails involved).
"""

import numpy as np
import pytest

sla = pytest.importorskip("scipy.linalg")
sopt = pytest.importorskip("scipy.optimize")
st = pytest.importorskip("scipy.stats")

from real_simple_stats import _rss  # noqa: E402


def rel(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    denom = np.abs(b).max() or 1.0
    return float(np.abs(a - b).max() / denom)


@pytest.fixture
def rng():
    return np.random.default_rng(5)


@pytest.fixture
def spd(rng):
    a = rng.normal(size=(6, 6))
    return a @ a.T + 6 * np.eye(6)


def test_inv(spd):
    got = np.array(_rss.mat_inv(6, 6, spd.ravel().tolist())).reshape(6, 6)
    assert rel(got, np.linalg.inv(spd)) < 1e-12


def test_pinv_and_lstsq(rng):
    b = rng.normal(size=(8, 3))
    y = rng.normal(size=8)
    r, c, d = _rss.mat_pinv(8, 3, b.ravel().tolist())
    assert rel(np.array(d).reshape(r, c), np.linalg.pinv(b)) < 1e-12
    got = _rss.mat_lstsq(8, 3, b.ravel().tolist(), y.tolist())
    assert rel(got, np.linalg.lstsq(b, y, rcond=None)[0]) < 1e-12


def test_eigh(spd):
    w, v = _rss.mat_eigh(6, 6, spd.ravel().tolist())
    nw, nv = np.linalg.eigh(spd)
    assert rel(w, nw) < 1e-12
    # Eigenvectors are only defined up to sign.
    assert rel(np.abs(np.array(v).reshape(6, 6)), np.abs(nv)) < 1e-11


def test_svd_values(rng):
    b = rng.normal(size=(8, 3))
    _, s, _, _, _, _, _ = _rss.mat_svd(8, 3, b.ravel().tolist())
    assert rel(s[:3], np.linalg.svd(b, compute_uv=False)) < 1e-12


def test_sqrtm_spd(spd):
    got = np.array(_rss.mat_sqrtm_spd(6, 6, spd.ravel().tolist())).reshape(6, 6)
    assert rel(got, np.real(sla.sqrtm(spd))) < 1e-11


def test_cov_and_corr(rng):
    x = rng.normal(size=(30, 4))
    got = np.array(_rss.mat_cov(30, 4, x.ravel().tolist(), 1)).reshape(4, 4)
    assert rel(got, np.cov(x, rowvar=False, ddof=1)) < 1e-13
    got = np.array(_rss.mat_corr(30, 4, x.ravel().tolist())).reshape(4, 4)
    assert rel(got, np.corrcoef(x, rowvar=False)) < 1e-13


def test_linregress(rng):
    x = np.sort(rng.normal(size=50))
    y = 2.5 * x - 1.2 + rng.normal(scale=0.4, size=50)
    slope, icept, r, p, se, _ = _rss.linregress(x.tolist(), y.tolist())
    ref = st.linregress(x, y)
    assert abs(slope - ref.slope) < 1e-12
    assert abs(icept - ref.intercept) < 1e-12
    assert abs(r - ref.rvalue) < 1e-12
    assert abs(p - ref.pvalue) < 1e-12
    assert abs(se - ref.stderr) < 1e-12


def test_ols(rng):
    x = np.sort(rng.normal(size=50))
    y = 2.5 * x - 1.2 + rng.normal(scale=0.4, size=50)
    design = np.column_stack([np.ones(50), x, x**2])
    coef = _rss.ols(50, 3, design.ravel().tolist(), y.tolist())[0]
    assert rel(coef, np.linalg.lstsq(design, y, rcond=None)[0]) < 1e-12


def test_t_tests(rng):
    a = rng.normal(0, 1, 40)
    b = rng.normal(0.5, 1.3, 45)
    assert rel(_rss.ttest_1samp(a.tolist(), 0.0), st.ttest_1samp(a, 0.0)) < 1e-12
    assert rel(_rss.ttest_ind(a.tolist(), b.tolist(), True), st.ttest_ind(a, b)) < 1e-12
    assert (
        rel(_rss.ttest_ind(a.tolist(), b.tolist(), False), st.ttest_ind(a, b, equal_var=False))
        < 1e-12
    )
    p1 = rng.normal(0, 1, 30)
    p2 = p1 + rng.normal(0.2, 0.5, 30)
    assert rel(_rss.ttest_rel(p1.tolist(), p2.tolist()), st.ttest_rel(p1, p2)) < 1e-12


def test_f_oneway(rng):
    groups = [rng.normal(i * 0.3, 1, 25).tolist() for i in range(4)]
    ref = st.f_oneway(*[np.array(g) for g in groups])
    assert rel(_rss.f_oneway(groups), ref) < 1e-12


@pytest.mark.parametrize(
    "table",
    [
        [[34.0, 52.0], [41.0, 29.0]],  # 2x2 -> Yates correction applies
        [[10.0, 20.0, 30.0], [15.0, 25.0, 20.0], [12.0, 18.0, 22.0]],
    ],
)
def test_chi2_contingency(table):
    t = np.array(table)
    r, c = t.shape
    chi2, p, dof, expected = _rss.chi2_contingency(r, c, t.ravel().tolist(), True)
    rchi2, rp, rdof, rexpected = st.chi2_contingency(t)
    assert abs(chi2 - rchi2) < 1e-12
    assert abs(p - rp) < 1e-12
    assert dof == rdof
    assert rel(np.array(expected).reshape(r, c), rexpected) < 1e-13


def test_brentq():
    f = lambda v: v**3 - 2 * v - 5  # noqa: E731
    assert abs(_rss.brentq(f, 1.0, 3.0) - sopt.brentq(f, 1, 3)) < 1e-12


def test_brentq_rejects_unbracketed():
    with pytest.raises(ValueError):
        _rss.brentq(lambda v: v * v + 1.0, -1.0, 1.0)


def test_curve_fit_bounded(rng):
    """Bounded Levenberg-Marquardt on a spherical-variogram-shaped model."""
    h = np.linspace(0.1, 10, 40)

    def sph(h, n, s, r):
        return np.where(h < r, n + (s - n) * (1.5 * h / r - 0.5 * (h / r) ** 3), s)

    obs = sph(h, 0.5, 3.0, 2.5) + rng.normal(scale=0.02, size=40)
    got = _rss.curve_fit_lm(
        lambda p: (sph(h, *p) - obs).tolist(), [0.0, 3.0, 3.0], [0.0, 0.0, 1e-6], [3.0, 6.0, 20.0], 40
    )
    ref, _ = sopt.curve_fit(
        sph, h, obs, p0=[0.0, 3.0, 3.0], bounds=([0, 0, 1e-6], [3, 6, 20]), maxfev=5000
    )
    assert rel(got, ref) < 1e-4


def test_simplex_least_squares(rng):
    """Synthetic-control weights: on the simplex, and as good as SLSQP."""
    y_ctrl = rng.normal(size=(40, 5))
    target = y_ctrl @ np.array([0.4, 0.3, 0.2, 0.1, 0.0]) + rng.normal(scale=0.01, size=40)
    w = np.array(
        _rss.simplex_least_squares(target.tolist(), [y_ctrl[:, j].tolist() for j in range(5)])
    )

    # Feasibility is structural, not approximate.
    assert abs(w.sum() - 1.0) < 1e-12
    assert (w >= 0.0).all()

    ref = sopt.minimize(
        lambda v: ((target - y_ctrl @ v) ** 2).sum(),
        np.full(5, 0.2),
        method="SLSQP",
        bounds=[(0, 1)] * 5,
        constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    ours_cost = ((target - y_ctrl @ w) ** 2).sum()
    ref_cost = ((target - y_ctrl @ ref.x) ** 2).sum()
    # Judge by objective value: the argmin can be non-unique, the minimum is not.
    assert ours_cost <= ref_cost * (1 + 1e-6)


def test_project_to_simplex_is_exact():
    """Projection must land exactly on the simplex even for adversarial input."""
    w = _rss.simplex_least_squares([1.0, 2.0, 3.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert abs(sum(w) - 1.0) < 1e-12
    assert all(v >= 0.0 for v in w)
