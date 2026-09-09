mod arr;

use arr::{with_slice, with_two_slices, Arr};
use pyo3::prelude::*;
use rss_core::descriptive as ds;
use rss_core::linalg as la;
use rss_core::rng as rrng;
use rss_core::simulate as sim;
use rss_core::optimize as opt;
use rss_core::regression as reg;
use rss_core::resample as rsmp;
use rss_core::dist as d;
use rss_core::noncentral as nc;
use rss_core::special as sp;

macro_rules! wrap1 {
    ($name:ident, $path:path) => {
        #[pyfunction]
        fn $name(x: f64) -> f64 { $path(x) }
    };
}
macro_rules! wrap2 {
    ($name:ident, $path:path) => {
        #[pyfunction]
        fn $name(a: f64, b: f64) -> f64 { $path(a, b) }
    };
}
macro_rules! wrap3 {
    ($name:ident, $path:path) => {
        #[pyfunction]
        fn $name(a: f64, b: f64, c: f64) -> f64 { $path(a, b, c) }
    };
}

wrap1!(ln_gamma, sp::ln_gamma);
wrap1!(gamma_fn, sp::gamma_fn);
wrap1!(erf, sp::erf);
wrap1!(erfc, sp::erfc);
wrap1!(norm_cdf, sp::norm_cdf);
wrap1!(norm_sf, sp::norm_sf);
wrap1!(norm_pdf, sp::norm_pdf);
wrap1!(norm_ppf, sp::norm_ppf);
wrap2!(ln_beta, sp::ln_beta);
wrap2!(gammainc_p, sp::gammainc_p);
wrap2!(gammainc_q, sp::gammainc_q);
wrap2!(gammaincinv, sp::gammaincinv);
wrap3!(betainc, sp::betainc);
wrap3!(betaincinv, sp::betaincinv);

#[pyfunction]
fn normal_pdf(a0: f64, a1: f64, a2: f64) -> f64 { d::normal_pdf(a0, a1, a2) }
#[pyfunction]
fn normal_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::normal_cdf(a0, a1, a2) }
#[pyfunction]
fn normal_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::normal_sf(a0, a1, a2) }
#[pyfunction]
fn normal_ppf(a0: f64, a1: f64, a2: f64) -> f64 { d::normal_ppf(a0, a1, a2) }
#[pyfunction]
fn t_pdf(a0: f64, a1: f64) -> f64 { d::t_pdf(a0, a1) }
#[pyfunction]
fn t_cdf(a0: f64, a1: f64) -> f64 { d::t_cdf(a0, a1) }
#[pyfunction]
fn t_sf(a0: f64, a1: f64) -> f64 { d::t_sf(a0, a1) }
#[pyfunction]
fn t_ppf(a0: f64, a1: f64) -> f64 { d::t_ppf(a0, a1) }
#[pyfunction]
fn chi2_pdf(a0: f64, a1: f64) -> f64 { d::chi2_pdf(a0, a1) }
#[pyfunction]
fn chi2_cdf(a0: f64, a1: f64) -> f64 { d::chi2_cdf(a0, a1) }
#[pyfunction]
fn chi2_sf(a0: f64, a1: f64) -> f64 { d::chi2_sf(a0, a1) }
#[pyfunction]
fn chi2_ppf(a0: f64, a1: f64) -> f64 { d::chi2_ppf(a0, a1) }
#[pyfunction]
fn chi2_isf(a0: f64, a1: f64) -> f64 { d::chi2_isf(a0, a1) }
#[pyfunction]
fn f_pdf(a0: f64, a1: f64, a2: f64) -> f64 { d::f_pdf(a0, a1, a2) }
#[pyfunction]
fn f_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::f_cdf(a0, a1, a2) }
#[pyfunction]
fn f_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::f_sf(a0, a1, a2) }
#[pyfunction]
fn f_ppf(a0: f64, a1: f64, a2: f64) -> f64 { d::f_ppf(a0, a1, a2) }
#[pyfunction]
fn gamma_pdf(a0: f64, a1: f64, a2: f64) -> f64 { d::gamma_pdf(a0, a1, a2) }
#[pyfunction]
fn gamma_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::gamma_cdf(a0, a1, a2) }
#[pyfunction]
fn gamma_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::gamma_sf(a0, a1, a2) }
#[pyfunction]
fn gamma_ppf(a0: f64, a1: f64, a2: f64) -> f64 { d::gamma_ppf(a0, a1, a2) }
#[pyfunction]
fn beta_pdf(a0: f64, a1: f64, a2: f64) -> f64 { d::beta_pdf(a0, a1, a2) }
#[pyfunction]
fn beta_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::beta_cdf(a0, a1, a2) }
#[pyfunction]
fn beta_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::beta_sf(a0, a1, a2) }
#[pyfunction]
fn beta_ppf(a0: f64, a1: f64, a2: f64) -> f64 { d::beta_ppf(a0, a1, a2) }
#[pyfunction]
fn expon_pdf(a0: f64, a1: f64) -> f64 { d::expon_pdf(a0, a1) }
#[pyfunction]
fn expon_cdf(a0: f64, a1: f64) -> f64 { d::expon_cdf(a0, a1) }
#[pyfunction]
fn expon_sf(a0: f64, a1: f64) -> f64 { d::expon_sf(a0, a1) }
#[pyfunction]
fn expon_ppf(a0: f64, a1: f64) -> f64 { d::expon_ppf(a0, a1) }
#[pyfunction]
fn lognorm_pdf(a0: f64, a1: f64, a2: f64) -> f64 { d::lognorm_pdf(a0, a1, a2) }
#[pyfunction]
fn lognorm_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::lognorm_cdf(a0, a1, a2) }
#[pyfunction]
fn lognorm_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::lognorm_sf(a0, a1, a2) }
#[pyfunction]
fn lognorm_ppf(a0: f64, a1: f64, a2: f64) -> f64 { d::lognorm_ppf(a0, a1, a2) }
#[pyfunction]
fn weibull_pdf(a0: f64, a1: f64, a2: f64) -> f64 { d::weibull_pdf(a0, a1, a2) }
#[pyfunction]
fn weibull_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::weibull_cdf(a0, a1, a2) }
#[pyfunction]
fn weibull_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::weibull_sf(a0, a1, a2) }
#[pyfunction]
fn weibull_ppf(a0: f64, a1: f64, a2: f64) -> f64 { d::weibull_ppf(a0, a1, a2) }
#[pyfunction]
fn fisk_pdf(a0: f64, a1: f64, a2: f64) -> f64 { d::fisk_pdf(a0, a1, a2) }
#[pyfunction]
fn fisk_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::fisk_cdf(a0, a1, a2) }
#[pyfunction]
fn fisk_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::fisk_sf(a0, a1, a2) }
#[pyfunction]
fn fisk_ppf(a0: f64, a1: f64, a2: f64) -> f64 { d::fisk_ppf(a0, a1, a2) }
#[pyfunction]
fn binom_pmf(a0: f64, a1: f64, a2: f64) -> f64 { d::binom_pmf(a0, a1, a2) }
#[pyfunction]
fn binom_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::binom_cdf(a0, a1, a2) }
#[pyfunction]
fn binom_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::binom_sf(a0, a1, a2) }
#[pyfunction]
fn poisson_pmf(a0: f64, a1: f64) -> f64 { d::poisson_pmf(a0, a1) }
#[pyfunction]
fn poisson_cdf(a0: f64, a1: f64) -> f64 { d::poisson_cdf(a0, a1) }
#[pyfunction]
fn poisson_sf(a0: f64, a1: f64) -> f64 { d::poisson_sf(a0, a1) }
#[pyfunction]
fn geom_pmf(a0: f64, a1: f64) -> f64 { d::geom_pmf(a0, a1) }
#[pyfunction]
fn geom_cdf(a0: f64, a1: f64) -> f64 { d::geom_cdf(a0, a1) }
#[pyfunction]
fn geom_sf(a0: f64, a1: f64) -> f64 { d::geom_sf(a0, a1) }
#[pyfunction]
fn nbinom_pmf(a0: f64, a1: f64, a2: f64) -> f64 { d::nbinom_pmf(a0, a1, a2) }
#[pyfunction]
fn nbinom_cdf(a0: f64, a1: f64, a2: f64) -> f64 { d::nbinom_cdf(a0, a1, a2) }
#[pyfunction]
fn nbinom_sf(a0: f64, a1: f64, a2: f64) -> f64 { d::nbinom_sf(a0, a1, a2) }
#[pyfunction]
fn nct_cdf(a0: f64, a1: f64, a2: f64) -> f64 { nc::nct_cdf(a0, a1, a2) }
#[pyfunction]
fn nct_sf(a0: f64, a1: f64, a2: f64) -> f64 { nc::nct_sf(a0, a1, a2) }
#[pyfunction]
fn ncf_cdf(a0: f64, a1: f64, a2: f64, a3: f64) -> f64 { nc::ncf_cdf(a0, a1, a2, a3) }
#[pyfunction]
fn ncf_sf(a0: f64, a1: f64, a2: f64, a3: f64) -> f64 { nc::ncf_sf(a0, a1, a2, a3) }

#[pyfunction]
fn shapiro_wilk(data: Vec<f64>) -> (f64, f64) { nc::shapiro_wilk(&data) }


// ------------------------------------------------------- descriptive stats --

#[pyfunction]
fn sum_(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(py, x, ds::sum_par)
}
#[pyfunction]
fn mean(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(py, x, ds::mean)
}
#[pyfunction]
#[pyo3(signature = (x, ddof=1))]
fn variance(py: Python<'_>, x: &Bound<'_, PyAny>, ddof: usize) -> PyResult<f64> {
    with_slice(py, x, |s| ds::variance(s, ddof))
}
#[pyfunction]
#[pyo3(signature = (x, ddof=1))]
fn std_dev(py: Python<'_>, x: &Bound<'_, PyAny>, ddof: usize) -> PyResult<f64> {
    with_slice(py, x, |s| ds::std_dev(s, ddof))
}
#[pyfunction]
fn min_(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(py, x, ds::min)
}
#[pyfunction]
fn max_(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(py, x, ds::max)
}
#[pyfunction]
fn median(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(py, x, ds::median)
}
#[pyfunction]
fn quantile(py: Python<'_>, x: &Bound<'_, PyAny>, q: f64) -> PyResult<f64> {
    with_slice(py, x, |s| ds::quantile_linear(s, q))
}
#[pyfunction]
fn quantiles(py: Python<'_>, x: &Bound<'_, PyAny>, qs: Vec<f64>) -> PyResult<Vec<f64>> {
    with_slice(py, x, |s| {
        let sorted = ds::sorted_copy(s);
        qs.iter().map(|&q| ds::quantile_linear_sorted(&sorted, q)).collect()
    })
}
#[pyfunction]
fn five_number_summary(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Option<(f64, f64, f64, f64, f64)>> {
    with_slice(py, x, |s| ds::five_number_summary(s).map(|f| (f.min, f.q1, f.median, f.q3, f.max)))
}
#[pyfunction]
fn skewness(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(py, x, ds::skewness)
}
#[pyfunction]
fn kurtosis(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_slice(py, x, ds::kurtosis)
}
#[pyfunction]
fn modes(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    with_slice(py, x, ds::modes)
}
#[pyfunction]
fn sorted_copy(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    with_slice(py, x, ds::sorted_copy)
}
#[pyfunction]
#[pyo3(signature = (x, y, ddof=1))]
fn covariance(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, ddof: usize) -> PyResult<f64> {
    with_two_slices(py, x, y, |a, b| ds::covariance(a, b, ddof))
}
#[pyfunction]
fn pearson_r(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<f64> {
    with_two_slices(py, x, y, ds::pearson_r)
}

/// Diagnostic: did this input take the zero-copy path?
#[pyfunction]
fn is_zero_copy(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(Arr::from_py(x)?.is_zero_copy())
}


// ------------------------------------------------------------- resampling --

fn parse_stat(name: &str) -> PyResult<rsmp::Stat> {
    rsmp::Stat::from_name(name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported statistic {name:?}; expected one of \
             mean, median, std, var, min, max, sum"
        ))
    })
}

#[pyfunction]
fn bootstrap_dist(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    stat: &str,
    n_iterations: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    let st = parse_stat(stat)?;
    let a = Arr::from_py(data)?;
    let s = a.as_slice(py);
    // Release the GIL: the kernel touches no Python objects, so other threads
    // (and rayon's own pool) can run freely.
    let owned = s.to_vec();
    Ok(py.allow_threads(|| rsmp::bootstrap(&owned, st, n_iterations, seed)))
}

#[pyfunction]
fn jackknife_values(py: Python<'_>, data: &Bound<'_, PyAny>, stat: &str) -> PyResult<Vec<f64>> {
    let st = parse_stat(stat)?;
    let a = Arr::from_py(data)?;
    let owned = a.as_slice(py).to_vec();
    Ok(py.allow_threads(|| rsmp::jackknife(&owned, st)))
}

#[pyfunction]
fn permutation_dist(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    stat: &str,
    n_permutations: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    let st = parse_stat(stat)?;
    let xa = Arr::from_py(a)?;
    let xb = Arr::from_py(b)?;
    let va = xa.as_slice(py).to_vec();
    let vb = xb.as_slice(py).to_vec();
    Ok(py.allow_threads(|| rsmp::permutation_diff(&va, &vb, st, n_permutations, seed)))
}

#[pyfunction]
fn percentile_ci(py: Python<'_>, dist: &Bound<'_, PyAny>, confidence: f64) -> PyResult<(f64, f64)> {
    with_slice(py, dist, |s| rsmp::percentile_ci(s, confidence))
}

#[pyfunction]
fn permutation_pvalue(
    py: Python<'_>,
    dist: &Bound<'_, PyAny>,
    observed: f64,
    alternative: &str,
) -> PyResult<f64> {
    with_slice(py, dist, |s| rsmp::permutation_pvalue(s, observed, alternative))
}


// -------------------------------------------------- linalg / regression ----

fn mat(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<la::Matrix> {
    la::Matrix::new(rows, cols, data).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "data length does not match shape {rows}x{cols}"
        ))
    })
}
fn singular() -> PyErr {
    pyo3::exceptions::PyValueError::new_err("matrix is singular")
}

#[pyfunction]
fn mat_inv(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<Vec<f64>> {
    la::inv(&mat(rows, cols, data)?).map(|m| m.data).ok_or_else(singular)
}
#[pyfunction]
#[pyo3(signature = (rows, cols, data, rcond=1e-15))]
fn mat_pinv(rows: usize, cols: usize, data: Vec<f64>, rcond: f64) -> PyResult<(usize, usize, Vec<f64>)> {
    let m = la::pinv(&mat(rows, cols, data)?, rcond);
    Ok((m.rows, m.cols, m.data))
}
#[pyfunction]
fn mat_matmul(ar: usize, ac: usize, a: Vec<f64>, br: usize, bc: usize, b: Vec<f64>) -> PyResult<(usize, usize, Vec<f64>)> {
    let m = la::matmul(&mat(ar, ac, a)?, &mat(br, bc, b)?)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("shape mismatch in matmul"))?;
    Ok((m.rows, m.cols, m.data))
}
#[pyfunction]
fn mat_lstsq(rows: usize, cols: usize, data: Vec<f64>, y: Vec<f64>) -> PyResult<Vec<f64>> {
    la::lstsq(&mat(rows, cols, data)?, &y)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("lstsq shape mismatch"))
}
#[pyfunction]
fn mat_eigh(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (v, vecs) = la::eigh(&mat(rows, cols, data)?)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eigh requires a square matrix"))?;
    Ok((v, vecs.data))
}
#[pyfunction]
fn mat_svd(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, usize, usize)> {
    let (u, s, vt) = la::svd(&mat(rows, cols, data)?);
    Ok((u.data, s, vt.data, u.rows, u.cols, vt.rows, vt.cols))
}
#[pyfunction]
fn mat_sqrtm_spd(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<Vec<f64>> {
    la::sqrtm_spd(&mat(rows, cols, data)?).map(|m| m.data).ok_or_else(singular)
}
#[pyfunction]
#[pyo3(signature = (rows, cols, data, ddof=1))]
fn mat_cov(rows: usize, cols: usize, data: Vec<f64>, ddof: usize) -> PyResult<Vec<f64>> {
    Ok(la::cov_matrix(&mat(rows, cols, data)?, ddof).data)
}
#[pyfunction]
fn mat_corr(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(la::corr_matrix(&mat(rows, cols, data)?).data)
}

#[pyfunction]
fn linregress(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let r = with_two_slices(py, x, y, reg::linregress)?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("linregress needs >= 2 distinct x values"))?;
    Ok((r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr, r.intercept_stderr))
}

/// OLS. Returns a flat tuple; the Python layer reshapes it into a dict.
#[allow(clippy::type_complexity)]
#[pyfunction]
fn ols(rows: usize, cols: usize, data: Vec<f64>, y: Vec<f64>)
    -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64, f64, Vec<f64>)> {
    let f = reg::ols(&mat(rows, cols, data)?, &y).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("OLS requires more rows than columns")
    })?;
    Ok((f.coefficients, f.std_errors, f.t_values, f.p_values, f.residuals,
        f.r_squared, f.adj_r_squared, f.df_resid, f.sigma2, f.cov))
}

#[pyfunction]
fn ttest_1samp(py: Python<'_>, x: &Bound<'_, PyAny>, mu: f64) -> PyResult<(f64, f64)> {
    with_slice(py, x, |s| reg::ttest_1samp(s, mu))
}
#[pyfunction]
#[pyo3(signature = (a, b, equal_var=true))]
fn ttest_ind(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, equal_var: bool) -> PyResult<(f64, f64)> {
    let xa = Arr::from_py(a)?;
    let xb = Arr::from_py(b)?;
    Ok(reg::ttest_ind(xa.as_slice(py), xb.as_slice(py), equal_var))
}
#[pyfunction]
fn ttest_rel(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    with_two_slices(py, a, b, reg::ttest_rel)
}
#[pyfunction]
fn f_oneway(groups: Vec<Vec<f64>>) -> (f64, f64) {
    reg::f_oneway(&groups)
}
#[pyfunction]
#[pyo3(signature = (rows, cols, data, correction=true))]
fn chi2_contingency(rows: usize, cols: usize, data: Vec<f64>, correction: bool)
    -> PyResult<(f64, f64, f64, Vec<f64>)> {
    Ok(reg::chi2_contingency(&mat(rows, cols, data)?, correction))
}

// ------------------------------------------------------------- optimize ----

#[pyfunction]
#[pyo3(signature = (f, a, b, xtol=1e-12, max_iter=200))]
fn brentq(py: Python<'_>, f: PyObject, a: f64, b: f64, xtol: f64, max_iter: usize) -> PyResult<f64> {
    let mut err: Option<PyErr> = None;
    let root = opt::brentq(
        |x| match f.call1(py, (x,)).and_then(|v| v.extract::<f64>(py)) {
            Ok(v) => v,
            Err(e) => {
                if err.is_none() {
                    err = Some(e);
                }
                f64::NAN
            }
        },
        a, b, xtol, max_iter,
    );
    if let Some(e) = err {
        return Err(e);
    }
    root.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "f(a) and f(b) must have opposite signs; got a={a}, b={b}"
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (f, p0, lower, upper, n_resid, max_iter=200))]
fn curve_fit_lm(py: Python<'_>, f: PyObject, p0: Vec<f64>, lower: Vec<f64>, upper: Vec<f64>,
                n_resid: usize, max_iter: usize) -> PyResult<Vec<f64>> {
    let mut err: Option<PyErr> = None;
    let out = opt::levenberg_marquardt(
        |p: &[f64], out: &mut [f64]| {
            match f.call1(py, (p.to_vec(),)).and_then(|v| v.extract::<Vec<f64>>(py)) {
                Ok(v) if v.len() == out.len() => out.copy_from_slice(&v),
                Ok(v) => {
                    if err.is_none() {
                        err = Some(pyo3::exceptions::PyValueError::new_err(format!(
                            "residual function returned {} values, expected {}", v.len(), out.len()
                        )));
                    }
                    out.fill(f64::NAN);
                }
                Err(e) => {
                    if err.is_none() { err = Some(e); }
                    out.fill(f64::NAN);
                }
            }
        },
        &p0, &lower, &upper, n_resid, max_iter,
    );
    if let Some(e) = err { return Err(e); }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (y, columns, max_iter=20000, tol=1e-14))]
fn simplex_least_squares(y: Vec<f64>, columns: Vec<Vec<f64>>, max_iter: usize, tol: f64) -> Vec<f64> {
    opt::simplex_least_squares(&y, &columns, max_iter, tol)
}


// ------------------------------------------------------------------- rng ---

/// The library's random number generator: a seeded PCG64 stream.
///
/// Exposed as a real type rather than as a NumPy-shaped compatibility object.
/// Constructing with the same seed always reproduces the same sequence.
#[pyclass(name = "Rng")]
struct PyRng {
    inner: rrng::Sampler,
}

#[pymethods]
impl PyRng {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> PyRng {
        PyRng {
            inner: match seed {
                Some(s) => rrng::Sampler::new(s),
                None => rrng::Sampler::from_entropy(),
            },
        }
    }

    /// n uniform draws from [0, 1).
    fn random(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.uniform01()).collect()
    }
    #[pyo3(signature = (low=0.0, high=1.0, n=1))]
    fn uniform(&mut self, low: f64, high: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.uniform(low, high)).collect()
    }
    #[pyo3(signature = (low, high, n=1))]
    fn integers(&mut self, low: i64, high: i64, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.inner.integer(low, high)).collect()
    }
    #[pyo3(signature = (mean=0.0, sd=1.0, n=1))]
    fn normal(&mut self, mean: f64, sd: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.normal(mean, sd)).collect()
    }
    #[pyo3(signature = (scale=1.0, n=1))]
    fn exponential(&mut self, scale: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.exponential(scale)).collect()
    }
    #[pyo3(signature = (mu=0.0, sigma=1.0, n=1))]
    fn lognormal(&mut self, mu: f64, sigma: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.lognormal(mu, sigma)).collect()
    }
    #[pyo3(signature = (shape, scale=1.0, n=1))]
    fn gamma(&mut self, shape: f64, scale: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.gamma(shape, scale)).collect()
    }
    #[pyo3(signature = (a, b, n=1))]
    fn beta(&mut self, a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.beta(a, b)).collect()
    }
    #[pyo3(signature = (lam=1.0, n=1))]
    fn poisson(&mut self, lam: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.poisson(lam)).collect()
    }
    #[pyo3(signature = (trials, p, n=1))]
    fn binomial(&mut self, trials: u64, p: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.binomial(trials, p)).collect()
    }
    #[pyo3(signature = (shape, scale=1.0, n=1))]
    fn weibull(&mut self, shape: f64, scale: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.inner.weibull(shape, scale)).collect()
    }
    #[pyo3(signature = (data, n, replace=true))]
    fn choice(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        n: usize,
        replace: bool,
    ) -> PyResult<Vec<f64>> {
        let a = Arr::from_py(data)?;
        let s = a.as_slice(py);
        if s.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot choose from an empty sequence",
            ));
        }
        if replace {
            let mut out = vec![0.0; n];
            self.inner.choice_with_replacement(s, &mut out);
            Ok(out)
        } else {
            if n > s.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cannot take a sample larger than the population without replacement",
                ));
            }
            Ok(self.inner.choice_without_replacement(s, n))
        }
    }
    /// A random permutation of 0..n.
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        self.inner.permutation(n)
    }
    /// Return a shuffled copy of the input.
    fn shuffled(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let a = Arr::from_py(data)?;
        let mut v = a.as_slice(py).to_vec();
        self.inner.shuffle_slice(&mut v);
        Ok(v)
    }
}

#[pyfunction]
fn gbm_paths(py: Python<'_>, s0: f64, mu: f64, sigma: f64, t: f64, n_steps: usize, n_sims: usize, seed: u64) -> Vec<f64> {
    py.allow_threads(|| sim::gbm_paths(s0, mu, sigma, t, n_steps, n_sims, seed))
}

#[pyfunction]
fn uniform_box(py: Python<'_>, lower: Vec<f64>, upper: Vec<f64>, n_samples: usize, seed: u64) -> Vec<f64> {
    py.allow_threads(|| sim::uniform_box(&lower, &upper, n_samples, seed))
}

#[pymodule]
fn _rss(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ln_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_fn, m)?)?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(norm_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_sf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(ln_beta, m)?)?;
    m.add_function(wrap_pyfunction!(gammainc_p, m)?)?;
    m.add_function(wrap_pyfunction!(gammainc_q, m)?)?;
    m.add_function(wrap_pyfunction!(gammaincinv, m)?)?;
    m.add_function(wrap_pyfunction!(betainc, m)?)?;
    m.add_function(wrap_pyfunction!(betaincinv, m)?)?;
    m.add_function(wrap_pyfunction!(normal_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(normal_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(normal_sf, m)?)?;
    m.add_function(wrap_pyfunction!(normal_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(t_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(t_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(t_sf, m)?)?;
    m.add_function(wrap_pyfunction!(t_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_sf, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_isf, m)?)?;
    m.add_function(wrap_pyfunction!(f_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(f_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(f_sf, m)?)?;
    m.add_function(wrap_pyfunction!(f_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_sf, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(beta_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(beta_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(beta_sf, m)?)?;
    m.add_function(wrap_pyfunction!(beta_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(expon_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(expon_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(expon_sf, m)?)?;
    m.add_function(wrap_pyfunction!(expon_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(lognorm_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(lognorm_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(lognorm_sf, m)?)?;
    m.add_function(wrap_pyfunction!(lognorm_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(weibull_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(weibull_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(weibull_sf, m)?)?;
    m.add_function(wrap_pyfunction!(weibull_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(fisk_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(fisk_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(fisk_sf, m)?)?;
    m.add_function(wrap_pyfunction!(fisk_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(binom_pmf, m)?)?;
    m.add_function(wrap_pyfunction!(binom_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(binom_sf, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_pmf, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_sf, m)?)?;
    m.add_function(wrap_pyfunction!(geom_pmf, m)?)?;
    m.add_function(wrap_pyfunction!(geom_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(geom_sf, m)?)?;
    m.add_function(wrap_pyfunction!(nbinom_pmf, m)?)?;
    m.add_function(wrap_pyfunction!(nbinom_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(nbinom_sf, m)?)?;
    m.add_function(wrap_pyfunction!(nct_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(nct_sf, m)?)?;
    m.add_function(wrap_pyfunction!(ncf_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(ncf_sf, m)?)?;
    m.add_function(wrap_pyfunction!(shapiro_wilk, m)?)?;
    m.add_function(wrap_pyfunction!(sum_, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(min_, m)?)?;
    m.add_function(wrap_pyfunction!(max_, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    m.add_function(wrap_pyfunction!(quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(five_number_summary, m)?)?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(modes, m)?)?;
    m.add_function(wrap_pyfunction!(sorted_copy, m)?)?;
    m.add_function(wrap_pyfunction!(covariance, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_r, m)?)?;
    m.add_function(wrap_pyfunction!(is_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_dist, m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_values, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_dist, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_ci, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_pvalue, m)?)?;
    m.add_function(wrap_pyfunction!(mat_inv, m)?)?;
    m.add_function(wrap_pyfunction!(mat_pinv, m)?)?;
    m.add_function(wrap_pyfunction!(mat_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(mat_lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(mat_eigh, m)?)?;
    m.add_function(wrap_pyfunction!(mat_svd, m)?)?;
    m.add_function(wrap_pyfunction!(mat_sqrtm_spd, m)?)?;
    m.add_function(wrap_pyfunction!(mat_cov, m)?)?;
    m.add_function(wrap_pyfunction!(mat_corr, m)?)?;
    m.add_function(wrap_pyfunction!(linregress, m)?)?;
    m.add_function(wrap_pyfunction!(ols, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_1samp, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_ind, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_rel, m)?)?;
    m.add_function(wrap_pyfunction!(f_oneway, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_contingency, m)?)?;
    m.add_function(wrap_pyfunction!(brentq, m)?)?;
    m.add_function(wrap_pyfunction!(curve_fit_lm, m)?)?;
    m.add_function(wrap_pyfunction!(simplex_least_squares, m)?)?;
    m.add_class::<PyRng>()?;
    m.add_function(wrap_pyfunction!(gbm_paths, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_box, m)?)?;
    Ok(())
}
