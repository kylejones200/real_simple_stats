mod arr;

use arr::{with_slice, with_two_slices, Arr};
use pyo3::prelude::*;
use rss_core::descriptive as ds;
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
    Ok(())
}
