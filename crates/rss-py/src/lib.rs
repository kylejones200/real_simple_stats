use pyo3::prelude::*;
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
    Ok(())
}
