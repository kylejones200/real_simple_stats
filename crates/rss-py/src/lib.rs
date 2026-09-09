use pyo3::prelude::*;
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
    Ok(())
}
