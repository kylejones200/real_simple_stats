//! Maximum-likelihood fits for the survival families, with location fixed at 0.
//!
//! Closed forms are used where they exist (exponential, lognormal); the Weibull
//! shape comes from a one-dimensional root solve, and the log-logistic is fitted
//! as a logistic on log-durations, which is exactly what it is.

use crate::descriptive as ds;
use crate::optimize::{brentq, nelder_mead};

/// `(shape_params, scale, log_likelihood)`. `shape_params` is empty for the
/// exponential, which has scale only.
pub struct FitResult {
    pub shape: Option<f64>,
    pub scale: f64,
    pub log_likelihood: f64,
}

/// Exponential MLE: the scale is just the sample mean.
pub fn fit_exponential(t: &[f64]) -> FitResult {
    let scale = ds::mean(t);
    let n = t.len() as f64;
    // sum ln f = -n ln(scale) - sum(t)/scale
    let ll = -n * scale.ln() - ds::sum(t) / scale;
    FitResult { shape: None, scale, log_likelihood: ll }
}

/// Lognormal MLE: normal MLE on log-durations (population sd, as SciPy uses).
pub fn fit_lognormal(t: &[f64]) -> FitResult {
    let logs: Vec<f64> = t.iter().map(|v| v.ln()).collect();
    let mu = ds::mean(&logs);
    let s = ds::std_dev(&logs, 0);
    let n = t.len() as f64;
    // sum ln f = -sum ln(t) - n ln(s) - n/2 ln(2 pi) - sum((ln t - mu)^2)/(2 s^2)
    let ss: f64 = logs.iter().map(|l| (l - mu) * (l - mu)).sum();
    let ll = -logs.iter().sum::<f64>() - n * s.ln()
        - 0.5 * n * (2.0 * std::f64::consts::PI).ln()
        - ss / (2.0 * s * s);
    FitResult { shape: Some(s), scale: mu.exp(), log_likelihood: ll }
}

/// Weibull MLE. The shape solves
/// `sum(t^c ln t)/sum(t^c) - 1/c - mean(ln t) = 0`, after which the scale is
/// available in closed form.
pub fn fit_weibull(t: &[f64]) -> FitResult {
    let n = t.len() as f64;
    let mean_log = t.iter().map(|v| v.ln()).sum::<f64>() / n;

    let g = |c: f64| -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for &v in t {
            let p = v.powf(c);
            num += p * v.ln();
            den += p;
        }
        if den == 0.0 {
            return f64::NAN;
        }
        num / den - 1.0 / c - mean_log
    };

    // Bracket the root by expanding outward from 1.
    let mut lo = 0.05f64;
    let mut hi = 1.0f64;
    while g(hi) < 0.0 && hi < 1e4 {
        lo = hi;
        hi *= 2.0;
    }
    while g(lo) > 0.0 && lo > 1e-6 {
        hi = lo;
        lo /= 2.0;
    }
    let c = brentq(g, lo, hi, 1e-12, 200).unwrap_or(1.0);
    let scale = (t.iter().map(|v| v.powf(c)).sum::<f64>() / n).powf(1.0 / c);

    // sum ln f = n ln(c) - n c ln(scale) + (c-1) sum ln t - sum (t/scale)^c
    let ll = n * c.ln() - n * c * scale.ln()
        + (c - 1.0) * t.iter().map(|v| v.ln()).sum::<f64>()
        - t.iter().map(|v| (v / scale).powf(c)).sum::<f64>();
    FitResult { shape: Some(c), scale, log_likelihood: ll }
}

/// Log-logistic (Fisk) MLE.
///
/// If T is log-logistic with shape c and scale a, then ln T is logistic with
/// location ln(a) and spread 1/c -- so this fits a logistic to the logs.
pub fn fit_loglogistic(t: &[f64]) -> FitResult {
    let logs: Vec<f64> = t.iter().map(|v| v.ln()).collect();
    let mu0 = ds::mean(&logs);
    // Moment start: logistic variance = pi^2 s^2 / 3.
    let s0 = (ds::std_dev(&logs, 0) * (3.0f64).sqrt() / std::f64::consts::PI).max(1e-6);

    let neg_ll = |p: &[f64]| -> f64 {
        let (mu, s) = (p[0], p[1]);
        if s <= 0.0 {
            return f64::INFINITY;
        }
        let mut acc = 0.0;
        for &y in &logs {
            let z = (y - mu) / s;
            // ln pdf = -z - ln s - 2 ln(1 + e^-z), written via ln(1+e^-|z|)
            // to avoid overflow for large |z|.
            let softplus = if z > 0.0 {
                (-z).exp().ln_1p()
            } else {
                -z + (z).exp().ln_1p()
            };
            acc -= -z - s.ln() - 2.0 * softplus;
        }
        acc
    };

    let best = nelder_mead(neg_ll, &[mu0, s0], 0.1, 2000, 1e-12);
    let (mu, s) = (best[0], best[1].abs().max(1e-12));
    let ll_logistic = -neg_ll(&[mu, s]);
    // Change of variables ln T = Y contributes -sum ln t to the density in t.
    let ll = ll_logistic - logs.iter().sum::<f64>();
    FitResult { shape: Some(1.0 / s), scale: mu.exp(), log_likelihood: ll }
}
