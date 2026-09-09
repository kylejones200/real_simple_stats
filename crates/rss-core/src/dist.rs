//! Continuous and discrete distributions, built on `special`.
//!
//! Every CDF has a matching survival function computed in the tail-accurate
//! direction rather than as `1 - cdf`, so upper-tail p-values keep full
//! relative precision.

use crate::special::*;

// ---------------------------------------------------------------- normal ----

pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NAN;
    }
    norm_pdf((x - mu) / sigma) / sigma
}
pub fn normal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NAN;
    }
    norm_cdf((x - mu) / sigma)
}
pub fn normal_sf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NAN;
    }
    norm_sf((x - mu) / sigma)
}
pub fn normal_ppf(p: f64, mu: f64, sigma: f64) -> f64 {
    mu + sigma * norm_ppf(p)
}

// --------------------------------------------------------------- student t --

pub fn t_pdf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let c = -0.5 * (df + 1.0) * (1.0 + x * x / df).ln();
    (c - 0.5 * (df * std::f64::consts::PI).ln() + ln_gamma(0.5 * (df + 1.0)) - ln_gamma(0.5 * df))
        .exp()
}

/// Student-t CDF via the incomplete beta.
pub fn t_cdf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.5;
    }
    // I_{df/(df+x^2)}(df/2, 1/2) gives twice the tail area.
    let z = df / (df + x * x);
    let tail = 0.5 * betainc(0.5 * df, 0.5, z);
    if x > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Student-t survival function, evaluated directly in the upper tail.
pub fn t_sf(x: f64, df: f64) -> f64 {
    t_cdf(-x, df)
}

pub fn t_ppf(p: f64, df: f64) -> f64 {
    if df <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Work in the lower tail, then mirror; keeps the beta inverse well conditioned.
    let (tail, sign) = if p < 0.5 { (p, -1.0) } else { (1.0 - p, 1.0) };
    let z = betaincinv(0.5 * df, 0.5, 2.0 * tail);
    let x = (df * (1.0 - z) / z).sqrt();
    sign * x
}

// ------------------------------------------------------------- chi-squared --

pub fn chi2_pdf(x: f64, df: f64) -> f64 {
    if x < 0.0 || df <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if df < 2.0 {
            f64::INFINITY
        } else if df == 2.0 {
            0.5
        } else {
            0.0
        };
    }
    let k = 0.5 * df;
    ((k - 1.0) * x.ln() - 0.5 * x - k * std::f64::consts::LN_2 - ln_gamma(k)).exp()
}
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    gammainc_p(0.5 * df, 0.5 * x)
}
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    gammainc_q(0.5 * df, 0.5 * x)
}
pub fn chi2_ppf(p: f64, df: f64) -> f64 {
    2.0 * gammaincinv(0.5 * df, p)
}
pub fn chi2_isf(q: f64, df: f64) -> f64 {
    2.0 * gammainc_cinv(0.5 * df, q)
}

/// Thin alias so callers outside `special` need not import it directly.
#[inline]
fn gammainc_cinv(a: f64, q: f64) -> f64 {
    crate::special::gammainccinv(a, q)
}

// ------------------------------------------------------------------- F -----

pub fn f_pdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x < 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if d1 < 2.0 {
            f64::INFINITY
        } else if d1 == 2.0 {
            1.0
        } else {
            0.0
        };
    }
    let lg = 0.5 * d1 * (d1 / d2).ln() + (0.5 * d1 - 1.0) * x.ln()
        - 0.5 * (d1 + d2) * (1.0 + d1 * x / d2).ln()
        - ln_beta(0.5 * d1, 0.5 * d2);
    lg.exp()
}
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let z = d1 * x / (d1 * x + d2);
    betainc(0.5 * d1, 0.5 * d2, z)
}
/// F survival function, computed via the mirrored beta so the upper tail keeps
/// full precision instead of cancelling against 1.
pub fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let z = d2 / (d1 * x + d2);
    betainc(0.5 * d2, 0.5 * d1, z)
}
pub fn f_ppf(p: f64, d1: f64, d2: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let z = betaincinv(0.5 * d1, 0.5 * d2, p);
    if z >= 1.0 {
        return f64::INFINITY;
    }
    d2 * z / (d1 * (1.0 - z))
}

// ---------------------------------------------------------------- gamma ----

pub fn gamma_pdf(x: f64, shape: f64, scale: f64) -> f64 {
    if x < 0.0 || shape <= 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if shape < 1.0 {
            f64::INFINITY
        } else if shape == 1.0 {
            1.0 / scale
        } else {
            0.0
        };
    }
    let z = x / scale;
    ((shape - 1.0) * z.ln() - z - ln_gamma(shape) - scale.ln()).exp()
}
pub fn gamma_cdf(x: f64, shape: f64, scale: f64) -> f64 {
    gammainc_p(shape, x / scale)
}
pub fn gamma_sf(x: f64, shape: f64, scale: f64) -> f64 {
    gammainc_q(shape, x / scale)
}
pub fn gamma_ppf(p: f64, shape: f64, scale: f64) -> f64 {
    scale * gammaincinv(shape, p)
}

// ----------------------------------------------------------------- beta ----

pub fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta(a, b)).exp()
}
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    betainc(a, b, x)
}
pub fn beta_sf(x: f64, a: f64, b: f64) -> f64 {
    betainc(b, a, 1.0 - x)
}
pub fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    betaincinv(a, b, p)
}

// ----------------------------------------------------------- exponential ---

pub fn expon_pdf(x: f64, scale: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    (-x / scale).exp() / scale
}
pub fn expon_cdf(x: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    -(-x / scale).exp_m1()
}
pub fn expon_sf(x: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    (-x / scale).exp()
}
pub fn expon_ppf(p: f64, scale: f64) -> f64 {
    -scale * (-p).ln_1p()
}

// ------------------------------------------------------------- lognormal ---

pub fn lognorm_pdf(x: f64, s: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    norm_pdf((x / scale).ln() / s) / (x * s)
}
pub fn lognorm_cdf(x: f64, s: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    norm_cdf((x / scale).ln() / s)
}
pub fn lognorm_sf(x: f64, s: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    norm_sf((x / scale).ln() / s)
}
pub fn lognorm_ppf(p: f64, s: f64, scale: f64) -> f64 {
    scale * (s * norm_ppf(p)).exp()
}

// --------------------------------------------------------------- weibull ---

pub fn weibull_pdf(x: f64, c: f64, scale: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    let z = x / scale;
    (c / scale) * z.powf(c - 1.0) * (-z.powf(c)).exp()
}
pub fn weibull_cdf(x: f64, c: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    -(-(x / scale).powf(c)).exp_m1()
}
pub fn weibull_sf(x: f64, c: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    (-(x / scale).powf(c)).exp()
}
pub fn weibull_ppf(p: f64, c: f64, scale: f64) -> f64 {
    scale * (-(-p).ln_1p()).powf(1.0 / c)
}

// ------------------------------------------------- log-logistic (fisk) -----

pub fn fisk_pdf(x: f64, c: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let z = x / scale;
    let zc = z.powf(c);
    (c / scale) * z.powf(c - 1.0) / ((1.0 + zc) * (1.0 + zc))
}
pub fn fisk_cdf(x: f64, c: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let zc = (x / scale).powf(c);
    zc / (1.0 + zc)
}
pub fn fisk_sf(x: f64, c: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    1.0 / (1.0 + (x / scale).powf(c))
}
pub fn fisk_ppf(p: f64, c: f64, scale: f64) -> f64 {
    scale * (p / (1.0 - p)).powf(1.0 / c)
}

// -------------------------------------------------------------- binomial ---

pub fn binom_pmf(k: f64, n: f64, p: f64) -> f64 {
    if k < 0.0 || k > n || k.fract() != 0.0 {
        return 0.0;
    }
    if p == 0.0 {
        return if k == 0.0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    let lc = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0);
    (lc + k * p.ln() + (n - k) * (1.0 - p).ln()).exp()
}
/// P(X <= k) via the incomplete beta identity.
pub fn binom_cdf(k: f64, n: f64, p: f64) -> f64 {
    let kf = k.floor();
    if kf < 0.0 {
        return 0.0;
    }
    if kf >= n {
        return 1.0;
    }
    betainc(n - kf, kf + 1.0, 1.0 - p)
}
pub fn binom_sf(k: f64, n: f64, p: f64) -> f64 {
    let kf = k.floor();
    if kf < 0.0 {
        return 1.0;
    }
    if kf >= n {
        return 0.0;
    }
    betainc(kf + 1.0, n - kf, p)
}

// --------------------------------------------------------------- poisson ---

pub fn poisson_pmf(k: f64, mu: f64) -> f64 {
    if k < 0.0 || k.fract() != 0.0 {
        return 0.0;
    }
    (k * mu.ln() - mu - ln_gamma(k + 1.0)).exp()
}
pub fn poisson_cdf(k: f64, mu: f64) -> f64 {
    let kf = k.floor();
    if kf < 0.0 {
        return 0.0;
    }
    gammainc_q(kf + 1.0, mu)
}
pub fn poisson_sf(k: f64, mu: f64) -> f64 {
    let kf = k.floor();
    if kf < 0.0 {
        return 1.0;
    }
    gammainc_p(kf + 1.0, mu)
}

// ------------------------------------------------------------- geometric ---

pub fn geom_pmf(k: f64, p: f64) -> f64 {
    if k < 1.0 || k.fract() != 0.0 {
        return 0.0;
    }
    p * (1.0 - p).powf(k - 1.0)
}
pub fn geom_cdf(k: f64, p: f64) -> f64 {
    let kf = k.floor();
    if kf < 1.0 {
        return 0.0;
    }
    -(kf * (-p).ln_1p()).exp_m1()
}
pub fn geom_sf(k: f64, p: f64) -> f64 {
    let kf = k.floor();
    if kf < 1.0 {
        return 1.0;
    }
    (kf * (-p).ln_1p()).exp()
}

// ------------------------------------------------- negative binomial ------

pub fn nbinom_pmf(k: f64, n: f64, p: f64) -> f64 {
    if k < 0.0 || k.fract() != 0.0 {
        return 0.0;
    }
    let lc = ln_gamma(k + n) - ln_gamma(n) - ln_gamma(k + 1.0);
    (lc + n * p.ln() + k * (1.0 - p).ln()).exp()
}
pub fn nbinom_cdf(k: f64, n: f64, p: f64) -> f64 {
    let kf = k.floor();
    if kf < 0.0 {
        return 0.0;
    }
    betainc(n, kf + 1.0, p)
}
pub fn nbinom_sf(k: f64, n: f64, p: f64) -> f64 {
    let kf = k.floor();
    if kf < 0.0 {
        return 1.0;
    }
    betainc(kf + 1.0, n, 1.0 - p)
}
