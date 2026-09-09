//! Noncentral t and F distributions, and the Shapiro-Wilk normality test.
//!
//! These are the three pieces of the SciPy surface with no elementary closed
//! form; they are what made a full SciPy removal a risk item. `nct`/`ncf` back
//! `power_analysis`, and `shapiro` backs `assumptions`.

use crate::special::*;

/// Noncentral t CDF, P(T <= t | df, delta) — Lenth (1989), AS 243.
///
/// The series alternates contributions from odd and even incomplete-beta terms
/// with Poisson-like weights; `s` tracks the unused probability mass and gives
/// the termination criterion.
pub fn nct_cdf(t: f64, df: f64, delta: f64) -> f64 {
    const ITRMAX: usize = 1000;
    const ERRBD: f64 = 1e-14;

    if df <= 0.0 {
        return f64::NAN;
    }
    // Reflect so the series always runs on the positive side.
    if t < 0.0 {
        return 1.0 - nct_cdf(-t, df, -delta);
    }

    let x = t * t / (t * t + df);
    if x <= 0.0 {
        return norm_cdf(-delta);
    }

    let lambda = delta * delta;
    let mut p = 0.5 * (-0.5 * lambda).exp();
    let mut q = (2.0 / std::f64::consts::PI).sqrt() * p * delta;
    let mut s = 0.5 - p;
    // Guard against total cancellation for large delta.
    if s < 1e-300 {
        s = 0.0;
    }

    let mut a = 0.5;
    let b = 0.5 * df;
    let rxb = (1.0 - x).powf(b);
    let albeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);

    let mut xodd = betainc(a, b, x);
    let mut godd = 2.0 * rxb * (a * x.ln() - albeta).exp();
    let mut xeven = 1.0 - rxb;
    let mut geven = b * x * rxb;

    let mut tnc = p * xodd + q * xeven;

    for it in 1..=ITRMAX {
        a += 1.0;
        xodd -= godd;
        xeven -= geven;
        godd *= x * (a + b - 1.0) / a;
        geven *= x * (a + b - 0.5) / (a + 0.5);
        p *= lambda / (2.0 * it as f64);
        q *= lambda / (2.0 * it as f64 + 1.0);
        s -= p;
        tnc += p * xodd + q * xeven;
        // Remaining mass bounds the truncation error.
        let errbd = 2.0 * s * (xodd - godd);
        if errbd.abs() < ERRBD || s <= 0.0 {
            break;
        }
    }
    (tnc + norm_cdf(-delta)).clamp(0.0, 1.0)
}

pub fn nct_sf(t: f64, df: f64, delta: f64) -> f64 {
    1.0 - nct_cdf(t, df, delta)
}

/// Noncentral F CDF: a Poisson(lambda/2)-weighted mixture of central incomplete
/// beta terms.
///
/// The weights are summed outward from the Poisson mode rather than from j = 0,
/// so large noncentrality does not start the recurrence in an underflowed term.
pub fn ncf_cdf(f: f64, dfn: f64, dfd: f64, nc: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    if dfn <= 0.0 || dfd <= 0.0 || nc < 0.0 {
        return f64::NAN;
    }
    if nc == 0.0 {
        return crate::dist::f_cdf(f, dfn, dfd);
    }

    let x = dfn * f / (dfn * f + dfd);
    let half_nc = 0.5 * nc;
    let b = 0.5 * dfd;

    // Poisson weight in log space, so no term underflows on the way to the mode.
    let ln_w = |j: f64| -half_nc + j * half_nc.ln() - ln_gamma(j + 1.0);

    let jmode = half_nc.floor().max(0.0);
    let mut total = 0.0;
    let mut weight_seen = 0.0;

    // Outward from the mode in both directions until the weights stop mattering.
    for dir in [0i64, 1i64] {
        let mut j = if dir == 0 { jmode as i64 } else { jmode as i64 - 1 };
        loop {
            if j < 0 || j > 1_000_000 {
                break;
            }
            let jf = j as f64;
            let w = ln_w(jf).exp();
            if w < 1e-18 && weight_seen > 0.5 {
                break;
            }
            total += w * betainc(0.5 * dfn + jf, b, x);
            weight_seen += w;
            if weight_seen > 1.0 - 1e-16 {
                break;
            }
            if dir == 0 {
                j += 1;
            } else {
                j -= 1;
            }
        }
    }
    total.clamp(0.0, 1.0)
}

pub fn ncf_sf(f: f64, dfn: f64, dfd: f64, nc: f64) -> f64 {
    1.0 - ncf_cdf(f, dfn, dfd, nc)
}

/// Shapiro-Wilk test for normality — Royston (1995), AS R94.
///
/// Returns `(W, p_value)`. Valid for 3 <= n <= 5000; `data` need not be sorted.
///
/// Weight indexing follows AS R94 exactly: `a[i]` is the coefficient for the
/// i-th widest pair `(x[n-1-i] - x[i])`, so `a[0]` is the largest weight and
/// pairs with the extreme spread. Getting this backwards silently produces a
/// plausible-looking but wrong W.
pub fn shapiro_wilk(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n < 3 {
        return (f64::NAN, f64::NAN);
    }
    let mut x: Vec<f64> = data.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;
    let nh = n / 2;

    // Sum of squares about the mean; W is undefined for a constant sample.
    let mean: f64 = x.iter().sum::<f64>() / nf;
    let ssq: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    if ssq <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let mut a = vec![0.0f64; nh];
    if n == 3 {
        a[0] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        // Expected normal order statistics for the lower half (all negative).
        let an25 = nf + 0.25;
        let m: Vec<f64> = (0..nh)
            .map(|j| norm_ppf((j as f64 + 1.0 - 0.375) / an25))
            .collect();
        let summ2: f64 = 2.0 * m.iter().map(|v| v * v).sum::<f64>();
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / nf.sqrt();

        // Royston's polynomial corrections for the one or two extreme weights.
        let c1 = poly_asc(rsn, &[0.0, 0.221_157, -0.147_981, -2.071_190, 4.434_685, -2.706_056]);
        let a0 = c1 - m[0] / ssumm2;

        let (i1, fac);
        if n > 5 {
            let c2 =
                poly_asc(rsn, &[0.0, 0.042_981, -0.293_762, -1.752_461, 5.682_633, -3.582_633]);
            let a1 = c2 - m[1] / ssumm2;
            fac = ((summ2 - 2.0 * m[0] * m[0] - 2.0 * m[1] * m[1])
                / (1.0 - 2.0 * a0 * a0 - 2.0 * a1 * a1))
                .sqrt();
            a[0] = a0;
            a[1] = a1;
            i1 = 2;
        } else {
            fac = ((summ2 - 2.0 * m[0] * m[0]) / (1.0 - 2.0 * a0 * a0)).sqrt();
            a[0] = a0;
            i1 = 1;
        }
        for i in i1..nh {
            a[i] = -m[i] / fac;
        }
    }

    let mut num = 0.0;
    for i in 0..nh {
        num += a[i] * (x[n - 1 - i] - x[i]);
    }
    let w = (num * num / ssq).min(1.0);

    let p = if n == 3 {
        // Exact null distribution at n = 3.
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = (0.75f64).sqrt().asin();
        (pi6 * (w.sqrt().asin() - stqr)).clamp(0.0, 1.0)
    } else if n <= 11 {
        let gma = poly_asc(nf, &[-2.273, 0.459]);
        let mu = poly_asc(nf, &[0.5440, -0.39978, 0.025_054, -6.714e-4]);
        let sigma = poly_asc(nf, &[1.3822, -0.77857, 0.062_767, -0.002_0322]).exp();
        // NOTE: the argument is gma - ln(1-w), not gma - w.
        let y = -(gma - (-w).ln_1p()).ln();
        norm_sf((y - mu) / sigma).clamp(0.0, 1.0)
    } else {
        let ln_n = nf.ln();
        let mu = poly_asc(ln_n, &[-1.5861, -0.31082, -0.083_751, 0.003_8915]);
        let sigma = poly_asc(ln_n, &[-0.4803, -0.082_676, 0.003_0302]).exp();
        let y = (-w).ln_1p();
        norm_sf((y - mu) / sigma).clamp(0.0, 1.0)
    };
    (w, p)
}

/// Horner evaluation with coefficients given lowest-order first.
#[inline]
fn poly_asc(x: f64, c: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &ci in c.iter().rev() {
        acc = acc * x + ci;
    }
    acc
}
