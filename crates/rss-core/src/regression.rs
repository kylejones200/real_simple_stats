//! Regression: simple linear, OLS with inference, and the hypothesis tests
//! that share its machinery.

use crate::descriptive as ds;
use crate::dist;
use crate::linalg::{lstsq, pinv, Matrix};

pub struct LinRegress {
    pub slope: f64,
    pub intercept: f64,
    pub rvalue: f64,
    pub pvalue: f64,
    pub stderr: f64,
    pub intercept_stderr: f64,
}

/// Simple least-squares regression of `y` on `x`, matching `scipy.stats.linregress`.
pub fn linregress(x: &[f64], y: &[f64]) -> Option<LinRegress> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }
    let nf = n as f64;
    let mx = ds::mean(x);
    let my = ds::mean(y);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx == 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let r = if syy > 0.0 {
        (sxy / (sxx.sqrt() * syy.sqrt())).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // Two-sided t test on the slope, df = n - 2.
    let df = nf - 2.0;
    let (stderr, icept_stderr, p) = if df > 0.0 {
        let ss_res = (syy - slope * sxy).max(0.0);
        let s2 = ss_res / df;
        let se = (s2 / sxx).sqrt();
        let se_i = (s2 * (1.0 / nf + mx * mx / sxx)).sqrt();
        let p = if se > 0.0 {
            2.0 * dist::t_sf((slope / se).abs(), df)
        } else {
            0.0
        };
        (se, se_i, p)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    Some(LinRegress {
        slope,
        intercept,
        rvalue: r,
        pvalue: p,
        stderr,
        intercept_stderr: icept_stderr,
    })
}

pub struct OlsFit {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub df_resid: f64,
    pub sigma2: f64,
    /// Row-major (k x k) coefficient covariance matrix.
    pub cov: Vec<f64>,
}

/// Ordinary least squares with the usual inference.
///
/// `x` is the full design matrix including any intercept column -- this
/// function does not add one, so callers control the parameterisation.
pub fn ols(x: &Matrix, y: &[f64]) -> Option<OlsFit> {
    let n = x.rows;
    let k = x.cols;
    // n == k is permitted: the fit is exact (zero residual) and the degrees of
    // freedom are floored at 1 below, matching numpy.linalg.lstsq behaviour.
    if y.len() != n || n < k {
        return None;
    }
    let beta = lstsq(x, y)?;

    let resid: Vec<f64> = (0..n)
        .map(|i| y[i] - (0..k).map(|j| x.at(i, j) * beta[j]).sum::<f64>())
        .collect();
    let ss_res: f64 = resid.iter().map(|r| r * r).sum();
    let my = ds::mean(y);
    let ss_tot: f64 = y.iter().map(|v| (v - my) * (v - my)).sum();
    let df = ((n - k) as f64).max(1.0);
    let sigma2 = ss_res / df;

    // cov(beta) = sigma^2 (X'X)^-1, via pinv so rank-deficient designs degrade
    // gracefully instead of raising.
    let xt = x.transpose();
    let xtx = crate::linalg::matmul(&xt, x)?;
    let xtx_inv = crate::linalg::inv(&xtx).unwrap_or_else(|| pinv(&xtx, 1e-15));

    let mut cov = vec![0.0; k * k];
    let mut se = vec![0.0; k];
    let mut tv = vec![0.0; k];
    let mut pv = vec![0.0; k];
    for a in 0..k {
        for b in 0..k {
            cov[a * k + b] = sigma2 * xtx_inv.at(a, b);
        }
        se[a] = cov[a * k + a].max(0.0).sqrt();
        tv[a] = if se[a] > 0.0 {
            beta[a] / se[a]
        } else {
            f64::NAN
        };
        pv[a] = if se[a] > 0.0 {
            2.0 * dist::t_sf(tv[a].abs(), df)
        } else {
            f64::NAN
        };
    }

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        f64::NAN
    };
    let adj = if ss_tot > 0.0 && n > k {
        1.0 - (1.0 - r2) * (n as f64 - 1.0) / df
    } else {
        f64::NAN
    };

    Some(OlsFit {
        coefficients: beta,
        std_errors: se,
        t_values: tv,
        p_values: pv,
        residuals: resid,
        r_squared: r2,
        adj_r_squared: adj,
        df_resid: df,
        sigma2,
        cov,
    })
}

/// One-sample t test against `mu`.
pub fn ttest_1samp(x: &[f64], mu: f64) -> (f64, f64) {
    let n = x.len() as f64;
    if x.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let m = ds::mean(x);
    let se = ds::std_dev(x, 1) / n.sqrt();
    if se == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let t = (m - mu) / se;
    (t, 2.0 * dist::t_sf(t.abs(), n - 1.0))
}

/// Two-sample t test. `equal_var = false` gives Welch's version.
pub fn ttest_ind(a: &[f64], b: &[f64], equal_var: bool) -> (f64, f64) {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    if a.len() < 2 || b.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let (ma, mb) = (ds::mean(a), ds::mean(b));
    let (va, vb) = (ds::variance(a, 1), ds::variance(b, 1));

    let (t, df) = if equal_var {
        let df = na + nb - 2.0;
        let sp2 = ((na - 1.0) * va + (nb - 1.0) * vb) / df;
        let se = (sp2 * (1.0 / na + 1.0 / nb)).sqrt();
        ((ma - mb) / se, df)
    } else {
        // Welch-Satterthwaite degrees of freedom.
        let se2 = va / na + vb / nb;
        let df =
            se2 * se2 / ((va / na) * (va / na) / (na - 1.0) + (vb / nb) * (vb / nb) / (nb - 1.0));
        ((ma - mb) / se2.sqrt(), df)
    };
    (t, 2.0 * dist::t_sf(t.abs(), df))
}

/// Paired-sample t test.
pub fn ttest_rel(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.len() != b.len() {
        return (f64::NAN, f64::NAN);
    }
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    ttest_1samp(&d, 0.0)
}

/// One-way ANOVA across `groups`.
pub fn f_oneway(groups: &[Vec<f64>]) -> (f64, f64) {
    let k = groups.len();
    if k < 2 {
        return (f64::NAN, f64::NAN);
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total <= k {
        return (f64::NAN, f64::NAN);
    }
    let grand: f64 = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;

    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for g in groups {
        let m = ds::mean(g);
        ss_between += g.len() as f64 * (m - grand) * (m - grand);
        ss_within += g.iter().map(|v| (v - m) * (v - m)).sum::<f64>();
    }
    let df_b = (k - 1) as f64;
    let df_w = (n_total - k) as f64;
    if ss_within == 0.0 {
        return (f64::INFINITY, 0.0);
    }
    let f = (ss_between / df_b) / (ss_within / df_w);
    (f, dist::f_sf(f, df_b, df_w))
}

/// Chi-squared test of independence on a row-major contingency table.
///
/// Returns `(chi2, p, dof, expected)`. Applies Yates' continuity correction on
/// 2x2 tables, matching `scipy.stats.chi2_contingency`'s default.
pub fn chi2_contingency(table: &Matrix, correction: bool) -> (f64, f64, f64, Vec<f64>) {
    let (r, c) = (table.rows, table.cols);
    let total: f64 = table.data.iter().sum();
    let row_sums: Vec<f64> = (0..r)
        .map(|i| (0..c).map(|j| table.at(i, j)).sum())
        .collect();
    let col_sums: Vec<f64> = (0..c)
        .map(|j| (0..r).map(|i| table.at(i, j)).sum())
        .collect();

    let mut expected = vec![0.0; r * c];
    let mut chi2 = 0.0;
    let use_yates = correction && r == 2 && c == 2;
    for i in 0..r {
        for j in 0..c {
            let e = row_sums[i] * col_sums[j] / total;
            expected[i * c + j] = e;
            if e > 0.0 {
                let mut d = (table.at(i, j) - e).abs();
                if use_yates {
                    d = (d - 0.5).max(0.0);
                }
                chi2 += d * d / e;
            }
        }
    }
    let dof = ((r - 1) * (c - 1)) as f64;
    (chi2, dist::chi2_sf(chi2, dof), dof, expected)
}
