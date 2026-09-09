//! Spatial statistics: Moran's I and the empirical variogram.
//!
//! Both are O(n^2) over point pairs, which is exactly the kind of work worth
//! moving out of Python. Neither materialises the full distance matrix; pairs
//! are streamed, so memory stays O(n) instead of O(n^2).

use crate::descriptive as ds;
use crate::special::norm_sf;
use rayon::prelude::*;

pub struct MoranResult {
    pub moran_i: f64,
    pub expected_i: f64,
    pub variance_i: f64,
    pub z_score: f64,
    pub p_value: f64,
}

/// Moran's I with binary spatial weights, and its normality-assumption
/// variance (Moran 1950).
///
/// Weights are 1 for distinct points within `threshold` (or all distinct pairs
/// when `threshold` is None), 0 otherwise. The matrix is symmetric with a zero
/// diagonal, which lets S1 and S2 be accumulated from row sums alone rather
/// than from a stored matrix.
pub fn morans_i(
    x: &[f64],
    y: &[f64],
    values: &[f64],
    threshold: Option<f64>,
) -> Result<MoranResult, String> {
    let n = values.len();
    if x.len() != n || y.len() != n {
        return Err("x, y, and values must have the same length.".into());
    }
    if n < 3 {
        return Err("Need at least 3 observations.".into());
    }

    let mean = ds::mean(values);
    let z: Vec<f64> = values.iter().map(|v| v - mean).collect();

    let within = |i: usize, j: usize| -> bool {
        if i == j {
            return false;
        }
        let d = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
        match threshold {
            // Distances are strictly positive for distinct points, matching
            // the original `(D > 0) & (D <= threshold)` mask.
            Some(t) => d > 0.0 && d <= t,
            None => d > 0.0,
        }
    };

    // Row sums double as column sums: the weight matrix is symmetric.
    let row_sums: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| (0..n).filter(|&j| within(i, j)).count() as f64)
        .collect();
    let w_sum: f64 = row_sums.iter().sum();
    if w_sum == 0.0 {
        return Err(
            "Spatial weights matrix is all zeros - no neighbours found. \
             Try increasing distance_threshold."
                .into(),
        );
    }

    let numerator: f64 = (0..n)
        .into_par_iter()
        .map(|i| (0..n).filter(|&j| within(i, j)).map(|j| z[i] * z[j]).sum::<f64>())
        .sum();
    let denominator: f64 = z.iter().map(|v| v * v).sum();
    if denominator == 0.0 {
        return Err("All values are identical; Moran's I is undefined.".into());
    }

    let nf = n as f64;
    let moran_i = (nf / w_sum) * (numerator / denominator);
    let e_i = -1.0 / (nf - 1.0);

    // S1 = 0.5 * sum (w_ij + w_ji)^2 = 2 * sum w_ij^2 for a symmetric 0/1 matrix.
    let s1 = 2.0 * w_sum;
    // S2 = sum_i (row_i + col_i)^2 = 4 * sum_i row_i^2, likewise.
    let s2: f64 = row_sums.iter().map(|r| 4.0 * r * r).sum();

    let m2 = denominator / nf;
    let m4: f64 = z.iter().map(|v| v.powi(4)).sum::<f64>() / nf;
    let b2 = if m2 > 0.0 { m4 / (m2 * m2) } else { 0.0 };

    let a = nf * ((nf * nf - 3.0 * nf + 3.0) * s1 - nf * s2 + 3.0 * w_sum * w_sum);
    let b = b2 * ((nf * nf - nf) * s1 - 2.0 * nf * s2 + 6.0 * w_sum * w_sum);
    let c = (nf - 1.0) * (nf - 2.0) * (nf - 3.0) * w_sum * w_sum;
    let var_i = ((a - b) / c - e_i * e_i).max(1e-12);

    let z_score = (moran_i - e_i) / var_i.sqrt();
    Ok(MoranResult {
        moran_i,
        expected_i: e_i,
        variance_i: var_i,
        z_score,
        p_value: 2.0 * norm_sf(z_score.abs()),
    })
}

pub struct VariogramResult {
    pub lags: Vec<f64>,
    pub gamma: Vec<f64>,
    pub n_pairs: Vec<usize>,
    pub max_lag: f64,
    pub total_variance: f64,
}

/// Empirical semivariogram over equal-width distance bins.
pub fn variogram(
    x: &[f64],
    y: &[f64],
    values: &[f64],
    n_lags: usize,
    max_lag: Option<f64>,
) -> Result<VariogramResult, String> {
    let n = values.len();
    if x.len() != n || y.len() != n {
        return Err("x, y, and values must have the same length.".into());
    }
    if n < 4 {
        return Err("Need at least 4 observations.".into());
    }
    if n_lags < 2 {
        return Err("n_lags must be at least 2.".into());
    }

    // One pass to find the largest pair distance when no cap was supplied.
    let max_lag = match max_lag {
        Some(m) => m,
        None => {
            let dmax = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut local = 0.0f64;
                    for j in i + 1..n {
                        let d = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
                        if d > local {
                            local = d;
                        }
                    }
                    local
                })
                .reduce(|| 0.0f64, f64::max);
            dmax / 2.0
        }
    };

    let width = max_lag / n_lags as f64;
    // Accumulate per-row then combine, so the pair loop parallelises.
    let (sums, counts) = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut s = vec![0.0f64; n_lags];
            let mut c = vec![0usize; n_lags];
            for j in i + 1..n {
                let d = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
                if width <= 0.0 {
                    continue;
                }
                let bin = (d / width).floor();
                if bin < 0.0 || bin >= n_lags as f64 {
                    continue;
                }
                let b = bin as usize;
                let diff = values[i] - values[j];
                s[b] += diff * diff;
                c[b] += 1;
            }
            (s, c)
        })
        .reduce(
            || (vec![0.0; n_lags], vec![0usize; n_lags]),
            |mut acc, item| {
                for k in 0..n_lags {
                    acc.0[k] += item.0[k];
                    acc.1[k] += item.1[k];
                }
                acc
            },
        );

    let lags: Vec<f64> = (0..n_lags).map(|i| (i as f64 + 0.5) * width).collect();
    let gamma: Vec<f64> = (0..n_lags)
        .map(|i| if counts[i] > 0 { 0.5 * sums[i] / counts[i] as f64 } else { 0.0 })
        .collect();

    Ok(VariogramResult {
        lags,
        gamma,
        n_pairs: counts,
        max_lag,
        total_variance: ds::variance(values, 1),
    })
}
