//! Descriptive statistics kernels.
//!
//! Summation is pairwise rather than naive: a sequential sum accumulates
//! O(n) rounding error, which is visible by n = 1e6. Pairwise recursion drops
//! that to O(log n) at effectively no cost, matching NumPy's accuracy.

use rayon::prelude::*;

/// Below this length the recursion overhead outweighs the accuracy win.
const PAIRWISE_BLOCK: usize = 128;
/// Below this length threads cost more than they save.
const PAR_THRESHOLD: usize = 50_000;

/// Pairwise (cascade) summation: O(log n) error growth instead of O(n).
pub fn sum(x: &[f64]) -> f64 {
    if x.len() <= PAIRWISE_BLOCK {
        // Unrolled by 4 to break the serial dependency chain on the adder.
        let mut a = 0.0;
        let mut b = 0.0;
        let mut c = 0.0;
        let mut d = 0.0;
        let mut ch = x.chunks_exact(4);
        for k in &mut ch {
            a += k[0];
            b += k[1];
            c += k[2];
            d += k[3];
        }
        let mut tail = 0.0;
        for &v in ch.remainder() {
            tail += v;
        }
        return (a + b) + (c + d) + tail;
    }
    let mid = x.len() / 2;
    sum(&x[..mid]) + sum(&x[mid..])
}

/// Parallel pairwise sum for large inputs.
pub fn sum_par(x: &[f64]) -> f64 {
    if x.len() < PAR_THRESHOLD {
        return sum(x);
    }
    x.par_chunks(16_384).map(sum).sum()
}

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    sum_par(x) / x.len() as f64
}

/// Sum of squared deviations about the mean, computed in two passes.
///
/// The textbook one-pass form `E[x^2] - E[x]^2` cancels catastrophically when
/// the mean is large relative to the spread; two passes cost one extra read
/// and are exact to rounding.
pub fn sum_sq_dev(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let m = mean(x);
    if x.len() < PAR_THRESHOLD {
        sum_sq_dev_block(x, m)
    } else {
        x.par_chunks(16_384).map(|c| sum_sq_dev_block(c, m)).sum()
    }
}

fn sum_sq_dev_block(x: &[f64], m: f64) -> f64 {
    if x.len() <= PAIRWISE_BLOCK {
        let mut a = 0.0;
        let mut b = 0.0;
        let mut ch = x.chunks_exact(2);
        for k in &mut ch {
            let d0 = k[0] - m;
            let d1 = k[1] - m;
            a += d0 * d0;
            b += d1 * d1;
        }
        let mut tail = 0.0;
        for &v in ch.remainder() {
            let d = v - m;
            tail += d * d;
        }
        return a + b + tail;
    }
    let mid = x.len() / 2;
    sum_sq_dev_block(&x[..mid], m) + sum_sq_dev_block(&x[mid..], m)
}

/// Variance with `ddof` degrees of freedom removed (ddof=1 -> sample variance).
pub fn variance(x: &[f64], ddof: usize) -> f64 {
    let n = x.len();
    if n <= ddof {
        return f64::NAN;
    }
    sum_sq_dev(x) / (n - ddof) as f64
}

pub fn std_dev(x: &[f64], ddof: usize) -> f64 {
    variance(x, ddof).sqrt()
}

pub fn min(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::INFINITY, f64::min)
}
pub fn max(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Sort a copy ascending, NaNs last. Uses a parallel sort for large inputs.
pub fn sorted_copy(x: &[f64]) -> Vec<f64> {
    let mut v = x.to_vec();
    if v.len() >= PAR_THRESHOLD {
        v.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    v
}

/// Median of an already-sorted slice.
pub fn median_sorted(s: &[f64]) -> f64 {
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

pub fn median(x: &[f64]) -> f64 {
    median_sorted(&sorted_copy(x))
}

/// Linear-interpolation quantile — NumPy's default `method="linear"`.
pub fn quantile_linear(x: &[f64], q: f64) -> f64 {
    let s = sorted_copy(x);
    quantile_linear_sorted(&s, q)
}

pub fn quantile_linear_sorted(s: &[f64], q: f64) -> f64 {
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return s[0];
    }
    let h = (n as f64 - 1.0) * q.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    s[lo] + frac * (s[hi] - s[lo])
}

/// Tukey (median-of-halves) five-number summary.
///
/// This is deliberately NOT NumPy's linear-interpolation quantile. It is the
/// convention `real_simple_stats` has always used and what its docstrings and
/// tests specify, including the special cases at n <= 3. Swapping in the NumPy
/// convention here would silently change every existing user's Q1/Q3.
pub struct FiveNumber {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
}

pub fn five_number_summary(x: &[f64]) -> Option<FiveNumber> {
    let n = x.len();
    if n == 0 {
        return None;
    }
    let s = sorted_copy(x);
    if n == 1 {
        let v = s[0];
        return Some(FiveNumber {
            min: v,
            q1: v,
            median: v,
            q3: v,
            max: v,
        });
    }
    let med = median_sorted(&s);
    if n == 2 {
        return Some(FiveNumber {
            min: s[0],
            q1: s[0],
            median: med,
            q3: s[1],
            max: s[1],
        });
    }
    if n == 3 {
        return Some(FiveNumber {
            min: s[0],
            q1: s[0],
            median: med,
            q3: s[2],
            max: s[2],
        });
    }
    let mid = n / 2;
    let lower = &s[..mid];
    // Odd n drops the median itself from both halves.
    let upper = if n % 2 == 1 { &s[mid + 1..] } else { &s[mid..] };
    Some(FiveNumber {
        min: s[0],
        q1: median_sorted(lower),
        median: med,
        q3: median_sorted(upper),
        max: s[n - 1],
    })
}

/// Fisher-Pearson standardized moment coefficient (population skewness).
pub fn skewness(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    if m2 <= 0.0 {
        return f64::NAN;
    }
    m3 / m2.powf(1.5)
}

/// Excess kurtosis (normal distribution -> 0).
pub fn kurtosis(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    if m2 <= 0.0 {
        return f64::NAN;
    }
    m4 / (m2 * m2) - 3.0
}

/// Covariance with `ddof` degrees of freedom removed.
pub fn covariance(x: &[f64], y: &[f64], ddof: usize) -> f64 {
    let n = x.len();
    if n != y.len() || n <= ddof {
        return f64::NAN;
    }
    let mx = mean(x);
    let my = mean(y);
    let mut acc = 0.0;
    for i in 0..n {
        acc += (x[i] - mx) * (y[i] - my);
    }
    acc / (n - ddof) as f64
}

/// Pearson product-moment correlation.
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n < 2 {
        return f64::NAN;
    }
    let mx = mean(x);
    let my = mean(y);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    // sqrt(sxx * syy) rather than sqrt(sxx) * sqrt(syy): the latter cannot
    // return exactly 1.0 for a perfectly correlated pair, because squaring a
    // rounded square root does not recover the original product.
    (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
}

/// Mode(s): every value tied for the highest frequency, ascending.
///
/// Floats are keyed by bit pattern after normalizing -0.0 to 0.0, so equal
/// values always land in the same bucket.
pub fn modes(x: &[f64]) -> Vec<f64> {
    use std::collections::HashMap;
    let mut counts: HashMap<u64, (f64, usize)> = HashMap::new();
    for &v in x {
        let key = (if v == 0.0 { 0.0 } else { v }).to_bits();
        let e = counts.entry(key).or_insert((v, 0));
        e.1 += 1;
    }
    let best = counts.values().map(|(_, c)| *c).max().unwrap_or(0);
    let mut out: Vec<f64> = counts
        .values()
        .filter(|(_, c)| *c == best)
        .map(|(v, _)| *v)
        .collect();
    out.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}
