//! Bootstrap, permutation, and jackknife resampling.
//!
//! The statistic is selected by enum rather than passed as a callback: calling
//! back into Python once per iteration would reacquire the GIL 10,000 times and
//! throw away the entire speedup. Custom Python statistics still work -- the
//! binding layer falls back to a Python loop for those -- but the common cases
//! run entirely in Rust and in parallel.

use crate::descriptive as ds;
use crate::rng::{sample_with_replacement, shuffle, stream};
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stat {
    Mean,
    Median,
    Std,
    Var,
    Min,
    Max,
    Sum,
}

impl Stat {
    /// Parse the statistic name used by the Python layer.
    pub fn from_name(s: &str) -> Option<Stat> {
        Some(match s {
            "mean" => Stat::Mean,
            "median" => Stat::Median,
            "std" => Stat::Std,
            "var" => Stat::Var,
            "min" => Stat::Min,
            "max" => Stat::Max,
            "sum" => Stat::Sum,
            _ => return None,
        })
    }

    /// Apply to a scratch buffer. `Median` is allowed to reorder it.
    pub fn apply(self, buf: &mut [f64]) -> f64 {
        match self {
            Stat::Mean => ds::mean(buf),
            Stat::Median => median_inplace(buf),
            Stat::Std => ds::std_dev(buf, 1),
            Stat::Var => ds::variance(buf, 1),
            Stat::Min => ds::min(buf),
            Stat::Max => ds::max(buf),
            Stat::Sum => ds::sum(buf),
        }
    }
}

/// Median via quickselect: O(n) per call instead of the O(n log n) of a full
/// sort. Over 10,000 bootstrap iterations that difference is the whole cost.
pub fn median_inplace(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        *select_nth(buf, n / 2)
    } else {
        let hi = *select_nth(buf, n / 2);
        // The lower half is now everything left of n/2; its max is the partner.
        let lo = buf[..n / 2]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

fn select_nth(buf: &mut [f64], k: usize) -> &mut f64 {
    let (_, nth, _) =
        buf.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    nth
}

/// Bootstrap distribution of `stat` over `n_iterations` resamples.
pub fn bootstrap(data: &[f64], stat: Stat, n_iterations: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    (0..n_iterations)
        .into_par_iter()
        .map(|i| {
            // Independent stream per iteration => thread-count independent.
            let mut rng = stream(seed, i as u64);
            let mut buf = vec![0.0f64; n];
            sample_with_replacement(&mut buf, data, &mut rng);
            stat.apply(&mut buf)
        })
        .collect()
}

/// Jackknife (leave-one-out) values of `stat`.
pub fn jackknife(data: &[f64], stat: Stat) -> Vec<f64> {
    let n = data.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut buf = Vec::with_capacity(n - 1);
            buf.extend_from_slice(&data[..i]);
            buf.extend_from_slice(&data[i + 1..]);
            stat.apply(&mut buf)
        })
        .collect()
}

/// Permutation distribution of the difference in a statistic between two groups.
///
/// The pooled sample is shuffled and re-split at the original group sizes.
pub fn permutation_diff(
    a: &[f64],
    b: &[f64],
    stat: Stat,
    n_permutations: usize,
    seed: u64,
) -> Vec<f64> {
    let na = a.len();
    let mut pooled = Vec::with_capacity(na + b.len());
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);

    (0..n_permutations)
        .into_par_iter()
        .map(|i| {
            let mut rng = stream(seed, i as u64);
            let mut p = pooled.clone();
            shuffle(&mut p, &mut rng);
            let (left, right) = p.split_at_mut(na);
            stat.apply(left) - stat.apply(right)
        })
        .collect()
}

/// Percentile confidence interval from a bootstrap distribution.
pub fn percentile_ci(dist: &[f64], confidence: f64) -> (f64, f64) {
    let sorted = ds::sorted_copy(dist);
    let alpha = 1.0 - confidence;
    (
        ds::quantile_linear_sorted(&sorted, alpha / 2.0),
        ds::quantile_linear_sorted(&sorted, 1.0 - alpha / 2.0),
    )
}

/// Two-sided p-value: the share of permuted statistics at least as extreme as
/// the observed one, with the standard +1 correction so it is never exactly 0.
pub fn permutation_pvalue(dist: &[f64], observed: f64, alternative: &str) -> f64 {
    let n = dist.len() as f64;
    let count = match alternative {
        "greater" => dist.iter().filter(|&&v| v >= observed).count(),
        "less" => dist.iter().filter(|&&v| v <= observed).count(),
        _ => dist
            .iter()
            .filter(|&&v| v.abs() >= observed.abs())
            .count(),
    } as f64;
    (count + 1.0) / (n + 1.0)
}
