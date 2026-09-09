//! Monte Carlo path simulation.

use crate::rng::stream;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Geometric Brownian motion paths.
///
/// Returns `(n_steps + 1) * n_simulations` values in row-major order, so row
/// `t` holds every simulation's price at time `t` -- the layout the Python
/// layer reports as `paths`.
///
/// Each simulation gets its own derived RNG stream, so results are identical
/// no matter how the work is distributed across cores.
pub fn gbm_paths(
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_sims: usize,
    seed: u64,
) -> Vec<f64> {
    let dt = t / n_steps as f64;
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let rows = n_steps + 1;

    // Simulate per column, then transpose into row-major on the way out.
    let columns: Vec<Vec<f64>> = (0..n_sims)
        .into_par_iter()
        .map(|i| {
            let mut rng = stream(seed, i as u64);
            let normal = Normal::new(0.0, 1.0).expect("valid normal");
            let mut col = Vec::with_capacity(rows);
            let mut price = s0;
            col.push(price);
            for _ in 1..rows {
                let z: f64 = normal.sample(&mut rng);
                price *= (drift + vol * z).exp();
                col.push(price);
            }
            col
        })
        .collect();

    let mut out = vec![0.0; rows * n_sims];
    for (i, col) in columns.iter().enumerate() {
        for (t_idx, &v) in col.iter().enumerate() {
            out[t_idx * n_sims + i] = v;
        }
    }
    out
}

/// Uniform quasi-independent samples over a box, for Monte Carlo integration.
///
/// Returns `n_samples * n_dims` values in row-major order.
pub fn uniform_box(lower: &[f64], upper: &[f64], n_samples: usize, seed: u64) -> Vec<f64> {
    let d = lower.len();
    (0..n_samples)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut rng = stream(seed, i as u64);
            (0..d)
                .map(|j| lower[j] + (upper[j] - lower[j]) * rng.gen::<f64>())
                .collect::<Vec<f64>>()
        })
        .collect()
}
