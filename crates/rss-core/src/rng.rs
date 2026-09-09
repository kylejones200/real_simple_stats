//! Deterministic random number generation.
//!
//! Reproducibility is a hard requirement: `random_seed=42` must give the same
//! answer on every machine, every run, and -- critically -- regardless of how
//! many threads rayon happens to use. That rules out sharing one generator
//! across a parallel loop, since the interleaving would vary. Instead each
//! iteration derives its own independent stream from (seed, iteration), so the
//! result depends only on the seed.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// Build the generator for iteration `i` of a job seeded with `seed`.
///
/// The seed is mixed with SplitMix64 so that sequential iteration indices
/// produce well-separated streams rather than correlated ones.
pub fn stream(seed: u64, i: u64) -> Pcg64 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    Pcg64::seed_from_u64(z)
}

/// Fisher-Yates shuffle in place.
pub fn shuffle(v: &mut [f64], rng: &mut Pcg64) {
    for i in (1..v.len()).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}

/// Draw `n` indices uniformly with replacement from `0..n`.
pub fn sample_with_replacement(out: &mut [f64], src: &[f64], rng: &mut Pcg64) {
    let n = src.len();
    for slot in out.iter_mut() {
        *slot = src[rng.gen_range(0..n)];
    }
}
