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

// ---------------------------------------------------------------- sampler --

use rand_distr::{Beta, Binomial, Distribution, Exp, Gamma, LogNormal, Normal, Poisson};

/// The library's random number generator.
///
/// A PCG64 stream plus the variate families the statistics modules need.
/// Seeded construction is fully reproducible across platforms and runs.
pub struct Sampler {
    rng: Pcg64,
}

impl Sampler {
    pub fn new(seed: u64) -> Sampler {
        Sampler {
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// Seed from entropy, for when the caller passes no seed.
    pub fn from_entropy() -> Sampler {
        Sampler {
            rng: Pcg64::from_entropy(),
        }
    }

    pub fn uniform01(&mut self) -> f64 {
        self.rng.gen::<f64>()
    }

    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.rng.gen::<f64>()
    }

    pub fn integer(&mut self, low: i64, high: i64) -> i64 {
        if high <= low {
            return low;
        }
        self.rng.gen_range(low..high)
    }

    pub fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        match Normal::new(mean, sd) {
            Ok(d) => d.sample(&mut self.rng),
            Err(_) => f64::NAN,
        }
    }

    pub fn exponential(&mut self, scale: f64) -> f64 {
        match Exp::new(1.0 / scale) {
            Ok(d) => d.sample(&mut self.rng),
            Err(_) => f64::NAN,
        }
    }

    pub fn lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        match LogNormal::new(mu, sigma) {
            Ok(d) => d.sample(&mut self.rng),
            Err(_) => f64::NAN,
        }
    }

    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        match Gamma::new(shape, scale) {
            Ok(d) => d.sample(&mut self.rng),
            Err(_) => f64::NAN,
        }
    }

    pub fn beta(&mut self, a: f64, b: f64) -> f64 {
        match Beta::new(a, b) {
            Ok(d) => d.sample(&mut self.rng),
            Err(_) => f64::NAN,
        }
    }

    pub fn poisson(&mut self, lambda: f64) -> f64 {
        match Poisson::new(lambda) {
            Ok(d) => d.sample(&mut self.rng),
            Err(_) => f64::NAN,
        }
    }

    pub fn binomial(&mut self, n: u64, p: f64) -> f64 {
        match Binomial::new(n, p) {
            Ok(d) => d.sample(&mut self.rng) as f64,
            Err(_) => f64::NAN,
        }
    }

    pub fn weibull(&mut self, shape: f64, scale: f64) -> f64 {
        // Inverse-CDF: scale * (-ln(1-u))^(1/shape)
        let u = self.uniform01();
        scale * (-(1.0 - u).ln()).powf(1.0 / shape)
    }

    /// Fill `out` with a uniform sample of `src` drawn with replacement.
    pub fn choice_with_replacement(&mut self, src: &[f64], out: &mut [f64]) {
        let n = src.len();
        for slot in out.iter_mut() {
            *slot = src[self.rng.gen_range(0..n)];
        }
    }

    /// Sample `k` distinct elements (partial Fisher-Yates).
    pub fn choice_without_replacement(&mut self, src: &[f64], k: usize) -> Vec<f64> {
        let mut pool: Vec<f64> = src.to_vec();
        let k = k.min(pool.len());
        for i in 0..k {
            let j = self.rng.gen_range(i..pool.len());
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.rng.gen_range(0..=i);
            idx.swap(i, j);
        }
        idx
    }

    pub fn shuffle_slice(&mut self, v: &mut [f64]) {
        shuffle(v, &mut self.rng)
    }
}
