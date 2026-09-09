//! Pure-Rust statistics kernels backing `real_simple_stats`.
//!
//! No Python, no BLAS, no SciPy. Everything here operates on plain `&[f64]`
//! slices so the Python binding layer can hand over buffers zero-copy.

pub mod descriptive;
pub mod dist;
pub mod noncentral;
pub mod resample;
pub mod rng;
pub mod special;
