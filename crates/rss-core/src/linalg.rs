//! Dense linear algebra, backed by `faer` (pure Rust, no BLAS/LAPACK).
//!
//! Matrices cross the Python boundary as flat row-major `Vec<f64>` plus their
//! shape, which keeps the binding layer free of any array type dependency.

use faer::prelude::{SpSolver, SpSolverLstsq};
use faer::{Mat, MatRef, Side};

/// A row-major dense matrix.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Option<Matrix> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Matrix { rows, cols, data })
    }
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }
    fn to_faer(&self) -> Mat<f64> {
        Mat::from_fn(self.rows, self.cols, |i, j| self.at(i, j))
    }
    fn from_faer(m: MatRef<'_, f64>) -> Matrix {
        let (rows, cols) = (m.nrows(), m.ncols());
        let mut out = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                out.set(i, j, m[(i, j)]);
            }
        }
        out
    }
    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.at(i, j));
            }
        }
        t
    }
}

/// Matrix product A @ B.
pub fn matmul(a: &Matrix, b: &Matrix) -> Option<Matrix> {
    if a.cols != b.rows {
        return None;
    }
    let out = a.to_faer() * b.to_faer();
    Some(Matrix::from_faer(out.as_ref()))
}

/// Inverse via LU with partial pivoting. `None` if singular.
pub fn inv(a: &Matrix) -> Option<Matrix> {
    if a.rows != a.cols {
        return None;
    }
    let m = a.to_faer();
    let lu = m.partial_piv_lu();
    let identity = Mat::<f64>::identity(a.rows, a.rows);
    let x = lu.solve(&identity);
    let out = Matrix::from_faer(x.as_ref());
    if out.data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(out)
}

/// Moore-Penrose pseudo-inverse via SVD, with the standard relative cutoff on
/// small singular values.
pub fn pinv(a: &Matrix, rcond: f64) -> Matrix {
    let m = a.to_faer();
    let svd = m.svd();
    let u = svd.u();
    let v = svd.v();
    let s = svd.s_diagonal();
    let k = s.nrows();
    let smax = (0..k).map(|i| s[i]).fold(0.0f64, f64::max);
    let cutoff = rcond * smax;

    // pinv(A) = V * diag(1/s) * U^T, dropping directions below the cutoff.
    let mut out = Matrix::zeros(a.cols, a.rows);
    for i in 0..a.cols {
        for j in 0..a.rows {
            let mut acc = 0.0;
            for t in 0..k {
                let sv = s[t];
                if sv > cutoff {
                    acc += v[(i, t)] * u[(j, t)] / sv;
                }
            }
            out.set(i, j, acc);
        }
    }
    out
}

/// Least-squares solution of A x = b (minimum residual norm).
///
/// Uses a QR solve when A has full column rank and falls back to the
/// pseudo-inverse when it does not, matching `numpy.linalg.lstsq` behaviour on
/// rank-deficient designs instead of failing.
pub fn lstsq(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    if a.rows != b.len() || a.rows < a.cols {
        // Fall through to pinv for underdetermined systems.
        if a.rows != b.len() {
            return None;
        }
    }
    let m = a.to_faer();
    let rhs = Mat::from_fn(b.len(), 1, |i, _| b[i]);

    if a.rows >= a.cols {
        let qr = m.qr();
        let x = qr.solve_lstsq(&rhs);
        let sol: Vec<f64> = (0..a.cols).map(|i| x[(i, 0)]).collect();
        if sol.iter().all(|v| v.is_finite()) {
            return Some(sol);
        }
    }
    // Rank-deficient or underdetermined: minimum-norm solution.
    let p = pinv(a, 1e-15);
    let sol: Vec<f64> = (0..a.cols)
        .map(|i| (0..a.rows).map(|j| p.at(i, j) * b[j]).sum())
        .collect();
    Some(sol)
}

/// Eigendecomposition of a symmetric matrix, ascending eigenvalues.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors in columns, matching
/// `numpy.linalg.eigh`.
pub fn eigh(a: &Matrix) -> Option<(Vec<f64>, Matrix)> {
    if a.rows != a.cols {
        return None;
    }
    let n = a.rows;
    let m = a.to_faer();
    let e = m.selfadjoint_eigendecomposition(Side::Lower);
    let vals_raw = e.s().column_vector();
    let vecs_raw = e.u();

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        vals_raw[i]
            .partial_cmp(&vals_raw[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let vals: Vec<f64> = idx.iter().map(|&i| vals_raw[i]).collect();
    let mut vecs = Matrix::zeros(n, n);
    for (col, &src) in idx.iter().enumerate() {
        for row in 0..n {
            vecs.set(row, col, vecs_raw[(row, src)]);
        }
    }
    Some((vals, vecs))
}

/// Singular value decomposition. Returns `(U, singular values, V^T)`.
pub fn svd(a: &Matrix) -> (Matrix, Vec<f64>, Matrix) {
    let m = a.to_faer();
    let d = m.svd();
    let s: Vec<f64> = (0..d.s_diagonal().nrows())
        .map(|i| d.s_diagonal()[i])
        .collect();
    (
        Matrix::from_faer(d.u()),
        s,
        Matrix::from_faer(d.v().transpose()),
    )
}

/// Principal square root of a symmetric positive semi-definite matrix.
///
/// Computed through the eigendecomposition (`V diag(sqrt(lambda)) V^T`) rather
/// than a general Denman-Beavers iteration, because every caller here supplies
/// a covariance matrix. Negative eigenvalues from round-off are clamped to zero.
pub fn sqrtm_spd(a: &Matrix) -> Option<Matrix> {
    let (vals, vecs) = eigh(a)?;
    let n = a.rows;
    // V diag(sqrt(lambda)) V^T. Take the square roots once up front rather
    // than n^2 times inside the accumulation. Negative eigenvalues are
    // round-off on a semi-definite matrix, so they clamp to zero.
    let roots: Vec<f64> = vals.iter().map(|v| v.max(0.0).sqrt()).collect();
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let acc = roots
                .iter()
                .enumerate()
                .map(|(k, r)| vecs.at(i, k) * r * vecs.at(j, k))
                .sum();
            out.set(i, j, acc);
        }
    }
    Some(out)
}

/// Column-wise covariance matrix of an observations-by-variables matrix.
pub fn cov_matrix(x: &Matrix, ddof: usize) -> Matrix {
    let (n, p) = (x.rows, x.cols);
    let means: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| x.at(i, j)).sum::<f64>() / n as f64)
        .collect();
    let denom = (n - ddof) as f64;
    let mut c = Matrix::zeros(p, p);
    for j in 0..p {
        for k in j..p {
            let mut acc = 0.0;
            for i in 0..n {
                acc += (x.at(i, j) - means[j]) * (x.at(i, k) - means[k]);
            }
            let v = acc / denom;
            c.set(j, k, v);
            c.set(k, j, v);
        }
    }
    c
}

/// Correlation matrix derived from the covariance matrix.
pub fn corr_matrix(x: &Matrix) -> Matrix {
    let c = cov_matrix(x, 1);
    let p = c.rows;
    let sd: Vec<f64> = (0..p).map(|i| c.at(i, i).sqrt()).collect();
    let mut r = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            let d = sd[i] * sd[j];
            r.set(
                i,
                j,
                if d > 0.0 {
                    (c.at(i, j) / d).clamp(-1.0, 1.0)
                } else {
                    f64::NAN
                },
            );
        }
    }
    r
}
