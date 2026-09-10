//! The three optimizers this library actually needs.
//!
//! Deliberately not a general optimization suite: `power_analysis` needs scalar
//! root-finding, `spatial_stats` needs bounded nonlinear least squares for
//! variogram fitting, and `causal_inference` needs the simplex-constrained
//! least squares that defines a synthetic control. Each gets a purpose-built
//! solver rather than a port of SciPy's generic machinery.

/// Brent's method for a root of `f` bracketed by `[a, b]`.
///
/// Combines bisection, secant, and inverse quadratic interpolation: superlinear
/// convergence with bisection's guarantee that the bracket never escapes.
pub fn brentq<F: FnMut(f64) -> f64>(
    mut f: F,
    mut a: f64,
    mut b: f64,
    xtol: f64,
    max_iter: usize,
) -> Option<f64> {
    let mut fa = f(a);
    let mut fb = f(b);
    if fa == 0.0 {
        return Some(a);
    }
    if fb == 0.0 {
        return Some(b);
    }
    if fa * fb > 0.0 {
        return None; // not bracketed
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    for _ in 0..max_iter {
        if fb * fc > 0.0 {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * xtol;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return Some(b);
        }
        if e.abs() < tol || fa.abs() <= fb.abs() {
            // Bisect.
            d = m;
            e = d;
        } else {
            let s = fb / fa;
            let (p, q) = if a == c {
                // Secant.
                (2.0 * m * s, 1.0 - s)
            } else {
                // Inverse quadratic interpolation.
                let q0 = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q0 * (q0 - r) - (b - a) * (r - 1.0)),
                    (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol {
            d
        } else if m > 0.0 {
            tol
        } else {
            -tol
        };
        fb = f(b);
    }
    Some(b)
}

/// Bounded Levenberg-Marquardt for nonlinear least squares.
///
/// `residual(params, out)` fills `out` with (model - observed) / sigma. Bounds
/// are enforced by clamping each trial step back into the box, which is enough
/// for the low-dimensional, well-posed variogram fits this backs.
pub fn levenberg_marquardt<F>(
    mut residual: F,
    p0: &[f64],
    lower: &[f64],
    upper: &[f64],
    n_resid: usize,
    max_iter: usize,
) -> Vec<f64>
where
    F: FnMut(&[f64], &mut [f64]),
{
    let np = p0.len();
    let mut p: Vec<f64> = p0
        .iter()
        .enumerate()
        .map(|(i, &v)| v.clamp(lower[i], upper[i]))
        .collect();

    let mut r = vec![0.0; n_resid];
    residual(&p, &mut r);
    let mut cost: f64 = r.iter().map(|v| v * v).sum();
    let mut lambda = 1e-3;

    let mut jac = vec![0.0; n_resid * np];
    let mut rp = vec![0.0; n_resid];

    for _ in 0..max_iter {
        // Forward-difference Jacobian; the parameter counts here are tiny.
        for j in 0..np {
            let h = (1e-7 * p[j].abs()).max(1e-9);
            let mut pj = p.clone();
            pj[j] = (pj[j] + h).min(upper[j]);
            let hh = pj[j] - p[j];
            if hh == 0.0 {
                for i in 0..n_resid {
                    jac[i * np + j] = 0.0;
                }
                continue;
            }
            residual(&pj, &mut rp);
            for i in 0..n_resid {
                jac[i * np + j] = (rp[i] - r[i]) / hh;
            }
        }

        // Normal equations J^T J + lambda*diag(J^T J), solved by Gaussian
        // elimination -- np is at most a handful here.
        let mut jtj = vec![0.0; np * np];
        let mut jtr = vec![0.0; np];
        for i in 0..n_resid {
            for a in 0..np {
                jtr[a] += jac[i * np + a] * r[i];
                for b in 0..np {
                    jtj[a * np + b] += jac[i * np + a] * jac[i * np + b];
                }
            }
        }

        let mut improved = false;
        for _ in 0..30 {
            let mut m = jtj.clone();
            for a in 0..np {
                m[a * np + a] *= 1.0 + lambda;
                m[a * np + a] += lambda * 1e-12;
            }
            let step = match solve_small(&mut m, &jtr, np) {
                Some(s) => s,
                None => break,
            };
            let cand: Vec<f64> = (0..np)
                .map(|j| (p[j] - step[j]).clamp(lower[j], upper[j]))
                .collect();
            residual(&cand, &mut rp);
            let c_new: f64 = rp.iter().map(|v| v * v).sum();
            if c_new < cost {
                let rel = (cost - c_new) / cost.max(1e-300);
                p = cand;
                r.copy_from_slice(&rp);
                cost = c_new;
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                if rel < 1e-12 {
                    return p;
                }
                break;
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                return p;
            }
        }
        if !improved {
            return p;
        }
    }
    p
}

/// Gaussian elimination with partial pivoting for a small dense system.
fn solve_small(m: &mut [f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut b = rhs.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r * n + col].abs() > m[piv * n + col].abs() {
                piv = r;
            }
        }
        if m[piv * n + col].abs() < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = m[col * n + col];
        for r in col + 1..n {
            let f = m[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                m[r * n + c] -= f * m[col * n + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in i + 1..n {
            acc -= m[i * n + j] * x[j];
        }
        x[i] = acc / m[i * n + i];
    }
    Some(x)
}

/// Euclidean projection onto the probability simplex {w : w >= 0, sum w = 1}.
///
/// Duchi et al. (2008): sort descending, find how many coordinates stay
/// positive after a uniform shift, then apply that shift and clamp. O(k log k)
/// and exact -- not an approximate renormalisation.
pub fn project_to_simplex(v: &[f64]) -> Vec<f64> {
    let k = v.len();
    if k == 0 {
        return vec![];
    }
    let mut u = v.to_vec();
    u.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut css = 0.0;
    let mut rho = 0usize;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            rho = j + 1;
            theta = t;
        }
    }
    if rho == 0 {
        // Degenerate input (all -inf/NaN); fall back to uniform.
        return vec![1.0 / k as f64; k];
    }
    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

/// Minimise ||y - X w||^2 subject to w >= 0 and sum(w) = 1.
///
/// This is exactly the synthetic-control weight problem. Projected gradient
/// descent with the exact simplex projection above: every iterate is feasible
/// by construction, so the equality and non-negativity constraints can never
/// drift or need repair. The step size comes from the Lipschitz constant of the
/// gradient (2 * largest eigenvalue of X'X, found by power iteration), which
/// makes convergence monotone without a line search.
pub fn simplex_least_squares(y: &[f64], x: &[Vec<f64>], max_iter: usize, tol: f64) -> Vec<f64> {
    let k = x.len();
    if k == 0 {
        return vec![];
    }
    let n = y.len();

    // Largest eigenvalue of X'X by power iteration -> gradient Lipschitz bound.
    let mut v = vec![1.0 / (k as f64).sqrt(); k];
    let mut lmax = 1.0;
    for _ in 0..100 {
        // t = X v ; then w = X' t  == (X'X) v
        let mut t = vec![0.0; n];
        for j in 0..k {
            let vj = v[j];
            if vj != 0.0 {
                for i in 0..n {
                    t[i] += vj * x[j][i];
                }
            }
        }
        let mut w2 = vec![0.0; k];
        for j in 0..k {
            w2[j] = (0..n).map(|i| x[j][i] * t[i]).sum();
        }
        let norm = w2.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm <= 0.0 || !norm.is_finite() {
            break;
        }
        for j in 0..k {
            v[j] = w2[j] / norm;
        }
        if (norm - lmax).abs() <= 1e-12 * norm {
            lmax = norm;
            break;
        }
        lmax = norm;
    }
    let step = 1.0 / (2.0 * lmax).max(1e-12);

    let mut w = vec![1.0 / k as f64; k];
    let mut prev_cost = f64::NAN;
    let mut resid = vec![0.0; n];

    for _ in 0..max_iter {
        for i in 0..n {
            let mut fit = 0.0;
            for j in 0..k {
                fit += w[j] * x[j][i];
            }
            resid[i] = fit - y[i];
        }
        let cost: f64 = resid.iter().map(|r| r * r).sum();
        // NOTE: prev_cost starts as NaN, not infinity. Seeding it with infinity
        // makes the very first relative-change test `inf <= tol*inf` -> true,
        // which exits before taking a single step.
        if prev_cost.is_finite() && (prev_cost - cost).abs() <= tol * prev_cost.max(1e-300) {
            break;
        }
        prev_cost = cost;

        // grad = 2 X' (Xw - y)
        let grad: Vec<f64> = (0..k)
            .map(|j| 2.0 * (0..n).map(|i| resid[i] * x[j][i]).sum::<f64>())
            .collect();

        let trial: Vec<f64> = (0..k).map(|j| w[j] - step * grad[j]).collect();
        w = project_to_simplex(&trial);
    }
    w
}

/// Nelder-Mead simplex minimisation for low-dimensional unconstrained problems.
///
/// Derivative-free, which suits maximum-likelihood fits where the gradient is
/// awkward and the parameter count is two or three.
pub fn nelder_mead<F>(mut f: F, x0: &[f64], step: f64, max_iter: usize, tol: f64) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return vec![];
    }
    // Initial simplex: the start point plus one offset vertex per dimension.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += if v[i].abs() > 1e-12 {
            step * v[i].abs()
        } else {
            step
        };
        simplex.push(v);
    }
    let mut fx: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..max_iter {
        // Order vertices best -> worst.
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| {
            fx[a]
                .partial_cmp(&fx[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let simplex_sorted: Vec<Vec<f64>> = idx.iter().map(|&i| simplex[i].clone()).collect();
        let fx_sorted: Vec<f64> = idx.iter().map(|&i| fx[i]).collect();
        simplex = simplex_sorted;
        fx = fx_sorted;

        if (fx[n] - fx[0]).abs() <= tol * (fx[0].abs() + tol) {
            break;
        }

        // Centroid of all but the worst vertex.
        let mut centroid = vec![0.0; n];
        for v in simplex.iter().take(n) {
            for j in 0..n {
                centroid[j] += v[j] / n as f64;
            }
        }

        let reflect: Vec<f64> = (0..n)
            .map(|j| centroid[j] + (centroid[j] - simplex[n][j]))
            .collect();
        let f_reflect = f(&reflect);

        if f_reflect < fx[0] {
            // Expand.
            let expand: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (centroid[j] - simplex[n][j]))
                .collect();
            let f_expand = f(&expand);
            if f_expand < f_reflect {
                simplex[n] = expand;
                fx[n] = f_expand;
            } else {
                simplex[n] = reflect;
                fx[n] = f_reflect;
            }
        } else if f_reflect < fx[n - 1] {
            simplex[n] = reflect;
            fx[n] = f_reflect;
        } else {
            // Contract.
            let contract: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 0.5 * (simplex[n][j] - centroid[j]))
                .collect();
            let f_contract = f(&contract);
            if f_contract < fx[n] {
                simplex[n] = contract;
                fx[n] = f_contract;
            } else {
                // Shrink toward the best vertex. `simplex[0]` is cloned first
                // so the loop is not borrowing it while mutating `simplex[i]`.
                let best = simplex[0].clone();
                for i in 1..=n {
                    for (j, b) in best.iter().enumerate() {
                        simplex[i][j] = b + 0.5 * (simplex[i][j] - b);
                    }
                    fx[i] = f(&simplex[i]);
                }
            }
        }
    }

    let best = (0..=n)
        .min_by(|&a, &b| {
            fx[a]
                .partial_cmp(&fx[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    simplex[best].clone()
}
