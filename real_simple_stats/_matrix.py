"""Small dense-matrix helpers over plain nested lists.

Matrices are ``list[list[float]]`` -- rows of floats. The heavy operations
(product, inverse, pseudo-inverse, symmetric eigendecomposition, SVD, matrix
square root) are delegated to the Rust backend; what lives here is the shaping,
centering, and elementwise work that would be pointless to cross the FFI
boundary for.

This is internal infrastructure for the multivariate routines, not a public
API, and not a NumPy compatibility layer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import _rss

Matrix = list[list[float]]


def as_matrix(x: Sequence[Sequence[float]] | Sequence[float]) -> Matrix:
    """Coerce rows-of-numbers (or a flat sequence) into a rectangular matrix."""
    rows: Matrix = []
    for row in x:
        if isinstance(row, (int, float)):
            rows.append([float(row)])
        else:
            rows.append([float(v) for v in row])
    if rows:
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("all rows must have the same length")
    return rows


def shape(a: Matrix) -> tuple[int, int]:
    return (len(a), len(a[0]) if a else 0)


def flatten(a: Matrix) -> list[float]:
    return [v for row in a for v in row]


def unflatten(flat: Sequence[float], rows: int, cols: int) -> Matrix:
    return [list(flat[i * cols : (i + 1) * cols]) for i in range(rows)]


def transpose(a: Matrix) -> Matrix:
    n, m = shape(a)
    return [[a[i][j] for i in range(n)] for j in range(m)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    ar, ac = shape(a)
    br, bc = shape(b)
    r, c, flat = _rss.mat_matmul(ar, ac, flatten(a), br, bc, flatten(b))
    return unflatten(flat, r, c)


def matvec(a: Matrix, v: Sequence[float]) -> list[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def col_means(a: Matrix) -> list[float]:
    n, m = shape(a)
    return [_rss.mean([a[i][j] for i in range(n)]) for j in range(m)]


def col_stds(a: Matrix, ddof: int = 0) -> list[float]:
    n, m = shape(a)
    return [_rss.std_dev([a[i][j] for i in range(n)], ddof) for j in range(m)]


def center(a: Matrix) -> tuple[Matrix, list[float]]:
    """Subtract each column's mean. Returns the centered matrix and the means."""
    means = col_means(a)
    return [[v - mu for v, mu in zip(row, means)] for row in a], means


def standardize(a: Matrix) -> tuple[Matrix, list[float], list[float]]:
    """Center and scale each column. Zero-variance columns are left centered."""
    centered, means = center(a)
    stds = col_stds(a, 0)
    safe = [s if s > 0 else 1.0 for s in stds]
    return [[v / s for v, s in zip(row, safe)] for row in centered], means, stds


def cov(a: Matrix, ddof: int = 1) -> Matrix:
    n, m = shape(a)
    return unflatten(_rss.mat_cov(n, m, flatten(a), ddof), m, m)


def corr(a: Matrix) -> Matrix:
    n, m = shape(a)
    return unflatten(_rss.mat_corr(n, m, flatten(a)), m, m)


def inv(a: Matrix) -> Matrix:
    n, _ = shape(a)
    return unflatten(_rss.mat_inv(n, n, flatten(a)), n, n)


def pinv(a: Matrix, rcond: float = 1e-15) -> Matrix:
    n, m = shape(a)
    r, c, flat = _rss.mat_pinv(n, m, flatten(a), rcond)
    return unflatten(flat, r, c)


def inv_or_pinv(a: Matrix) -> Matrix:
    """Inverse where possible, pseudo-inverse where the matrix is singular."""
    try:
        return inv(a)
    except ValueError:
        return pinv(a)


def eigh(a: Matrix) -> tuple[list[float], Matrix]:
    """Symmetric eigendecomposition. Eigenvalues ascending, vectors in columns."""
    n, _ = shape(a)
    vals, vecs = _rss.mat_eigh(n, n, flatten(a))
    return vals, unflatten(vecs, n, n)


def eigh_descending(a: Matrix) -> tuple[list[float], Matrix]:
    """As :func:`eigh`, but ordered from largest eigenvalue to smallest."""
    vals, vecs = eigh(a)
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i], reverse=True)
    sorted_vals = [vals[i] for i in order]
    sorted_vecs = [[vecs[r][i] for i in order] for r in range(len(vecs))]
    return sorted_vals, sorted_vecs


def svd(a: Matrix) -> tuple[Matrix, list[float], Matrix]:
    n, m = shape(a)
    u, s, vt, ur, uc, vr, vc = _rss.mat_svd(n, m, flatten(a))
    return unflatten(u, ur, uc), s, unflatten(vt, vr, vc)


def sqrtm_spd(a: Matrix) -> Matrix:
    """Principal square root of a symmetric positive semi-definite matrix."""
    n, _ = shape(a)
    return unflatten(_rss.mat_sqrtm_spd(n, n, flatten(a)), n, n)


def add_ridge(a: Matrix, eps: float) -> Matrix:
    """Add `eps` to the diagonal, to keep a near-singular matrix invertible."""
    out = [list(row) for row in a]
    for i in range(len(out)):
        out[i][i] += eps
    return out


def take_columns(a: Matrix, k: int) -> Matrix:
    return [row[:k] for row in a]


def norm2(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))
