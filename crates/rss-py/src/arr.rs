//! Zero-copy input adaptation.
//!
//! This layer is where the performance story is won or lost. Unboxing a Python
//! `list[float]` costs ~16 ns/element (about 16 ms for a million values), which
//! is roughly 30x the cost of the arithmetic the kernels then perform. So any
//! object exposing the buffer protocol with contiguous f64 data -- a NumPy
//! array, `array.array('d')`, a memoryview -- is borrowed in place and never
//! copied. Lists still work; they simply pay the unboxing floor.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Owned-or-borrowed float data obtained from a Python object.
pub enum Arr {
    /// Borrowed directly from a contiguous f64 buffer; no copy was made.
    Buf(PyBuffer<f64>),
    /// Materialised from a sequence of Python floats.
    Owned(Vec<f64>),
}

impl Arr {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Fast path: anything exporting a contiguous f64 buffer.
        if let Ok(buf) = PyBuffer::<f64>::get_bound(obj) {
            if buf.is_c_contiguous() {
                return Ok(Arr::Buf(buf));
            }
            // Strided or Fortran-ordered: fall through and copy elementwise.
        }
        // A plain list of floats is the common beginner input. PyO3's generic
        // `extract::<Vec<f64>>` runs the full conversion protocol per element,
        // which is slower than CPython's own `sum()` float fast path -- enough
        // that a naive binding makes `mean(list)` slower than the pure-Python
        // version it replaces. Unbox exact floats directly instead.
        if let Ok(list) = obj.downcast::<PyList>() {
            if let Some(v) = unbox_float_list(list) {
                return Ok(Arr::Owned(v));
            }
        }
        // Slow path: a tuple/iterable, or a list with non-float entries.
        match obj.extract::<Vec<f64>>() {
            Ok(v) => Ok(Arr::Owned(v)),
            Err(_) => Err(PyTypeError::new_err(
                "expected a sequence of numbers, or an object supporting the \
                 buffer protocol with float64 items",
            )),
        }
    }

    /// View the data as a plain slice.
    pub fn as_slice<'a>(&'a self, py: Python<'_>) -> &'a [f64] {
        match self {
            Arr::Buf(b) => {
                // `as_slice` yields `&[ReadOnlyCell<f64>]`, which is
                // `repr(transparent)` over `f64`. We hold the GIL for the whole
                // borrow, so no Python code can mutate or free the buffer while
                // this slice is alive.
                match b.as_slice(py) {
                    Some(cells) => unsafe {
                        std::slice::from_raw_parts(cells.as_ptr() as *const f64, cells.len())
                    },
                    None => &[],
                }
            }
            Arr::Owned(v) => v.as_slice(),
        }
    }

    /// True when the data was borrowed rather than copied. Exposed so the test
    /// suite can assert that the fast path is actually being taken.
    pub fn is_zero_copy(&self) -> bool {
        matches!(self, Arr::Buf(_))
    }
}

/// Convenience: pull a slice out of a Python object and hand it to `f`.
pub fn with_slice<T, F>(py: Python<'_>, obj: &Bound<'_, PyAny>, f: F) -> PyResult<T>
where
    F: FnOnce(&[f64]) -> T,
{
    let a = Arr::from_py(obj)?;
    Ok(f(a.as_slice(py)))
}

/// Same, for two aligned inputs; errors if the lengths differ.
pub fn with_two_slices<T, F>(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    f: F,
) -> PyResult<T>
where
    F: FnOnce(&[f64], &[f64]) -> T,
{
    let xa = Arr::from_py(a)?;
    let xb = Arr::from_py(b)?;
    let sa = xa.as_slice(py);
    let sb = xb.as_slice(py);
    if sa.len() != sb.len() {
        return Err(PyTypeError::new_err(format!(
            "inputs must have the same length; got {} and {}",
            sa.len(),
            sb.len()
        )));
    }
    Ok(f(sa, sb))
}

/// Unbox a list whose items are all exactly `float`.
///
/// Returns `None` on the first non-float item, so the caller can fall back to
/// the general conversion path (which handles ints, numpy scalars, Decimal,
/// objects with `__float__`, and so on).
fn unbox_float_list(list: &Bound<'_, PyList>) -> Option<Vec<f64>> {
    let n = list.len();
    let mut out = Vec::with_capacity(n);
    let ptr = list.as_ptr();
    for i in 0..n {
        unsafe {
            // Borrowed reference; the list holds it alive and we hold the GIL.
            let item = pyo3::ffi::PyList_GetItem(ptr, i as pyo3::ffi::Py_ssize_t);
            if item.is_null() || pyo3::ffi::PyFloat_CheckExact(item) == 0 {
                return None;
            }
            out.push(pyo3::ffi::PyFloat_AsDouble(item));
        }
    }
    Some(out)
}
