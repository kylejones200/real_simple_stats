"""Benchmark the Rust backend against NumPy/SciPy and against pure Python.

Run with the dev extra installed (NumPy and SciPy are used here only as
comparison points -- the library itself needs neither):

    python benchmarks/performance_comparison.py

Three columns are reported for each operation:

* **pure python** - the kind of implementation this library shipped before
  0.5.0, kept here so the speedup is measured against something real rather
  than asserted.
* **rss (list)** - the current backend given a plain list. Every element must
  still be unboxed from a Python float, which is the floor on this path.
* **rss (buffer)** - the current backend given a contiguous float64 buffer,
  which is read without copying. This is the fast path.
"""

from __future__ import annotations

import array
import math
import random
import statistics
import time

import real_simple_stats as rss
from real_simple_stats import _rss

try:
    import numpy as np

    NUMPY = True
except ImportError:  # pragma: no cover
    NUMPY = False

try:
    from scipy import stats as scipy_stats

    SCIPY = True
except ImportError:  # pragma: no cover
    SCIPY = False


def timed(fn, budget: float = 1.0, max_reps: int | None = None) -> float:
    """Milliseconds per call, averaged over as many calls as fit in `budget`."""
    fn()  # warm up
    start = time.perf_counter()
    reps = 0
    while time.perf_counter() - start < budget:
        fn()
        reps += 1
        if max_reps and reps >= max_reps:
            break
    return (time.perf_counter() - start) / reps * 1e3


# --- the pre-0.5.0 implementations, for an honest baseline --------------------


def py_variance(v):
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def py_std(v):
    return math.sqrt(py_variance(v))


def py_median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def py_five_number(v):
    s = sorted(v)
    n = len(s)
    mid = n // 2
    lower, upper = s[:mid], s[mid + 1 :] if n % 2 else s[mid:]
    return s[0], py_median(lower), py_median(s), py_median(upper), s[-1]


def py_bootstrap(data, iterations):
    rnd = random.Random(1)
    n = len(data)
    return [statistics.fmean(rnd.choices(data, k=n)) for _ in range(iterations)]


def py_permutation(a, b, iterations):
    rnd = random.Random(1)
    pooled = a + b
    n1 = len(a)
    out = []
    for _ in range(iterations):
        rnd.shuffle(pooled)
        out.append(statistics.fmean(pooled[:n1]) - statistics.fmean(pooled[n1:]))
    return out


def row(label: str, py: float, lst: float | None, buf: float | None) -> None:
    lst_s = f"{lst:10.4f}" if lst is not None else " " * 10
    buf_s = f"{buf:10.4f}" if buf is not None else " " * 10
    best = buf if buf is not None else lst
    speed = f"{py / best:7.1f}x" if best else ""
    print(f"{label:<40} {py:10.4f} {lst_s} {buf_s} {speed:>9}")


def main() -> None:
    random.seed(0)
    print(f"{'operation':<40} {'pure py':>10} {'rss(list)':>10} {'rss(buf)':>10} {'speedup':>9}")
    print("-" * 84)

    for n in (1_000, 100_000, 1_000_000):
        data = [random.gauss(0, 1) for _ in range(n)]
        buf = array.array("d", data)
        tag = f"(n={n:,})"
        row(f"sample_std_dev {tag}", timed(lambda: py_std(data), max_reps=40),
            timed(lambda: rss.sample_std_dev(data)), timed(lambda: rss.sample_std_dev(buf)))
        row(f"median {tag}", timed(lambda: py_median(data), max_reps=40),
            None, timed(lambda: _rss.median(buf)))
        row(f"five_number_summary {tag}", timed(lambda: py_five_number(data), max_reps=40),
            None, timed(lambda: rss.five_number_summary(buf)))
        print()

    sample = [random.gauss(0, 1) for _ in range(500)]
    group_a = [random.gauss(0, 1) for _ in range(300)]
    group_b = [random.gauss(0.3, 1) for _ in range(300)]

    from real_simple_stats.resampling import bootstrap, permutation_test

    for iterations in (2_000, 10_000):
        row(f"bootstrap ({iterations:,} iterations)",
            timed(lambda: py_bootstrap(sample, iterations), max_reps=5),
            timed(lambda: bootstrap(sample, "mean", n_iterations=iterations, random_seed=1),
                  max_reps=25),
            None)
    for permutations in (2_000, 10_000):
        row(f"permutation_test ({permutations:,} perms)",
            timed(lambda: py_permutation(group_a, group_b, permutations), max_reps=5),
            timed(lambda: permutation_test(group_a, group_b, "mean", permutations,
                                           random_seed=1), max_reps=25),
            None)

    if NUMPY:
        print()
        print("Against NumPy on the same million-element buffer (lower is better):")
        data = [random.gauss(0, 1) for _ in range(1_000_000)]
        arr = np.asarray(data)
        for label, ours, theirs in [
            ("std (ddof=1)", lambda: _rss.std_dev(arr, 1), lambda: np.std(arr, ddof=1)),
            ("mean", lambda: _rss.mean(arr), lambda: np.mean(arr)),
            ("median", lambda: _rss.median(arr), lambda: np.median(arr)),
        ]:
            a, b = timed(ours), timed(theirs)
            verdict = "faster" if a < b else "slower"
            print(f"  {label:<16} rss {a:8.4f} ms   numpy {b:8.4f} ms   ({b / a:.1f}x {verdict})")

    if SCIPY:
        print()
        print("Distribution accuracy is gated in tests/parity/, not here;")
        print(f"SciPy {scipy_stats.__name__} is installed and used there as an oracle.")


if __name__ == "__main__":
    main()
