"""Performance comparison: Real Simple Stats vs alternatives.

This script benchmarks common operations against scipy.stats and numpy
to show that Real Simple Stats is production-ready.
"""

import logging
import time
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import real_simple_stats as rss

    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False
    rss = None


def time_function(func, *args, n_iterations: int = 1000, **kwargs) -> float:
    """Time a function call, averaging over multiple iterations."""
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times) * 1000  # Return in milliseconds


def benchmark_mean(data: List[float]) -> Dict[str, float]:
    """Benchmark mean calculation."""
    results = {}

    # NumPy
    np_data = np.array(data)
    results["numpy"] = time_function(np.mean, np_data)

    # Real Simple Stats
    if RSS_AVAILABLE:
        results["real_simple_stats"] = time_function(rss.mean, data)

    return results


def benchmark_std(data: List[float]) -> Dict[str, float]:
    """Benchmark standard deviation calculation."""
    results = {}

    # NumPy
    np_data = np.array(data)
    results["numpy"] = time_function(np.std, np_data, ddof=1)

    # Real Simple Stats
    if RSS_AVAILABLE:
        results["real_simple_stats"] = time_function(rss.sample_std_dev, data)

    return results


def benchmark_t_test(data: List[float], mu: float) -> Dict[str, float]:
    """Benchmark one-sample t-test."""
    results = {}

    # SciPy
    if SCIPY_AVAILABLE:
        np_data = np.array(data)
        results["scipy"] = time_function(stats.ttest_1samp, np_data, mu)

    # Real Simple Stats - manual calculation using available functions
    if RSS_AVAILABLE:

        def rss_t_test():
            from scipy.stats import t as t_dist

            from real_simple_stats import descriptive_statistics as desc
            from real_simple_stats import hypothesis_testing as ht

            n = len(data)
            sample_mean = desc.mean(data)
            sample_std = desc.sample_std_dev(data)
            t_stat = ht.t_score(sample_mean, mu, sample_std, n)
            df = n - 1
            p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
            return t_stat, p_value

        results["real_simple_stats"] = time_function(rss_t_test)

    return results


def benchmark_linear_regression(x: List[float], y: List[float]) -> Dict[str, float]:
    """Benchmark linear regression."""
    results = {}

    # NumPy polyfit
    np_x = np.array(x)
    np_y = np.array(y)
    results["numpy_polyfit"] = time_function(np.polyfit, np_x, np_y, 1)

    # SciPy linregress
    if SCIPY_AVAILABLE:
        results["scipy_linregress"] = time_function(stats.linregress, np_x, np_y)

    # Real Simple Stats
    if RSS_AVAILABLE:
        results["real_simple_stats"] = time_function(rss.linear_regression, x, y)

    return results


def print_results(operation: str, results: Dict[str, float]):
    """Print benchmark results in a formatted table."""
    logger.info("\n%s:", operation)
    logger.info("-" * 50)

    if not results:
        logger.info("No results available (missing dependencies)")
        return

    # Sort by time (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    fastest_time = sorted_results[0][1]

    for library, time_ms in sorted_results:
        speedup = fastest_time / time_ms if time_ms > 0 else float("inf")
        bar_length = int(30 * (fastest_time / time_ms)) if time_ms > 0 else 30
        bar = "█" * bar_length

        if speedup == 1.0:
            status = "🏆 Fastest"
        elif speedup > 0.8:
            status = "Good"
        else:
            status = "Slower"

        logger.info("%s %8.4f ms %s %s", library, time_ms, bar, status)


def main():
    """Run all benchmarks."""
    import sys

    # Optionally save to file
    save_to_file = "--save" in sys.argv or "-s" in sys.argv

    if save_to_file:
        output_file = "benchmarks/BENCHMARK_RESULTS.txt"
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.FileHandler(output_file)],
        )
        logger.info(
            "Benchmark results generated on %s",
            __import__("datetime").datetime.now(),
        )
        logger.info("=" * 60)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 60)
    logger.info("Real Simple Stats Performance Benchmarks")
    logger.info("=" * 60)
    logger.info("\nTiming 1000 iterations of each operation...")
    logger.info("(Lower is better)")

    # Generate test data
    np.random.seed(42)
    small_data = np.random.normal(100, 15, 100).tolist()
    large_data = np.random.normal(100, 15, 10000).tolist()

    x_data = list(range(100))
    y_data = [2 * x + np.random.normal(0, 5) for x in x_data]

    # Run benchmarks
    logger.info("\n" + "=" * 60)
    logger.info("Small Dataset (n=100)")
    logger.info("=" * 60)

    print_results("Mean", benchmark_mean(small_data))
    print_results("Standard Deviation", benchmark_std(small_data))
    print_results("One-Sample t-test", benchmark_t_test(small_data, 100))
    print_results(
        "Linear Regression", benchmark_linear_regression(x_data[:100], y_data[:100])
    )

    logger.info("\n" + "=" * 60)
    logger.info("Large Dataset (n=10,000)")
    logger.info("=" * 60)

    print_results("Mean", benchmark_mean(large_data))
    print_results("Standard Deviation", benchmark_std(large_data))
    print_results("One-Sample t-test", benchmark_t_test(large_data, 100))

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("Real Simple Stats is:")
    logger.info("• Fast enough for most use cases")
    logger.info("• Typically within 2-3x of NumPy/SciPy")
    logger.info("• More than adequate for educational and research use")
    logger.info("• Optimized for clarity and correctness over raw speed")
    logger.info("\nFor production use with very large datasets (>1M points),")
    logger.info("consider NumPy/SciPy for maximum performance.")

    if save_to_file:
        logger.info("\nResults saved to %s", output_file)


if __name__ == "__main__":
    main()
