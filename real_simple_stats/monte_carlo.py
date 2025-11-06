"""Monte Carlo simulation methods for statistical inference and forecasting.

This module provides functions for Monte Carlo simulations including
geometric Brownian motion for financial modeling and general-purpose
simulation techniques.
"""

from typing import List, Tuple, Callable, Dict, Optional, Union
import numpy as np
from scipy import stats

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@jit(nopython=True)
def _gbm_simulation_jit(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_simulations: int,
    seed: int,
) -> np.ndarray:
    """JIT-compiled Geometric Brownian Motion simulation.

    Args:
        S0: Initial value
        mu: Drift (expected return)
        sigma: Volatility (standard deviation)
        T: Time horizon
        n_steps: Number of time steps
        n_simulations: Number of simulation paths
        seed: Random seed

    Returns:
        Array of shape (n_steps+1, n_simulations) with simulated paths
    """
    np.random.seed(seed)
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S0

    for i in range(n_simulations):
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal()
            paths[t, i] = paths[t - 1, i] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

    return paths


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_simulations: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict[str, any]:
    """Simulate paths using Geometric Brownian Motion.

    Geometric Brownian Motion (GBM) is commonly used to model stock prices
    and other financial time series. The model assumes:
    dS = μS dt + σS dW

    where:
    - S is the asset price
    - μ (mu) is the drift (expected return)
    - σ (sigma) is the volatility
    - dW is a Wiener process (Brownian motion)

    Args:
        S0: Initial value (e.g., current stock price)
        mu: Drift coefficient (expected return, annualized)
        sigma: Volatility coefficient (standard deviation, annualized)
        T: Time horizon in years (e.g., 1.0 for one year)
        n_steps: Number of time steps (default: 252 for trading days)
        n_simulations: Number of simulation paths
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - paths: Array of simulated paths (n_steps+1, n_simulations)
            - times: Array of time points
            - final_values: Final values from all simulations
            - mean_path: Mean across all simulations
            - percentiles: 5th, 25th, 50th, 75th, 95th percentiles
            - statistics: Summary statistics

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> # Simulate stock price for 1 year
        >>> result = geometric_brownian_motion(
        ...     S0=100,           # Current price $100
        ...     mu=0.10,          # 10% expected annual return
        ...     sigma=0.20,       # 20% annual volatility
        ...     T=1.0,            # 1 year
        ...     n_steps=252,      # Daily steps
        ...     n_simulations=1000
        ... )
        >>> print(f"Expected final price: ${result['statistics']['mean']:.2f}")
        >>> print(f"95% CI: ${result['percentiles'][5]:.2f} - ${result['percentiles'][95]:.2f}")
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T <= 0:
        raise ValueError("T must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")

    seed = random_seed if random_seed is not None else np.random.randint(0, 2**31)

    # Use JIT-compiled version if available and beneficial
    if NUMBA_AVAILABLE and n_simulations >= 100:
        paths = _gbm_simulation_jit(S0, mu, sigma, T, n_steps, n_simulations, seed)
    else:
        # Standard NumPy implementation
        np.random.seed(seed)
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_simulations))
        paths[0] = S0

        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_simulations)
            paths[t] = paths[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

    # Calculate statistics
    times = np.linspace(0, T, n_steps + 1)
    final_values = paths[-1, :]
    mean_path = np.mean(paths, axis=1)

    percentiles = {
        5: np.percentile(final_values, 5),
        25: np.percentile(final_values, 25),
        50: np.percentile(final_values, 50),
        75: np.percentile(final_values, 75),
        95: np.percentile(final_values, 95),
    }

    statistics = {
        "mean": float(np.mean(final_values)),
        "median": float(np.median(final_values)),
        "std": float(np.std(final_values)),
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
    }

    return {
        "paths": paths,
        "times": times,
        "final_values": final_values,
        "mean_path": mean_path,
        "percentiles": percentiles,
        "statistics": statistics,
    }


def monte_carlo_from_data(
    data: List[float],
    n_steps: int,
    n_simulations: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict[str, any]:
    """Run Monte Carlo simulation using parameters estimated from historical data.

    This function estimates drift (mu) and volatility (sigma) from historical
    data and uses them to simulate future paths using Geometric Brownian Motion.

    Args:
        data: Historical data (e.g., stock prices)
        n_steps: Number of steps to simulate into the future
        n_simulations: Number of simulation paths
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing simulation results and estimated parameters

    Raises:
        ValueError: If data is insufficient

    Examples:
        >>> # Simulate future stock prices based on historical data
        >>> historical_prices = [100, 102, 101, 105, 103, 107, 110]
        >>> result = monte_carlo_from_data(
        ...     data=historical_prices,
        ...     n_steps=30,  # Simulate 30 days ahead
        ...     n_simulations=1000
        ... )
        >>> print(f"Estimated drift: {result['parameters']['mu']:.4f}")
        >>> print(f"Estimated volatility: {result['parameters']['sigma']:.4f}")
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 values")

    data_array = np.asarray(data)

    # Calculate log returns
    log_returns = np.log(data_array[1:] / data_array[:-1])

    # Estimate parameters
    mu = np.mean(log_returns)
    sigma = np.std(log_returns, ddof=1)

    # Adjust drift for bias correction
    drift = mu - (0.5 * sigma**2)

    # Run simulation
    S0 = data_array[-1]
    T = n_steps / 252  # Assume daily data, convert to years

    result = geometric_brownian_motion(
        S0=S0,
        mu=drift,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    # Add estimated parameters to result
    result["parameters"] = {
        "mu": float(mu),
        "sigma": float(sigma),
        "drift": float(drift),
        "S0": float(S0),
    }

    return result


def monte_carlo_integration(
    func: Callable[[np.ndarray], np.ndarray],
    lower_bounds: Union[float, List[float]],
    upper_bounds: Union[float, List[float]],
    n_samples: int = 10000,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate integral using Monte Carlo integration.

    Monte Carlo integration estimates the integral of a function by
    randomly sampling points in the integration domain and computing
    the average function value.

    Args:
        func: Function to integrate (must accept numpy arrays)
        lower_bounds: Lower bounds for each dimension
        upper_bounds: Upper bounds for each dimension
        n_samples: Number of random samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - integral: Estimated integral value
            - std_error: Standard error of the estimate
            - confidence_interval: 95% confidence interval

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> # Integrate x^2 from 0 to 1 (analytical answer: 1/3)
        >>> result = monte_carlo_integration(
        ...     func=lambda x: x**2,
        ...     lower_bounds=0,
        ...     upper_bounds=1,
        ...     n_samples=10000
        ... )
        >>> print(f"Estimated integral: {result['integral']:.4f}")
        >>> print(f"True value: {1/3:.4f}")
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    # Convert bounds to arrays
    if not isinstance(lower_bounds, (list, np.ndarray)):
        lower_bounds = [lower_bounds]
    if not isinstance(upper_bounds, (list, np.ndarray)):
        upper_bounds = [upper_bounds]

    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("lower_bounds and upper_bounds must have same length")

    n_dims = len(lower_bounds)

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random samples
    samples = np.random.uniform(
        lower_bounds,
        upper_bounds,
        size=(n_samples, n_dims) if n_dims > 1 else n_samples,
    )

    # Evaluate function
    if n_dims == 1:
        function_values = func(samples)
    else:
        function_values = func(samples.T)

    # Calculate volume of integration domain
    volume = np.prod(upper_bounds - lower_bounds)

    # Estimate integral
    integral_estimate = volume * np.mean(function_values)
    std_error = volume * np.std(function_values, ddof=1) / np.sqrt(n_samples)

    # 95% confidence interval
    ci_lower = integral_estimate - 1.96 * std_error
    ci_upper = integral_estimate + 1.96 * std_error

    return {
        "integral": float(integral_estimate),
        "std_error": float(std_error),
        "confidence_interval": (float(ci_lower), float(ci_upper)),
    }


def monte_carlo_probability(
    condition: Callable[[np.ndarray], np.ndarray],
    lower_bounds: Union[float, List[float]],
    upper_bounds: Union[float, List[float]],
    n_samples: int = 10000,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate probability using Monte Carlo simulation.

    Estimates P(condition is True) by randomly sampling points
    and computing the fraction that satisfy the condition.

    Args:
        condition: Function that returns True/False for each sample
        lower_bounds: Lower bounds for sampling
        upper_bounds: Upper bounds for sampling
        n_samples: Number of random samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - probability: Estimated probability
            - std_error: Standard error
            - confidence_interval: 95% confidence interval

    Examples:
        >>> # Estimate P(x^2 + y^2 <= 1) for x,y in [0,1]
        >>> # This estimates pi/4
        >>> result = monte_carlo_probability(
        ...     condition=lambda xy: xy[0]**2 + xy[1]**2 <= 1,
        ...     lower_bounds=[0, 0],
        ...     upper_bounds=[1, 1],
        ...     n_samples=10000
        ... )
        >>> pi_estimate = result['probability'] * 4
        >>> print(f"Estimated pi: {pi_estimate:.4f}")
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    # Convert bounds to arrays
    if not isinstance(lower_bounds, (list, np.ndarray)):
        lower_bounds = [lower_bounds]
    if not isinstance(upper_bounds, (list, np.ndarray)):
        upper_bounds = [upper_bounds]

    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    n_dims = len(lower_bounds)

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random samples
    samples = np.random.uniform(
        lower_bounds,
        upper_bounds,
        size=(n_samples, n_dims) if n_dims > 1 else n_samples,
    )

    # Evaluate condition
    if n_dims == 1:
        satisfies_condition = condition(samples)
    else:
        satisfies_condition = condition(samples.T)

    # Estimate probability
    probability = np.mean(satisfies_condition)
    std_error = np.sqrt(probability * (1 - probability) / n_samples)

    # 95% confidence interval
    ci_lower = max(0, probability - 1.96 * std_error)
    ci_upper = min(1, probability + 1.96 * std_error)

    return {
        "probability": float(probability),
        "std_error": float(std_error),
        "confidence_interval": (float(ci_lower), float(ci_upper)),
        "n_successes": int(np.sum(satisfies_condition)),
        "n_samples": int(n_samples),
    }


__all__ = [
    "geometric_brownian_motion",
    "monte_carlo_from_data",
    "monte_carlo_integration",
    "monte_carlo_probability",
]
