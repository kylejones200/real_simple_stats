import logging

from scipy.stats import expon, geom, nbinom, poisson

logger = logging.getLogger(__name__)

# --- POISSON DISTRIBUTION ---


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for Poisson distribution with rate λ."""
    return float(poisson.pmf(k, lam))


def poisson_cdf(k: int, lam: float) -> float:
    """P(X ≤ k) for Poisson distribution."""
    return float(poisson.cdf(k, lam))


# --- GEOMETRIC DISTRIBUTION ---


def geometric_pmf(k: int, p: float) -> float:
    """P(X = k) for geometric distribution (first success on trial k)."""
    return float(geom.pmf(k, p))


def geometric_cdf(k: int, p: float) -> float:
    """P(X ≤ k) for geometric distribution."""
    return float(geom.cdf(k, p))


# --- EXPONENTIAL DISTRIBUTION ---


def exponential_pdf(x: float, lam: float) -> float:
    """f(x) for exponential distribution. λ = 1/mean"""
    return float(expon.pdf(x, scale=1 / lam))


def exponential_cdf(x: float, lam: float) -> float:
    """P(X ≤ x) for exponential distribution."""
    return float(expon.cdf(x, scale=1 / lam))


# --- NEGATIVE BINOMIAL DISTRIBUTION ---


def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """P(X = k failures before r successes)"""
    return float(nbinom.pmf(k, r, p))


# --- SUMMARY OF EXPECTATIONS AND VARIANCE ---


def expected_value_poisson(lam: float) -> float:
    return lam


def variance_poisson(lam: float) -> float:
    return lam


def expected_value_geometric(p: float) -> float:
    return 1 / p


def variance_geometric(p: float) -> float:
    return (1 - p) / p**2


def expected_value_exponential(lam: float) -> float:
    return 1 / lam


def variance_exponential(lam: float) -> float:
    return 1 / lam**2


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Poisson
    logger.info("Poisson P(X=3), λ=2: %s", poisson_pmf(3, 2))
    logger.info("Poisson P(X≤3), λ=2: %s", poisson_cdf(3, 2))

    # Geometric
    logger.info("Geometric P(X=4), p=0.2: %s", geometric_pmf(4, 0.2))
    logger.info("Geometric P(X≤4), p=0.2: %s", geometric_cdf(4, 0.2))

    # Exponential
    logger.info("Exponential f(x=2), λ=0.5: %s", exponential_pdf(2, 0.5))
    logger.info("Exponential P(X≤2), λ=0.5: %s", exponential_cdf(2, 0.5))

    # Negative Binomial
    logger.info(
        "Negative Binomial P(k=3 failures before 2 successes, p=0.5): %s",
        negative_binomial_pmf(3, 2, 0.5),
    )

    # Expectations
    logger.info("E[X] for Poisson(λ=4): %s", expected_value_poisson(4))
    logger.info("Var[X] for Geometric(p=0.2): %s", variance_geometric(0.2))
