import math
from typing import List, Tuple, Dict

# --- BASIC PROBABILITY FUNCTIONS ---


def probability_not(p: float) -> float:
    """Calculate the probability of an event NOT happening.

    Args:
        p: Probability of the event (between 0 and 1)

    Returns:
        Probability of the complement event

    Raises:
        ValueError: If p is not between 0 and 1

    Example:
        >>> probability_not(0.3)
        0.7
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability must be between 0 and 1")
    return 1 - p


def joint_probability(p_a: float, p_b: float) -> float:
    """Calculate the joint probability P(A and B) for independent events.

    Args:
        p_a: Probability of event A
        p_b: Probability of event B

    Returns:
        Joint probability P(A and B) = P(A) × P(B)

    Example:
        >>> joint_probability(0.5, 0.3)
        0.15
    """
    return p_a * p_b


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """Calculate conditional probability P(A|B) = P(A and B) / P(B).

    Args:
        p_a_and_b: Joint probability P(A and B)
        p_b: Probability of event B

    Returns:
        Conditional probability P(A|B)

    Raises:
        ValueError: If P(B) is zero

    Example:
        >>> conditional_probability(0.15, 0.3)
        0.5
    """
    if p_b == 0:
        raise ValueError("Cannot divide by zero: P(B) cannot be 0")
    return p_a_and_b / p_b


def mutually_exclusive(p_a: float, p_b: float) -> float:
    """Returns P(A or B) for mutually exclusive events."""
    return p_a + p_b


def general_addition_rule(p_a: float, p_b: float, p_a_and_b: float) -> float:
    """Returns P(A or B) = P(A) + P(B) - P(A and B)"""
    return p_a + p_b - p_a_and_b


# --- COUNTING PRINCIPLE AND COMBINATORICS ---


def fundamental_counting(outcomes: List[int]) -> int:
    """Multiplies choices across stages to get total outcomes."""
    result = 1
    for o in outcomes:
        result *= o
    return result


def combinations(n: int, k: int) -> int:
    """Calculate the number of combinations (n choose k).

    Args:
        n: Total number of items
        k: Number of items to choose

    Returns:
        Number of ways to choose k items from n items

    Raises:
        ValueError: If n < 0, k < 0, or k > n

    Example:
        >>> combinations(5, 2)
        10
    """
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        raise ValueError("k cannot be greater than n")
    return math.comb(n, k)


def permutations(n: int, k: int) -> int:
    """Calculate the number of permutations of k items from n.

    Args:
        n: Total number of items
        k: Number of items to arrange

    Returns:
        Number of ways to arrange k items from n items

    Raises:
        ValueError: If n < 0, k < 0, or k > n

    Example:
        >>> permutations(5, 2)
        20
    """
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        raise ValueError("k cannot be greater than n")
    return math.perm(n, k)


# --- BAYES' THEOREM ---


def bayes_theorem(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """Computes P(A|B) using Bayes' Theorem."""
    if p_b == 0:
        raise ValueError("Cannot divide by zero")
    return (p_b_given_a * p_a) / p_b


# --- PROBABILITY TREES ---


def probability_tree(branches: List[Tuple[float, float]]) -> float:
    """Calculates total probability of desired outcomes through tree branches.

    Args:
        branches: list of tuples (P(path1), P(subpath|path1))

    Returns:
        Total probability of reaching desired outcome.
    """
    return sum(p1 * p2 for p1, p2 in branches)


# --- DISCRETE PROBABILITY DISTRIBUTIONS ---


def probability_distribution_table(
    values: List[int], probabilities: List[float]
) -> Dict[int, float]:
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("Probabilities must sum to 1")
    return dict(zip(values, probabilities))


def expected_value(values: List[float], probabilities: List[float]) -> float:
    return sum(v * p for v, p in zip(values, probabilities))


# Example usage
if __name__ == "__main__":
    print("Probability not happening:", probability_not(0.4))
    print("Joint probability of A and B:", joint_probability(0.8, 0.5))
    print("Conditional probability P(A|B):", conditional_probability(0.25, 0.5))
    print("Mutually exclusive OR:", mutually_exclusive(0.3, 0.4))
    print("General addition rule:", general_addition_rule(0.3, 0.4, 0.1))

    print("Combinations (5 choose 3):", combinations(5, 3))
    print("Permutations (5P3):", permutations(5, 3))
    print(
        "Counting meals:", fundamental_counting([4, 3, 2, 5])
    )  # Sandwich, side, dessert, drink

    print("Bayes' Theorem:", bayes_theorem(0.7, 0.5, 0.4))

    tree_branches = [(0.5, 0.7), (0.25, 0.25), (0.25, 0.25)]
    print("Tree probability (passenger plane):", probability_tree(tree_branches))

    values = [0, 1, 2, 3]
    probs = [0.1, 0.3, 0.4, 0.2]
    dist = probability_distribution_table(values, probs)
    print("Probability distribution:", dist)
    print("Expected value:", expected_value(values, probs))
