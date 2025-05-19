import math
from typing import List, Dict, Tuple

# --- BASIC PROBABILITY FUNCTIONS ---

def probability_not(p: float) -> float:
    """Returns the probability of an event NOT happening."""
    return 1 - p

def joint_probability(p_a: float, p_b: float) -> float:
    """Returns the joint probability P(A and B) for independent events."""
    return p_a * p_b

def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """Returns P(A|B) = P(A and B) / P(B)"""
    if p_b == 0:
        raise ValueError("Cannot divide by zero")
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
    """Returns number of combinations (n choose k)."""
    return math.comb(n, k)

def permutations(n: int, k: int) -> int:
    """Returns number of permutations of k items from n."""
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

def probability_distribution_table(values: List[int], probabilities: List[float]) -> Dict[int, float]:
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
    print("Counting meals:", fundamental_counting([4, 3, 2, 5]))  # Sandwich, side, dessert, drink

    print("Bayes' Theorem:", bayes_theorem(0.7, 0.5, 0.4))

    tree_branches = [(0.5, 0.7), (0.25, 0.25), (0.25, 0.25)]
    print("Tree probability (passenger plane):", probability_tree(tree_branches))

    values = [0, 1, 2, 3]
    probs = [0.1, 0.3, 0.4, 0.2]
    dist = probability_distribution_table(values, probs)
    print("Probability distribution:", dist)
    print("Expected value:", expected_value(values, probs))
