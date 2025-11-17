import math
from typing import Sequence, Tuple, Dict

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

    Raises:
        ValueError: If probabilities are not between 0 and 1

    Example:
        >>> joint_probability(0.5, 0.3)
        0.15
    """
    if not 0 <= p_a <= 1:
        raise ValueError("p_a must be between 0 and 1")
    if not 0 <= p_b <= 1:
        raise ValueError("p_b must be between 0 and 1")
    return p_a * p_b


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """Calculate conditional probability P(A|B) = P(A and B) / P(B).

    Args:
        p_a_and_b: Joint probability P(A and B)
        p_b: Probability of event B

    Returns:
        Conditional probability P(A|B)

    Raises:
        ValueError: If P(B) is zero or probabilities are invalid

    Example:
        >>> conditional_probability(0.15, 0.3)
        0.5
    """
    if not 0 <= p_a_and_b <= 1:
        raise ValueError("p_a_and_b must be between 0 and 1")
    if not 0 <= p_b <= 1:
        raise ValueError("p_b must be between 0 and 1")
    if p_b == 0:
        raise ValueError("Cannot divide by zero: P(B) cannot be 0")
    if p_a_and_b > p_b:
        raise ValueError("P(A and B) cannot be greater than P(B)")
    return p_a_and_b / p_b


def mutually_exclusive(p_a: float, p_b: float) -> float:
    """Returns P(A or B) for mutually exclusive events.
    
    Args:
        p_a: Probability of event A
        p_b: Probability of event B
        
    Returns:
        P(A or B) = P(A) + P(B)
        
    Raises:
        ValueError: If probabilities are not between 0 and 1, or sum exceeds 1
        
    Example:
        >>> mutually_exclusive(0.3, 0.4)
        0.7
    """
    if not 0 <= p_a <= 1:
        raise ValueError("p_a must be between 0 and 1")
    if not 0 <= p_b <= 1:
        raise ValueError("p_b must be between 0 and 1")
    result = p_a + p_b
    if result > 1:
        raise ValueError("For mutually exclusive events, P(A) + P(B) cannot exceed 1")
    return result


def general_addition_rule(p_a: float, p_b: float, p_a_and_b: float) -> float:
    """Returns P(A or B) = P(A) + P(B) - P(A and B)
    
    Args:
        p_a: Probability of event A
        p_b: Probability of event B
        p_a_and_b: Joint probability P(A and B)
        
    Returns:
        P(A or B)
        
    Raises:
        ValueError: If probabilities are not between 0 and 1, or relationships are invalid
        
    Example:
        >>> general_addition_rule(0.3, 0.4, 0.1)
        0.6
    """
    if not 0 <= p_a <= 1:
        raise ValueError("p_a must be between 0 and 1")
    if not 0 <= p_b <= 1:
        raise ValueError("p_b must be between 0 and 1")
    if not 0 <= p_a_and_b <= 1:
        raise ValueError("p_a_and_b must be between 0 and 1")
    if p_a_and_b > min(p_a, p_b):
        raise ValueError("P(A and B) cannot be greater than min(P(A), P(B))")
    result = p_a + p_b - p_a_and_b
    if result > 1:
        raise ValueError("P(A or B) cannot exceed 1")
    return result


# --- COUNTING PRINCIPLE AND COMBINATORICS ---


def fundamental_counting(outcomes: Sequence[int]) -> int:
    """Multiplies choices across stages to get total outcomes.
    
    Args:
        outcomes: Sequence of number of choices at each stage
        
    Returns:
        Total number of possible outcomes
        
    Raises:
        ValueError: If any outcome count is not positive
        
    Example:
        >>> fundamental_counting([4, 3, 2, 5])
        120
    """
    if not outcomes:
        raise ValueError("outcomes sequence cannot be empty")
    result = 1
    for o in outcomes:
        if o <= 0:
            raise ValueError(f"All outcome counts must be positive, got {o}")
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
    """Computes P(A|B) using Bayes' Theorem.
    
    Args:
        p_b_given_a: Conditional probability P(B|A)
        p_a: Prior probability P(A)
        p_b: Prior probability P(B)
        
    Returns:
        Posterior probability P(A|B)
        
    Raises:
        ValueError: If probabilities are not between 0 and 1, or P(B) is zero
        
    Example:
        >>> bayes_theorem(0.9, 0.01, 0.05)
        0.18
    """
    if not 0 <= p_b_given_a <= 1:
        raise ValueError("p_b_given_a must be between 0 and 1")
    if not 0 <= p_a <= 1:
        raise ValueError("p_a must be between 0 and 1")
    if not 0 <= p_b <= 1:
        raise ValueError("p_b must be between 0 and 1")
    if p_b == 0:
        raise ValueError("Cannot divide by zero: P(B) cannot be 0")
    return (p_b_given_a * p_a) / p_b


# --- PROBABILITY TREES ---


def probability_tree(branches: Sequence[Tuple[float, float]]) -> float:
    """Calculates total probability of desired outcomes through tree branches.

    Args:
        branches: list of tuples (P(path1), P(subpath|path1))

    Returns:
        Total probability of reaching desired outcome.
        
    Raises:
        ValueError: If probabilities are not between 0 and 1, or result exceeds 1
        
    Example:
        >>> probability_tree([(0.5, 0.7), (0.25, 0.25), (0.25, 0.25)])
        0.5
    """
    if not branches:
        raise ValueError("branches sequence cannot be empty")
    result = 0.0
    for p1, p2 in branches:
        if not 0 <= p1 <= 1:
            raise ValueError(f"Path probability must be between 0 and 1, got {p1}")
        if not 0 <= p2 <= 1:
            raise ValueError(f"Subpath probability must be between 0 and 1, got {p2}")
        result += p1 * p2
    if result > 1:
        raise ValueError(f"Total probability cannot exceed 1, got {result}")
    return result


# --- DISCRETE PROBABILITY DISTRIBUTIONS ---


def probability_distribution_table(
    values: Sequence[int], probabilities: Sequence[float]
) -> Dict[int, float]:
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("Probabilities must sum to 1")
    return dict(zip(values, probabilities))


def expected_value(values: Sequence[float], probabilities: Sequence[float]) -> float:
    """Calculate the expected value of a discrete random variable.
    
    Args:
        values: Possible values of the random variable
        probabilities: Corresponding probabilities for each value
        
    Returns:
        Expected value E[X] = Σ(x * P(x))
        
    Raises:
        ValueError: If lengths don't match, probabilities are invalid, or don't sum to 1
        
    Example:
        >>> expected_value([0, 1, 2], [0.1, 0.3, 0.6])
        1.5
    """
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have the same length")
    if not values:
        raise ValueError("values sequence cannot be empty")
    
    prob_sum = sum(probabilities)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(f"Probabilities must sum to 1, got {prob_sum}")
    
    for p in probabilities:
        if not 0 <= p <= 1:
            raise ValueError(f"All probabilities must be between 0 and 1, got {p}")
    
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
