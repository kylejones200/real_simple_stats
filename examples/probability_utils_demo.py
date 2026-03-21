"""Demo: Probability rules - Bayes, combinations, conditional probability."""

import logging

from real_simple_stats.probability_utils import (
    bayes_theorem,
    combinations,
    conditional_probability,
    expected_value,
    joint_probability,
    permutations,
    probability_not,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Complement rule: P(not A) = 1 - P(A)
logger.info("P(not rain) if P(rain)=0.3: %.2f", probability_not(0.3))

# Joint probability (independent events)
p_a, p_b = 0.5, 0.4
logger.info("P(A and B) if independent: %.2f", joint_probability(p_a, p_b))

# Conditional probability: P(A|B) = P(A and B) / P(B)
p_a_and_b, p_b_given = 0.2, 0.5
logger.info("P(A|B): %.2f", conditional_probability(p_a_and_b, p_b_given))

# Bayes' theorem: P(A|B) from P(B|A), P(A), P(B)
p_b_given_a, p_a_prior, p_b_total = 0.9, 0.01, 0.1
posterior = bayes_theorem(p_b_given_a, p_a_prior, p_b_total)
logger.info("Bayes: P(A|B) = %.4f", posterior)

# Combinatorics
n, k = 10, 3
logger.info("C(10,3) = %s", combinations(n, k))
logger.info("P(10,3) = %s", permutations(n, k))

# Expected value
values = [0, 100, 500]
probs = [0.7, 0.2, 0.1]
logger.info("Expected value: %.2f", expected_value(values, probs))
