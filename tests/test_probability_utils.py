import unittest

from real_simple_stats.probability_utils import (
    bayes_theorem,
    combinations,
    conditional_probability,
    expected_value,
    fundamental_counting,
    general_addition_rule,
    joint_probability,
    mutually_exclusive,
    permutations,
    probability_not,
    probability_tree,
)


class TestProbabilityUtils(unittest.TestCase):
    def test_probability_not(self):
        """Test probability of NOT event."""
        result = probability_not(0.3)
        self.assertEqual(result, 0.7)

    def test_probability_not_edge_cases(self):
        """Test edge cases for probability not."""
        # P(not certain) = 0
        self.assertEqual(probability_not(1.0), 0.0)
        # P(not impossible) = 1
        self.assertEqual(probability_not(0.0), 1.0)

    def test_joint_probability_independent(self):
        """Test joint probability for independent events."""
        # P(A and B) = P(A) * P(B) for independent events
        result = joint_probability(p_a=0.5, p_b=0.3)
        self.assertEqual(result, 0.15)

    def test_conditional_probability(self):
        """Test conditional probability calculation."""
        # P(A|B) = P(A and B) / P(B)
        result = conditional_probability(p_a_and_b=0.15, p_b=0.3)
        self.assertEqual(result, 0.5)

    def test_bayes_theorem(self):
        """Test Bayes' theorem calculation."""
        # P(A|B) = P(B|A) * P(A) / P(B)
        result = bayes_theorem(p_b_given_a=0.8, p_a=0.3, p_b=0.4)
        expected = (0.8 * 0.3) / 0.4
        self.assertEqual(result, expected)

    def test_combinations(self):
        """Test combinations calculation."""
        # C(5,2) = 5!/(2!(5-2)!) = 10
        result = combinations(n=5, k=2)
        self.assertEqual(result, 10)

    def test_combinations_edge_cases(self):
        """Test combinations edge cases."""
        # C(n,0) = 1
        self.assertEqual(combinations(5, 0), 1)
        # C(n,n) = 1
        self.assertEqual(combinations(5, 5), 1)

    def test_permutations(self):
        """Test permutations calculation."""
        # P(5,2) = 5!/(5-2)! = 20
        result = permutations(n=5, k=2)
        self.assertEqual(result, 20)

    def test_permutations_edge_cases(self):
        """Test permutations edge cases."""
        # P(n,0) = 1
        self.assertEqual(permutations(5, 0), 1)
        # P(n,n) = n!
        self.assertEqual(permutations(5, 5), 120)

    def test_mutually_exclusive(self):
        """Test mutually exclusive probability."""
        result = mutually_exclusive(p_a=0.3, p_b=0.4)
        self.assertEqual(result, 0.7)

    def test_general_addition_rule(self):
        """Test general addition rule."""
        result = general_addition_rule(p_a=0.5, p_b=0.4, p_a_and_b=0.2)
        self.assertEqual(result, 0.7)  # 0.5 + 0.4 - 0.2

    def test_fundamental_counting(self):
        """Test fundamental counting principle."""
        result = fundamental_counting([3, 4, 2])
        self.assertEqual(result, 24)  # 3 * 4 * 2

    def test_invalid_probability_values(self):
        """Test that invalid probability values are handled."""
        # Test zero division in conditional probability
        with self.assertRaises(ValueError):
            conditional_probability(0.5, 0)

    def test_zero_division_handling(self):
        """Test zero division scenarios."""
        with self.assertRaises(ValueError):
            bayes_theorem(0.8, 0.3, 0)  # P(B) = 0

    def test_joint_probability_validation(self):
        """Test validation in joint_probability."""
        with self.assertRaises(ValueError):
            joint_probability(-0.1, 0.5)
        with self.assertRaises(ValueError):
            joint_probability(0.5, 1.5)

    def test_conditional_probability_validation(self):
        """Test validation in conditional_probability."""
        with self.assertRaises(ValueError):
            conditional_probability(1.5, 0.3)
        with self.assertRaises(ValueError):
            conditional_probability(0.5, 0.3)  # p_a_and_b > p_b

    def test_mutually_exclusive_validation(self):
        """Test validation in mutually_exclusive."""
        with self.assertRaises(ValueError):
            mutually_exclusive(-0.1, 0.5)
        with self.assertRaises(ValueError):
            mutually_exclusive(0.6, 0.5)  # Sum > 1

    def test_general_addition_rule_validation(self):
        """Test validation in general_addition_rule."""
        with self.assertRaises(ValueError):
            general_addition_rule(1.5, 0.4, 0.2)
        with self.assertRaises(ValueError):
            general_addition_rule(0.3, 0.4, 0.5)  # p_a_and_b > min(p_a, p_b)

    def test_fundamental_counting_validation(self):
        """Test validation in fundamental_counting."""
        with self.assertRaises(ValueError):
            fundamental_counting([])
        with self.assertRaises(ValueError):
            fundamental_counting([4, 0, 2])

    def test_bayes_theorem_validation(self):
        """Test validation in bayes_theorem."""
        with self.assertRaises(ValueError):
            bayes_theorem(1.5, 0.3, 0.4)
        with self.assertRaises(ValueError):
            bayes_theorem(0.8, -0.1, 0.4)

    def test_probability_tree_validation(self):
        """Test validation in probability_tree."""
        with self.assertRaises(ValueError):
            probability_tree([])
        with self.assertRaises(ValueError):
            probability_tree([(1.5, 0.7)])
        with self.assertRaises(ValueError):
            probability_tree([(1.0, 1.0), (0.1, 1.0)])  # 1.0 + 0.1 = 1.1 > 1

    def test_expected_value_validation(self):
        """Test validation in expected_value."""
        with self.assertRaises(ValueError):
            expected_value([1, 2], [0.3])  # Length mismatch
        with self.assertRaises(ValueError):
            expected_value([], [])
        with self.assertRaises(ValueError):
            expected_value([1, 2], [0.3, 0.5])  # Sum != 1
        with self.assertRaises(ValueError):
            expected_value([1, 2], [1.5, -0.5])  # Invalid probabilities


if __name__ == "__main__":
    unittest.main()
