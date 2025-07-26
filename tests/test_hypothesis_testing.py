import unittest
from real_simple_stats.hypothesis_testing import (
    t_score,
    f_test,
    critical_value_z,
    critical_value_t,
    p_value_method,
    reject_null,
    state_null_hypothesis,
)


class TestHypothesisTesting(unittest.TestCase):

    def setUp(self):
        self.sample_data = [23, 25, 28, 30, 32, 35, 38, 40]
        self.large_sample = list(range(1, 101))  # 1 to 100

    def test_t_score(self):
        """Test t-score calculation."""
        from real_simple_stats.descriptive_statistics import mean, sample_std_dev

        sample_mean = mean(self.sample_data)
        sample_std = sample_std_dev(self.sample_data)
        mu_null = 30
        n = len(self.sample_data)

        t_stat = t_score(sample_mean, mu_null, sample_std, n)

        # Should return reasonable values
        self.assertIsInstance(t_stat, float)

    def test_t_score_exact_mean(self):
        """Test t-score when sample mean equals null hypothesis."""
        from real_simple_stats.descriptive_statistics import mean, sample_std_dev

        # Create data with known mean
        data = [5, 5, 5, 5, 5]
        sample_mean = mean(data)
        sample_std = sample_std_dev(data)
        n = len(data)

        # When sample std is 0, we can't calculate t-score
        # This will raise an error, which is expected
        with self.assertRaises(ZeroDivisionError):
            t_score(sample_mean, 5, sample_std, n)

    def test_critical_value_z(self):
        """Test critical z-value calculation."""
        alpha = 0.05

        # Two-tailed test
        z_crit = critical_value_z(alpha, "two-tailed")
        self.assertIsInstance(z_crit, float)
        self.assertGreater(z_crit, 0)
        self.assertAlmostEqual(z_crit, 1.96, places=2)

        # One-tailed test
        z_crit_one = critical_value_z(alpha, "right-tailed")
        self.assertAlmostEqual(z_crit_one, 1.645, places=2)

    def test_critical_value_t(self):
        """Test critical t-value calculation."""
        alpha = 0.05
        df = 10

        # Two-tailed test
        t_crit = critical_value_t(alpha, df, "two-tailed")
        self.assertIsInstance(t_crit, float)
        self.assertGreater(t_crit, 0)

        # Should be larger than z-critical for same alpha
        z_crit = critical_value_z(alpha, "two-tailed")
        self.assertGreater(t_crit, z_crit)

    def test_f_test(self):
        """Test F-test calculation."""
        var1 = 25.0
        var2 = 16.0

        f_stat = f_test(var1, var2)
        expected_f = 25.0 / 16.0

        self.assertAlmostEqual(f_stat, expected_f, places=6)

    def test_p_value_method(self):
        """Test p-value calculation from test statistic."""
        test_stat = 2.0

        # Two-tailed test
        p_val = p_value_method(test_stat, "two-tailed")
        self.assertIsInstance(p_val, float)
        self.assertGreaterEqual(p_val, 0)
        self.assertLessEqual(p_val, 1)

        # Right-tailed should give smaller p-value for positive test stat
        p_val_right = p_value_method(test_stat, "right-tailed")
        self.assertLess(p_val_right, p_val)

    def test_reject_null(self):
        """Test null hypothesis rejection decision."""
        alpha = 0.05

        # Should reject when p < alpha
        self.assertTrue(reject_null(0.01, alpha))

        # Should not reject when p >= alpha
        self.assertFalse(reject_null(0.10, alpha))
        self.assertFalse(reject_null(0.05, alpha))  # Equal case

    def test_state_hypotheses(self):
        """Test hypothesis statement functions."""
        null_desc = "μ = 50"

        null_statement = state_null_hypothesis(null_desc)
        self.assertEqual(null_statement, "H0: μ = 50")

        # Test that we can create hypothesis statements
        self.assertIsInstance(null_statement, str)
        self.assertIn("H0:", null_statement)

    def test_invalid_test_type(self):
        """Test error handling for invalid test types."""
        with self.assertRaises(ValueError):
            p_value_method(2.0, "invalid-test-type")


if __name__ == "__main__":
    unittest.main()
