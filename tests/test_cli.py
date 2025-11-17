"""Tests for CLI module."""

from io import StringIO
from unittest.mock import patch

import pytest

from real_simple_stats.cli import (
    descriptive_stats_command,
    glossary_command,
    hypothesis_test_command,
    main,
    parse_numbers,
    probability_command,
)


class TestParseNumbers:
    def test_parse_comma_separated(self):
        result = parse_numbers("1,2,3,4,5")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_space_separated(self):
        result = parse_numbers("1 2 3 4 5")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_with_spaces_and_commas(self):
        result = parse_numbers("1, 2, 3, 4, 5")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_parse_floats(self):
        result = parse_numbers("1.5, 2.7, 3.2")
        assert result == [1.5, 2.7, 3.2]

    def test_parse_negative_numbers(self):
        result = parse_numbers("-1, -2, 3, 4")
        assert result == [-1.0, -2.0, 3.0, 4.0]

    def test_parse_invalid_format(self):
        with pytest.raises(ValueError):
            parse_numbers("1, 2, abc, 4")


class TestDescriptiveStatsCommand:
    @patch("sys.stdout", new_callable=StringIO)
    def test_mean_calculation(self, mock_stdout):
        class Args:
            data = "1,2,3,4,5"
            stat = "mean"
            all = False

        descriptive_stats_command(Args())
        output = mock_stdout.getvalue()
        assert "Mean: 3.0" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_median_calculation(self, mock_stdout):
        class Args:
            data = "1,2,3,4,5"
            stat = "median"
            all = False

        descriptive_stats_command(Args())
        output = mock_stdout.getvalue()
        assert "Median: 3" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_std_calculation(self, mock_stdout):
        class Args:
            data = "1,2,3,4,5"
            stat = "std"
            all = False

        descriptive_stats_command(Args())
        output = mock_stdout.getvalue()
        assert "Standard Deviation:" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_std_single_value(self, mock_stdout):
        class Args:
            data = "5"
            stat = "std"
            all = False

        descriptive_stats_command(Args())
        output = mock_stdout.getvalue()
        assert "Error" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_summary_calculation(self, mock_stdout):
        class Args:
            data = "1,2,3,4,5"
            stat = "summary"
            all = False

        descriptive_stats_command(Args())
        output = mock_stdout.getvalue()
        assert "Min:" in output
        assert "Max:" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_all_statistics(self, mock_stdout):
        class Args:
            data = "1,2,3,4,5"
            stat = None
            all = True

        descriptive_stats_command(Args())
        output = mock_stdout.getvalue()
        assert "mean" in output
        assert "median" in output
        assert "std_dev" in output


class TestProbabilityCommand:
    @patch("sys.stdout", new_callable=StringIO)
    def test_binomial_probability(self, mock_stdout):
        class Args:
            type = "binomial"
            n = 10
            k = 3
            p = 0.5
            cdf = False
            x = None
            mean = 0
            std = 1
            p_b_given_a = None
            p_a = None
            p_b = None

        probability_command(Args())
        output = mock_stdout.getvalue()
        assert "P(X = 3)" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_bayes_theorem(self, mock_stdout):
        class Args:
            type = "bayes"
            p_b_given_a = 0.9
            p_a = 0.01
            p_b = 0.05
            n = None
            k = None
            p = None
            cdf = False
            x = None
            mean = 0
            std = 1

        probability_command(Args())
        output = mock_stdout.getvalue()
        assert "P(A|B)" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_normal_pdf(self, mock_stdout):
        class Args:
            type = "normal"
            x = 0.0
            mean = 0
            std = 1
            cdf = False
            n = None
            k = None
            p = None
            p_b_given_a = None
            p_a = None
            p_b = None

        probability_command(Args())
        output = mock_stdout.getvalue()
        assert "PDF(X = 0.0)" in output
        assert "0.398942" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_normal_cdf(self, mock_stdout):
        class Args:
            type = "normal"
            x = 1.96
            mean = 0
            std = 1
            cdf = True
            n = None
            k = None
            p = None
            p_b_given_a = None
            p_a = None
            p_b = None

        probability_command(Args())
        output = mock_stdout.getvalue()
        assert "P(X ≤ 1.96)" in output
        assert "0.975002" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_normal_missing_x(self, mock_stdout):
        class Args:
            type = "normal"
            x = None
            mean = 0
            std = 1
            cdf = False
            n = None
            k = None
            p = None
            p_b_given_a = None
            p_a = None
            p_b = None

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --x is required" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_binomial_missing_n(self, mock_stdout):
        class Args:
            type = "binomial"
            n = None
            k = 3
            p = 0.5
            cdf = False
            x = None
            mean = 0
            std = 1
            p_b_given_a = None
            p_a = None
            p_b = None

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --n (number of trials) is required" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_binomial_missing_k(self, mock_stdout):
        class Args:
            type = "binomial"
            n = 10
            k = None
            p = 0.5
            cdf = False
            x = None
            mean = 0
            std = 1
            p_b_given_a = None
            p_a = None
            p_b = None

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --k (number of successes) is required" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_binomial_missing_p(self, mock_stdout):
        class Args:
            type = "binomial"
            n = 10
            k = 3
            p = None
            cdf = False
            x = None
            mean = 0
            std = 1
            p_b_given_a = None
            p_a = None
            p_b = None

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --p (probability of success) is required" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_binomial_invalid_k(self, mock_stdout):
        class Args:
            type = "binomial"
            n = 10
            k = 15
            p = 0.5
            cdf = False
            x = None
            mean = 0
            std = 1
            p_b_given_a = None
            p_a = None
            p_b = None

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --k (number of successes) must be between 0 and 10" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_binomial_invalid_p(self, mock_stdout):
        class Args:
            type = "binomial"
            n = 10
            k = 3
            p = 1.5
            cdf = False
            x = None
            mean = 0
            std = 1
            p_b_given_a = None
            p_a = None
            p_b = None

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --p (probability of success) must be between 0 and 1" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_bayes_missing_args(self, mock_stdout):
        class Args:
            type = "bayes"
            p_b_given_a = None
            p_a = 0.01
            p_b = 0.05
            n = None
            k = None
            p = None
            cdf = False
            x = None
            mean = 0
            std = 1

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --p_b_given_a is required" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_bayes_invalid_p_b(self, mock_stdout):
        class Args:
            type = "bayes"
            p_b_given_a = 0.9
            p_a = 0.01
            p_b = 0
            n = None
            k = None
            p = None
            cdf = False
            x = None
            mean = 0
            std = 1

        with pytest.raises(SystemExit):
            probability_command(Args())
        output = mock_stdout.getvalue()
        assert "Error: --p_b cannot be zero" in output


class TestHypothesisTestCommand:
    @patch("sys.stdout", new_callable=StringIO)
    def test_t_test_with_mu(self, mock_stdout):
        class Args:
            data = "23,25,28,30,32"
            type = "t_test"
            mu = 30.0
            alpha = 0.05

        hypothesis_test_command(Args())
        output = mock_stdout.getvalue()
        assert "One-sample t-test" in output
        assert "Null hypothesis mean: 30.0" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_t_test_without_mu(self, mock_stdout):
        class Args:
            data = "23,25,28,30,32"
            type = "t_test"
            mu = None
            alpha = 0.05

        hypothesis_test_command(Args())
        output = mock_stdout.getvalue()
        assert "Error" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_t_test_custom_alpha(self, mock_stdout):
        class Args:
            data = "23,25,28,30,32"
            type = "t_test"
            mu = 30.0
            alpha = 0.01

        hypothesis_test_command(Args())
        output = mock_stdout.getvalue()
        assert "α = 0.01" in output


class TestGlossaryCommand:
    @patch("sys.stdout", new_callable=StringIO)
    def test_lookup_existing_term(self, mock_stdout):
        class Args:
            term = "mean"

        glossary_command(Args())
        output = mock_stdout.getvalue()
        assert "MEAN" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_lookup_nonexistent_term(self, mock_stdout):
        class Args:
            term = "nonexistent_term_xyz"

        glossary_command(Args())
        output = mock_stdout.getvalue()
        assert "not found" in output


class TestMainCLI:
    @patch("sys.argv", ["rss-calc", "stats", "--data", "1,2,3,4,5", "--stat", "mean"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_stats_mean(self, mock_stdout):
        main()
        output = mock_stdout.getvalue()
        assert "Mean: 3.0" in output

    @patch("sys.argv", ["rss-calc", "stats", "--data", "1,2,3,4,5", "--all"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_stats_all(self, mock_stdout):
        main()
        output = mock_stdout.getvalue()
        assert "mean" in output

    @patch(
        "sys.argv",
        [
            "rss-calc",
            "prob",
            "--type",
            "binomial",
            "--n",
            "10",
            "--k",
            "3",
            "--p",
            "0.5",
        ],
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_prob_binomial(self, mock_stdout):
        main()
        output = mock_stdout.getvalue()
        assert "P(X = 3)" in output

    @patch(
        "sys.argv",
        [
            "rss-calc",
            "prob",
            "--type",
            "bayes",
            "--p_b_given_a",
            "0.9",
            "--p_a",
            "0.01",
            "--p_b",
            "0.05",
        ],
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_prob_bayes(self, mock_stdout):
        main()
        output = mock_stdout.getvalue()
        assert "P(A|B)" in output

    @patch(
        "sys.argv",
        [
            "rss-calc",
            "test",
            "--data",
            "23,25,28,30,32",
            "--type",
            "t_test",
            "--mu",
            "30",
        ],
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_test_t_test(self, mock_stdout):
        main()
        output = mock_stdout.getvalue()
        assert "One-sample t-test" in output

    @patch("sys.argv", ["rss-calc", "glossary", "--term", "mean"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_glossary(self, mock_stdout):
        main()
        output = mock_stdout.getvalue()
        assert "MEAN" in output

    @patch("sys.argv", ["rss-calc"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_no_command(self, mock_stdout):
        main()
        output = mock_stdout.getvalue()
        # Should print help when no command is given
        assert len(output) > 0

    @patch("sys.argv", ["rss-calc", "stats", "--data", "invalid", "--stat", "mean"])
    def test_main_invalid_data(self):
        with pytest.raises(SystemExit):
            main()
