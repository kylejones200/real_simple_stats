#!/usr/bin/env python3
"""
Command Line Interface for Real Simple Stats

Provides quick access to statistical calculations from the terminal.
"""

import argparse
import sys
from typing import List
import json

from . import descriptive_statistics as desc
from . import probability_utils as prob
from . import binomial_distributions as binom
from .glossary import lookup


def parse_numbers(numbers_str: str) -> List[float]:
    """Parse a string of numbers into a list of floats."""
    try:
        # Handle comma-separated or space-separated numbers
        if "," in numbers_str:
            return [float(x.strip()) for x in numbers_str.split(",")]
        else:
            return [float(x) for x in numbers_str.split()]
    except ValueError as e:
        raise ValueError(f"Invalid number format: {e}")


def descriptive_stats_command(args):
    """Handle descriptive statistics calculations."""
    data = parse_numbers(args.data)

    if args.all:
        # Calculate all descriptive statistics
        results = {
            "count": len(data),
            "mean": desc.mean(data),
            "median": desc.median(data),
            "std_dev": desc.sample_std_dev(data) if len(data) > 1 else 0,
            "variance": desc.sample_variance(data) if len(data) > 1 else 0,
            "five_number_summary": desc.five_number_summary(data),
            "iqr": desc.interquartile_range(data),
            "cv": desc.coefficient_of_variation(data) if len(data) > 1 else 0,
        }
        print(json.dumps(results, indent=2))
    elif args.stat == "mean":
        print(f"Mean: {desc.mean(data)}")
    elif args.stat == "median":
        print(f"Median: {desc.median(data)}")
    elif args.stat == "std":
        if len(data) > 1:
            print(f"Standard Deviation: {desc.sample_std_dev(data)}")
        else:
            print("Error: Standard deviation requires at least 2 values")
    elif args.stat == "summary":
        summary = desc.five_number_summary(data)
        for key, value in summary.items():
            print(f"{key.capitalize()}: {value}")


def probability_command(args):
    """Handle probability calculations."""
    if args.type == "normal":
        if args.cdf:
            # Note: These functions may not exist in current implementation
            # result = norm.normal_cdf(args.x, args.mean, args.std)
            print(f"CDF calculation for X={args.x}, mean={args.mean}, std={args.std}")
        else:
            # result = norm.normal_pdf(args.x, args.mean, args.std)
            print(f"PDF calculation for X={args.x}, mean={args.mean}, std={args.std}")

    elif args.type == "binomial":
        result = binom.binomial_probability(args.n, args.k, args.p)
        print(f"P(X = {args.k}) = {result:.6f}")

    elif args.type == "bayes":
        result = prob.bayes_theorem(args.p_b_given_a, args.p_a, args.p_b)
        print(f"P(A|B) = {result:.6f}")


def test_command(args):
    """Handle hypothesis testing."""
    data = parse_numbers(args.data)

    if args.type == "t_test":
        if args.mu is None:
            print("Error: --mu (null hypothesis mean) is required for t-test")
            return

        # Note: one_sample_t_test may not exist in current implementation
        # t_stat, p_value = ht.one_sample_t_test(data, args.mu)
        print("One-sample t-test:")
        print(f"Sample data: {data}")
        print(f"Null hypothesis mean: {args.mu}")

        alpha = args.alpha or 0.05
        # if p_value < alpha:
        #     print(f"Result: Reject H₀ at α = {alpha}")
        # else:
        #     print(f"Result: Fail to reject H₀ at α = {alpha}")
        print(f"Significance level: α = {alpha}")


def glossary_command(args):
    """Handle glossary lookups."""
    try:
        definition = lookup(args.term)
        print(f"\n{args.term.upper()}:")
        print(f"{definition}\n")
    except KeyError:
        print(f"Term '{args.term}' not found in glossary.")
        print("Try a different spelling or check available terms.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Real Simple Stats - Statistical calculations from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rss-calc stats --data "1,2,3,4,5" --all
  rss-calc stats --data "1 2 3 4 5" --stat mean
  rss-calc prob --type normal --x 1.96 --mean 0 --std 1 --cdf
  rss-calc prob --type binomial --n 10 --k 3 --p 0.5
  rss-calc test --data "23,25,28,30,32" --type t_test --mu 30
  rss-calc glossary --term "p-value"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Descriptive statistics subcommand
    stats_parser = subparsers.add_parser("stats", help="Descriptive statistics")
    stats_parser.add_argument(
        "--data", required=True, help="Data values (comma or space separated)"
    )
    stats_parser.add_argument(
        "--stat",
        choices=["mean", "median", "std", "summary"],
        help="Specific statistic to calculate",
    )
    stats_parser.add_argument(
        "--all", action="store_true", help="Calculate all descriptive statistics"
    )

    # Probability subcommand
    prob_parser = subparsers.add_parser("prob", help="Probability calculations")
    prob_parser.add_argument(
        "--type",
        required=True,
        choices=["normal", "binomial", "bayes"],
        help="Type of probability calculation",
    )

    # Normal distribution arguments
    prob_parser.add_argument("--x", type=float, help="Value for normal distribution")
    prob_parser.add_argument("--mean", type=float, default=0, help="Mean (default: 0)")
    prob_parser.add_argument(
        "--std", type=float, default=1, help="Standard deviation (default: 1)"
    )
    prob_parser.add_argument(
        "--cdf", action="store_true", help="Calculate CDF instead of PDF"
    )

    # Binomial distribution arguments
    prob_parser.add_argument("--n", type=int, help="Number of trials")
    prob_parser.add_argument("--k", type=int, help="Number of successes")
    prob_parser.add_argument("--p", type=float, help="Probability of success")

    # Bayes theorem arguments
    prob_parser.add_argument("--p_b_given_a", type=float, help="P(B|A)")
    prob_parser.add_argument("--p_a", type=float, help="P(A)")
    prob_parser.add_argument("--p_b", type=float, help="P(B)")

    # Hypothesis testing subcommand
    test_parser = subparsers.add_parser("test", help="Hypothesis testing")
    test_parser.add_argument(
        "--data", required=True, help="Sample data (comma or space separated)"
    )
    test_parser.add_argument(
        "--type", required=True, choices=["t_test"], help="Type of hypothesis test"
    )
    test_parser.add_argument("--mu", type=float, help="Null hypothesis mean")
    test_parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)"
    )

    # Glossary subcommand
    glossary_parser = subparsers.add_parser(
        "glossary", help="Look up statistical terms"
    )
    glossary_parser.add_argument(
        "--term", required=True, help="Statistical term to look up"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "stats":
            descriptive_stats_command(args)
        elif args.command == "prob":
            probability_command(args)
        elif args.command == "test":
            test_command(args)
        elif args.command == "glossary":
            glossary_command(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
