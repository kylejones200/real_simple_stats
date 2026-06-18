"""Demo: market basket analysis.

Run with:  python examples/market_basket_demo.py

Discovers product co-purchase patterns in simulated grocery transactions.
Drawn from the Python for Business Analytics book, Ch. 6.
"""

import numpy as np

import real_simple_stats as rss


def main() -> None:
    rng = np.random.default_rng(42)
    products = [
        "Bread", "Milk", "Eggs", "Butter", "Cheese",
        "Apples", "Bananas", "Diapers", "Beer", "Coffee",
    ]

    # Simulate 500 grocery transactions with two planted associations
    transactions = []
    for _ in range(500):
        size = int(rng.integers(2, 6))
        basket = set(rng.choice(products, size=size, replace=False))
        if rng.random() < 0.28:
            basket.update({"Diapers", "Beer"})      # classic example
        if rng.random() < 0.32:
            basket.update({"Bread", "Butter"})
        if rng.random() < 0.20:
            basket.update({"Milk", "Eggs", "Bread"})
        transactions.append(list(basket))

    # Step 1: encode into a binary transaction matrix
    matrix, items = rss.encode_transactions(transactions)
    print(f"Transactions: {matrix.shape[0]}   Items: {matrix.shape[1]}")
    print(f"Items: {items}")
    print()

    # Step 2: find frequent itemsets
    fs = rss.frequent_itemsets(matrix, items, min_support=0.10)
    print(f"Frequent itemsets (min_support=0.10): {len(fs)}")
    print()
    print(f"{'Itemset':<40} {'Support':>8}")
    print("-" * 50)
    for r in fs[:12]:
        print(f"{str(set(r['itemset'])):<40} {r['support']:>8.3f}")
    print()

    # Step 3: derive association rules
    rules = rss.association_rules(fs, min_confidence=0.55)
    print(f"Association rules (min_confidence=0.55): {len(rules)}")
    print()
    print(f"{'Antecedents':<20} {'Consequents':<20} {'Conf':>6} {'Lift':>6} {'Supp':>6}")
    print("-" * 60)
    for r in rules[:10]:
        ant = ", ".join(sorted(r["antecedents"]))
        con = ", ".join(sorted(r["consequents"]))
        print(
            f"{ant:<20} {con:<20} "
            f"{r['confidence']:>6.3f} {r['lift']:>6.2f} {r['support']:>6.3f}"
        )
    print()

    # Business interpretation
    print("Top insight:")
    if rules:
        top = rules[0]
        ant = " + ".join(sorted(top["antecedents"]))
        con = " + ".join(sorted(top["consequents"]))
        print(
            f"  Customers who buy [{ant}] also buy [{con}] "
            f"{top['confidence']*100:.0f}% of the time "
            f"({top['lift']:.1f}x more than chance)."
        )


if __name__ == "__main__":
    main()
