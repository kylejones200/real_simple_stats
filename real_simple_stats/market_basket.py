"""Market basket analysis — association rule mining.

Discovers co-purchase patterns in transactional data:

- :func:`encode_transactions` — convert a list of baskets into a binary
  (transaction × item) matrix.

- :func:`frequent_itemsets` — find all itemsets appearing in at least
  ``min_support`` fraction of transactions (Apriori algorithm).

- :func:`association_rules` — derive ``if {A} then {B}`` rules from frequent
  itemsets, filtered by minimum confidence and optional minimum lift.

All functions return plain Python types (lists of dicts) so results are easy
to inspect or convert: ``import pandas as pd; pd.DataFrame(results)``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "encode_transactions",
    "frequent_itemsets",
    "association_rules",
]


def encode_transactions(
    transactions: Sequence[Iterable[str]],
) -> tuple[np.ndarray, list[str]]:
    """Convert a list of baskets into a binary transaction matrix.

    Args:
        transactions: List of baskets, each basket an iterable of item names
            (strings).  Duplicate items within a basket are ignored.

    Returns:
        A tuple ``(matrix, items)`` where:
            matrix: Boolean numpy array of shape (n_transactions, n_items).
                ``matrix[i, j]`` is True if item ``items[j]`` appears in
                transaction i.
            items: Sorted list of all unique item names.

    Example:
        >>> baskets = [["milk", "bread"], ["milk", "eggs"], ["bread", "eggs"]]
        >>> matrix, items = encode_transactions(baskets)
        >>> items
        ['bread', 'eggs', 'milk']
        >>> matrix.shape
        (3, 3)
    """
    all_items: set[str] = set()
    basket_sets: list[set[str]] = []
    for tx in transactions:
        s = {str(item) for item in tx}
        basket_sets.append(s)
        all_items.update(s)

    items = sorted(all_items)
    item_idx = {item: i for i, item in enumerate(items)}
    n_tx = len(basket_sets)
    n_items = len(items)

    matrix: np.ndarray = np.zeros((n_tx, n_items), dtype=bool)
    for row, basket in enumerate(basket_sets):
        for item in basket:
            matrix[row, item_idx[item]] = True

    return matrix, items


def frequent_itemsets(
    matrix: np.ndarray,
    items: list[str],
    min_support: float,
    max_length: int = 4,
) -> list[dict[str, Any]]:
    """Find all frequent itemsets using the Apriori algorithm.

    An itemset is *frequent* if it appears in at least ``min_support`` fraction
    of transactions.

    Args:
        matrix: Binary transaction matrix from :func:`encode_transactions`,
            shape (n_transactions, n_items).
        items: Item name list from :func:`encode_transactions`.
        min_support: Minimum support threshold in [0, 1].
        max_length: Maximum itemset size to consider (default 4).  Larger values
            are exponentially slower; keep this ≤ 5 for teaching datasets.

    Returns:
        List of dicts, each with keys:
            itemset: frozenset of item names.
            support: Fraction of transactions containing the itemset.
        Sorted by support descending.

    Example:
        >>> import numpy as np
        >>> baskets = [["milk","bread"],["milk","eggs"],["bread","eggs"],["milk","bread","eggs"]]
        >>> mat, its = encode_transactions(baskets)
        >>> fs = frequent_itemsets(mat, its, min_support=0.5)
        >>> {frozenset(["milk","bread"])} <= {r["itemset"] for r in fs}
        True
    """
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be in [0, 1].")

    n_tx, n_items = matrix.shape
    if n_tx == 0:
        return []

    supports: dict[frozenset[str], float] = {}

    # Size-1 candidates
    for j, item in enumerate(items):
        sup = float(matrix[:, j].mean())
        if sup >= min_support:
            supports[frozenset([item])] = sup

    # Size-k candidates (generate from size-(k-1) frequent sets)
    prev_frequent = list(supports.keys())
    for k in range(2, min(max_length, n_items) + 1):
        if not prev_frequent:
            break
        # Collect all unique items appearing in previous frequent sets
        candidate_items = sorted({item for fs in prev_frequent for item in fs})
        new_frequent = []
        for combo in itertools.combinations(candidate_items, k):
            fs = frozenset(combo)
            if fs in supports:
                continue
            col_indices = [items.index(c) for c in combo]
            sup = float(matrix[:, col_indices].all(axis=1).mean())
            if sup >= min_support:
                supports[fs] = sup
                new_frequent.append(fs)
        prev_frequent = new_frequent

    result: list[dict[str, Any]] = [
        {"itemset": itemset, "support": sup}
        for itemset, sup in supports.items()
    ]
    result.sort(key=lambda r: float(r["support"]), reverse=True)
    return result


def association_rules(
    itemsets: list[dict[str, Any]],
    min_confidence: float,
    min_lift: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate association rules from a list of frequent itemsets.

    For every itemset of size ≥ 2, generates all possible antecedent →
    consequent splits and keeps rules meeting the confidence and lift thresholds.

    *Confidence* is P(consequent | antecedent) — how often the rule is correct.
    *Lift* measures how much more likely the consequent is given the antecedent
    compared to chance: lift = confidence / support(consequent).  Lift > 1 means
    the items co-occur more than expected under independence.

    Args:
        itemsets: Output of :func:`frequent_itemsets`.
        min_confidence: Minimum confidence threshold in [0, 1].
        min_lift: Minimum lift threshold (default 0.0 = no filter).

    Returns:
        List of dicts, each with keys:
            antecedents: frozenset of item names.
            consequents: frozenset of item names.
            support: Support of the full itemset.
            confidence: P(consequents | antecedents).
            lift: confidence / support(consequents).
        Sorted by lift descending, then confidence descending.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> baskets = []
        >>> for _ in range(200):
        ...     b = list(rng.choice(["A","B","C","D"], size=3, replace=False))
        ...     if rng.random() < 0.4: b += ["A","B"]
        ...     baskets.append(b)
        >>> mat, its = encode_transactions(baskets)
        >>> fs = frequent_itemsets(mat, its, min_support=0.1)
        >>> rules = association_rules(fs, min_confidence=0.5)
        >>> all("confidence" in r for r in rules)
        True
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1].")

    support_map: dict[frozenset[str], float] = {
        r["itemset"]: r["support"] for r in itemsets
    }

    rules: list[dict[str, Any]] = []
    for itemset, sup in support_map.items():
        if len(itemset) < 2:
            continue
        item_list = list(itemset)
        for r in range(1, len(item_list)):
            for antecedent_tuple in itertools.combinations(item_list, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                sup_ant = support_map.get(antecedent)
                sup_con = support_map.get(consequent)
                if sup_ant is None or sup_ant == 0 or sup_con is None or sup_con == 0:
                    continue
                conf = sup / sup_ant
                lift = conf / sup_con
                if conf >= min_confidence and lift >= min_lift:
                    rules.append(
                        {
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "support": sup,
                            "confidence": conf,
                            "lift": lift,
                        }
                    )

    rules.sort(key=lambda r: (r["lift"], r["confidence"]), reverse=True)
    return rules
