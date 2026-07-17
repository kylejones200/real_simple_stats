"""Tests for the market basket analysis module."""

import numpy as np
import pytest

from real_simple_stats.market_basket import (
    association_rules,
    encode_transactions,
    frequent_itemsets,
)

# ---------------------------------------------------------------------------
# encode_transactions
# ---------------------------------------------------------------------------


class TestEncodeTransactions:
    def test_basic_shape(self):
        baskets = [["milk", "bread"], ["milk", "eggs"], ["bread", "eggs"]]
        mat, items = encode_transactions(baskets)
        assert mat.shape == (3, 3)
        assert sorted(items) == ["bread", "eggs", "milk"]

    def test_items_sorted(self):
        baskets = [["z", "a", "m"], ["a", "z"]]
        _, items = encode_transactions(baskets)
        assert items == sorted(items)

    def test_correct_values(self):
        baskets = [["A", "B"], ["B", "C"]]
        mat, items = encode_transactions(baskets)
        idx = {item: i for i, item in enumerate(items)}
        assert mat[0, idx["A"]]
        assert mat[0, idx["B"]]
        assert not mat[0, idx["C"]]
        assert not mat[1, idx["A"]]

    def test_duplicates_within_basket_ignored(self):
        baskets = [["A", "A", "B"]]
        mat, items = encode_transactions(baskets)
        assert mat.shape == (1, 2)

    def test_empty_transactions(self):
        mat, items = encode_transactions([])
        assert mat.shape == (0, 0)
        assert items == []


# ---------------------------------------------------------------------------
# frequent_itemsets
# ---------------------------------------------------------------------------


class TestFrequentItemsets:
    def _make_data(self, seed=42):
        rng = np.random.default_rng(seed)
        products = ["Bread", "Milk", "Eggs", "Butter", "Cheese", "Beer", "Diapers"]
        rows = []
        for _ in range(300):
            size = int(rng.integers(2, 5))
            basket = set(rng.choice(products, size=size, replace=False))
            if rng.random() < 0.25:
                basket.update({"Diapers", "Beer"})
            if rng.random() < 0.3:
                basket.update({"Bread", "Butter"})
            rows.append(list(basket))
        return encode_transactions(rows)

    def test_returns_list_of_dicts(self):
        mat, items = self._make_data()
        fs = frequent_itemsets(mat, items, min_support=0.1)
        assert isinstance(fs, list)
        assert all("itemset" in r and "support" in r for r in fs)

    def test_support_meets_threshold(self):
        mat, items = self._make_data()
        threshold = 0.1
        fs = frequent_itemsets(mat, items, min_support=threshold)
        for r in fs:
            assert r["support"] >= threshold - 1e-9

    def test_sorted_by_support_descending(self):
        mat, items = self._make_data()
        fs = frequent_itemsets(mat, items, min_support=0.1)
        supports = [r["support"] for r in fs]
        assert supports == sorted(supports, reverse=True)

    def test_size1_itemsets_present(self):
        mat, items = self._make_data()
        fs = frequent_itemsets(mat, items, min_support=0.05)
        sizes = {len(r["itemset"]) for r in fs}
        assert 1 in sizes

    def test_high_support_returns_nothing(self):
        mat, items = self._make_data()
        fs = frequent_itemsets(mat, items, min_support=0.99)
        assert len(fs) == 0

    def test_invalid_support_raises(self):
        mat, items = self._make_data()
        with pytest.raises(ValueError, match="min_support"):
            frequent_itemsets(mat, items, min_support=1.5)

    def test_known_pair_appears(self):
        # Diapers+Beer injected with p=0.25 — should be frequent at 0.15
        mat, items = self._make_data()
        fs = frequent_itemsets(mat, items, min_support=0.15)
        found_items = {frozenset(r["itemset"]) for r in fs}
        assert frozenset({"Diapers", "Beer"}) in found_items

    def test_itemsets_are_frozensets(self):
        mat, items = self._make_data()
        fs = frequent_itemsets(mat, items, min_support=0.1)
        for r in fs:
            assert isinstance(r["itemset"], frozenset)


# ---------------------------------------------------------------------------
# association_rules
# ---------------------------------------------------------------------------


class TestAssociationRules:
    def _make_rules(self, min_support=0.15, min_confidence=0.5, seed=42):
        rng = np.random.default_rng(seed)
        products = ["Bread", "Milk", "Eggs", "Butter", "Beer", "Diapers"]
        rows = []
        for _ in range(400):
            size = int(rng.integers(2, 5))
            basket = set(rng.choice(products, size=size, replace=False))
            if rng.random() < 0.35:
                basket.update({"Diapers", "Beer"})
            rows.append(list(basket))
        mat, items = encode_transactions(rows)
        fs = frequent_itemsets(mat, items, min_support=min_support)
        return association_rules(fs, min_confidence=min_confidence)

    def test_returns_list_of_dicts(self):
        rules = self._make_rules()
        assert isinstance(rules, list)
        for r in rules:
            for key in ("antecedents", "consequents", "support", "confidence", "lift"):
                assert key in r

    def test_confidence_meets_threshold(self):
        threshold = 0.5
        rules = self._make_rules(min_confidence=threshold)
        for r in rules:
            assert r["confidence"] >= threshold - 1e-9

    def test_lift_filter(self):
        rules_lift = association_rules(
            [{"itemset": frozenset(["A", "B"]), "support": 0.4},
             {"itemset": frozenset(["A"]), "support": 0.5},
             {"itemset": frozenset(["B"]), "support": 0.5}],
            min_confidence=0.0,
            min_lift=1.5,
        )
        for r in rules_lift:
            assert r["lift"] >= 1.5

    def test_sorted_by_lift_descending(self):
        rules = self._make_rules()
        if len(rules) > 1:
            lifts = [r["lift"] for r in rules]
            assert lifts == sorted(lifts, reverse=True)

    def test_antecedents_and_consequents_disjoint(self):
        rules = self._make_rules()
        for r in rules:
            assert r["antecedents"].isdisjoint(r["consequents"])

    def test_antecedents_and_consequents_are_frozensets(self):
        rules = self._make_rules()
        for r in rules:
            assert isinstance(r["antecedents"], frozenset)
            assert isinstance(r["consequents"], frozenset)

    def test_invalid_confidence_raises(self):
        fs = [{"itemset": frozenset(["A", "B"]), "support": 0.5}]
        with pytest.raises(ValueError, match="min_confidence"):
            association_rules(fs, min_confidence=1.5)

    def test_no_rules_from_size1_only(self):
        # Single-item frequent sets produce no rules
        fs = [
            {"itemset": frozenset(["A"]), "support": 0.8},
            {"itemset": frozenset(["B"]), "support": 0.7},
        ]
        rules = association_rules(fs, min_confidence=0.0)
        assert rules == []

    def test_beer_diapers_rule_found(self):
        rules = self._make_rules(min_support=0.1, min_confidence=0.3)
        pairs = {
            (frozenset(r["antecedents"]), frozenset(r["consequents"]))
            for r in rules
        }
        beer = frozenset({"Beer"})
        diapers = frozenset({"Diapers"})
        assert (beer, diapers) in pairs or (diapers, beer) in pairs
