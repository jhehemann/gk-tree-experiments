"""Smoke tests for the display module.

``display`` reads trees exclusively through the observer metadata
interface (``dimension``, ``dummy_item``, ``inner_tree_of``), so both
tree variants are rendered here: G⁺-trees and Gᵏ⁺-trees, including a
Gᵏ⁺-tree with an expanded (deeper-dimension) node set.
"""

import random
import re
import unittest

from gplus_trees.base import ItemData, LeafItem
from gplus_trees.display import collect_leaf_keys, print_pretty, print_structure
from gplus_trees.factory import create_gplustree
from gplus_trees.g_k_plus.factory import create_gkplus_tree

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI colour codes for content assertions."""
    return _ANSI_RE.sub("", text)


def _make_item(key: int) -> LeafItem:
    return LeafItem(ItemData(key, f"val_{key}"))


def _fill(tree, keys, ranks):
    for key, rank in zip(keys, ranks, strict=True):
        tree.insert(_make_item(key), rank)
    return tree


def _fill_expanded_gk_tree():
    """A K=2 Gᵏ⁺-tree whose root node set is an expanded dimension-2 tree.

    Deterministic: fixed seed and rank cycle drive several node sets over
    the conversion threshold (l_factor * K = 2).
    """
    keys = random.Random(2080).sample(range(1, 100_000), 80)
    tree = _fill(create_gkplus_tree(K=2), keys, [1 + (i % 5) for i in range(len(keys))])
    return tree, keys


class TestPrintPrettyEdgeCases(unittest.TestCase):
    def test_none_input(self):
        self.assertEqual(print_pretty(None), "NoneType: None")

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            print_pretty("not a tree")

    def test_empty_gplus_tree(self):
        result = print_pretty(create_gplustree(K=4))
        self.assertIn("Empty", result)

    def test_empty_gkplus_tree(self):
        result = print_pretty(create_gkplus_tree(K=4))
        self.assertIn("Empty", result)

    def test_klist(self):
        tree = _fill(create_gplustree(K=4), [5, 6, 7], [1, 1, 1])
        result = _plain(print_pretty(tree.node.set))
        self.assertIn("[-1 | 5 | 6 | 7]", result)


class TestPrintPrettyGPlusTree(unittest.TestCase):
    def test_single_leaf(self):
        tree = _fill(create_gplustree(K=4), [10, 20], [1, 1])
        result = _plain(print_pretty(tree))
        self.assertIn("Rank 1:", result)
        # Dummy leads the leaf, keys follow in order.
        self.assertIn("-1 | 10 | 20", result)

    def test_multi_rank_tree(self):
        tree = _fill(create_gplustree(K=4), [1, 2, 3, 4, 5, 6], [1, 2, 1, 3, 1, 2])
        result = _plain(print_pretty(tree))
        self.assertIn("Rank 3:", result)
        self.assertIn("Rank 2:", result)
        self.assertIn("Rank 1:", result)
        for key in range(1, 7):
            self.assertIn(str(key), result)
        self.assertNotIn("Other Dims", result)


class TestPrintPrettyGKPlusTree(unittest.TestCase):
    def test_multi_rank_tree(self):
        tree = _fill(create_gkplus_tree(K=4), [1, 2, 3, 4, 5, 6], [1, 2, 1, 3, 1, 2])
        result = _plain(print_pretty(tree))
        self.assertIn("Rank 3:", result)
        self.assertIn("Rank 1:", result)
        for key in range(1, 7):
            self.assertIn(str(key), result)

    def test_expanded_tree_renders_deeper_dimension(self):
        tree, keys = _fill_expanded_gk_tree()
        result = _plain(print_pretty(tree))
        self.assertIn("Other Dims:", result)
        self.assertIn("(D2R", result)  # inner-tree nodes labelled with dimension 2
        self.assertIn("-2", result)  # dimension-2 dummy rendered
        for key in keys:
            self.assertIn(str(key), result)


class TestPrintStructure(unittest.TestCase):
    def test_empty_gplus_tree(self):
        result = print_structure(create_gplustree(K=4))
        self.assertIn("Empty", result)

    def test_gplus_tree(self):
        tree = _fill(create_gplustree(K=4), [1, 2, 3, 4], [1, 2, 1, 2])
        result = print_structure(tree, max_depth=10)
        self.assertIn("rank=2", result)
        self.assertIn("rank=1", result)
        self.assertIn("Right:", result)
        self.assertIn("Next:", result)

    def test_gkplus_tree_with_expanded_set(self):
        tree, _ = _fill_expanded_gk_tree()
        result = print_structure(tree, max_depth=10)
        self.assertIn("rank=", result)
        # The expanded root node set is a dimension-2 tree, not a k-list.
        self.assertIn("set=GKPlusTree_K2_Dim2", result)
        self.assertIn("Right:", result)


class TestCollectLeafKeys(unittest.TestCase):
    def test_gplus_tree(self):
        keys = [3, 1, 4, 5, 9, 2, 6]
        tree = _fill(create_gplustree(K=4), keys, [1, 2, 1, 1, 2, 1, 1])
        self.assertEqual(collect_leaf_keys(tree), sorted(keys))


if __name__ == "__main__":
    unittest.main()
