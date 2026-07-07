"""Regression tests for the Merkle Gᵏ⁺-tree factory and root hashing.

These tests guard the ``make_merkle_gk_plustree_classes`` factory, which
previously omitted the ``KListClass`` / ``DIM`` class attributes that
``GKPlusTreeBase._insert_new_item`` relies on. Building any Merkle
Gᵏ⁺-tree then failed at the first new-item insert with
``AttributeError: ... has no attribute 'KListClass'`` — reproducible with
as few as three keys.

The larger cases deliberately push past the KList→tree conversion
threshold (K=4, l_factor=1.0, ~40 keys) so that ``node.set`` becomes an
inner Gᵏ⁺-tree of a deeper dimension. Conversion builds those inner
trees through the *non-Merkle* ``create_gkplus_tree`` path, so inner
nodes are plain ``GKPlusNode`` instances without ``get_hash()``. We
assert that root hashing nonetheless works and stays
insertion-order-invariant: ``compute_hash`` iterates each node set
*flatly* (only ``left_subtree`` / ``right_subtree`` nodes are hashed
recursively, and those are always Merkle nodes), so it never dereferences
``get_hash`` on a plain inner node. ``test_hashing_survives_dimensional_nesting``
pins that this path is actually exercised (``get_max_dim() > 1``).
"""

import random
import unittest
from typing import ClassVar

from benchmarks.benchmark_utils import BenchmarkUtils

from gplus_trees.g_k_plus.utils import calc_ranks
from gplus_trees.merkle_gk_plus.factory import make_merkle_gk_plustree_classes


def _build_tree(K: int, keys: list[int], l_factor: float = 1.0):
    """Build a Merkle Gᵏ⁺-tree by inserting ``keys`` in the given order."""
    TreeClass, _ = make_merkle_gk_plustree_classes(K)
    tree = TreeClass(l_factor=l_factor)
    items = BenchmarkUtils.create_test_items(keys)
    ranks = calc_ranks(keys, K)
    for item, rank in zip(items, ranks, strict=True):
        tree.insert(item, rank)
    return tree


class TestMerkleGKPlusFactory(unittest.TestCase):
    """The factory must produce trees that can be built and hashed."""

    def test_insert_small_does_not_raise(self):
        """Regression: three inserts used to raise AttributeError (no KListClass)."""
        tree = _build_tree(16, [10, 20, 30])
        root = tree.get_root_hash()
        assert root is not None  # narrows bytes | None for the type checker
        self.assertIsInstance(root, bytes)
        self.assertEqual(len(root), 32)  # sha256 digest

    def test_factory_sets_klist_and_dim(self):
        """The Merkle tree class mirrors the plain factory's KListClass/DIM."""
        TreeClass, _ = make_merkle_gk_plustree_classes(8)
        self.assertTrue(hasattr(TreeClass, "KListClass"))
        self.assertEqual(TreeClass.DIM, 1)
        # The attribute the insert path reads must resolve.
        self.assertIsInstance(TreeClass.KListClass.KListNodeClass.CAPACITY, int)


class TestMerkleGKPlusRootHash(unittest.TestCase):
    """Root hashing past a conversion: stable and order-invariant."""

    K = 4
    KEYS: ClassVar[list[int]] = list(range(1, 41))  # forces KList→tree conversions at K=4

    def test_root_hash_is_deterministic(self):
        """Two independent builds of the same key set share a root hash."""
        h1 = _build_tree(self.K, self.KEYS).get_root_hash()
        h2 = _build_tree(self.K, self.KEYS).get_root_hash()
        assert h1 is not None  # narrows bytes | None for the type checker
        self.assertIsInstance(h1, bytes)
        self.assertEqual(len(h1), 32)
        self.assertEqual(h1, h2)

    def test_root_hash_is_insertion_order_invariant(self):
        """Shuffling the insertion order yields the same canonical root hash."""
        ordered = _build_tree(self.K, self.KEYS).get_root_hash()
        shuffled = self.KEYS[:]
        random.Random(42).shuffle(shuffled)
        self.assertNotEqual(shuffled, self.KEYS)  # guard: order actually changed
        self.assertEqual(_build_tree(self.K, shuffled).get_root_hash(), ordered)

    def test_hashing_survives_dimensional_nesting(self):
        """The ~40-key K=4 tree really nests, and hashing/integrity still hold.

        ``get_max_dim() > 1`` proves ``node.set`` became an inner Gᵏ⁺-tree
        (built via the non-Merkle path), so this exercises the path where
        inner nodes lack ``get_hash()``. Hashing must not touch them.
        """
        tree = _build_tree(self.K, self.KEYS)
        self.assertGreater(tree.get_max_dim(), 1, "expected dimensional nesting")
        self.assertGreater(tree.expanded_count(), 0)
        self.assertIsNotNone(tree.get_root_hash())
        self.assertTrue(tree.verify_integrity())

    def test_different_keys_change_root_hash(self):
        """Soundness under nesting: a different key set yields a different root."""
        base = _build_tree(self.K, self.KEYS).get_root_hash()
        altered = _build_tree(self.K, [*self.KEYS[:-1], 999]).get_root_hash()
        self.assertNotEqual(base, altered)


if __name__ == "__main__":
    unittest.main()
