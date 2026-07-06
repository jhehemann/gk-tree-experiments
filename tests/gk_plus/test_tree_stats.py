"""Regression tests for expansion-aware flags in ``gtree_stats_``."""

import random

from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
from gplus_trees.g_k_plus.utils import calc_ranks
from gplus_trees.tree_stats import gtree_stats_
from tests.test_base import GKPlusTreeTestCase


class TestStatsLeafValuesExpansionAware(GKPlusTreeTestCase):
    """``all_leaf_values_present`` must be derived from the root-level
    leaf walk, not from per-node checks merged across inner trees.

    Regression: the per-node rank-1 check treated every rank-1 node as a
    data leaf and fired on the value-less replicas in inner trees of
    internal nodes; the flag was then merged into the outer stats, so
    correct GK+-trees were misreported once an internal node's set
    expanded.  Same bug class as the ``check_leaf_keys_and_values`` fix
    (commit c8ed7ee).
    """

    def _build_tree_with_expanded_internal_set(self):
        """K=4, threshold 4: eight rank-2 items put 8 replicas + dummy in
        the root's set, which therefore expands into an inner tree whose
        rank-1 nodes hold value-less replicas — the misfire case."""
        tree = create_gkplus_tree(K=4)
        for key in range(1, 9):
            tree, _, _ = tree.insert(self.make_item(key * 10, f"v{key * 10}"), rank=2)
        self.assertIsNotNone(tree.node.right_subtree, "premise: root is an internal node")
        self.assertIsInstance(tree.node.set, GKPlusTreeBase, "premise: internal node's set expanded")
        return tree

    def test_expanded_internal_set_keeps_flag_true(self):
        tree = self._build_tree_with_expanded_internal_set()

        stats = gtree_stats_(tree)

        self.assertTrue(
            stats.all_leaf_values_present,
            "value-less replicas in an internal node's inner tree must not flip all_leaf_values_present",
        )
        self.assertEqual(stats.real_item_count, 8)
        self.assertTrue(stats.leaf_keys_in_order)

    def test_missing_leaf_value_still_flips_flag(self):
        # The root-level walk must stay sharp: a real data item without a
        # value in the dimension-1 leaf chain is a genuine violation.
        tree = self._build_tree_with_expanded_internal_set()
        nulled = False
        for leaf in tree.iter_leaf_nodes():
            for entry in leaf.set:
                if entry.item.key >= 0:
                    entry.item.value = None
                    nulled = True
                    break
            if nulled:
                break
        self.assertTrue(nulled, "premise: found a real item to null")

        stats = gtree_stats_(tree)

        self.assertFalse(stats.all_leaf_values_present, "missing leaf value must flip all_leaf_values_present")

    def test_repro_5000_random_keys_all_flags_hold(self):
        # Deterministic repro: the single flagged run of the 120-run
        # structural de-risking campaign (2026-07-06, see docs/journal/).
        keys = random.Random(1).sample(range(1, 2**60), 5000)
        ranks = calc_ranks(keys, 4)
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1.0)
        for key, rank in zip(keys, ranks, strict=True):
            tree, _, _ = tree.insert(self.make_item(key, f"v{key}"), rank)

        stats = gtree_stats_(tree)

        # Ground truth via the leaf chain: every real item carries a value.
        missing = sum(
            1
            for leaf in tree.iter_leaf_nodes()
            for entry in leaf.set
            if entry.item.key >= 0 and entry.item.value is None
        )
        self.assertEqual(missing, 0, "premise: leaf chain is complete")
        self.assertTrue(stats.all_leaf_values_present, "flag must match the leaf-chain ground truth")
        self.assertEqual(stats.real_item_count, 5000)
        self.assertTrue(stats.leaf_keys_in_order)
        self.assertTrue(stats.linked_leaf_nodes)
