import copy
import logging
from typing import ClassVar

from gplus_trees.base import Entry
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.utils import calc_rank
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import (
    BaseTestCase,
)
from tests.test_base import (
    GKPlusTreeTestCase as TreeTestCase,
)

logger = logging.getLogger(__name__)


class TestInsertMultipleDimensions(TreeTestCase):
    ASSERTION_MESSAGE_TEMPLATE = "TREE RESULT\nTREE: {tree}\n\n\nROOT SET: {root}\n"

    # Initialize items once to avoid re-creating them in each test
    _KEYS: ClassVar[list[int]] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ITEMS: ClassVar[dict] = {k: BaseTestCase.make_item(BaseTestCase, k, "val") for k in _KEYS}

    def _run_insert_case_multi_dim(
        self, keys, rank_combo, insert_pair, exp_keys, case_name, gnode_capacity=2, l_factor: float = 1.0
    ):
        if len(rank_combo) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")

        # build the tree once
        base_tree = create_gkplus_tree(K=gnode_capacity, dimension=1, l_factor=l_factor)

        for key, rank in zip(keys, rank_combo, strict=False):
            base_tree, _, _ = base_tree.insert(self.ITEMS[key], rank)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after initial insertions: {print_pretty(base_tree)}")
            logger.debug(f"Root node: {print_pretty(base_tree.node.set)}")

        msg_head = (
            f"\n\nKey-Rank combo:\nK: {keys}\nR: {rank_combo}"
            # f"\n\nTREE AFTER INITIAL INSERTIONS: {print_pretty(base_tree)}\n"
        )

        # deep-copy and split
        tree_copy = copy.deepcopy(base_tree)
        tree, _, _ = tree_copy.insert(self.ITEMS[insert_pair[0]], rank=insert_pair[1])

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after inserting {insert_pair[0]} with rank {insert_pair[1]}: {print_pretty(tree)}")

        msg = f"\n\nInsert {case_name}" + msg_head
        msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
            tree=print_pretty(tree),
            root=print_pretty(tree.node.set),
        )

        dummies_tree = self.get_dummies(tree)
        exp_tree_keys = sorted(dummies_tree + exp_keys)

        # assertions
        self.validate_tree(tree, exp_tree_keys, msg)

    def test_early_return_dim_2(self):
        """Test size is correctly maintained in a larger tree with random insertions"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [2, 2],
            [1, 2],
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=2)
        logger.debug(f"Keys: {keys}")
        insert_key_idx = 0
        logger.debug(f"Insert key: {keys[insert_key_idx]}")
        insert_item = self.make_item(keys[insert_key_idx])

        # Insert all items
        inserted_count = 0
        for i, key in enumerate(keys):
            # for i in range(1, 1000):
            if keys[i] == keys[insert_key_idx]:
                logger.debug(f"Skipping key {keys[i]} as it is the insert key")
                continue
            item = self.make_item(key, "val")
            rank = rank_lists[0][i]
            tree, _, _ = tree.insert(item, rank=rank)
            inserted_count += 1
            max_dim = tree.get_max_dim()
            dummy_cnt = self.get_dummy_count(tree)
            expanded_leafs = tree.get_expanded_leaf_count()
            expected_keys = [entry.item.key for entry in tree]
            expected_item_count = inserted_count + dummy_cnt
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tree after inserting {inserted_count} items: {print_pretty(tree)}")
                logger.debug(
                    f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}"
                )

            self.assertEqual(
                expected_item_count,
                tree.item_count(),
                f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs} (dummy count {dummy_cnt}). Leaf keys: {expected_keys}",
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after initial insertions: {print_pretty(tree)}")
        tree, _, _ = tree.insert(insert_item, rank=rank_lists[0][insert_key_idx])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after initial insertions + {insert_item.key}: {print_pretty(tree)}")

        max_dim = tree.get_max_dim()
        expanded_leafs = tree.get_expanded_leaf_count()
        dummy_cnt = self.get_dummy_count(tree)
        inserted_count += 1
        expected_keys = [entry.item.key for entry in tree]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after inserting {inserted_count} items: {print_pretty(tree)}")
            logger.debug(f"Tree structure: {tree.print_structure()}")
        expected_item_count = inserted_count + dummy_cnt
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}"
            )

        self.assertEqual(
            expected_item_count,
            tree.item_count(),
            f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs} (dummy count {dummy_cnt}). Leaf keys: {expected_keys}",
        )

        text = " | ".join(str(e.item.key) for e in tree.node.set)
        logger.debug(f"Root set keys after all inserts: {text}")

        for e in tree:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Entry: {e.item.key}, value: {e.item.value}, left_subtree: {e.left_subtree}")

        self.assertTrue(self.verify_subtree_sizes(tree))

    def test_specific_keys(self):
        """Test inserting specific keys into a tree with multiple dimensions"""
        k = 2
        tree = create_gkplus_tree(K=k)
        keys = [419, 533, 555, 719, 883, 120, 181, 389]
        ranks = [calc_rank(key=key, k=k, dim=1) for key in keys]

        # Insert items into the tree
        for i, key in enumerate(keys):
            item = self.make_item(key)
            rank = ranks[i]
            tree, _, _ = tree.insert(item, rank=rank)

        dum_keys = self.get_dummies(tree)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Keys ({len(keys)}): {keys}")
            logger.debug(f"Dummies ({len(dum_keys)}): {dum_keys}")
        exp_keys = sorted(dum_keys + keys)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Expected keys after insertions ({len(exp_keys)}): {exp_keys}")
            logger.debug(f"Tree after inserting items: {print_pretty(tree)}")
            logger.debug(f"Root node: {print_pretty(tree.node.set)}")

        self.validate_tree(tree, exp_keys)

    def test_insert_middle(self):
        """Test inserting specific keys into a tree with multiple dimensions"""
        keys = [1, 3, 7, 9, 11]
        ranks = [1, 1, 2, 2, 2]

        insert_key = 5
        insert_rank = 2
        insert_pair = (insert_key, insert_rank)

        # array of tuples with (case_name, split_key)
        insert_cases = [(f"Insert key {insert_key} rank {insert_rank}", insert_pair)]

        for case_name, insert_pair in insert_cases:
            exp_keys = sorted([k for k in keys] + [insert_key])
            with self.subTest(case=case_name, insert_pair=insert_pair):
                self._run_insert_case_multi_dim(
                    keys, ranks, insert_pair, exp_keys, case_name, gnode_capacity=4, l_factor=1.0
                )


class TestGrowthAcrossConversionThreshold(TreeTestCase):
    """Interface-level coverage for the bulk build of deeper dimensions.

    Growing a tree past the conversion threshold (k · l_factor) through
    ``insert_entry`` drives the entire internal growth path: KList
    expansion, conversion, and the bottom-up bulk build of the
    dimension-d+1 tree.  These tests assert only observable behaviour
    (max dimension, expanded-leaf count, item counts, iteration order,
    retrievability), so they exercise the conversion / bulk-create
    internals without importing them and survive internal refactors.
    """

    def _insert_all(self, tree, keys, ranks):
        for key, rank in zip(keys, ranks, strict=True):
            entry = Entry(self.make_item(key, f"val_{key}"), None)
            tree, inserted, _ = tree.insert_entry(entry, rank)
            self.assertTrue(inserted, f"Failed to insert key {key}")
        return tree

    def _assert_all_retrievable(self, tree, keys):
        for key in keys:
            entry, _ = tree.retrieve(key)
            self.assertIsNotNone(entry, f"Key {key} must be retrievable after growth")
            self.assertEqual(entry.item.key, key)
            self.assertEqual(entry.item.value, f"val_{key}", f"Value of key {key} must survive expansion")

    def _assert_sorted_real_keys(self, tree, keys):
        actual = [e.item.key for e in tree.iter_real_entries()]
        self.assertEqual(sorted(keys), actual, "Real entries must iterate in sorted key order")

    def test_growth_across_threshold_expands_leaf(self):
        """Crossing k · l_factor in a leaf set must expand it into a
        dimension-2 tree (built via the internal bulk path); staying at
        the threshold must not."""
        k = 4  # threshold = k * l_factor = 4 (leaf set holds the dummy too)
        keys = self.find_keys_for_rank_lists([[1, 1, 1, 1]], k=k)

        # Below/at threshold: dummy + 3 keys == 4 items -> no expansion.
        tree = create_gkplus_tree(K=k, dimension=1, l_factor=1.0)
        tree = self._insert_all(tree, keys[:3], [1] * 3)
        self.assertEqual(tree.get_max_dim(), 1, "At the threshold no deeper dimension may exist")
        self.assertEqual(tree.get_expanded_leaf_count(), 0)

        # One more same-rank key pushes the leaf set past the threshold.
        tree = self._insert_all(tree, keys[3:], [1])
        self.assertGreaterEqual(tree.get_max_dim(), 2, "Crossing the threshold must create dimension >= 2")
        self.assertGreaterEqual(tree.get_expanded_leaf_count(), 1)

        exp_keys = sorted(self.get_dummies(tree) + keys)
        self.validate_tree(tree, exp_keys)
        self._assert_all_retrievable(tree, keys)
        self._assert_sorted_real_keys(tree, keys)
        self.assertEqual(tree.real_item_count(), len(keys))

    def test_growth_threshold_scales_with_l_factor(self):
        """With l_factor=2.0 the same capacity tolerates twice the items
        before expanding: dummy + 7 == 8 items stay flat, the 8th key
        triggers the dimension-2 bulk build."""
        k = 4  # threshold = k * l_factor = 8
        keys = self.find_keys_for_rank_lists([[1] * 8], k=k)

        tree = create_gkplus_tree(K=k, dimension=1, l_factor=2.0)
        tree = self._insert_all(tree, keys[:7], [1] * 7)
        self.assertEqual(tree.get_max_dim(), 1, "l_factor=2.0 must defer expansion until 8 items")
        self.assertEqual(tree.get_expanded_leaf_count(), 0)

        tree = self._insert_all(tree, keys[7:], [1])
        self.assertGreaterEqual(tree.get_max_dim(), 2)
        self.assertGreaterEqual(tree.get_expanded_leaf_count(), 1)

        exp_keys = sorted(self.get_dummies(tree) + keys)
        self.validate_tree(tree, exp_keys)
        self._assert_all_retrievable(tree, keys)

    def test_growth_into_nested_dimensions(self):
        """Keys colliding at rank 1 in dimensions 1 AND 2 force the bulk
        build to recurse: the expanded dimension-2 set itself exceeds the
        threshold and expands further."""
        k = 4
        keys = self.find_keys_for_rank_lists([[1] * 5, [1] * 5], k=k)

        tree = create_gkplus_tree(K=k, dimension=1, l_factor=1.0)
        tree = self._insert_all(tree, keys, [1] * len(keys))

        self.assertGreaterEqual(tree.get_max_dim(), 3, "Double rank collision must nest beyond dimension 2")
        exp_keys = sorted(self.get_dummies(tree) + keys)
        self.validate_tree(tree, exp_keys)
        self._assert_all_retrievable(tree, keys)
        self._assert_sorted_real_keys(tree, keys)

    def test_growth_sequential_keys_natural_ranks(self):
        """Sequential keys with their natural (digest-derived) ranks grow
        past the threshold in many leaves; structure, counts, ordering
        and retrievability must hold throughout."""
        k = 4
        keys = list(range(1, 301))
        ranks = [calc_rank(key=key, k=k, dim=1) for key in keys]

        tree = create_gkplus_tree(K=k, dimension=1, l_factor=1.0)
        tree = self._insert_all(tree, keys, ranks)

        self.assertGreaterEqual(tree.get_max_dim(), 2, "300 keys at K=4 must expand at least one leaf")
        self.assertGreaterEqual(tree.get_expanded_leaf_count(), 1)
        self.assertEqual(tree.real_item_count(), len(keys))

        exp_keys = sorted(self.get_dummies(tree) + keys)
        self.validate_tree(tree, exp_keys)
        self._assert_sorted_real_keys(tree, keys)
        self._assert_all_retrievable(tree, keys)
        self.assertTrue(self.verify_subtree_sizes(tree))
