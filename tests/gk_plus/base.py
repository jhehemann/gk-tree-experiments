"""Tests for G+-trees with factory pattern"""
# pylint: skip-file

from typing import Tuple, Optional, List
import unittest
import logging

# Import factory function instead of concrete classes
from gplus_trees.factory import make_gplustree_classes, create_gplustree
from gplus_trees.gplus_tree_base import (
    DUMMY_ITEM,
    gtree_stats_,
    collect_leaf_keys,
    Stats
)
from gplus_trees.base import (
    Item,
    Entry,
    AbstractSetDataStructure,
    _create_replica
)
from stats.stats_gplus_tree import check_leaf_keys_and_values
from tests.utils import assert_tree_invariants_tc
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty


# Configure logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreeTestCase(unittest.TestCase):
    """Base class for all GPlusTree factory tests"""
    def setUp(self):
        # Use the factory to create a tree with the test capacity
        self.K = 4  # Default capacity for tests
        self.TreeClass, self.NodeClass, _, _ = make_gplustree_classes(self.K)
        self.tree = self.TreeClass()
        logger.debug(f"Created GPlusTree test with K={self.K}, using class {self.TreeClass.__name__}")

    def tearDown(self):
        # nothing to do if no tree or it's empty
        if not getattr(self, 'tree', None) or self.tree.is_empty():
            return

        stats = gtree_stats_(self.tree, {})
        assert_tree_invariants_tc(self, self.tree, stats)
        
        # --- optional invariants ---
        expected_item_count = getattr(self, 'expected_item_count', None)
        if expected_item_count is not None:
            self.assertEqual(
                stats.item_count, expected_item_count,
                f"Item count {stats.item_count} does not match"
                f"expected {expected_item_count}\n"
                f"Tree structure:\n{self.tree.print_structure()}"
            )

        expected_root_rank = getattr(self, 'expected_root_rank', None)
        if expected_root_rank is not None:
            self.assertEqual(
                self.tree.node.rank, expected_root_rank,
                f"Root rank {self.tree.node.rank} does not match expected"
                f"{expected_root_rank}"
            )

        # expected_keys = getattr(self, 'expected_keys', None)
        # if expected_keys is not None:
        #     keys = collect_leaf_keys(self.tree)
        #     self.assertEqual(
        #         sorted(keys), sorted(expected_keys),
        #         f"Leaf keys {keys} do not match expected {expected_keys}"
        #     )

        expected_gnode_count = getattr(self, 'expected_gnode_count', None)
        if expected_gnode_count is not None:
            self.assertEqual(
                stats.gnode_count, expected_gnode_count,
                f"GNode count {stats.gnode_count} does not match expected"
                f"{expected_gnode_count}"
            )

        # Leaf invariants
        expected_keys = getattr(self, 'expected_leaf_keys', None)
        keys, presence_ok, have_values, order_ok = (
            check_leaf_keys_and_values(self.tree, expected_keys)
        )

        # Values and ordering must always hold
        self.assertTrue(have_values, "Leaf items must have non-None values")
        self.assertTrue(order_ok, "Leaf keys must be in sorted order")

        # If expected_leaf_keys was set, also enforce presence
        if expected_keys is not None:
            self.assertTrue(
                presence_ok,
                f"Leaf keys {keys} do not match expected {expected_keys}"
            )
    
    def _assert_internal_node_properties(
            self, node, items: List[Item], rank: int
        )-> Optional[Entry]:
        """
        Verify that `node` is an internal node with the expected rank
        containing exactly `items`in order, with the first item's left subtree
        being empty and the next pointer being None.
        Returns:
            first_non_dummy: the first non-dummy entry in `node.set`
        """
        # print(f"Node to assert internal properties: {print_pretty(node.set)}")

        # must exist and be an internal node
        self.assertIsNotNone(node, "Node should not be None")
        self.assertEqual(node.rank, rank, f"Node rank should be {rank}")
        self.assertIsNone(node.next,
                          "Internal node should have no next pointer")

        # correct number of items
        actual_len = node.set.item_count()
        expected_len = len(items)
        self.assertEqual(
            actual_len, expected_len,
            f"Expected {expected_len} entries in node.set, found {actual_len}"
        )

        # verify each entry's key, value=0 and empty left subtree for min item
        for i, (entry, expected_item) in enumerate(zip(node.set, items)):
            self.assertEqual(
                entry.item.key, expected_item.key,
                f"Entry #{i} key mismatch: "
                f"expected {expected_item.key}, got {entry.item.key}"
            )
            self.assertEqual(
                entry.item.value, expected_item.value,
                f"Entry #{i} value for key {entry.item.key} should be None"
            )
            # the first entry's left subtree must be None
            # all replicas of real items must have a left subtree
            if i == 0:
                self.assertIsNone(
                    entry.left_subtree,
                    "Expected first (min) entry's left subtree to be None"
                )
            # elif expected_item.key >= 0:
            #     self.assertIsNotNone(
            #         entry.left_subtree,
            #         f"Expected non-first item #{i} ({expected_item.key}), which is a replica of a real entry NOT to have a None left subtree"
            #     )
                
            #     self.assertIsNotNone(
            #         entry.left_subtree,
            #         f"Expected Entry #{i}'s ({expected_item.key}) left subtree NOT to be None"
            #     )
            #     self.assertFalse(
            #         entry.left_subtree.is_empty(),
            #         f"Expected Entry #{i}'s ({expected_item.key}) left subtree NOT to be empty"
            #     )
        # find the first real entry (not a dummy)
        # and verify that it has no left subtree
        first_non_dummy = None
        for idx, entry in enumerate(node.set):
            if entry.item.key >= 0:
                first_non_dummy = entry
                break

        # # collect and return the first two entries
        # entries = list(node.set)
        # min_entry = entries[0]
        # next_entry = entries[1] if len(entries) > 1 else None
        
        return first_non_dummy
    
    def _assert_leaf_node_properties_old(
            self, node, items: List[Item]
        ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Verify that `node` is a rank 1 leaf containing exactly `items` in order,
        and that all its subtrees are empty.

        Returns:
            (min_entry, next_entry): the first two entries in `node.set`
                (next_entry is None if there's only one).
        """
        # must exist and be a leaf
        self.assertIsNotNone(node, "Node should not be None")
        self.assertEqual(node.rank, 1, f"Leaf node rank should be 1")
        
        # correct number of items
        actual_len   = node.set.item_count()
        expected_len = len(items)
        self.assertEqual(
            actual_len, expected_len,
            f"Leaf node has {actual_len} items; expected {expected_len}"
        )

        # no children at a leaf
        self.assertIsNone(node.right_subtree, 
                          "Expected leaf node's right_subtree to be None")

        # verify each entry's key/value and left subtree == None
        for i, (entry, expected) in enumerate(zip(node.set, items)):
            self.assertEqual(
                entry.item.key, expected.key,
                f"Entry #{i} key: expected {expected.key}, got {entry.item.key}"
            )
            self.assertEqual(
                entry.item.value, expected.value,
                f"Entry #{i} ({expected.key}) value: expected "
                f"{expected.value!r}, "
                f"got {entry.item.value!r}"
            )
            self.assertIsNone(
                entry.left_subtree,
                f"Expected Entry #{i}'s ({expected.key}) left subtree NOT to be empty"
            )

        # collect and return the first two entries
        entries = list(node.set)
        min_entry = entries[0]
        next_entry = entries[1] if len(entries) > 1 else None
        return min_entry, next_entry
    
    def _assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
            self, node, items: List[Item]
        ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Verify that `node` is a rank 1 leaf containing exactly `items` in order,
        and that all its subtrees are empty.

        Returns:
            (min_entry, next_entry): the first two entries in `node.set`
                (next_entry is None if there's only one).
        """
        # must exist and be a leaf
        self.assertIsNotNone(node, "Node should not be None")
        self.assertEqual(node.rank, 1, f"Leaf node rank should be 1")
        
        # correct number of items
        actual_len   = node.set.item_count()
        expected_len = len(items)
        self.assertEqual(
            actual_len, expected_len,
            f"Leaf node has {actual_len} items; expected {expected_len}"
        )

        # no right subtree at a leaf
        self.assertIsNone(node.right_subtree, 
                          "Expected leaf node's right_subtree to be None")

        # verify each entry's key/value
        for i, (entry, expected) in enumerate(zip(node.set, items)):
            self.assertEqual(
                entry.item.key, expected.key,
                f"Entry #{i} key: expected {expected.key}, got {entry.item.key}"
            )
            self.assertEqual(
                entry.item.value, expected.value,
                f"Entry #{i} ({expected.key}) value: expected "
                f"{expected.value!r}, "
                f"got {entry.item.value!r}"
            )
            # if i == 0:
            #     # the first entry's left subtree must be None in internal nodes
            #     self.assertIsNone(
            #         entry.left_subtree,
            #         f"Expected #1's ({entry.item.key}) entry's left subtree to be None"
            #     )
            # self.assertIsNone(
            #     entry.left_subtree,
            #     f"Expected Entry #{i}'s ({expected.key}) left subtree NOT to be empty"
            # )

        # # collect and return the first two entries
        # entries = list(node.set)
        # min_entry = entries[0]
        # next_entry = entries[1] if len(entries) > 1 else None
        # return min_entry, next_entry

    def _assert_leaf_node_properties(
            self, node, items: List[Item]
        ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Verify that `node` is a rank 1 leaf containing exactly `items` in order,
        and that all its subtrees are empty.

        Returns:
            (min_entry, next_entry): the first two entries in `node.set`
                (next_entry is None if there's only one).
        """
        self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
            node, items
        )
        # no children at a leaf
        self.assertIsNone(node.right_subtree, 
                          "Expected leaf node's right_subtree to be None")

        # verify each entry's key/value and left subtree == None
        for i, (entry, expected) in enumerate(zip(node.set, items)):
            self.assertIsNone(
                entry.left_subtree,
                f"Expected Entry #{i}'s ({expected.key}) left subtree NOT to be empty"
            )
