"""Unified test base classes for all tree types."""

from typing import Tuple, Optional, List, TYPE_CHECKING
import unittest
import hashlib
from dataclasses import asdict
from pprint import pformat


from gplus_trees.base import Item, Entry
from gplus_trees.factory import make_gplustree_classes
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, get_dummy
from gplus_trees.g_k_plus.utils import (
    calc_rank_from_digest,
    calc_ranks_multi_dims,
    calculate_group_size,
    count_trailing_zero_bits,
)
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from gplus_trees.klist_base import KListBase
from stats.stats_gplus_tree import check_leaf_keys_and_values
from tests.utils import assert_tree_invariants_tc
# from gplus_trees.base import logger

if TYPE_CHECKING:
    from gplus_trees.gplus_tree_base import GPlusTreeBase

from gplus_trees.logging_config import get_test_logger

import logging
logger = get_test_logger("TestBase")

class BaseTestCase(unittest.TestCase):
    """Base class for all tests with common functionality."""
    
    def validate_klist(
        self,
        klist: KListBase,
        exp_entries: Optional[List[Entry]] = None,
        err_msg: Optional[str] = "",
    ):
        """Validate tree invariants and structure."""
        # Check tree invariants
        self.assertIsNotNone(klist, "KList should not be None")
        self.assertIsInstance(klist, KListBase, "KList should be a KListBase instance")
        klist.check_invariant()

        # Verify expected entries if provided
        if exp_entries is not None:
            actual_entries = list(klist)
            self.assertEqual(
                len(exp_entries), len(actual_entries),
                f"Expected {len(exp_entries)} entries, got {len(actual_entries)}\n{err_msg}"
            )
            self.assertEqual(klist.item_count(), len(exp_entries),
                             f"Expected {len(exp_entries)} items in klist, got {klist.item_count()}\n{err_msg}")
            for i, entry in enumerate(actual_entries):
                self.assertIs(entry.item, exp_entries[i].item)
                self.assertEqual(entry.item.key, exp_entries[i].item.key,
                                 f"Entry #{i} key mismatch: "
                                 f"expected {exp_entries[i].item.key}, got {entry.item.key}")
                self.assertEqual(entry.item.value, exp_entries[i].item.value,
                                 f"Entry #{i} value mismatch: "
                                 f"expected {exp_entries[i].item.value}, got {entry.item.value}")
                self.assertIs(entry.left_subtree, exp_entries[i].left_subtree,
                                 f"Entry #{i} left subtree mismatch: "
                                 f"expected {exp_entries[i].left_subtree}, got {entry.left_subtree}")
                self.assertIs(entry, exp_entries[i],
                                 f"Entry #{i} mismatch: expected {exp_entries[i]}, got {entry}")
            


class BaseTreeTestCase(BaseTestCase):
    """Base class for all tree tests with common functionality."""
    
    def create_item(self, key: int, value: str = "val") -> Item:
        """Helper to create test items."""
        return Item(key, value)
    
    def tearDown(self):
        """Common tearDown logic for tree tests."""
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
        

# Legacy alias for backward compatibility - will inherit all BaseTreeTestCase methods
class TreeTestCase(BaseTreeTestCase):
    """Legacy alias for BaseTreeTestCase to maintain backward compatibility."""
    pass


class GPlusTreeTestCase(BaseTreeTestCase):
    """Test case for standard G+-trees - inherits common functionality."""
    
    def setUp(self):
        # Use the factory to create a tree with the test capacity
        self.K = 4  # Default capacity for tests
        self.TreeClass, self.NodeClass, _, _ = make_gplustree_classes(self.K)
        self.tree = self.TreeClass()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Created GPlusTree test with K={self.K}, using class {self.TreeClass.__name__}")

    def _assert_internal_node_properties(
        self, node, items: List[Item], rank: int
    ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Assert properties of internal nodes."""
        # Implementation from tests/gplus/base.py
        expected_len = len(items)
        actual_len = node.set.item_count()  # Use item_count() instead of len()

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
            if i == 0:
                self.assertIsNone(
                    entry.left_subtree,
                    "Expected first (min) entry's left subtree to be None"
                )
            else:
                self.assertIsNotNone(
                    entry.left_subtree,
                    f"Expected Entry #{i}'s ({expected_item.key}) left subtree NOT to be None"
                )
                self.assertFalse(
                    entry.left_subtree.is_empty(),
                    f"Expected Entry #{i}'s ({expected_item.key}) left subtree NOT to be empty"
                )

        # collect and return the first two entries
        entries = list(node.set)
        min_entry = entries[0]
        next_entry = entries[1] if len(entries) > 1 else None
        return min_entry, next_entry

    def _assert_leaf_node_properties(
        self, node, items: List[Item]
    ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Assert properties of leaf nodes."""
        # Implementation from tests/gplus/base.py
        expected_len = len(items)
        actual_len = node.set.item_count()  # Use item_count() instead of len()

        self.assertEqual(
            actual_len, expected_len,
            f"Expected {expected_len} entries in leaf node.set, found {actual_len}"
        )

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


# Legacy alias for backward compatibility with existing G+ tests
class TreeTestCase(GPlusTreeTestCase):
    """Legacy alias - provides G+ tree functionality by default."""
    pass


class GKPlusTreeTestCase(BaseTreeTestCase):
    """Test case for GK+-trees with extended functionality."""
    
    def setUp(self):
        # Create trees with different K values for testing
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)
        self.tree_k16 = create_gkplus_tree(K=16)

        # Create a dummy item map for dummy keys and their ranks in different dimensions
        dummy_range = range(-1, -11, -1)
        self.dummy_map = {i: get_dummy(dim=abs(i)) for i in dummy_range}
        # abs_dummies = [abs(self.dummy_map[i].key) for i in dummy_range]
        # self.dummy_ranks_k2 = calc_ranks_multi_dims(abs_dummies, 2, dimensions=10)
        # # log dummy ranks pretty for debugging
        # logger.debug(f"Dummy ranks for dummies: {list(dummy_range)}")
        # for dim, ranks in enumerate(self.dummy_ranks):
        #     logger.debug(f"Dimension {dim + 1}: {ranks}")

    def get_dummy_count(self, tree: 'GKPlusTreeBase') -> int:
        """Count the number of dummy items in the tree."""
        # Count the number of recursively instantiated trees
        # 1 dummy per tree instance + 1 for the initial tree
        expanded_leaf_count = tree.get_expanded_leaf_count()
        dummy_count = expanded_leaf_count + 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Expanded leaf count for tree {print_pretty(tree)}: {expanded_leaf_count}")
        return dummy_count

    def get_dummies(self, tree: 'GKPlusTreeBase') -> List[int]:
        """Collect all dummy keys from a tree."""
        if tree.is_empty():
            return []
        
        if tree.node.right_subtree is None:
            # If the tree is a leaf, return empty list
            return [e.item.key for e in tree if e.item.key < 0]

        dummy_keys = []
        for e in tree.node.set:
            if e.left_subtree is not None:
                dummy_keys.extend(self.get_dummies(e.left_subtree))
        dummy_keys.extend(self.get_dummies(tree.node.right_subtree))

        return dummy_keys

    def validate_tree(
        self,
        tree,
        expected_keys: Optional[List[int]] = None,
        err_msg: Optional[str] = "",
    ):
        """Validate tree invariants and structure."""
        # Check tree invariants
        self.assertIsNotNone(tree, "Tree should not be None")
        if not tree.is_empty():
            self.assertTrue(tree.node.set, "Root node set should not be empty")
            for entry in tree.node.set:
                self.assertIsInstance(entry, Entry, "Tree node set should contain Entry instances")
                self.assertIsNotNone(entry.item, "Root node item should not be None")
                if entry.left_subtree is not None:
                    self.assertIsInstance(entry.left_subtree, GKPlusTreeBase,
                                          "Left subtree should be a GKPlusTreeBase instance")
            if tree.node.right_subtree is not None:
                self.assertIsInstance(tree.node.right_subtree, GKPlusTreeBase,
                                  "Right subtree should be a GKPlusTreeBase instance")
            
        stats = gtree_stats_(tree, {})
        # add stats to the error message if provided
        err_msg += str(asdict(stats))
        assert_tree_invariants_tc(self, tree, stats, err_msg)
        
        # Verify expected keys if provided
        if expected_keys is not None:
            actual_keys = sorted(self.collect_keys(tree))
            self.assertEqual(expected_keys, actual_keys, 
                            f"Expected keys {expected_keys} don't match actual keys {actual_keys}\n{err_msg}")
            self.assertEqual(len(expected_keys), tree.item_count(),
                            f"Expected {len(expected_keys)} items in tree, got {tree.item_count()}\n{err_msg}")

    def collect_keys(self, tree):
        """Collect all keys from a tree."""
        if tree.is_empty():
            return []
            
        keys = []

        # Collect keys from leaf nodes
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                keys.append(entry.item.key)
        
        return sorted(keys)

    def verify_subtree_sizes(self, tree):
        """
        Recursive helper to verify that the size of each node equals the sum of its children's sizes
        plus the number of real items in the node itself.
        """
        if tree is None:
            return True
        
        if tree.is_empty():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Empty tree: Use None instead of empty tree")
            
        # For leaf nodes, size should be the count of all items (incl. dummies)
        node = tree.node
        if node.right_subtree is None:
            calculated_size = sum(1 for entry in node.set)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Leaf node at rank {node.rank} has {calculated_size} items; node set (print_pretty): {print_pretty(node.set)}\n node set (print_structure): {node.set.print_structure()}")
            if calculated_size != tree.item_cnt:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Leaf node has size {tree.item_cnt} but contains {calculated_size} items, node set: {print_pretty(node.set)}")
                return False
            return True
            
        # For internal nodes, size should be sum of child sizes
        calculated_size = 0
        
        # Add sizes from left subtrees
        for entry in node.set:
            if entry.left_subtree is not None:
                if entry.left_subtree.item_cnt is None:
                    size = entry.left_subtree.item_count()
                else:
                    size = entry.left_subtree.item_cnt
                calculated_size += size
                # Recursively verify this subtree
                if not self.verify_subtree_sizes(entry.left_subtree):
                    return False
        
        # Add size from right subtree
        if node.right_subtree is not None:
            if node.right_subtree.item_cnt is None:
                size = node.right_subtree.item_count()
            else:
                size = node.right_subtree.item_cnt
            calculated_size += size
            # Recursively verify right subtree
            if not self.verify_subtree_sizes(node.right_subtree):
                return False
        
        # Check if the stored size matches calculated size
        if calculated_size != tree.item_cnt:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Node at rank {node.rank} has size {tree.item_cnt} but calculated size is {calculated_size}")
            return False
            
        return True

    def print_hash_info(self, key: int, k: int, num_levels: int = 1):
        """
        Prints the binary representation, trailing zeros, and rank for each hash level of the key.
        
        Parameters:
            key (int): The key to inspect.
            k (int): Group size parameter (must be a power of 2).
            num_levels (int): How many times to re-hash.
        """
        group_size = calculate_group_size(k)
        current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Key: {key}")
            logger.debug(f"Group Size: {group_size}")
        for level in range(num_levels):
            binary_hash = bin(int.from_bytes(current_hash, 'big'))[2:].zfill(256)
            trailing_zeros = count_trailing_zero_bits(current_hash)
            rank = calc_rank_from_digest(current_hash, group_size)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Level {level + 1}:")
                logger.debug(f"  Binary Hash   : {binary_hash}")
                logger.debug(f"  Trailing Zeros: {trailing_zeros}")
                logger.debug(f"  Rank          : {rank}")
            current_hash = hashlib.sha256(current_hash).digest()

    def find_keys_for_rank_lists(self, rank_lists, k, spacing=False):
        """Delegate to the library utility for finding keys matching rank lists."""
        from gplus_trees.utils import find_keys_for_rank_lists as util_find
        return util_find(rank_lists, k, spacing)
    
    def sort_and_calculate_rank_lists_from_keys(self, keys, k, dim_limit):
        """Validate that keys match expected rank lists."""
        group_size = calculate_group_size(k)
        keys = sorted(keys)
        logger.debug(f"Sorted keys: {keys}")
        
        # Create a list to hold rank lists for each dimension
        if dim_limit < 1:
            raise ValueError("dim_limit must be at least 1")

        rank_lists = [[] for _ in range(dim_limit)]
        for key_idx, key in enumerate(keys):
            current_dim = 1
            current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
            while current_dim <= dim_limit:
                rank = calc_rank_from_digest(current_hash, group_size)
                rank_lists[current_dim - 1].append(rank)
                current_hash = hashlib.sha256(current_hash).digest()
                current_dim += 1
        
        # Ensure all rank lists are of equal length
        for dim_idx, dim in enumerate(rank_lists):
            if len(dim) != len(keys):
                raise ValueError(f"Rank list for dimension {dim_idx + 1} has length "
                                 f"{len(dim)}, expected {len(keys)}")

        return rank_lists

    def validate_key_ranks(self, keys, rank_lists, k):
        """Validate that keys match expected rank lists."""
        group_size = calculate_group_size(k)
        for key_idx, key in enumerate(keys):
            current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
            for dim_idx, dim in enumerate(rank_lists):
                expected_rank = dim[key_idx]
                actual_rank = calc_rank_from_digest(current_hash, group_size)
                self.assertEqual(actual_rank, expected_rank,
                               f"Key {key} in dimension {dim_idx+1}: expected rank {expected_rank}, got {actual_rank}")
                current_hash = hashlib.sha256(current_hash).digest()

    # Extended assertion methods for GK+ trees
    def _assert_internal_node_properties(
        self, node, items: List[Item], rank: int
    ) -> Optional[Entry]:
        """Assert properties of internal nodes for GK+ trees."""
        self.assertIsNotNone(node, "Node should not be None")
        self.assertEqual(node.rank, rank, f"Node rank should be {rank}")
        self.assertIsNone(node.next, "Internal nodes should not have a next pointer")

        # check correct number of items
        expected_len = len(items)
        actual_len = node.set.item_count()

        self.assertEqual(
            actual_len, expected_len,
            f"Expected {expected_len} entries in node.set, found {actual_len}"
        )
        
        pivot = node.set.find_pivot()
        # verify each entry's key, value=0 and empty left subtree for min item
        for i, (entry, expected_item) in enumerate(zip(node.set, items)):
            self.assertEqual(
                entry.item.key, expected_item.key,
                f"Entry at pos {i} key mismatch: "
                f"expected {expected_item.key}, got {entry.item.key}"
            )
            self.assertEqual(
                entry.item.value, expected_item.value,
                f"Entry at pos {i}'s value for key {entry.item.key} should be None"
            )

            # verify each entry: key=expected, value is None, empty left subtree for min and pivot
            # all replicas of real items must have a left subtree
            if i == 0:
                self.assertIsNone(
                    entry.left_subtree,
                    "Expected first (min) entry's left subtree to be None"
                )
            elif i == pivot:
                self.assertIsNone(
                    entry.left_subtree,
                    "Expected pivot entry's left subtree to be None"
                )

        # collect and return the first two entries
        entries = list(node.set)
        min_entry = entries[0] if entries else None
        next_entry = entries[1] if len(entries) > 1 else None
        return min_entry, next_entry

    def _assert_leaf_node_properties(
        self, node, items: List[Item]
    ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Assert properties of leaf nodes for GK+ trees."""
        expected_len = len(items)
        actual_len = node.set.item_count()  # Use item_count() instead of len()

        self.assertEqual(
            actual_len, expected_len,
            f"Expected {expected_len} entries in leaf node.set, found {actual_len}"
        )

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
                f"Expected Entry #{i}'s ({expected.key}) left subtree to be None"
            )

        # collect and return the first two entries
        entries = list(node.set)
        min_entry = entries[0] if entries else None
        next_entry = entries[1] if len(entries) > 1 else None
        return min_entry, next_entry
