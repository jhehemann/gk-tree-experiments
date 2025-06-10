"""Tests for G+-trees with factory pattern"""
# pylint: skip-file

from typing import Tuple, Optional, List
import unittest
import logging
import hashlib

# Import factory function instead of concrete classes
from gplus_trees.base import calculate_group_size, count_trailing_zero_bits
from gplus_trees.gplus_tree_base import (
    Item,
    Entry,
    gtree_stats_,
    print_pretty,
)
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import (
    GKPlusTreeBase,
    get_dummy
)
from gplus_trees.g_k_plus.utils import calc_rank_from_digest, calc_rank, calc_ranks_for_multiple_dimensions
from stats.stats_gplus_tree import check_leaf_keys_and_values

from tests.utils import assert_tree_invariants_tc

# Configure logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreeTestCase(unittest.TestCase):
    """Base class for all GPlusTree factory tests"""
    def setUp(self):
        # Create trees with different K values for testing
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)
        self.tree_k16 = create_gkplus_tree(K=16)

        # Create dummy ranks for each of the dimensions
        # dummies = [get_dummy(i).key for i in range(1, 5)]
        # Convert dummy key to positive integer to calculate a rank from its hash
        # pos_dummies = [abs(dummies[i]) for i in range(len(dummies))]
        # dummy_keys = [-1, -2]
        
        # Create a dummy item map for dummy keys and their ranks in different dimensions
        dummy_range = range(-1, -11, -1)
        self.dummy_map = {i: get_dummy(dim=abs(i)) for i in dummy_range}
        abs_dummies = [abs(self.dummy_map[i].key) for i in dummy_range]
        self.dummy_ranks = calc_ranks_for_multiple_dimensions(abs_dummies, 2, dimensions=5)
        logger.debug(f"Calculate ranks for dummies of dims: {abs_dummies}")
        for dim_idx, ranks in enumerate(self.dummy_ranks):
            logger.debug(f"Dim {dim_idx+1}: {ranks}")

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
    
    def create_item(self, key, value="val"):
        """Helper to create test items"""
        return Item(key, value)
    
    def get_dummy_count(self, tree: 'GKPlusTreeBase') -> int:
        """Count the number of dummy items in the tree."""
        # Count the number of recursively instantiated trees
        # 1 dummy per tree instance + 1 for the initial tree
        expanded_leaf_count = tree.get_expanded_leaf_count()
        dummy_count = expanded_leaf_count + 1
        logger.debug(f"Expanded leaf count for tree {print_pretty(tree)}: {expanded_leaf_count}")
        return dummy_count
    
    def collect_keys(self, tree):
        """Collect all keys from a tree, excluding dummy keys."""
        if tree.is_empty():
            return []
            
        # dummy_key = get_dummy(type(tree).DIM).key
        keys = []

        # Collect keys from leaf nodes
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                # if entry.item.key != dummy_key:
                keys.append(entry.item.key)
        
        return sorted(keys)

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

    def verify_subtree_sizes(self, tree):
        """
        Recursive helper to verify that the size of each node equals the sum of its children's sizes
        plus the number of real items in the node itself.
        """
        if tree is None:
            return True
        
        if tree.is_empty():
            logger.debug("Empty tree: Use None instead of empty tree")
            
        node = tree.node
        
        # For leaf nodes, size should be the count of all items (incl. dummies)
        # dummy_key = get_dummy(tree.__class__.DIM).key
        if node.right_subtree is None:
            calculated_size = sum(1 for entry in node.set)
            logger.debug(f"Leaf node at rank {node.rank} has {calculated_size} items; node set (print_pretty): {print_pretty(node.set)}\n node set (print_structure): {node.set.print_structure()}")
            if calculated_size != tree.item_cnt:
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
            logger.debug(f"Node at rank {node.rank} has size {tree.item_cnt} but calculated size is {calculated_size}")
            return False
            
        return True
    
    def verify_calculated_sizes_match(self, tree):
        """
        Recursive helper to verify that node.size matches what calculate_tree_size() returns
        for each node in the tree.
        """
        if tree is None:
            return True
        
        if tree.is_empty():
            logger.debug("Empty tree: Use None instead of empty tree")
            
        node = tree.node
        node_size = node.size
        
        # Check if node.size matches calculated size
        calculated = node.calculate_tree_size()
        if calculated != node_size:
            logger.debug(f"Node size {node_size} doesn't match calculated size {calculated}")
            return False
            
        # Recursively check all subtrees
        for entry in node.set:
            if entry.left_subtree is not None:
                if not self.verify_calculated_sizes_match(entry.left_subtree):
                    return False
        
        if node.right_subtree is not None:
            if not self.verify_calculated_sizes_match(node.right_subtree):
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

        logger.debug(f"Key: {key}")
        logger.debug(f"Group Size: {group_size}")
        for level in range(num_levels):
            binary_hash = bin(int.from_bytes(current_hash, 'big'))[2:].zfill(256)
            trailing_zeros = count_trailing_zero_bits(current_hash)
            rank = calc_rank_from_digest(current_hash, group_size)
            logger.debug(f"Level {level + 1}:")
            logger.debug(f"  Binary Hash   : {binary_hash}")
            logger.debug(f"  Trailing Zeros: {trailing_zeros}")
            logger.debug(f"  Rank          : {rank}")
            current_hash = hashlib.sha256(current_hash).digest()
    
    def find_keys_for_rank_lists(self, rank_lists, k):
        """
        Brute force find unique and strictly increasing keys such that each key, when hashed repeatedly, produces the specified ranks in rank_lists.
        
        Parameters:
            rank_lists (List[List[int]]): Rank values per level and position.
            k (int): Must be a power of 2, used to derive group size.
        
        Returns:
            List[int]: Unique, ascending list of keys matching the rank pattern.
        """
        group_size = calculate_group_size(k)
        num_positions = len(rank_lists[0])
        num_levels = len(rank_lists)

        result_keys = []
        next_candidate_key = 1

        for pos in range(num_positions):
            key = next_candidate_key
            while True:
                current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
                match = True
                for level in range(num_levels):
                    expected_rank = rank_lists[level][pos]
                    actual_rank = calc_rank_from_digest(current_hash, group_size)
                    if actual_rank != expected_rank:
                        match = False
                        break
                    current_hash = hashlib.sha256(current_hash).digest()
                if match:
                    result_keys.append(key)
                    next_candidate_key = key + 1  # ensure next key is greater
                    break
                key += 1

        return result_keys
    
    def validate_keys(self, keys, rank_lists, k):
        """
        Validate whether each key in the list produces the correct ranks through repeated SHA-256 hashing.
        
        Parameters:
            keys (List[int]): List of integer keys to validate.
            rank_lists (List[List[int]]): List of rank lists per hashing level.
            k (int): The group size parameter (must be power of 2).
            
        Returns:
            bool: True if all keys match the expected rank sequences, False otherwise.
        """
        group_size = calculate_group_size(k)
        num_hashes = len(rank_lists)
        
        self.assertTrue(num_hashes > 0, "Rank lists must have at least one level")
        self.assertTrue(len(rank_lists[0]) == len(keys), "Rank lists and keys must have the same length")
        
        for i, key in enumerate(keys):
            current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
            for level in range(num_hashes):
                expected_rank = rank_lists[level][i]
                actual_rank = calc_rank_from_digest(current_hash, group_size)
                self.assertEqual(actual_rank, expected_rank, f"Rank mismatch for key {key} at level {level}")
                current_hash = hashlib.sha256(current_hash).digest()

        return True
