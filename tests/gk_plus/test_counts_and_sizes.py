import sys
import os
import unittest
import random
from typing import List, Tuple, Optional, Iterator, TYPE_CHECKING
from itertools import product, islice
from pprint import pprint
import copy
from tqdm import tqdm
from statistics import median_low

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
from tests.gk_plus.base import TreeTestCase
from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from tests.utils import assert_tree_invariants_tc
from gplus_trees.base import calculate_group_size, count_trailing_zero_bits
from gplus_trees.g_k_plus.utils import calc_rank_from_digest, calc_rank, calc_ranks_for_multiple_dimensions

from gplus_trees.base import (
    Item,
    Entry,
    AbstractSetDataStructure,
    _create_replica
)

from tests.logconfig import logger

class TestGKPlusNodeItemCounts(TreeTestCase):

    def setUp(self):
        # Create trees with different K values for testing
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)

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

    def create_item(self, key, value="val"):
        """Helper to create test items"""
        return Item(key, f"{value}_{key}")

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
    
    def collect_keys(self, tree):
        """Collect all keys from a tree, excluding dummy keys."""
        if tree.is_empty():
            return []
            
        dummy_key = get_dummy(type(tree).DIM).key
        keys = []

        # Collect keys from leaf nodes
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                if entry.item.key != dummy_key:
                    keys.append(entry.item.key)
        
        return sorted(keys)
    
    def get_dummy_count(self, tree: 'GKPlusTreeBase') -> int:
        """Count the number of dummy items in the tree."""
        # Count the number of recursively instantiated trees
        # 1 dummy per tree instance + 1 for the initial tree
        expanded_leaf_count = tree.get_expanded_leaf_count()
        dummy_count = expanded_leaf_count + 1
        logger.debug(f"Expanded leaf count for tree {print_pretty(tree)}: {expanded_leaf_count}")
        return dummy_count

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
            if calculated_size != node.size:
                logger.debug(f"Leaf node has size {node.size} but contains {calculated_size} items, node set: {print_pretty(node.set)}")
                return False
            return True
            
        # For internal nodes, size should be sum of child sizes
        calculated_size = 0
        
        # Add sizes from left subtrees
        for entry in node.set:
            if entry.left_subtree is not None:
                if entry.left_subtree.node.size is None:
                    size = entry.left_subtree.node.get_size()
                else:
                    size = entry.left_subtree.node.size
                calculated_size += size
                # Recursively verify this subtree
                if not self.verify_subtree_sizes(entry.left_subtree):
                    return False
        
        # Add size from right subtree
        if node.right_subtree is not None:
            if node.right_subtree.node.size is None:
                size = node.right_subtree.node.get_size()
            else:
                size = node.right_subtree.node.size
            calculated_size += size
            # Recursively verify right subtree
            if not self.verify_subtree_sizes(node.right_subtree):
                return False
        
        # Check if the stored size matches calculated size
        if calculated_size != node.size:
            logger.debug(f"Node at rank {node.rank} has size {node.size} but calculated size is {calculated_size}")
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
        
    def test_single_insertion_size(self):
        """Test size is 1 after inserting a single item"""
        tree = self.tree_k2
        key = 1
        item = self.create_item(key)
        tree, _ = tree.insert(item, rank=1)

        self.assertFalse(tree.is_empty(), 
                         "Tree should not be empty after insertion")
        self.assertEqual(tree.node.get_node_item_count(), 2, 
                         "Node size should be 2 after single insertion. "
                         "Item + 1 dummy item")

    def test_insertion_triggers_single_expansion_in_leaf(self):
        """Test that inserting an item triggers expansion when necessary"""
        tree = self.tree_k2
        k = 2
        rank_lists = [
            [1, 1],  # Dimension 1
            [1, 2],  # Dimension 2
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        logger.debug(f"Keys: {keys}")
        self.item_map = { k: self.create_item(k) for k in keys}

        for i in range(2):
            key = keys[i]
            rank = rank_lists[0][i]
            item = self.item_map[key]
            tree, _ = tree.insert(item, rank=rank)
            
        with self.subTest("Dim 1 single node"):
            self.assertFalse(tree.is_empty(), "Tree should not be empty after insertions")
            logger.debug(f"Tree after insertions: {print_pretty(tree)}")
            logger.debug(f"Node set: {print_pretty(tree.node.set)}")
            
            node_size = tree.node.get_node_item_count()
            self.assertEqual(node_size, 4,
                            f"Node size should be 4 (including dummy) after inserting 2 items triggering a single expansion to dimension 2, got: "
                            f"{node_size} for set: {print_pretty(tree.node.set)}")
            
            self._assert_leaf_node_properties(
                    tree.node,
                    [self.dummy_map[-2], self.dummy_map[-1], self.item_map[5], self.item_map[9]]
                )
        
        with self.subTest("Dim 2 root"):
            root_dim2 = tree.node.set.node
            root_dim2_size = root_dim2.get_node_item_count()
            self.assertEqual(root_dim2_size, 2,
                             f"Root size in dim 2 should be 2 (including dummy) after inserting 2 items, got: {root_dim2_size} for set: {print_pretty(root_dim2.set)}")
            self._assert_internal_node_properties(
                root_dim2,
                [self.dummy_map[-2], self.dummy_map[-1]],
                3
            )

        with self.subTest("Dim 2 right internal node"):
            right_internal = root_dim2.right_subtree.node
            right_internal_size = right_internal.get_node_item_count()
            self.assertEqual(right_internal_size, 2,
                             f"Right internal node size in dim 2 should be 2, got: {right_internal_size} for set: {print_pretty(right_internal.set)}")
            self._assert_internal_node_properties(
                right_internal,
                [self.dummy_map[-1], _create_replica(9)],
                2
            )
        with self.subTest("Dim 2 leaf 1"):
            # Check leaf 1 in dimension 2
            root_entries = list(root_dim2.set)
            leaf1 = root_entries[1].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 1,
                             f"Leaf 1 size in dim 2 should be 1, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")
            self._assert_leaf_node_properties(
                leaf1,
                [self.dummy_map[-2]],
            )

        with self.subTest("Dim 2 leaf 2"):
            # Check leaf 2 in dimension 2
            right_internal_entries = list(right_internal.set)
            leaf2 = right_internal_entries[1].left_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 2,
                             f"Leaf 2 size in dim 2 should be 2 got: {leaf2_size} for set: {print_pretty(leaf2.set)}")
            self._assert_leaf_node_properties(
                leaf2,
                [self.dummy_map[-1], self.item_map[5]],
            )

        with self.subTest("Dim 2 leaf 3"):
            # Check leaf 3 in dimension 2
            leaf3 = right_internal.right_subtree.node
            leaf3_size = leaf3.get_node_item_count()
            self.assertEqual(leaf3_size, 1,
                             f"Leaf 3 size in dim 2 should be 1, got: {leaf3_size} for set: {print_pretty(leaf3.set)}")
            self._assert_leaf_node_properties(
                leaf3,
                [self.item_map[9]],
            )
        
    def test_insertion_triggers_multiple_expansions_in_leaf(self):
        """Test that inserting an item triggers expansion when necessary"""
        tree = self.tree_k2
        k = 2
        rank_lists = [
            [1, 1],  # Dimension 1
            [1, 1],  # Dimension 2
            [1, 2],  # Dimension 3
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        logger.debug(f"Keys: {keys}")
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}

        for i in range(2):
            key = keys[i]
            rank = rank_lists[0][i]
            item = self.create_item(key)
            tree, _ = tree.insert(item, rank=rank)

        logger.debug(f"Tree after insertions: {print_pretty(tree)}")
        logger.debug(f"Node set: {print_pretty(tree.node.set)}")

        self.assertFalse(tree.is_empty(), 
                         f"Tree should not be empty after inserting items for keys {keys}")
        
        with self.subTest("Dim 1 single node"):
            root_d1 = tree.node
            node_size = root_d1.get_node_item_count()
            self.assertEqual(node_size, 5,
                            f"Leaf size in dim 1 should be 5, got: "
                            f"{node_size} for set: {print_pretty(root_d1.set)}\n Nodes right subtree: {print_pretty(root_d1.set.node.right_subtree.node.set)}")

            self._assert_leaf_node_properties(
                    tree.node,
                    [self.dummy_map[-2], self.dummy_map[-3], self.dummy_map[-1], self.item_map[11], self.item_map[22]]
                )
            
        with self.subTest("Dim 2 root"):
            root_d2 = root_d1.set.node
            root_d2_size = root_d2.get_node_item_count()
            self.assertEqual(root_d2_size, 2,
                             f"Root size in dim 2 should be 2, got: {root_d2_size} for set: {print_pretty(root_d2.set)}")
            self._assert_internal_node_properties(
                root_d2,
                [self.dummy_map[-2], self.dummy_map[-1]],
                3
            )
            root_d2_entries = list(root_d2.set)
            self.assertIsNotNone(root_d2_entries[1].left_subtree,
                              f"Root entry #1's ({root_d2_entries[1].item.key}) left subtree in dim 2 should not be None, "
                              f"got: {print_pretty(root_d2_entries[1].left_subtree)}")
        
        with self.subTest("Dim 2 leaf 1"):
            root_entries_d2 = list(root_d2.set)
            leaf1 = root_entries_d2[1].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 1,
                             f"Leaf 1 size in dim 2 should be 1, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")
            self._assert_leaf_node_properties(
                leaf1,
                [self.dummy_map[-2]],
            )
        
        with self.subTest("Dim 2 leaf 2"):
            leaf2 = root_d2.right_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 4,
                             f"Leaf 2 size in dim 2 should be 4, got: {leaf2_size} for set: {print_pretty(leaf2.set)}")
            self._assert_leaf_node_properties(
                leaf2,
                [self.dummy_map[-3], self.dummy_map[-1], self.item_map[11], self.item_map[22]]
            )

        with self.subTest("Dim 3 root"):
            root_d3 = leaf2.set.node
            root_d3_size = root_d3.get_node_item_count()
            self.assertEqual(root_d3_size, 2,
                             f"Root size in dim 3 should be 2, got: {root_d3_size} for set: {print_pretty(root_d3.set)}")
            self._assert_internal_node_properties(
                root_d3,
                [self.dummy_map[-3], self.dummy_map[-1]],
                6
            )

        with self.subTest("Dim 3 leaf 1"):
            # Check leaf 1 in dimension 3
            root_entries_d3 = list(root_d3.set)
            leaf1 = root_entries_d3[1].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 1,
                             f"Leaf 1 size in dim 3 should be 2, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")
            self._assert_leaf_node_properties(
                leaf1,
                [self.dummy_map[-3]],
            )
        
        with self.subTest("Dim 3 right internal node"):
            # Check right internal node in dimension 3
            right_internal = root_d3.right_subtree.node
            right_internal_size = right_internal.get_node_item_count()
            self.assertEqual(right_internal_size, 2,
                             f"Right internal node size in dim 3 should be 2, got: {right_internal_size} for set: {print_pretty(right_internal.set)}")
            self._assert_internal_node_properties(
                right_internal,
                [self.dummy_map[-1], _create_replica(22)],
                2
            )
        
        with self.subTest("Dim 3 leaf 2"):
            # Check leaf 2 in dimension 3
            right_internal_entries = list(right_internal.set)
            leaf2 = right_internal_entries[1].left_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 2,
                             f"Leaf 2 size in dim 3 should be 2, got: {leaf2_size} for set: {print_pretty(leaf2.set)}")
            self._assert_leaf_node_properties(
                leaf2,
                [self.dummy_map[-1], self.item_map[11]]
            )
        
        with self.subTest("Dim 3 leaf 3"):
            # Check leaf 3 in dimension 3
            leaf3 = right_internal.right_subtree.node
            leaf3_size = leaf3.get_node_item_count()
            self.assertEqual(leaf3_size, 1,
                             f"Leaf 3 size in dim 3 should be 1, got: {leaf3_size} for set: {print_pretty(leaf3.set)}")
            self._assert_leaf_node_properties(
                leaf3,
                [self.item_map[22]],
            )
        
    def test_insertion_triggers_single_expansion_in_root(self):
        """Test that inserting an item triggers expansion when necessary"""
        tree = self.tree_k2
        k = 2
        rank_lists = [
            [2, 2],  # Dimension 1
            [1, 2],  # Dimension 2
        ]

        keys = self.find_keys_for_rank_lists(rank_lists, k)
        logger.debug(f"Keys: {keys}")

        # Create items
        item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        
        for i in range(2):
            key = keys[i]
            rank = rank_lists[0][i]
            item = self.create_item(key)
            tree, _ = tree.insert(item, rank=rank)        
        
        logger.debug(f"Tree after insertions: {print_pretty(tree)}")
        self.assertFalse(tree.is_empty(), "Tree should not be empty after insertions")
        logger.debug(f"Root set: {print_pretty(tree.node.set)}")
        
        with self.subTest("Dim 1 root"):
            # Insert one more item to trigger expansion in the root
            root_d1 = tree.node
            root_d1_entries = list(root_d1.set)
            root_d1_size = root_d1.get_node_item_count()
            self.assertEqual(root_d1_size, 4,
                            "Root size should be 4, got: "
                            f"{root_d1_size} for set: {print_pretty(root_d1.set)}"
            )
            self._assert_internal_node_properties(
                root_d1,
                [self.dummy_map[-2], self.dummy_map[-1], _create_replica(2), _create_replica(30)],
                2
            )

        with self.subTest("Dim 1 leaf 1"):
            # Check leaf 1 in dimension 1

            leaf1 = root_d1_entries[2].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 1,
                             f"Leaf 1 size should be 1, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")

            self._assert_leaf_node_properties(
                leaf1,
                [self.dummy_map[-1]],
            )
        with self.subTest("Dim 1 leaf 2"):
            # Check leaf 2 in dimension 1
            leaf2 = root_d1_entries[3].left_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 1,
                             f"Leaf 2 size should be 1, got: {leaf2_size} for set: {print_pretty(leaf2.set)}")

            self._assert_leaf_node_properties(
                leaf2,
                [item_map[2]],
            )
        with self.subTest("Dim 1 leaf 3"):
            # Check leaf 3 in dimension 1
            leaf3 = root_d1.right_subtree.node
            leaf3_size = leaf3.get_node_item_count()
            self.assertEqual(leaf3_size, 1,
                             f"Leaf 3 size should be 1, got: {leaf3_size} for set: {print_pretty(leaf3.set)}")

            self._assert_leaf_node_properties(
                leaf3,
                [item_map[30]],
            )

        with self.subTest("Dim 2 root"):
            root_d2 = root_d1.set.node
            root_d2_size = root_d2.get_node_item_count()
            self.assertEqual(root_d2_size, 2,
                             f"Root size in dim 2 should be 2, got: {root_d2_size} for set: {print_pretty(root_d2.set)}")
            self._assert_internal_node_properties(
                root_d2,
                [self.dummy_map[-2], self.dummy_map[-1]],
                3
            )
        
        with self.subTest("Dim 2 leaf 1"):
            # Check leaf 1 in dimension 2
            root_entries_d2 = list(root_d2.set)
            leaf1 = root_entries_d2[1].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 1,
                             f"Leaf 1 size in dim 2 should be 1, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf1,
                [self.dummy_map[-2]],
            )

        with self.subTest("Dim 2 right internal node"):
            right_internal_d2 = root_d2.right_subtree.node
            right_internal_d2_size = right_internal_d2.get_node_item_count()
            self.assertEqual(right_internal_d2_size, 2,
                             f"Right internal node size in dim 2 should be 2, got: {right_internal_d2_size} for set: {print_pretty(right_internal_d2.set)}")
            self._assert_internal_node_properties(
                right_internal_d2,
                [self.dummy_map[-1], _create_replica(30)],
                2
            )
        with self.subTest("Dim 2 leaf 2"):
            # Check leaf 2 in dimension 2
            right_internal_entries_d2 = list(right_internal_d2.set)
            leaf2 = right_internal_entries_d2[1].left_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 2,
                             f"Leaf 2 size in dim 2 should be 2, got: {leaf2_size} for set: {print_pretty(leaf2.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf2,
                [self.dummy_map[-1], _create_replica(2)]
            )
        with self.subTest("Dim 2 leaf 3"):
            # Check leaf 3 in dimension 2
            leaf3 = right_internal_d2.right_subtree.node
            leaf3_size = leaf3.get_node_item_count()
            self.assertEqual(leaf3_size, 1,
                             f"Leaf 3 size in dim 2 should be 1, got: {leaf3_size} for set: {print_pretty(leaf3.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf3,
                [_create_replica(30)]
            )

    def test_insertion_triggers_multiple_expansions_in_root(self):
        """Test that inserting an item triggers expansion when necessary"""
        tree = self.tree_k2
        k = 2
        rank_lists = [
            [2, 2],  # Dimension 1
            [3, 3],  # Dimension 2
            [1, 2],  # Dimension 3
        ]

        keys = self.find_keys_for_rank_lists(rank_lists, k)
        logger.debug(f"Keys: {keys}")

        # Create items
        item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        
        for i in range(2):
            key = keys[i]
            rank = rank_lists[0][i]
            item = self.create_item(key)
            tree, _ = tree.insert(item, rank=rank)        
        
        logger.debug(f"Tree after insertions: {print_pretty(tree)}")
        self.assertFalse(tree.is_empty(), "Tree should not be empty after insertions")
        # logger.debug(f"Tree structure: {tree.print_structure()}")
        logger.debug(f"Root set: {print_pretty(tree.node.set)}")
        

        with self.subTest("Dim 1 root"):
            # Insert one more item to trigger expansion in the root
            root_d1 = tree.node
            logger.debug(f"Root set in dim 1: {print_pretty(root_d1.set)}")
            root_d1_entries = list(root_d1.set)
            # logger.debug(f"Root entries in root set dim 1: {root_d1_entries}")
            root_d1_size = root_d1.get_node_item_count()
            self.assertEqual(root_d1_size, 4,
                            "Root size should be 4, got: "
                            f"{root_d1_size} for set: {print_pretty(root_d1.set)}"
            )
            self._assert_internal_node_properties(
                root_d1,
                [self.dummy_map[-2], self.dummy_map[-1], _create_replica(15), _create_replica(69)],
                2
            )

        with self.subTest("Dim 1 leaf 1"):
            leaf1 = root_d1_entries[2].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 1,
                             f"Leaf 1 size should be 1, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")

            self._assert_leaf_node_properties(
                leaf1,
                [self.dummy_map[-1]],
            )

        with self.subTest("Dim 1 leaf 2"):
            # Check leaf 2 in dimension 1
            leaf2 = root_d1_entries[3].left_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 1,
                             f"Leaf 2 size should be 1, got: {leaf2_size} for set: {print_pretty(leaf2.set)}")

            self._assert_leaf_node_properties(
                leaf2,
                [item_map[15]],
            )
            
        with self.subTest("Dim 1 leaf 3"):
            # Check leaf 3 in dimension 1
            leaf3 = root_d1.right_subtree.node
            leaf3_size = leaf3.get_node_item_count()
            self.assertEqual(leaf3_size, 1,
                             f"Leaf 3 size should be 1, got: {leaf3_size} for set: {print_pretty(leaf3.set)}")

            self._assert_leaf_node_properties(
                leaf3,
                [item_map[69]],
            )

        with self.subTest("Dim 2 root"):
            root_d2 = root_d1.set.node
            root_d2_size = root_d2.get_node_item_count()
            self.assertEqual(root_d2_size, 5,
                             f"Root size in dim 2 should be 5, got: {root_d2_size} for set: {print_pretty(root_d2.set)}")
            self._assert_internal_node_properties(
                root_d2,
                [self.dummy_map[-3], self.dummy_map[-2], self.dummy_map[-1], _create_replica(15), _create_replica(69)],
                3
            )

        with self.subTest("Dim 2 leaf 1"):
            # Check leaf 1 in dimension 2
            root_entries_d2 = list(root_d2.set)
            leaf1 = root_entries_d2[2].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 1,
                             f"Leaf 1 size in dim 2 should be 1, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf1,
                [self.dummy_map[-2]],
            )

        with self.subTest("Dim 2 leaf 2"):
            # Check leaf 2 in dimension 2
            leaf2 = root_entries_d2[3].left_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 1,
                             f"Leaf 2 size in dim 2 should be 1, got: {leaf2_size} for set: {print_pretty(leaf2.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf2,
                [self.dummy_map[-1]],
            )

        with self.subTest("Dim 2 leaf 3"):
            # Check leaf 3 in dimension 2
            leaf3 = root_entries_d2[4].left_subtree.node
            leaf3_size = leaf3.get_node_item_count()
            self.assertEqual(leaf3_size, 1,
                             f"Leaf 3 size in dim 2 should be 1, got: {leaf3_size} for set: {print_pretty(leaf3.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf3,
                [_create_replica(15)],
            )

        with self.subTest("Dim 2 leaf 4"):
            # Check leaf 4 in dimension 2
            leaf4 = root_d2.right_subtree.node
            leaf4_size = leaf4.get_node_item_count()
            self.assertEqual(leaf4_size, 1,
                             f"Leaf 4 size in dim 2 should be 1, got: {leaf4_size} for set: {print_pretty(leaf4.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf4,
                [_create_replica(69)],
            )

        with self.subTest("Dim 3 root"):
            root_d3 = root_d2.set.node
            root_d3_size = root_d3.get_node_item_count()
            self.assertEqual(root_d3_size, 2,
                             f"Root size in dim 3 should be 2, got: {root_d3_size} for set: {print_pretty(root_d3.set)}")
            self._assert_internal_node_properties(
                root_d3,
                [self.dummy_map[-3], self.dummy_map[-1]],
                6
            )

        with self.subTest("Dim 3 leaf 1"):
            # Check leaf 1 in dimension 3
            root_entries_d3 = list(root_d3.set)
            leaf1 = root_entries_d3[1].left_subtree.node
            leaf1_size = leaf1.get_node_item_count()
            self.assertEqual(leaf1_size, 2,
                             f"Leaf 1 size in dim 3 should be 1, got: {leaf1_size} for set: {print_pretty(leaf1.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf1,
                [self.dummy_map[-3], self.dummy_map[-2]],
            )

        with self.subTest("Dim 3 right internal node"):
            # Check right internal node in dimension 3
            right_internal_d3 = root_d3.right_subtree.node
            right_internal_d3_size = right_internal_d3.get_node_item_count()
            self.assertEqual(right_internal_d3_size, 2,
                             f"Right internal node size in dim 3 should be 2, got: {right_internal_d3_size} for set: {print_pretty(right_internal_d3.set)}")
            self._assert_internal_node_properties(
                right_internal_d3,
                [self.dummy_map[-1], _create_replica(69)],
                2
            )

        with self.subTest("Dim 3 leaf 2"):
            # Check leaf 2 in dimension 3
            right_internal_entries_d3 = list(right_internal_d3.set)
            leaf2 = right_internal_entries_d3[1].left_subtree.node
            leaf2_size = leaf2.get_node_item_count()
            self.assertEqual(leaf2_size, 2,
                             f"Leaf 2 size in dim 3 should be 1, got: {leaf2_size} for set: {print_pretty(leaf2.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf2,
                [self.dummy_map[-1], _create_replica(15)],
            )

        with self.subTest("Dim 3 leaf 3"):
            # Check leaf 3 in dimension 3
            leaf3 = right_internal_d3.right_subtree.node
            leaf3_size = leaf3.get_node_item_count()
            self.assertEqual(leaf3_size, 1,
                             f"Leaf 3 size in dim 3 should be 1, got: {leaf3_size} for set: {print_pretty(leaf3.set)}")
            self._assert_leaf_node_properties_for_leaf_in_expanded_internal_tree(
                leaf3,
                [_create_replica(69)],
            )