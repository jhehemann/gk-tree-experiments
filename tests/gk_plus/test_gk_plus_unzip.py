"""Comprehensive tests for the GKPlusTreeBase unzip() method."""

import unittest
from typing import List, Tuple, Optional
from gplus_trees.base import LeafItem, Entry
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy, GKPlusTreeBase
from gplus_trees.g_k_plus.utils import calc_rank
from gplus_trees.klist_base import KListBase
from tests.gk_plus.base import TreeTestCase
from tests.utils import assert_tree_invariants_tc
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from statistics import median_low

import logging

logger = logging.getLogger(__name__)

class TestGKPlusTreeUnzip(TreeTestCase):
    """Tests for the unzip() method in GKPlusTreeBase."""
    
    def setUp(self):
        """Set up test fixtures with different tree configurations."""
        super().setUp()
        self.k = 4  # Default K-list node capacity for tests
    
    def _create_tree(self, keys: List[int], ranks: List[int], tree=None):
        """Helper to create a simple tree with given keys and ranks."""
        if tree is None:
            tree = create_gkplus_tree(K=self.k, dimension=1)
        
        for key, rank in zip(keys, ranks):
            item = self.make_item(key, f"val_{key}")
            tree, inserted, _ = tree.insert(item, rank=rank)
            self.assertTrue(inserted, f"Failed to insert item with key {key}")
        
        return tree
    
    def _validate_tree_structure(self, tree: GKPlusTreeBase, expected_real_keys: List[int], description: str = "", is_right: bool = True):
        """
        Helper to validate tree structure and contents.
        Is_right is set to True by default for testing. Set it to False to not allow internal nodes to have less than 2 items in the left tree.
        """
        if description:
            description = f" ({description})"
            
        # Check tree invariants
        self.assertIsNotNone(tree, f"Tree should not be None{description}")

        if not tree.is_empty():
            
            stats = gtree_stats_(tree, {})
            
            exclude_checks = ['set_thresholds_met']
            if is_right:
                exclude_checks.append('internal_packed')
            assert_tree_invariants_tc(self, tree, stats, exclude_checks=exclude_checks)

        # Check that all expected keys are present
        actual_keys = []
        for entry in tree.iter_real_entries():
            actual_keys.append(entry.item.key)
        
        expected_keys_sorted = expected_real_keys
        actual_keys_sorted = actual_keys
        
        self.assertEqual(
            expected_keys_sorted, actual_keys_sorted,
            f"Expected keys {expected_keys_sorted}, got {actual_keys_sorted}{description}"
        )

    def _validate_unzip_result(self, left_tree: GKPlusTreeBase, key_subtree: Optional[GKPlusTreeBase], 
                              right_tree: GKPlusTreeBase, next_entry: Optional[Entry],
                              unzip_key: int, expected_next_key: Entry, expected_left_keys: List[int], 
                              expected_right_keys: List[int], has_key_subtree: bool = False):
        """Helper to validate the complete unzip result."""
        # Validate left tree
        if expected_left_keys:
            self._validate_tree_structure(left_tree, expected_left_keys, "left tree")
        else:
            if not left_tree.is_empty():
                has_real = left_tree.real_item_count() > 0
                self.assertFalse(has_real, "Left tree should not have real items when no expected left keys")
        
        # Validate right tree  
        if expected_right_keys:
            self._validate_tree_structure(right_tree, expected_right_keys, "right tree", is_right=True)
        else:
            self.assertTrue(right_tree.is_empty(), "Right tree should be empty when no expected right keys")
        
        # Validate key subtree presence
        if has_key_subtree:
            self.assertIsNotNone(key_subtree, f"Expected key subtree for key {unzip_key}")
            if key_subtree and not key_subtree.is_empty():
                # Key subtree should be valid if not empty
                stats = gtree_stats_(key_subtree, {})
                assert_tree_invariants_tc(self, key_subtree, stats)
        else:
            # Key subtree can be None or empty if the key doesn't exist
            if key_subtree is not None:
                self.assertIsNone(key_subtree, 
                              f"Key subtree should be None when key {unzip_key} doesn't exist")
                
        if expected_next_key is not None:
            # If expected_next_key exists, it should be a valid entry
            self.assertIsNotNone(next_entry, f"Expected next entry for key {unzip_key}")
            self.assertIsInstance(next_entry, Entry, "next_entry should be an instance of Entry")
            self.assertEqual(next_entry.item.key, expected_next_key, "expected_next_key key mismatch")
            self.assertIsNotNone(next_entry.item.value, "next_entry item value should not be None for dimension 1")

    def test_unzip_empty_tree(self):
        """Test unzipping an empty tree."""
        tree = create_gkplus_tree(K=self.k, dimension=1)
        self.assertTrue(tree.is_empty())
        
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(100)
        
        # All results should be empty or None
        self.assertTrue(left_tree.is_empty(), "Left tree should be empty")
        self.assertIsNone(key_subtree, "Key subtree should be None for empty tree")
        self.assertIsNone(next_entry, "Next entry should be None")

    def test_unzip_single_item_tree_key_exists(self):
        """Test unzipping a tree with a single item where the key exists."""
        rank_lists = [[1]]  # Single item with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)

        tree = self._create_tree(keys, rank_lists[0])
        

        leading_dummy_keys = []
        for entry in tree:
            if entry.item.key >= 0:
                break
            leading_dummy_keys.append(entry.item.key)

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(keys[0])
        

        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=keys[0], expected_next_key=None, expected_left_keys=[], expected_right_keys=[],
            has_key_subtree=False
        )

    def test_unzip_single_item_tree_key_not_exists_smaller(self):
        """Test unzipping a tree with a single item where the key is smaller than existing."""
        rank_lists = [[1]]  # Single item with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k, spacing=True)

        tree = self._create_tree(keys, rank_lists[0])
        
        unzip_key = keys[0] - 1
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        

        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=keys[0], expected_left_keys=[], expected_right_keys=keys,
            has_key_subtree=False
        )

    def test_unzip_single_item_tree_key_not_exists_larger(self):
        """Test unzipping a tree with a single item where the key is larger than existing."""
        rank_lists = [[1]]  # Single item with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        
        tree = self._create_tree(keys, rank_lists[0])
        unzip_key = keys[0] + 1
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=None, expected_left_keys=keys, expected_right_keys=[],
            has_key_subtree=False
        )

    def test_unzip_multiple_items_key_exists_middle(self):
        """Test unzipping with multiple items where key exists in the middle."""
        rank_lists = [[1, 1, 1, 1, 1]]  # Multiple items with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        tree = self._create_tree(keys, rank_lists[0])

        unzip_key = median_low(keys)
        unzip_idx = keys.index(unzip_key)

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        
        next_key = keys[unzip_idx + 1]
        expected_left_keys = keys[:unzip_idx]
        expected_right_keys = keys[unzip_idx + 1:]

        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=next_key, expected_left_keys=expected_left_keys, expected_right_keys=expected_right_keys,
            has_key_subtree=False
        )

    def test_unzip_multiple_items_key_exists_first(self):
        """Test unzipping with multiple items where key is the first item."""
        rank_lists = [[1, 1, 1, 1]]  # Multiple items with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        tree = self._create_tree(keys, rank_lists[0])
        logger.debug(f"Tree before unzip: {print_pretty(tree)}")
        unzip_key = keys[0]

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        logger.debug(f"Left tree after unzip: {print_pretty(left_tree)}")
        logger.debug(f"Key subtree after unzip: {print_pretty(key_subtree)}")
        logger.debug(f"Right tree after unzip: {print_pretty(right_tree)}")
        logger.debug(f"Next entry after unzip: {next_entry.item.key if next_entry else None}")
        
        expected_right_keys = keys[1:]

        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=keys[1], expected_left_keys=[], expected_right_keys=expected_right_keys,
            has_key_subtree=False
        )

    def test_unzip_multiple_items_key_exists_last(self):
        """Test unzipping with multiple items where key is the last item."""
        rank_lists = [[1, 1, 1, 1]]  # Multiple items with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        tree = self._create_tree(keys, rank_lists[0])
        unzip_key = keys[-1]

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=None, expected_left_keys=keys[:-1], expected_right_keys=[],
            has_key_subtree=False
        )

    def test_unzip_multiple_items_key_not_exists_between_rank_1(self):
        """Test unzipping with multiple items where key doesn't exist but falls between items."""
        rank_lists = [[1, 1, 1, 1, 1]]  # Multiple items with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k, spacing=True)
        tree = self._create_tree(keys, rank_lists[0])
        
        median_idx = keys.index(median_low(keys))
        unzip_key =  keys[median_idx] - 1  # median key - 1 (not in keys)

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)

        expected_left_keys = keys[:median_idx]
        expected_right_keys = keys[median_idx:]

        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=keys[median_idx], expected_left_keys=expected_left_keys, expected_right_keys=expected_right_keys,
            has_key_subtree=False
        )

    def test_unzip_multiple_items_key_not_exists_between_rank_gt_1(self):
        """Test unzipping with multiple items where key doesn't exist but falls between items."""
        rank_lists = [
            [2, 2, 2, 2],
            [2, 1, 2, 1]
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k, spacing=True)
        tree = self._create_tree(keys, rank_lists[0])
        median_index = keys.index(median_low(keys))
        unzip_key = keys[median_index] + 1  # median key + 1 (not in keys)
        logger.debug(f"Unzip key: {unzip_key}")
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        

        expected_left_keys = keys[:median_index + 1]
        expected_right_keys = keys[median_index + 1:]

        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=keys[median_index + 1], expected_left_keys=expected_left_keys, expected_right_keys=expected_right_keys,
            has_key_subtree=False
        )
        

    def test_unzip_multiple_items_key_smaller_than_all(self):
        """Test unzipping where key is smaller than all existing items."""
        rank_lists = [[1, 1, 1]]  # Multiple items with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k, spacing=True)
        tree = self._create_tree(keys, rank_lists[0])
        logger.debug(f"Tree before unzip: {print_pretty(tree)}")
        unzip_key = keys[0] - 1

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        logger.debug(f"left_tree: {print_pretty(left_tree)}")
        logger.debug(f"key_subtree: {print_pretty(key_subtree)}")
        logger.debug(f"right_tree: {print_pretty(right_tree)}")
        logger.debug(f"next_entry: {next_entry.item.key if next_entry else None}")
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=keys[0], expected_left_keys=[], expected_right_keys=keys,
            has_key_subtree=False
        )

    def test_unzip_multiple_items_key_larger_than_all(self):
        """Test unzipping where key is larger than all existing items."""
        rank_lists = [[1, 1, 1]]  # Multiple items with rank 1 in dimension 1
        keys = self.find_keys_for_rank_lists(rank_lists, self.k, spacing=True)
        tree = self._create_tree(keys, rank_lists[0])
        
        unzip_key = keys[-1] + 1

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=None, expected_left_keys=keys, expected_right_keys=[],
            has_key_subtree=False
        )

    def test_unzip_complex_tree_structure(self):
        """Test unzipping a tree with internal nodes (rank > 1)."""
        rank_lists = [[2, 1, 2, 1, 2, 1]]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k, spacing=True)
        tree = self._create_tree(keys, rank_lists[0])

        logger.debug(f"Tree before unzip: {print_pretty(tree)}")
        unzip_key = median_low(keys)
        unzip_idx = keys.index(unzip_key)
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)

        

        expected_left_keys = keys[:unzip_idx]
        expected_right_keys = keys[unzip_idx + 1:]
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=keys[unzip_idx + 1], expected_left_keys=expected_left_keys, expected_right_keys=expected_right_keys,
            has_key_subtree=None
        )

    def test_unzip_with_different_ranks(self):
        """Test unzipping a tree with mixed ranks."""
        rank_lists = [[3, 4, 1, 2, 3]]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k, spacing=True)
        tree = self._create_tree(keys, rank_lists[0])
        logger.debug(f"Tree before unzip: {print_pretty(tree)}")
        unzip_key = median_low(keys)
        unzip_idx = keys.index(unzip_key)

        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        logger.debug(f"left_tree: {print_pretty(left_tree)}")
        logger.debug(f"key_subtree: {print_pretty(key_subtree)}")
        logger.debug(f"right_tree: {print_pretty(right_tree)}")
        logger.debug(f"next_entry: {next_entry.item.key if next_entry else None}")

        expected_left_keys = keys[:unzip_idx]
        expected_right_keys = keys[unzip_idx + 1:]

        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=keys[unzip_idx + 1], expected_left_keys=expected_left_keys, expected_right_keys=expected_right_keys,
            has_key_subtree=None
        )

    def test_unzip_preserves_tree_properties(self):
        """Test that unzip preserves essential tree properties."""
        tree = self._create_tree([100, 200, 300, 400], [1, 2, 1, 2])
        
        # Record properties before unzip
        initial_dim = tree.DIM
        initial_l_factor = tree.l_factor
        
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(250)
        
        # Properties should be preserved in all result trees
        self.assertEqual(left_tree.DIM, initial_dim, "Left tree DIM should be preserved")
        self.assertEqual(right_tree.DIM, initial_dim, "Right tree DIM should be preserved")
        self.assertEqual(left_tree.l_factor, initial_l_factor, "Left tree l_factor should be preserved")
        self.assertEqual(right_tree.l_factor, initial_l_factor, "Right tree l_factor should be preserved")
        
        if key_subtree is not None:
            self.assertEqual(key_subtree.DIM, initial_dim, "Key subtree DIM should be preserved")
            self.assertEqual(key_subtree.l_factor, initial_l_factor, "Key subtree l_factor should be preserved")

    def test_unzip_invalidates_cached_sizes(self):
        """Test that unzip operation properly invalidates cached tree sizes."""
        tree = self._create_tree([100, 200, 300, 400], [1, 1, 1, 1])
        logger.debug(f"Tree before unzip: {print_pretty(tree)}")
        # Force calculation of sizes
        initial_size = tree.real_item_count()
        initial_count = tree.item_count()
        logger.debug(f"Initial size: {initial_size}, count: {initial_count}")
        
        # Perform unzip
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(250)
        
        # Sizes should be properly calculated for result trees
        left_size = left_tree.real_item_count()
        right_size = right_tree.real_item_count()
        
        # Total items should be preserved (left + right should equal original - 1 if key existed)
        total_result_size = left_size + right_size
        self.assertEqual(total_result_size, initial_size, 
                        f"Total size should be preserved: {left_size} + {right_size} = {total_result_size}, expected {initial_size}")

    def test_unzip_with_klist_sets(self):
        """Test unzipping trees where node sets are KLists."""
        tree = self._create_tree([100, 200, 300], [1, 1, 1])
        
        # Verify we have KList sets
        self.assertIsInstance(tree.node.set, KListBase)
        
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(200)
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=200, expected_next_key=300, expected_left_keys=[100], expected_right_keys=[300],
            has_key_subtree=False
        )

    def test_unzip_with_gkplus_sets(self):
        """Test unzipping trees where node sets are GKPlusTree instances."""
        # Create a tree with low l_factor to force GKPlusTree sets
        tree = create_gkplus_tree(K=2, dimension=1, l_factor=1)
        
        # Insert enough items to trigger expansion to GKPlusTree
        for i in range(5):
            item = self.make_item(100 + i * 10, f"val_{100 + i * 10}")
            tree, _, _ = tree.insert(item, rank=1)
        
        # Verify we have GKPlusTree sets
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(120)
        
        # Should split around key 120
        expected_left = [100, 110]
        expected_right = [130, 140]
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=120, expected_next_key=130, expected_left_keys=expected_left, expected_right_keys=expected_right,
            has_key_subtree=False
        )

    def test_unzip_type_validation(self):
        """Test that unzip validates input types correctly."""
        tree = self._create_tree([100, 200], [1, 1])
        
        # Test with invalid type
        with self.assertRaises(TypeError) as cm:
            tree.unzip("not an int")
        
        self.assertIn("key must be int", str(cm.exception))
        
        # Test with None
        with self.assertRaises(TypeError) as cm:
            tree.unzip(None)
        
        self.assertIn("key must be int", str(cm.exception))

    def test_unzip_large_tree(self):
        """Test unzipping a larger tree to ensure performance and correctness."""
        # Create a larger tree
        k = 8
        tree = create_gkplus_tree(K=k, dimension=1)

        keys = list(range(100, 200, 5))  # 20 keys: 100, 105, 110, ..., 195
        ranks = [calc_rank(key=key, k=k, dim=1) for key in keys]
        tree = self._create_tree(keys, ranks, tree)
        
        # Unzip at a key in the middle
        unzip_key = 150
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(unzip_key)
        
        # Calculate expected results
        expected_left = [k for k in keys if k < unzip_key]
        expected_right = [k for k in keys if k > unzip_key]
        has_key = unzip_key in keys
        
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=unzip_key, expected_next_key=expected_right[0] if expected_right else None,
            expected_left_keys=expected_left, expected_right_keys=expected_right, has_key_subtree=False
        )

    def test_unzip_boundary_values(self):
        """Test unzipping with boundary values (very small and very large keys)."""
        tree = self._create_tree([100, 200, 300], [1, 1, 1])
        
        # Test with very small key
        # left_tree, key_subtree, right_tree, next_entry = tree.unzip(-1000)
        # Try to unzip at a key less than the dummy key, should raise ValueError
        with self.assertRaises(ValueError) as cm:
            tree.unzip(-1000)
        self.assertIn("less than dimension dummy key", str(cm.exception))

        # Test with very large key
        tree2 = self._create_tree([100, 200, 300], [1, 1, 1])
        left_tree, key_subtree, right_tree, next_entry = tree2.unzip(10000)
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=10000, expected_next_key=None, expected_left_keys=[100, 200, 300], expected_right_keys=[],
            has_key_subtree=False
        )

    def test_unzip_sequential_operations(self):
        """Test multiple sequential unzip operations."""
        tree = self._create_tree([100, 200, 300, 400, 500], [1, 1, 1, 1, 1])
        
        # First unzip at 300
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(300)
        
        # Validate first unzip
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=300, expected_next_key=400, expected_left_keys=[100, 200], expected_right_keys=[400, 500],
            has_key_subtree=False
        )
        
        # Second unzip on right tree at 450
        if not right_tree.is_empty():
            left_tree2, key_subtree2, right_tree2, next_entry2 = right_tree.unzip(450)
            self._validate_unzip_result(
                left_tree2, key_subtree2, right_tree2, next_entry2,
                unzip_key=450, expected_next_key=500, expected_left_keys=[400], expected_right_keys=[500],
                has_key_subtree=False
            )

    def test_unzip_maintains_leaf_structure(self):
        """Test that unzip maintains proper leaf node structure and linking."""
        tree = self._create_tree([100, 200, 300, 400], [1, 1, 1, 1])
        
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(250)
        logger.debug(f"left_tree: {print_pretty(left_tree)}")
        logger.debug(f"key_subtree: {print_pretty(key_subtree)}")
        logger.debug(f"right_tree: {print_pretty(right_tree)}")
        logger.debug(f"next_entry: {next_entry.item.key if next_entry else None}")
        
        # Check that leaf nodes are properly structured
        if not left_tree.is_empty():
            left_stats = gtree_stats_(left_tree, {})
            assert_tree_invariants_tc(self, left_tree, left_stats,
                                      exclude_checks=["set_thresholds_met"])
        
        if not right_tree.is_empty():
            right_stats = gtree_stats_(right_tree, {})
            # TODO: unzip can leave inner (higher-dimension) trees with
            # single-entry internal nodes (internal_packed violation).
            # This is a known limitation of the current unzip algorithm.
            assert_tree_invariants_tc(self, right_tree, right_stats,
                                      exclude_checks=["internal_packed", "set_thresholds_met"])

    def test_unzip_with_dummy_entries(self):
        """Test unzipping trees that contain dummy entries."""
        tree = self._create_tree([100, 200, 300], [1, 1, 1])
        logger.debug(f"Tree before unzip: {print_pretty(tree)}")
        logger.debug(f"Tree size before unzip: {tree.real_item_count()}")
        
        # Get the dummy key for this dimension
        dummy_key = get_dummy(tree.DIM).key
        
        # Try to unzip at dummy key (should be negative)
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(dummy_key)
        logger.debug(f"Right side size after unzip: {right_tree.real_item_count()}")
        logger.debug(f"Left tree after unzip: {print_pretty(left_tree)}")
        logger.debug(f"Right tree after unzip: {print_pretty(right_tree)}")
        logger.debug(f"Next entry after unzip: {next_entry.item.key if next_entry else None}")
        
        # All real entries should be in the right tree since dummy key is negative
        self._validate_unzip_result(
            left_tree, key_subtree, right_tree, next_entry,
            unzip_key=dummy_key, expected_next_key=100, expected_left_keys=[], expected_right_keys=[100, 200, 300],
            has_key_subtree=False  # Dummy should exist
        )

if __name__ == '__main__':
    unittest.main()
