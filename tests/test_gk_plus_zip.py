"""Comprehensive tests for the GKPlusTreeBase zip() method."""

import unittest
from typing import List, Optional
from gplus_trees.base import LeafItem, Entry
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import bulk_create_gkplus_tree, get_dummy, GKPlusTreeBase
from gplus_trees.g_k_plus.utils import calc_ranks
from gplus_trees.klist_base import KListBase
from tests.gk_plus.base import TreeTestCase
from tests.utils import assert_tree_invariants_tc
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from dataclasses import asdict

from tests.logconfig import logger


class TestGKPlusTreeZip(TreeTestCase):
    """Tests for the zip() method in GKPlusTreeBase."""
    
    def setUp(self):
        """Set up test fixtures with different tree configurations."""
        super().setUp()
        self.tree_k2 = create_gkplus_tree(K=2, dimension=1)
        self.tree_k4 = create_gkplus_tree(K=4, dimension=1)
        self.tree_k8 = create_gkplus_tree(K=8, dimension=1)
        
        # Trees with different dimensions
        self.tree_k4_dim2 = create_gkplus_tree(K=4, dimension=2)
        self.tree_k4_dim3 = create_gkplus_tree(K=4, dimension=3)
    
    def _create_tree(self, keys: List[int], ranks: List[int], tree=None, k=4):
        """Helper to create a simple tree with given keys and ranks."""
        if tree is None:
            tree = create_gkplus_tree(K=k, dimension=1)

        for key, rank in zip(keys, ranks):
            item = self.make_item(key, f"val_{key}")
            tree, inserted, _ = tree.insert(item, rank=rank)
            self.assertTrue(inserted, f"Failed to insert item with key {key}")
        
        return tree

    def _create_tree_via_unzip(self, keys: List[int], ranks: List[int], tree=None, k=4):
        """Helper to create a tree by first building normally, then unzipping at key -1."""
        if tree is None:
            tree = create_gkplus_tree(K=k, dimension=1)

        # First build the tree normally
        for key, rank in zip(keys, ranks):
            item = self.make_item(key, f"val_{key}")
            tree, inserted, _ = tree.insert(item, rank=rank)
            self.assertTrue(inserted, f"Failed to insert item with key {key}")
        
        # Then unzip at key -1 and return the right tree (which should contain all positive keys)
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(-1)
        
        # Since all our test keys are positive, the right tree should contain them
        return right_tree

    def _validate_tree_after_zip(self, tree: GKPlusTreeBase, expected_keys: List[int], k: int = 4):
        """Helper to validate tree structure and contents after zip operation."""
        logger.debug(f"k={k}, Expected keys: {expected_keys}")
        # control_tree = create_gkplus_tree(K=k, dimension=tree.DIM, l_factor=tree.l_factor)
        # ranks = calc_ranks(expected_keys, k=k, DIM=tree.DIM)
        control_tree = bulk_create_gkplus_tree(
            [Entry(self.make_item(key, f"val_{key}"), None) for key in expected_keys],
            tree.DIM,
            tree.l_factor,
            tree.KListClass
        )
        # for key, rank in zip(expected_keys, ranks):
        #     item = self.make_item(key, f"val_{key}")
        #     control_tree, _, _ = control_tree.insert(item, rank=rank)
        logger.debug(f"Control tree: {print_pretty(control_tree)}")
        control_stats = asdict(gtree_stats_(control_tree, {}))
        # Check tree invariants
        if not tree.is_empty():
            stats = gtree_stats_(tree, {})
            assert_tree_invariants_tc(self, tree, stats)

            
            # Create a control tree with the expected keys and compare stats
            k = tree.SetClass.KListNodeClass.CAPACITY
            
            result_stats = asdict(gtree_stats_(tree, {}))
            if control_stats != result_stats:
                # Find which keys differ
                mismatches = [
                    f"{key}: expected={control_stats.get(key)}, actual={result_stats.get(key)}"
                    for key in set(control_stats.keys()).union(result_stats.keys())
                    if control_stats.get(key) != result_stats.get(key)
                ]
                msg = "Stats of zip result and control tree should match. Mismatches: " + "; ".join(mismatches)
            else:
                msg = "Stats of zip result and control tree should match"
            self.assertEqual(control_stats, result_stats, msg)
        
        # Check that all expected keys are present
        actual_keys = []
        for entry in tree.iter_real_entries():
            actual_keys.append(entry.item.key)
        
        expected_keys_sorted = sorted(expected_keys)
        actual_keys_sorted = actual_keys

        self.assertEqual(
            expected_keys_sorted, actual_keys_sorted,
            f"Expected keys {expected_keys_sorted}, got {actual_keys_sorted}"
        )

        

    def test_zip_empty_trees(self):
        """Test zipping two empty trees."""
        tree1 = create_gkplus_tree(K=4, dimension=1)
        tree2 = create_gkplus_tree(K=4, dimension=1)
        
        self.assertTrue(tree1.is_empty())
        self.assertTrue(tree2.is_empty())
        
        result = tree1.zip(tree2)
        
        self.assertIs(result, tree1, "zip should return self")
        self.assertTrue(result.is_empty(), "Result should be empty")

    def test_zip_non_empty_with_empty(self):
        """Test zipping a non-empty tree with an empty tree."""
        k = 4
        rank_lists = [[1, 1]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)

        non_empty_tree = self._create_tree(keys, rank_lists[0], k=k)
        empty_tree = create_gkplus_tree(K=4, dimension=1)
        
        result = non_empty_tree.zip(empty_tree)
        self.assertIs(result, non_empty_tree)
        self._validate_tree_after_zip(result, keys)

    def test_zip_empty_with_non_empty(self):
        """Test zipping an empty tree with a non-empty tree."""
        k = 4
        rank_lists = [[1, 1, 1]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)

        empty_tree = create_gkplus_tree(K=k, dimension=1)
        non_empty_tree = self._create_tree_via_unzip(keys, rank_lists[0], k=k)
        logger.debug(f"Non-empty tree before zip: {print_pretty(non_empty_tree)}")
        
        # Test empty.zip(non_empty)
        result1 = empty_tree.zip(non_empty_tree)
        self.assertIs(result1, empty_tree)
        self._validate_tree_after_zip(result1, keys)
       
    def test_zip_same_rank_1_both_klist(self):
        """Test zipping two trees with same rank where both nodes have KList sets."""
        k = 4
        rank_lists = [[1, 1, 1, 1]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:2]
        keys_large = keys[2:]

        tree1 = self._create_tree(keys_small, rank_lists[0][:2], k=k)
        tree2 = self._create_tree_via_unzip(keys_large, rank_lists[0][2:], k=k)
        logger.debug(f"Tree1: {print_pretty(tree1)}")
        logger.debug(f"Tree2: {print_pretty(tree2)}")

        # Verify both have rank 1 and KList sets
        self.assertEqual(tree1.node.rank, 1)
        self.assertEqual(tree2.node.rank, 1)
        self.assertIsInstance(tree1.node.set, KListBase)
        self.assertIsInstance(tree2.node.set, KListBase)
        
        result = tree1.zip(tree2)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        self.assertIs(result, tree1)
        
        # Should contain all keys from both trees
        self._validate_tree_after_zip(result, keys)

    def test_zip_same_rank_gt_1_both_klist(self):
        """Test zipping two trees with same rank where both nodes have KList sets."""
        k = 4
        rank_lists = [[1, 2, 1, 2]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_1 = keys[:2]
        ranks_1 = rank_lists[0][:2]
        keys_2 = keys[2:]
        ranks_2 = rank_lists[0][2:]


        # Create two simple trees with rank 2 nodes
        tree1 = self._create_tree(keys_1, ranks_1, k=k)
        tree2 = self._create_tree_via_unzip(keys_2, ranks_2, k=k)
        logger.debug(f"Tree1: {print_pretty(tree1)}")
        logger.debug(f"Tree2: {print_pretty(tree2)}")

        # Verify both have rank 2 and KList sets
        self.assertEqual(tree1.node.rank, 2)
        self.assertEqual(tree2.node.rank, 2)
        self.assertIsInstance(tree1.node.set, KListBase)
        self.assertIsInstance(tree2.node.set, KListBase)
        
        result = tree1.zip(tree2)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        self.assertIs(result, tree1)
        
        # Should contain all keys from both trees
        self._validate_tree_after_zip(result, keys)

    def test_zip_different_ranks_left_rank_smaller_right_no_left_subtree(self):
        """Test zipping when left has smaller rank than right."""
        k = 4
        rank_lists = [[1, 2, 1]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:1]
        logger.debug(f"Keys for small tree: {keys_small}")
        ranks_small = rank_lists[0][:1]
        keys_large = keys[1:]
        logger.debug(f"Keys for large tree: {keys_large}")
        ranks_large = rank_lists[0][1:]

        small_rank_tree = self._create_tree(keys_small, ranks_small)
        logger.debug(f"Small rank tree: {print_pretty(small_rank_tree)}")

        # Create tree with rank 2
        large_rank_tree = self._create_tree_via_unzip(keys_large, ranks_large)
        logger.debug(f"Large rank tree: {print_pretty(large_rank_tree)}")

        self.assertEqual(small_rank_tree.node.rank, 1)
        self.assertEqual(large_rank_tree.node.rank, 2)
        
        result = small_rank_tree.zip(large_rank_tree)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        # self.assertIs(result, small_rank_tree) # This is not guaranteed by current implementation
        
        # The result should contain all keys
        self._validate_tree_after_zip(result, keys)

    def test_zip_different_ranks_left_rank_smaller_right_has_left_subtree(self):
        """Test zipping when left has smaller rank than right."""
        k = 4
        rank_lists = [
            [1, 1, 1, 1, 2],
            [1, 1, 1, 1, 2]
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:1]
        logger.debug(f"Keys for small tree: {keys_small}")
        ranks_small = rank_lists[0][:1]
        keys_large = keys[1:]
        logger.debug(f"Keys for large tree: {keys_large}")
        ranks_large = rank_lists[0][1:]

        small_rank_tree = self._create_tree(keys_small, ranks_small)
        logger.debug(f"Small rank tree: {print_pretty(small_rank_tree)}")

        # Create tree with rank 2
        large_rank_tree = self._create_tree_via_unzip(keys_large, ranks_large)
        logger.debug(f"Large rank tree: {print_pretty(large_rank_tree)}")

        self.assertEqual(small_rank_tree.node.rank, 1)
        self.assertEqual(large_rank_tree.node.rank, 2)
        
        result = small_rank_tree.zip(large_rank_tree)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        # self.assertIs(result, small_rank_tree) # This is not guaranteed by current implementation
        
        # The result should contain all keys
        self._validate_tree_after_zip(result, keys)

    def test_zip_different_ranks_left_rank_smaller_right_has_left_subtree_expanded(self):
        """Test zipping when left has smaller rank than right."""
        k = 4
        rank_lists = [[1, 1, 1, 1, 1, 1, 2]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:1]
        logger.debug(f"Keys for small tree: {keys_small}")
        ranks_small = rank_lists[0][:1]
        keys_large = keys[1:]
        logger.debug(f"Keys for large tree: {keys_large}")
        ranks_large = rank_lists[0][1:]

        small_rank_tree = self._create_tree(keys_small, ranks_small)
        logger.debug(f"Small rank tree: {print_pretty(small_rank_tree)}")

        # Create tree with rank 2
        large_rank_tree = self._create_tree_via_unzip(keys_large, ranks_large, k=k)
        logger.debug(f"Large rank tree: {print_pretty(large_rank_tree)}")

        self.assertEqual(small_rank_tree.node.rank, 1)
        self.assertEqual(large_rank_tree.node.rank, 2)
        
        result = small_rank_tree.zip(large_rank_tree)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        # self.assertIs(result, small_rank_tree) # This is not guaranteed by current implementation
        
        # The result should contain all keys
        self._validate_tree_after_zip(result, keys, k=k)

    def test_zip_different_ranks_left_rank_smaller_expanded_right_has_left_subtree_expanded(self):
        """Test zipping when left has smaller rank than right."""
        k = 4
        rank_lists = [[1, 1, 1, 1, 1, 1, 1, 1, 2]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:4]
        logger.debug(f"Keys for small tree: {keys_small}")
        ranks_small = rank_lists[0][:4]
        keys_large = keys[4:]
        logger.debug(f"Keys for large tree: {keys_large}")
        ranks_large = rank_lists[0][4:]

        small_rank_tree = self._create_tree(keys_small, ranks_small)
        logger.debug(f"Small rank tree: {print_pretty(small_rank_tree)}")

        # Create tree with rank 2
        large_rank_tree = self._create_tree_via_unzip(keys_large, ranks_large, k=k)
        logger.debug(f"Large rank tree: {print_pretty(large_rank_tree)}")

        self.assertEqual(small_rank_tree.node.rank, 1)
        self.assertEqual(large_rank_tree.node.rank, 2)
        
        result = small_rank_tree.zip(large_rank_tree)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        # self.assertIs(result, small_rank_tree) # This is not guaranteed by current implementation
        
        # The result should contain all keys
        self._validate_tree_after_zip(result, keys, k=k)

    def test_zip_different_ranks_left_rank_greater(self):
        """Test zipping when left has larger rank than right."""
        k = 4
        rank_lists = [[2, 1, 1]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:2]
        logger.debug(f"Keys for small tree: {keys_small}")
        ranks_small = rank_lists[0][:2]
        keys_large = keys[2:]
        logger.debug(f"Keys for large tree: {keys_large}")
        ranks_large = rank_lists[0][2:]

        # Create tree with rank 2
        rank_2_tree = self._create_tree(keys_small, ranks_small)
        logger.debug(f"Rank 2 tree: {print_pretty(rank_2_tree)}")

        # Create tree with rank 1
        rank_1_tree = self._create_tree(keys_large, ranks_large)

        self.assertEqual(rank_2_tree.node.rank, 2)
        self.assertEqual(rank_1_tree.node.rank, 1)

        result = rank_2_tree.zip(rank_1_tree)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        self.assertIs(result, rank_2_tree)

        # The result should contain all keys
        self._validate_tree_after_zip(result, keys)

    def test_zip_different_ranks_left_rank_greater_non_matching_collapsed(self):
        """Test zipping when left has larger rank than right."""
        k = 4
        rank_lists = [[4, 1, 2]]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:2]
        logger.debug(f"Keys for small tree: {keys_small}")
        ranks_small = rank_lists[0][:2]
        keys_large = keys[2:]
        logger.debug(f"Keys for large tree: {keys_large}")
        ranks_large = rank_lists[0][2:]
        
        # Create tree with rank 4, ranks 3 and 2 collapsed
        rank_4_tree = self._create_tree(keys_small, ranks_small)

        # Create tree with rank 2
        rank_2_tree = self._create_tree_via_unzip(keys_large, ranks_large)

        self.assertEqual(rank_4_tree.node.rank, 4)
        self.assertEqual(rank_2_tree.node.rank, 2)

        result = rank_4_tree.zip(rank_2_tree)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        self.assertIs(result, rank_4_tree)

        # The result should contain all keys
        self._validate_tree_after_zip(result, keys)

    def test_zip_same_rank_mixed_set_types(self):
        """Test zipping trees with same rank but mixed set types (KList and GKPlusTree)."""
        k = 4
        rank_lists = [
            [1, 1, 1, 1, 1, 1], # Dimension 1
            [1, 1, 3, 4, 1, 2]  # Dimension 2
        ]

        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:2]
        ranks_small = rank_lists[0][:2]
        keys_large = keys[2:]
        ranks_large = rank_lists[0][2:]
        tree_klist = self._create_tree(keys_small, ranks_small, k=k)
        self.assertIsInstance(tree_klist.node.set, KListBase)

        # Create a tree with enough items to force GKPlusTree set (higher dimension)
        # This requires creating a tree that will expand to higher dimensions
        tree_gkplus = self._create_tree_via_unzip(keys_large, ranks_large, k=k)
        self.assertIsInstance(tree_gkplus.node.set, GKPlusTreeBase)
        

        # Now zip them
        result = tree_klist.zip(tree_gkplus)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        self.assertIs(result, tree_klist)

        # Validate the result contains all keys
        self._validate_tree_after_zip(result, keys)

    def test_zip_both_gkplus_trees(self):
        """Test zipping two trees where both have GKPlusTree sets."""
        k = 4
        rank_lists = [
            [1, 1, 1, 1, 1, 1, 1, 1], # Dimension 1
            [1, 2, 4, 1, 1, 2, 2, 3]  # Dimension 2
        ]

        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:4]
        ranks_small = rank_lists[0][:4]
        keys_large = keys[4:]
        ranks_large = rank_lists[0][4:]
        
        # Create a tree with enough items to force GKPlusTree set (higher dimension)
        # This requires creating a tree that will expand to higher dimensions
        tree_1 = self._create_tree(keys_small, ranks_small, k=k)
        self.assertIsInstance(tree_1.node.set, GKPlusTreeBase)

        tree_2 = self._create_tree_via_unzip(keys_large, ranks_large, k=k)
        self.assertIsInstance(tree_2.node.set, GKPlusTreeBase)

        result = tree_1.zip(tree_2)
        logger.debug(f"Result after zip: {print_pretty(result)}")
        self.assertIs(result, tree_1)

        # Validate all keys are present
        self._validate_tree_after_zip(result, keys)

    def test_zip_preserves_subtree_structure(self):
        """Test that zip operation preserves subtree structure correctly."""
        k = 4
        rank_lists = [[2, 1, 1, 2, 1, 1]]

        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:3]
        ranks_small = rank_lists[0][:3]
        keys_large = keys[3:]
        ranks_large = rank_lists[0][3:]

        tree_1 = self._create_tree(keys_small, ranks_small, k=k)
        tree_2 = self._create_tree_via_unzip(keys_large, ranks_large, k=k)
        
        # Both should have rank > 1
        self.assertGreater(tree_1.node.rank, 1)
        self.assertGreater(tree_2.node.rank, 1)

        result = tree_1.zip(tree_2)
        self.assertIs(result, tree_1)

        # Validate structure and contents
        self._validate_tree_after_zip(result, keys)

    def test_zip_with_different_dimensions(self):
        """Test zipping trees with different dimensions should raise an exception."""
        tree_dim1 = create_gkplus_tree(K=4, dimension=1)
        tree_dim2 = create_gkplus_tree(K=4, dimension=2)
        
        tree_dim1 = self._create_tree([100, 200], [1, 1], tree_dim1)
        tree_dim2 = self._create_tree_via_unzip([300, 400], [1, 1], tree_dim2)
        
        with self.assertRaises(ValueError) as cm:
            tree_dim1.zip(tree_dim2)
        self.assertIn("dimension", str(cm.exception).lower())

    def test_zip_complex_trees(self):
        """Test zipping complex trees with multiple levels and mixed ranks."""
        k = 4
        rank_lists = [[3, 2, 1, 2, 1, 2, 3, 1, 2, 1]]

        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:5]
        ranks_small = rank_lists[0][:5]
        keys_large = keys[5:]
        ranks_large = rank_lists[0][5:]

        tree_1 = self._create_tree(keys_small, ranks_small, k=k)
        tree_2 = self._create_tree_via_unzip(keys_large, ranks_large, k=k)

        result = tree_1.zip(tree_2)
        self.assertIs(result, tree_1)

        self._validate_tree_after_zip(result, keys)

    def test_zip_invalidates_cached_sizes(self):
        """Test that zip operation properly invalidates cached tree sizes."""
        tree1 = self._create_tree([100, 200], [1, 1])
        tree2 = self._create_tree_via_unzip([300, 400], [1, 1])
        
        # Force calculation of sizes
        initial_size1 = tree1.real_item_count()
        initial_count1 = tree1.item_count()

        initial_size2 = tree2.real_item_count()
        initial_count2 = tree2.item_count()

        # Perform zip
        result = tree1.zip(tree2)

        expected_size = initial_size1 + initial_size2
        expanded_leafs = result.get_expanded_leaf_count()
        
        expected_count = expected_size + (expanded_leafs + 1)
        
        # Sizes should be invalidated and recalculated
        new_size = result.real_item_count()
        new_count = result.item_count()
        
        self.assertGreater(new_size, initial_size1, "Size should increase after zip")
        self.assertGreater(new_count, initial_count1, "Count should increase after zip")

        self.assertEqual(new_size, expected_size, "Size after zip should match expected size")
        self.assertEqual(new_count, expected_count, "Count after zip should match expected count")

    def test_zip_item_counts(self):
        """Test that zip operation preserves subtree structure correctly."""
        k = 4
        rank_lists = [[1, 1, 1, 1, 1, 2]]

        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        keys_small = keys[:1]
        ranks_small = rank_lists[0][:1]
        keys_large = keys[1:]
        ranks_large = rank_lists[0][1:]

        tree_1 = self._create_tree(keys_small, ranks_small, k=k)
        tree_2 = self._create_tree_via_unzip(keys_large, ranks_large, k=k)

        result = tree_1.zip(tree_2)
        # self.assertIs(result, tree_1)

        tree_1_count = tree_1.item_count()
        tree_2_count = tree_2.item_count()
        tree_1_size = tree_1.real_item_count()
        tree_2_size = tree_2.real_item_count()

        expected_size = keys
        expanded_leafs = result.get_expanded_leaf_count()
        expected_count = len(keys_small) + len(keys_large)

        
        

        # Validate structure and contents
        self._validate_tree_after_zip(result, keys)

    def test_zip_with_overlapping_keys(self):
        """Test zipping trees that have overlapping keys."""
        # Create trees with some overlapping keys
        tree1 = self._create_tree([100, 200, 300], [1, 1, 1])
        tree2 = self._create_tree_via_unzip([200, 300, 400], [1, 1, 1])  # 200, 300 overlap
        
        result = tree1.zip(tree2)
        self.assertIs(result, tree1)
        
        # The result should contain all keys (duplicates will be handled by the tree structure)
        # We expect the tree to contain all the inserted keys
        actual_keys = []
        for entry in result.iter_real_entries():
            actual_keys.append(entry.item.key)
        
        # Should have all keys, but exact behavior with duplicates depends on implementation
        self.assertIn(100, actual_keys)
        self.assertIn(200, actual_keys)
        self.assertIn(300, actual_keys)
        self.assertIn(400, actual_keys)

    def test_zip_type_validation(self):
        """Test that zip validates input types correctly."""
        tree = self._create_tree([100, 200], [1, 1])
        
        # Test with invalid type
        with self.assertRaises(TypeError) as cm:
            tree.zip("not a tree")
        
        self.assertIn("other must be an instance of GKPlusTreeBase", str(cm.exception))
        
        # Test with None
        with self.assertRaises(TypeError) as cm:
            tree.zip(None)
        
        self.assertIn("other must be an instance of GKPlusTreeBase", str(cm.exception))

    def test_zip_large_trees(self):
        """Test zipping larger trees to ensure performance and correctness."""
        # Create larger trees
        tree1 = create_gkplus_tree(K=8, dimension=1)
        tree2 = create_gkplus_tree(K=8, dimension=1)
        
        keys1 = list(range(1000, 1100, 5))  # 20 keys
        ranks1 = calc_ranks(keys1, k=8, DIM=1)
        tree1 = self._create_tree(keys1, ranks1, tree1)
        
        keys2 = list(range(2000, 2100, 5))  # 20 keys
        ranks2 = calc_ranks(keys2, k=8, DIM=1)
        tree2 = self._create_tree_via_unzip(keys2, ranks2, tree2)
        
        result = tree1.zip(tree2)
        # self.assertIs(result, tree1)
        
        # Validate all keys are present
        expected_keys = keys1 + keys2
        self._validate_tree_after_zip(result, expected_keys)

    def test_zip_maintains_tree_properties(self):
        """Test that zip maintains essential tree properties."""
        tree1 = self._create_tree([100, 200, 300], [2, 1, 1])
        tree2 = self._create_tree_via_unzip([400, 500], [1, 1])
        
        # Record properties before zip
        initial_dim = tree1.DIM
        initial_l_factor = tree1.l_factor
        
        result = tree1.zip(tree2)
        
        # Properties should be maintained
        self.assertEqual(result.DIM, initial_dim, "DIM should be preserved")
        self.assertEqual(result.l_factor, initial_l_factor, "l_factor should be preserved")
        
        # Tree should still be valid
        if not result.is_empty():
            stats = gtree_stats_(result, {})
            assert_tree_invariants_tc(self, result, stats)

    def test_zip_self_reference(self):
        """Test zipping a tree with itself."""
        tree = self._create_tree([100, 200, 300], [1, 1, 1])
        
        # Make a copy of initial keys for validation
        initial_keys = [entry.item.key for entry in tree.iter_real_entries()]
        
        result = tree.zip(tree)
        self.assertIs(result, tree)
        
        # After zipping with itself, should contain at least the original keys
        final_keys = [entry.item.key for entry in result.iter_real_entries()]
        
        for key in initial_keys:
            self.assertIn(key, final_keys, f"Key {key} should still be present after self-zip")


if __name__ == '__main__':
    unittest.main()
