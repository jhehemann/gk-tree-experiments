import sys
import os
import random

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase
from gplus_trees.logging_config import get_test_logger
import logging

logger = get_test_logger(__name__)

class TestGKPlusTreeExpandedCountTracking(GKPlusTreeTestCase):
    def test_expansion_empty_tree(self):
        """Test that an empty tree has no expanded leaf nodes"""
        tree = create_gkplus_tree(K=4)
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 0,
                         "Expanded count should be 0 after triggering expanded_count()")
        self.assertEqual(tree.expanded_cnt, 0,
                         "expanded_cnt should have changed to 0 after triggering expanded_count()")

    def test_expansion_empty_tree_insertion(self):
        """Test that the count of expanded nodes is 0 after inserting a single item"""
        item = self.create_item(1)
        tree, _, _ = self.tree_k2.insert(item, rank=1)
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 0,
                         "Expanded count should be 0 after inserting one item (1 dummy + 1 item)")
        self.assertEqual(tree.expanded_cnt, 0,
                         "expanded_cnt should be 0 after expanded_count() is called")

    def test_expansion_rank_1_dim_1(self):
        """Test that the count of expanded nodes is 1 for a single expansion in a leaf"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [1, 1], # Dimension 1 (insertion)
            [2, 1], # Dimension 2 (expanded)
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        items = [self.create_item(key) for key in keys]
        # entries = self.create_entries(keys)
        for i, item in enumerate(items):
            tree, _, _ = tree.insert(item, rank=rank_lists[0][i])
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 1,
                         "Expanded count should be 1 after inserting one item (1 dummy + 1 item)")
        self.assertEqual(tree.expanded_cnt, 1,
                         "expanded_cnt should be 1 after expanded_count() is called")
        
    def test_expansion_rank_gt_1_dim_1(self):
        """Test that the count of expanded nodes is 0 for an expansion at rank > 1 in DIM 1"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [2, 2], # Dimension 1 (insertion)
            [2, 1], # Dimension 2 (expanded)
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        items = [self.create_item(key) for key in keys]
        # entries = self.create_entries(keys)
        for i, item in enumerate(items):
            tree, _, _ = tree.insert(item, rank=rank_lists[0][i])
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 0,
                         "Expanded count should be 0 after inserting one item (1 dummy + 1 item)")
        self.assertEqual(tree.expanded_cnt, 0,
                         "expanded_cnt should be 0 after expanded_count() is called")

    def test_multi_expansion_dim1_leaf(self):
        """Test that the count of expanded nodes is 3 for 3 expansions at rank 1 in DIM 1"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [1, 1, 2, 1, 1, 3, 1, 1], # Dimension 1 (insertion)
            [2, 1, 2, 1, 2, 1, 2, 1], # Dimension 2 (expanded)
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        items = [self.create_item(key) for key in keys]
        for i, item in enumerate(items):
            tree, _, _ = tree.insert(item, rank=rank_lists[0][i])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after insertions: {print_pretty(tree)}")
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 3,
                         "Expanded count should be 3 after inserting three items (1 dummy + 3 items)")
        self.assertEqual(tree.expanded_cnt, 3,
                         "expanded_cnt should be 3 after expanded_count() is called")

    def test_multi_expansion_dim1_leaf_dim2_root(self):
        """Test that the count of expanded nodes is 1 for a single expansion at rank 1 in DIM 1
        and a single expansion at rank > 1 in DIM 2"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [1, 1], # Dimension 1 (insertion)
            [2, 2], # Dimension 2 (expanded)
            [2, 1], # Dimension 3 (expanded)
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        items = [self.create_item(key) for key in keys]
        # entries = self.create_entries(keys)
        for i, item in enumerate(items):
            tree, _, _ = tree.insert(item, rank=rank_lists[0][i])
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 1,
                         "Expanded count should be 1 after inserting one item (1 dummy + 1 item)")
        self.assertEqual(tree.expanded_cnt, 1,
                         "expanded_cnt should be 1 after expanded_count() is called")
        
    
    def test_multi_expansion_dim1_leaf_dim2_leaf(self):
        """Test that the count of expanded nodes is 2 for a single expansion at rank 1 in DIM 1
        and a single expansion at rank 1 in DIM 2"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [1, 1], # Dimension 1 (insertion)
            [1, 1], # Dimension 2 (expanded)
            [2, 1], # Dimension 3 (expanded)
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        items = [self.create_item(key) for key in keys]
        # entries = self.create_entries(keys)
        for i, item in enumerate(items):
            tree, _, _ = tree.insert(item, rank=rank_lists[0][i])
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 2,
                         "Expanded count should be 2 after inserting one item (1 dummy + 1 item)")
        self.assertEqual(tree.expanded_cnt, 2,
                         "expanded_cnt should be 2 after expanded_count() is called")
        

    def test_multi_expansion_different_dim_paths(self):
        """Test that the count of expanded nodes sums correctly for multiple expansions in different leafs"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [1, 1, 2, 1, 1], # Dimension 1 (insertion 2 leafs expanded)
            [1, 1, 2, 1, 1], # Dimension 2 (expanded, 2 leafs expanded)
            [2, 1, 3, 1, 1], # Dimension 3 (expanded, 1 leaf expanded)
            [2, 1, 3, 1, 2],
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        items = [self.create_item(key) for key in keys]
        # entries = self.create_entries(keys)
        for i, item in enumerate(items):
            tree, _, _ = tree.insert(item, rank=rank_lists[0][i])
        self.assertIsNone(tree.expanded_cnt,
                          "expanded_cnt should be None before triggering expanded_count()")
        self.assertEqual(tree.expanded_count(), 5,
                         "Expanded count should be 5 after inserting one item (1 dummy + 4 items)")
        self.assertEqual(tree.expanded_cnt, 5,
                         "expanded_cnt should be 5 after expanded_count() is called")
