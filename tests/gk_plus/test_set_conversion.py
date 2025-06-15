"""Tests for GK+ tree utility functions"""
import sys
import os
import unittest
import random

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
# from gplus_trees.g_k_plus.utils import _tree_to_klist, _klist_to_tree
from gplus_trees.klist_base import KListBase
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, get_dummy, _tree_to_klist, _klist_to_tree
from gplus_trees.g_k_plus.g_k_plus_base import print_pretty
from gplus_trees.g_k_plus.utils import calc_rank
from tests.test_base import GKPlusTreeTestCase
from gplus_trees.logging_config import get_test_logger

logger = get_test_logger(__name__)


class TestGKPlusUtils(GKPlusTreeTestCase):
    """Test the utility functions for GKPlus trees"""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        # Create trees with different K values for testing
        self.K = 2
        self.empty_tree = create_gkplus_tree(K=self.K)
        
    def create_populated_tree(self, num_items=10):
        """Helper to create a tree with a specific number of items"""
        tree = create_gkplus_tree(K=self.K)
        
        for i in range(num_items):
            item = Item(1000 + i, f"val_{i}")
            # Use rank 1 for simplicity
            # Note that insert returns (tree, inserted) tuple
            tree, _ = tree.insert(item, 1)
        # Provide dummy items from other dimensions
        # tree, _ = tree.insert(get_dummy(1), 1)  # Insert a dummy item
        # tree, _ = tree.insert(get_dummy(2), 1)  # Insert another dummy item
        return tree
    
    def test_tree_to_klist_type_validation(self):
        """Test that _tree_to_klist validates the input type"""
        # Test with a non-GKPlusTreeBase object
        with self.assertRaises(TypeError):
            _tree_to_klist("not a tree")
    
    def test__tree_to_klist_empty_tree(self):
        """Test that _tree_to_klist works with an empty tree"""
        empty_tree = self.empty_tree
        klist = _tree_to_klist(empty_tree)
        
        # Check that the result is a KList
        self.assertIsInstance(klist, KListBase)
        # Check that the KList is empty
        self.assertTrue(klist.is_empty())
    
    def test_tree_to_klist_populated_tree(self):
        """Test that _tree_to_klist correctly converts a populated tree to KList"""
        # Create a tree with some items
        num_items = 10
        populated_tree = self.create_populated_tree(num_items)
        dum_count_tree = self.get_dummy_count(populated_tree)
        exp_tree_count = num_items + dum_count_tree

        # Convert to KList
        klist = _tree_to_klist(populated_tree)
        
        # Check that the result is a KList
        self.assertIsInstance(klist, KListBase)
        
        tree_keys = [entry.item.key for entry in populated_tree]
        self.assertEqual(len(list(populated_tree)), exp_tree_count,
                         f"Tree should contain {exp_tree_count} items")

        # Check that the KList contains all keys greater than the corresponding dummy item from the tree
        tree_keys_to_klist = [entry.item.key for entry in populated_tree if entry.item.key > get_dummy(populated_tree.DIM).key]
        klist_keys = [entry.item.key for entry in klist]

        self.assertEqual(len(klist_keys), len(tree_keys_to_klist),
                         f"KList should contain {len(tree_keys_to_klist)} items")        
        self.assertEqual(tree_keys_to_klist, klist_keys,
                         "Keys in KList should be in the same order as in the tree")
    
    def test_klist_to_tree_type_validation(self):
        """Test that _klist_to_tree validates the input type"""
        # Test with a non-KListBase object
        with self.assertRaises(TypeError):
            _klist_to_tree("not a klist", self.K, 1)
    
    def test_klist_to_tree_empty_klist(self):
        """Test that _klist_to_tree works with an empty KList"""
        # Get the KList class from an empty tree
        empty_tree = self.empty_tree
        empty_klist = _tree_to_klist(empty_tree)
        
        # Convert back to tree
        new_tree = _klist_to_tree(empty_klist, self.K, 1)

        # Check that the result is a GKPlusTreeBase
        self.assertIsInstance(new_tree, GKPlusTreeBase)
        # Check that the tree is empty
        self.assertTrue(new_tree.is_empty())
    
    def test_klist_to_tree_populated_klist(self):
        """Test that _klist_to_tree correctly converts a populated KList to a tree"""
        # Create a tree with some items and convert to KList
        # tree = self.empty_tree
        tree = create_gkplus_tree(K=2, dimension=1)
        num_items = 8
        
        # create random items
        items = []
        random.seed(44)  # For reproducibility
        keys = []
        for i in range(num_items):
            key = random.randint(1, 1000)
            item = Item(key, f"val_{i}")
            items.append(item)
            keys.append(key)

        # create ranks
        # ranks = []
        # for item in items:
        #     # logger.debug(f"Calculating rank for key: {item.key}, K: {self.K}, DIM: 1")
        #     rank = calc_rank(item.key, self.K, 1)
        #     ranks.append(rank)
        ranks = [calc_rank(key=key, k=2, dim=1) for key in keys]
        logger.debug(f"Keys: {[item.key for item in items]}")
        logger.debug(f"Ranks: {ranks}")

        # Insert items into the tree
        for item, rank in zip(items, ranks):
            tree, _ = tree.insert(item, rank)

        logger.debug(f"Tree after inserting items: {print_pretty(tree)}")
        tree_keys = [entry.item.key for entry in tree]
        logger.debug(f"Tree keys: {tree_keys}")
        
        dum_keys = self.get_dummies(tree)
        logger.debug(f"Dummies: {[dum_key for dum_key in dum_keys]}")
        exp_keys = sorted(dum_keys + [item.key for item in items])
        logger.debug(f"Expected keys: {exp_keys}")

        self.validate_tree(tree, exp_keys)

        dum_count_tree = self.get_dummy_count(tree)
        exp_tree_count = num_items + dum_count_tree
        logger.debug(f"\nInitial tree: {print_pretty(tree)}")
        klist = _tree_to_klist(tree)
        logger.debug(f"\nKList: {print_pretty(klist)}")

        # Convert back to tree
        new_tree = _klist_to_tree(klist, self.K, 1)
        logger.debug(f"\nNew tree: {print_pretty(new_tree)}")

        # Check that the result is a GKPlusTreeBase
        self.assertIsInstance(new_tree, GKPlusTreeBase)

        init_tree_keys = [entry.item.key for entry in tree]
        new_tree_keys = [entry.item.key for entry in new_tree]
        logger.debug(f"Original tree keys: {init_tree_keys}")
        logger.debug(f"New tree keys: {new_tree_keys}")
        
        self.assertEqual(len(new_tree_keys), exp_tree_count, 
                         f"New tree should contain {exp_tree_count} items")
        self.assertEqual(init_tree_keys, new_tree_keys,
                         "New tree should contain the same items as the original")
    
    def test_roundtrip_tree_to_klist_to_tree(self):
        """Test the round trip from tree to KList and back to tree"""
        # Create a tree with some items
        num_items = 10
        original_tree = self.create_populated_tree(num_items)
        
        # Convert to KList
        klist = _tree_to_klist(original_tree)
        
        # Convert back to tree
        new_tree = _klist_to_tree(klist, self.K, original_tree.DIM)
        
        # Check that all keys are preserved
        original_keys = set(entry.item.key for entry in original_tree)
        new_keys = set(entry.item.key for entry in new_tree)
        
        self.assertEqual(original_keys, new_keys, 
                         "Round trip conversion should preserve all keys")


if __name__ == '__main__':
    unittest.main()
