"""Tests for GK+ tree utility functions"""
import sys
import os
import unittest
import random
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item, Entry
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.utils import tree_to_klist, klist_to_tree
from gplus_trees.klist_base import KListBase
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, get_dummy
from gplus_trees.g_k_plus.g_k_plus_base import print_pretty



class TestGKPlusUtils(unittest.TestCase):
    """Test the utility functions for GKPlus trees"""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        # Create trees with different K values for testing
        self.K = 4
        self.DIM = 3
        self.empty_tree = create_gkplus_tree(K=self.K, dimension=self.DIM)
        
    def create_populated_tree(self, num_items=10):
        """Helper to create a tree with a specific number of items"""
        tree = create_gkplus_tree(K=self.K, dimension=self.DIM)
        
        for i in range(num_items):
            item = Item(1000 + i, f"val_{i}")
            # Use rank 1 for simplicity
            # Note that insert returns (tree, inserted) tuple
            tree, _ = tree.insert(item, 1)
        # Provide dummy items from other dimensions
        tree, _ = tree.insert(get_dummy(1), 1)  # Insert a dummy item
        tree, _ = tree.insert(get_dummy(2), 1)  # Insert another dummy item
        return tree
    
    def test_tree_to_klist_type_validation(self):
        """Test that tree_to_klist validates the input type"""
        # Test with a non-GKPlusTreeBase object
        with self.assertRaises(TypeError):
            tree_to_klist("not a tree")
    
    def test_tree_to_klist_empty_tree(self):
        """Test that tree_to_klist works with an empty tree"""
        empty_tree = self.empty_tree
        klist = tree_to_klist(empty_tree)
        
        # Check that the result is a KList
        self.assertIsInstance(klist, KListBase)
        # Check that the KList is empty
        self.assertTrue(klist.is_empty())
    
    def test_tree_to_klist_populated_tree(self):
        """Test that tree_to_klist correctly converts a populated tree to KList"""
        # Create a tree with some items
        num_items = 10
        populated_tree = self.create_populated_tree(num_items)
        tree_entries = list(populated_tree)

        # Convert to KList
        klist = tree_to_klist(populated_tree)
        
        # Check that the result is a KList
        self.assertIsInstance(klist, KListBase)
        
        # Check that the KList contains all items from the tree
        tree_items = set(entry.item.key for entry in populated_tree)
        klist_items = set(entry.item.key for entry in klist)

        self.assertEqual(len(list(populated_tree)), num_items + 2,
                         f"Tree should contain {num_items} + 2 items")
        self.assertEqual(len(klist_items), num_items + 2, 
                         f"KList should contain {num_items} + 2 items")
        self.assertEqual(tree_items, klist_items, 
                         "KList should contain the same items as the tree")
        
        # Check that the items are in the same order (sorted by key)
        tree_keys = [entry.item.key for entry in populated_tree]
        klist_keys = [entry.item.key for entry in klist]
        
        self.assertEqual(tree_keys, klist_keys, 
                         "Keys in KList should be in the same order as in the tree")
    
    def test_klist_to_tree_type_validation(self):
        """Test that klist_to_tree validates the input type"""
        # Test with a non-KListBase object
        with self.assertRaises(TypeError):
            klist_to_tree("not a klist", self.K, self.DIM)
    
    def test_klist_to_tree_empty_klist(self):
        """Test that klist_to_tree works with an empty KList"""
        # Get the KList class from an empty tree
        empty_tree = self.empty_tree
        empty_klist = tree_to_klist(empty_tree)
        
        # Convert back to tree
        new_tree = klist_to_tree(empty_klist, self.K, self.DIM)
        
        # Check that the result is a GKPlusTreeBase
        self.assertIsInstance(new_tree, GKPlusTreeBase)
        # Check that the tree is empty
        self.assertTrue(new_tree.is_empty())
    
    def test_klist_to_tree_populated_klist(self):
        """Test that klist_to_tree correctly converts a populated KList to a tree"""
        # Create a tree with some items and convert to KList
        num_items = 10
        populated_tree = self.create_populated_tree(num_items)
        # print(f"\nPopulated tree: {print_pretty(populated_tree)}")
        klist = tree_to_klist(populated_tree)
        # print(f"\nKList: {print_pretty(klist)}")

        # Convert back to tree
        new_tree = klist_to_tree(klist, self.K, self.DIM)
        # print(f"\nNew tree structure: {print_pretty(new_tree)}")

        # Check that the result is a GKPlusTreeBase
        self.assertIsInstance(new_tree, GKPlusTreeBase)
        
        # The trees won't be identical due to random ranks in klist_to_tree,
        # but they should contain the same keys
        orig_tree_keys = set(entry.item.key for entry in populated_tree)
        new_tree_keys = set(entry.item.key for entry in new_tree)
        
        self.assertEqual(len(new_tree_keys), num_items + 2, 
                         f"New tree should contain {num_items} + 2 dummy items")
        self.assertEqual(orig_tree_keys, new_tree_keys, 
                         "New tree should contain the same items as the original")
    
    def test_roundtrip_tree_to_klist_to_tree(self):
        """Test the round trip from tree to KList and back to tree"""
        # Create a tree with some items
        num_items = 10
        original_tree = self.create_populated_tree(num_items)
        
        # Convert to KList
        klist = tree_to_klist(original_tree)
        
        # Convert back to tree
        new_tree = klist_to_tree(klist, self.K, self.DIM)
        
        # Check that all keys are preserved
        original_keys = set(entry.item.key for entry in original_tree)
        new_keys = set(entry.item.key for entry in new_tree)
        
        self.assertEqual(original_keys, new_keys, 
                         "Round trip conversion should preserve all keys")


if __name__ == '__main__':
    unittest.main()
