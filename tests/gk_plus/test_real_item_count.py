"""
Comprehensive tests for the real_item_count() method of GKPlusTreeBase.

The real_item_count() method should return the number of real items in leaf nodes,
excluding dummy items (which have negative keys).
"""

import logging

from gplus_trees.base import Entry
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase

logger = logging.getLogger(__name__)


class TestGKPlusTreeRealItemCount(GKPlusTreeTestCase):
    """Test cases for the real_item_count() method."""

    def test_empty_tree_real_item_count(self):
        """Test that an empty tree has 0 real items."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Initially, size should be None (not calculated yet)
        self.assertIsNone(tree.size, "size should be None before calculation")
        
        # real_item_count() should return 0 for empty tree
        count = tree.real_item_count()
        self.assertEqual(count, 0, "Empty tree should have 0 real items")
        
        # After calculation, size should be cached
        self.assertEqual(tree.size, 0, "size should be cached as 0")
        
        # Subsequent calls should return cached value
        count2 = tree.real_item_count()
        self.assertEqual(count2, 0, "Cached real item count should be 0")
        self.validate_tree(tree)

    def test_single_item_leaf_real_item_count(self):
        """Test real_item_count with a single real item in a leaf."""
        tree = create_gkplus_tree(K=4, dimension=1)
        item = self.make_item(5)
        
        # Insert item at rank 1 (leaf level)
        tree, inserted, _ = tree.insert(item, rank=1)
        self.assertTrue(inserted, "Item should be inserted successfully")
        
        # Tree should have 1 real item (5) and 1 dummy item (-1)
        # total item_count = 2, real_item_count = 1
        total_count = tree.item_count()
        real_count = tree.real_item_count()
        
        self.assertEqual(total_count, 2, "Total items should be 2 (1 dummy + 1 real)")
        self.assertEqual(real_count, 1, "Real item count should be 1")
        
        # Verify caching
        self.assertEqual(tree.size, 1, "Cached size should be 1")
        self.validate_tree(tree)

    def test_multiple_items_same_leaf_real_item_count(self):
        """Test real_item_count with multiple real items in the same leaf."""
        tree = create_gkplus_tree(K=4, dimension=1)
        items = [self.make_item(i) for i in [5, 10, 15]]
        
        # Insert all items at rank 1 (leaf level)
        for item in items:
            tree, inserted, _ = tree.insert(item, rank=1)
            self.assertTrue(inserted, f"Item {item.key} should be inserted successfully")
        
        # Tree should have 3 real items and 1 dummy item
        total_count = tree.item_count()
        real_count = tree.real_item_count()
        
        self.assertEqual(total_count, 4, "Total items should be 4 (1 dummy + 3 real)")
        self.assertEqual(real_count, 3, "Real item count should be 3")
        self.validate_tree(tree)

    def test_multi_level_tree_real_item_count(self):
        """Test real_item_count with a multi-level tree structure."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Insert items at different ranks to create a multi-level structure
        items_and_ranks = [
            (self.make_item(5), 1),   # leaf level
            (self.make_item(10), 2),  # internal level
            (self.make_item(15), 1),  # leaf level
            (self.make_item(20), 3),  # higher internal level
        ]
        
        for item, rank in items_and_ranks:
            tree, inserted, _ = tree.insert(item, rank=rank)
            self.assertTrue(inserted, f"Item {item.key} should be inserted successfully")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Multi-level tree structure: {print_pretty(tree)}")
        
        # All items are real items (no additional dummies beyond the original one)
        real_count = tree.real_item_count()
        self.assertEqual(real_count, 4, "Real item count should be 4")
        self.validate_tree(tree)

    def test_tree_with_only_dummy_items(self):
        """Test real_item_count with a tree containing only dummy items."""
        tree = create_gkplus_tree(K=4, dimension=2)  # dimension=2, so dummy key is -2
        
        # Insert only dummy items (simulate internal tree operations)
        dummy_item = get_dummy(2)  # key = -2
        entry = Entry(dummy_item, None)
        
        # Create a leaf with only dummy
        tree.node = tree.NodeClass(1, tree.SetClass(), None)
        tree.node.set, _, _ = tree.node.set.insert_entry(entry)
        
        real_count = tree.real_item_count()
        total_count = tree.item_count()
        
        self.assertEqual(total_count, 1, "Total items should be 1 (dummy only)")
        self.assertEqual(real_count, 0, "Real item count should be 0 (no positive keys)")
        self.validate_tree(tree)

    def test_mixed_real_and_dummy_items(self):
        """Test real_item_count with a mix of real and dummy items."""
        tree = create_gkplus_tree(K=4, dimension=2)  # dimension=2, so dummy key is -2
        
        # Insert real items
        real_items = [self.make_item(i) for i in [5, 10, 15]]
        for item in real_items:
            tree, inserted, _ = tree.insert(item, rank=1)
            self.assertTrue(inserted, f"Item {item.key} should be inserted successfully")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree with real and dummy items: {print_pretty(tree)}")
        
        # Should count only real items (positive keys), not dummy items (negative keys)
        real_count = tree.real_item_count()
        total_count = tree.item_count()
        
        # Total includes both real and dummy items
        self.assertGreaterEqual(total_count, 3, "Total items should include dummies")
        self.assertEqual(real_count, 3, "Real item count should be 3 (only positive keys)")
        self.validate_tree(tree)

    def test_hierarchical_tree_real_item_count(self):
        """Test real_item_count with a hierarchical tree containing subtrees."""
        tree = create_gkplus_tree(K=4, dimension=1)  # Use K=4 as requested
        
        # Insert enough items to create a hierarchical structure
        items = [self.make_item(i) for i in range(1, 12)]  # 11 items: 1,2,3,4,5,6,7,8,9,10,11
        
        for item in items:
            tree, inserted, _ = tree.insert(item, rank=1)
            self.assertTrue(inserted, f"Item {item.key} should be inserted successfully")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Hierarchical tree structure: {print_pretty(tree)}")
        
        # Should count all real items across all leaf nodes
        real_count = tree.real_item_count()
        self.assertEqual(real_count, 11, "Real item count should be 11")
        self.validate_tree(tree)

    def test_cache_invalidation_on_modification(self):
        """Test that the size cache is invalidated when the tree is modified."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Insert initial item
        item1 = self.make_item(5)
        tree, _, _ = tree.insert(item1, rank=1)

        # Get initial count (this caches the size)
        initial_count = tree.real_item_count()
        self.assertEqual(initial_count, 1, "Initial real item count should be 1")
        self.assertEqual(tree.size, 1, "Size should be cached as 1")
        
        # Insert another item - this should invalidate the cache
        item2 = self.make_item(10)
        tree, _, _ = tree.insert(item2, rank=1)

        # The cache should be invalidated after insertion
        self.assertIsNone(tree.size, "Size cache should be invalidated after insertion")
        
        # Getting count again should recalculate and cache new value
        new_count = tree.real_item_count()
        self.assertEqual(new_count, 2, "New real item count should be 2")
        self.assertEqual(tree.size, 2, "Size should be cached as 2")
        self.validate_tree(tree)

    def test_consistency_with_leaf_iteration(self):
        """Test that real_item_count is consistent with manually counting real items."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Insert various items
        items = [self.make_item(i) for i in [1, 5, 10, 15, 20]]
        for item in items:
            tree, _, _ = tree.insert(item, rank=1)

        # Count real items manually by iterating through all entries
        manual_count = 0
        for entry in tree:
            if entry.item.key >= 0:  # Real items have non-negative keys
                manual_count += 1
        
        # Compare with real_item_count() method
        method_count = tree.real_item_count()
        self.assertEqual(method_count, manual_count, 
                        f"Method count ({method_count}) should match manual count ({manual_count})")
        self.assertEqual(method_count, 5, "Should have 5 real items")
        self.validate_tree(tree)

    def test_real_item_count_vs_item_count(self):
        """Test the difference between real_item_count and item_count."""
        tree = create_gkplus_tree(K=4, dimension=2)  # dimension=2 creates dummy with key=-2
        
        # Insert real items
        real_items = [self.make_item(i) for i in [3, 7, 11]]
        for item in real_items:
            tree, _, _ = tree.insert(item, rank=1)

        total_count = tree.item_count()
        real_count = tree.real_item_count()
        dummy_count = total_count - real_count
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree: {print_pretty(tree)}")
            logger.debug(f"Total: {total_count}, Real: {real_count}, Dummy: {dummy_count}")
        
        self.assertEqual(real_count, 3, "Should have 3 real items")
        self.assertGreater(total_count, real_count, "Total count should be greater than real count")
        self.assertGreater(dummy_count, 0, "Should have at least one dummy item")
        self.validate_tree(tree)

    def test_complex_multi_dimensional_tree(self):
        """Test real_item_count with a complex multi-dimensional tree."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Create a complex structure by inserting items with various ranks
        test_data = [
            (1, 1), (2, 2), (3, 1), (4, 3), (5, 1),
            (6, 2), (7, 1), (8, 4), (9, 1), (10, 2)
        ]
        
        for key, rank in test_data:
            item = self.make_item(key)
            tree, inserted, _ = tree.insert(item, rank=rank)
            self.assertTrue(inserted, f"Item {key} should be inserted successfully")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Complex tree structure: {print_pretty(tree)}")
        
        # All inserted items are real items
        real_count = tree.real_item_count()
        self.assertEqual(real_count, len(test_data), 
                        f"Should have {len(test_data)} real items")
        self.validate_tree(tree)

    def test_get_size_method_consistency(self):
        """Test that real_item_count() is consistent with get_size() method."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Insert some items
        items = [self.make_item(i) for i in [2, 8, 12, 16]]
        for item in items:
            tree, _, _ = tree.insert(item, rank=1)

        # Both methods should return the same result
        real_count = tree.real_item_count()
        size_count = tree.get_size()
        
        self.assertEqual(real_count, size_count, 
                        "real_item_count() and get_size() should return the same value")
        self.assertEqual(real_count, 4, "Should have 4 real items")
        self.validate_tree(tree)

    def test_empty_subtrees_handling(self):
        """Test that real_item_count correctly handles empty subtrees."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Create a structure with potentially empty subtrees
        items_and_ranks = [
            (self.make_item(5), 3),   # High rank - creates internal structure
            (self.make_item(10), 1),  # Low rank - goes to leaf
        ]
        
        for item, rank in items_and_ranks:
            tree, _, _ = tree.insert(item, rank=rank)

        # Should count only the real items, ignoring empty subtrees
        real_count = tree.real_item_count()
        self.assertEqual(real_count, 2, "Should count 2 real items regardless of empty subtrees")
        self.validate_tree(tree)

    def test_performance_with_large_tree(self):
        """Test real_item_count performance and correctness with a larger tree."""
        tree = create_gkplus_tree(K=4, dimension=1)
        
        # Insert many items
        num_items = 50
        items = [self.make_item(i) for i in range(1, num_items + 1)]
        
        for item in items:
            tree, _, _ = tree.insert(item, rank=1)

        # Should count all real items efficiently
        real_count = tree.real_item_count()
        self.assertEqual(real_count, num_items, f"Should have {num_items} real items")
        
        # Verify caching works for subsequent calls
        cached_count = tree.real_item_count()
        self.assertEqual(cached_count, num_items, "Cached count should be the same")
        self.validate_tree(tree)


if __name__ == '__main__':
    import unittest
    unittest.main()
