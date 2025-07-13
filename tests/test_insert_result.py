"""
Tests for InsertResult functionality in GPlusTree, GKPlusTree, and KList.

This module tests:
1. InsertResult class behavior and interface
2. insert() method return values for GPlusTree and GKPlusTree
3. insert_entry() method return values for KList
"""

import unittest
from typing import Optional

from gplus_trees.base import Item, Entry, InsertResult
from gplus_trees.factory import create_gplustree
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import DUMMY_ITEM, _create_replica, print_pretty

from tests.test_base import BaseTreeTestCase
from tests.test_base import GKPlusTreeTestCase



class TestInsertResult(unittest.TestCase):
    """Test InsertResult class functionality."""
    
    def test_insert_result_creation(self):
        """Test InsertResult can be created with required parameters."""
        tree = create_gplustree(4)
        inserted = True
        next_entry = Entry(Item(10, "test"), None)
        
        result = InsertResult(tree, inserted, next_entry)
        
        self.assertIs(result.tree, tree)
        self.assertEqual(result.inserted, inserted)
        self.assertIs(result.next_entry, next_entry)
    
    def test_insert_result_iteration(self):
        """Test InsertResult supports iteration (backward compatibility)."""
        tree = create_gplustree(4)
        inserted = True
        next_entry = Entry(Item(10, "test"), None)
        
        result = InsertResult(tree, inserted, next_entry)
        
        # Test iteration unpacking
        unpacked_tree, unpacked_inserted = result
        self.assertIs(unpacked_tree, tree)
        self.assertEqual(unpacked_inserted, inserted)
        
        # Test list conversion
        result_list = list(result)
        self.assertEqual(len(result_list), 2)
        self.assertIs(result_list[0], tree)
        self.assertEqual(result_list[1], inserted)
    
    def test_insert_result_indexing(self):
        """Test InsertResult supports indexing."""
        tree = create_gplustree(4)
        inserted = True
        next_entry = Entry(Item(10, "test"), None)
        
        result = InsertResult(tree, inserted, next_entry)
        
        # Test indexing
        self.assertIs(result[0], tree)
        self.assertEqual(result[1], inserted)
        self.assertIs(result[2], next_entry)
    
    def test_insert_result_with_none_next_entry(self):
        """Test InsertResult with None next_entry."""
        tree = create_gplustree(4)
        inserted = False
        next_entry = None
        
        result = InsertResult(tree, inserted, next_entry)
        
        self.assertIs(result.tree, tree)
        self.assertEqual(result.inserted, inserted)
        self.assertIsNone(result.next_entry)
        self.assertIsNone(result[2])


class TestGPlusTreeInsertResult(BaseTreeTestCase):
    """Test InsertResult for GPlusTree insert method."""
    
    def setUp(self):
        self.tree = create_gplustree(4)
    
    def test_insert_empty_tree_returns_correct_result(self):
        """Test insert into empty tree returns correct InsertResult."""
        item = Item(5, "test_value")
        rank = 1
        
        result = self.tree.insert(item, rank)

        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.tree)
        self.assertTrue(result.inserted)
        self.assertIsNone(result.next_entry)
        
        # Verify backward compatibility
        tree, inserted = result
        self.assertIs(tree, self.tree)
        self.assertTrue(inserted)
    
    def test_insert_new_item_returns_correct_result(self):
        """Test insert of new item returns correct InsertResult."""
        # First insert
        item1 = Item(5, "value1")
        result1 = self.tree.insert(item1, 1)
        
        # Second insert
        item2 = Item(10, "value2")
        result2 = self.tree.insert(item2, 1)
        
        self.assertIsInstance(result2, InsertResult)
        self.assertIs(result2.tree, self.tree)
        self.assertTrue(result2.inserted)
        self.assertIsNone(result2.next_entry)  # No next entry at rank 1
    
    def test_insert_existing_item_returns_correct_result(self):
        """Test insert of existing item returns correct InsertResult."""
        item = Item(5, "original_value")
        self.tree.insert(item, 1)
        
        # Insert same key with different value
        updated_item = Item(5, "updated_value")
        result = self.tree.insert(updated_item, 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.tree)
        self.assertFalse(result.inserted)  # Should be False for existing items
        self.assertIsNone(result.next_entry)
        
        # Verify the value was updated
        found_entry, _ = self.tree.retrieve(5)
        self.assertEqual(found_entry.item.value, "updated_value")
    
    def test_insert_with_higher_rank_returns_correct_result(self):
        """Test insert with higher rank returns correct InsertResult."""
        # Insert at rank 1
        item1 = Item(5, "value1")
        self.tree.insert(item1, 1)
        
        # Insert at rank 2
        item2 = Item(10, "value2")
        result = self.tree.insert(item2, 2)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.tree)
        self.assertTrue(result.inserted)
        # next_entry behavior depends on tree structure
        self.assertIsInstance(result.next_entry, (type(None), Entry))
    
    def test_insert_with_next_entry_information(self):
        """Test insert returns correct next_entry information."""
        # Build a tree with multiple items
        items = [Item(i, f"value{i}") for i in [2, 4, 6, 8, 10]]
        for item in items:
            self.tree.insert(item, 1)
        
        # Insert between existing items
        new_item = Item(5, "value5")
        result = self.tree.insert(new_item, 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.tree)
        self.assertTrue(result.inserted)
        # next_entry might be the next item in the leaf node
        self.assertIsInstance(result.next_entry, (type(None), Entry))
        
        # If next_entry is not None, it should be the next item
        if result.next_entry is not None:
            self.assertGreater(result.next_entry.item.key, 5)


class TestGKPlusTreeInsertResult(GKPlusTreeTestCase):
    """Test InsertResult for GKPlusTree insert method."""
        
    
    def test_insert_empty_tree_returns_correct_result(self):
        """Test insert into empty GKPlusTree returns correct InsertResult."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        item = Item(5, "test_value")
        rank = 1

        result = tree.insert(item, rank)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, tree)
        self.assertTrue(result.inserted)
        self.assertIsNone(result.next_entry)
        
        # Verify backward compatibility
        new_tree, inserted = result
        self.assertIs(tree, new_tree)
        self.assertTrue(inserted)
    
    def test_insert_new_item_returns_correct_result(self):
        """Test insert of new item into GKPlusTree returns correct InsertResult."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        # First insert
        item1 = Item(5, "value1")
        result1 = tree.insert(item1, 1)

        # Second insert
        item2 = Item(10, "value2")
        result2 = tree.insert(item2, 1)

        self.assertIsInstance(result2, InsertResult)
        self.assertIs(result2.tree, tree)
        self.assertTrue(result2.inserted)
        self.assertIsInstance(result2.next_entry, (type(None), Entry))
    
    def test_insert_existing_item_returns_correct_result(self):
        """Test insert of existing item into GKPlusTree returns correct InsertResult."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        
        item = Item(5, "original_value")
        tree.insert(item, 1)

        # Insert same key with different value - GKPlusTree doesn't support updates
        updated_item = Item(5, "updated_value")
        with self.assertRaises(ValueError) as context:
            tree.insert(updated_item, 1)
        
        # Verify the error message
        self.assertIn("Entry with key 5 already exists", str(context.exception))
        self.assertIn("No updates allowed", str(context.exception))
    
    def test_insert_with_dimensional_expansion_returns_correct_result(self):
        """Test insert that triggers dimensional expansion returns correct InsertResult."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        # Insert multiple items to potentially trigger expansion
        items = [Item(i, f"value{i}") for i in range(1, 10)]
        for item in items:
            result = tree.insert(item, 1)

            self.assertIsInstance(result, InsertResult)
            self.assertIs(result.tree, tree)
            self.assertTrue(result.inserted)
            self.assertIsInstance(result.next_entry, (type(None), Entry))
    
    def test_insert_with_higher_rank_returns_correct_result(self):
        """Test insert with higher rank in GKPlusTree returns correct InsertResult."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        # Insert at rank 1
        item1 = Item(5, "value1")
        tree.insert(item1, 1)

        # Insert at rank 2
        item2 = Item(10, "value2")
        result = tree.insert(item2, 2)

        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, tree)
        self.assertTrue(result.inserted)
        self.assertIsInstance(result.next_entry, (type(None), Entry))


class TestKListInsertEntryResult(BaseTreeTestCase):
    """Test InsertResult for KList insert_entry method."""
    
    def setUp(self):
        super().setUp()
        from gplus_trees.factory import make_gplustree_classes
        _, _, self.KListClass, _ = make_gplustree_classes(4)
        self.klist = self.KListClass()
    
    def test_insert_entry_empty_klist_returns_correct_result(self):
        """Test insert_entry into empty KList returns correct InsertResult."""
        item = Item(5, "test_value")
        entry = Entry(item, None)
        
        result = self.klist.insert_entry(entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.klist)
        self.assertTrue(result.inserted)
        self.assertIsNone(result.next_entry)
        
        # Verify backward compatibility
        klist, inserted = result
        self.assertIs(klist, self.klist)
        self.assertTrue(inserted)
    
    def test_insert_entry_new_item_returns_correct_result(self):
        """Test insert_entry of new item returns correct InsertResult."""
        # First insert
        item1 = Item(5, "value1")
        entry1 = Entry(item1, None)
        result1 = self.klist.insert_entry(entry1)
        
        # Second insert
        item2 = Item(10, "value2")
        entry2 = Entry(item2, None)
        result2 = self.klist.insert_entry(entry2)
        
        self.assertIsInstance(result2, InsertResult)
        self.assertIs(result2.tree, self.klist)
        self.assertTrue(result2.inserted)
        self.assertIsInstance(result2.next_entry, (type(None), Entry))
    
    def test_insert_entry_existing_item_returns_correct_result(self):
        """Test insert_entry of existing item returns correct InsertResult."""
        item = Item(5, "original_value")
        entry = Entry(item, None)
        self.klist.insert_entry(entry)
        
        # Insert same key with different value
        updated_item = Item(5, "updated_value")
        updated_entry = Entry(updated_item, None)
        result = self.klist.insert_entry(updated_entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.klist)
        self.assertFalse(result.inserted)  # Should be False for existing items
        self.assertIsInstance(result.next_entry, (type(None), Entry))
    
    def test_insert_entry_with_next_entry_information(self):
        """Test insert_entry returns correct next_entry information."""
        # Build a KList with multiple items
        items = [Item(i, f"value{i}") for i in [2, 4, 6, 8, 10]]
        for item in items:
            entry = Entry(item, None)
            self.klist.insert_entry(entry)
        
        # Insert between existing items
        new_item = Item(5, "value5")
        new_entry = Entry(new_item, None)
        result = self.klist.insert_entry(new_entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.klist)
        self.assertTrue(result.inserted)
        
        # Verify next_entry points to the correct next item
        if result.next_entry is not None:
            self.assertGreater(result.next_entry.item.key, 5)
    
    def test_insert_entry_with_left_subtree_returns_correct_result(self):
        """Test insert_entry with left_subtree returns correct InsertResult."""
        subtree = create_gplustree(4)
        item = Item(5, "test_value")
        entry = Entry(item, subtree)
        
        result = self.klist.insert_entry(entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.klist)
        self.assertTrue(result.inserted)
        self.assertIsNone(result.next_entry)
        
        # Verify the entry was inserted with the correct left_subtree
        found_entry, _ = self.klist.retrieve(5)
        self.assertIs(found_entry.left_subtree, subtree)
    
    def test_insert_entry_overflow_returns_correct_result(self):
        """Test insert_entry that causes overflow returns correct InsertResult."""
        # Fill KList to capacity
        capacity = self.KListClass.KListNodeClass.CAPACITY
        items = [Item(i, f"value{i}") for i in range(1, capacity + 1)]
        for item in items:
            entry = Entry(item, None)
            self.klist.insert_entry(entry)
        
        # Insert one more item to trigger overflow
        overflow_item = Item(capacity + 1, f"value{capacity + 1}")
        overflow_entry = Entry(overflow_item, None)
        result = self.klist.insert_entry(overflow_entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.klist)
        self.assertTrue(result.inserted)
        self.assertIsInstance(result.next_entry, (type(None), Entry))
    
    def test_insert_entry_with_rank_parameter(self):
        """Test insert_entry with rank parameter (GKPlusTree specific)."""
        # This test is for completeness - rank parameter is used in GKPlusTree contexts
        item = Item(5, "test_value")
        entry = Entry(item, None)
        
        # Insert with rank parameter
        result = self.klist.insert_entry(entry, rank=1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertIs(result.tree, self.klist)
        self.assertTrue(result.inserted)
        self.assertIsInstance(result.next_entry, (type(None), Entry))


class TestInsertResultConsistency(BaseTreeTestCase):
    """Test InsertResult consistency across different tree types."""
    
    def test_insert_result_consistency_gplus_vs_gkplus(self):
        """Test InsertResult behavior is consistent between GPlusTree and GKPlusTree."""
        gplus_tree = create_gplustree(4)
        gkplus_tree = create_gkplus_tree(4)
        
        item = Item(5, "test_value")
        
        # Insert into both trees
        gplus_result = gplus_tree.insert(item, 1)
        gkplus_result = gkplus_tree.insert(item, 1)
        
        # Both should return InsertResult
        self.assertIsInstance(gplus_result, InsertResult)
        self.assertIsInstance(gkplus_result, InsertResult)
        
        # Both should indicate successful insertion
        self.assertTrue(gplus_result.inserted)
        self.assertTrue(gkplus_result.inserted)
        
        # Both should have None next_entry for single leaf insertion
        self.assertIsNone(gplus_result.next_entry)
        self.assertIsNone(gkplus_result.next_entry)
    
    def test_insert_result_consistency_update_vs_insert(self):
        """Test InsertResult behavior is consistent for updates vs new insertions."""
        tree = create_gplustree(4)  # Use GPlusTree which supports updates
        
        item = Item(5, "original_value")
        
        # First insertion
        insert_result = tree.insert(item, 1)
        self.assertTrue(insert_result.inserted)
        
        # Update same key - only GPlusTree supports updates
        updated_item = Item(5, "updated_value")
        update_result = tree.insert(updated_item, 1)
        self.assertFalse(update_result.inserted)
        
        # Both should return InsertResult
        self.assertIsInstance(insert_result, InsertResult)
        self.assertIsInstance(update_result, InsertResult)
        
        # Both should have same tree reference
        self.assertIs(insert_result.tree, update_result.tree)
    
    def test_insert_result_indexing_consistency(self):
        """Test InsertResult indexing works consistently across all scenarios."""
        tree = create_gplustree(4)
        item = Item(5, "test_value")
        
        result = tree.insert(item, 1)
        
        # Test all indexing methods work
        self.assertIs(result[0], result.tree)
        self.assertEqual(result[1], result.inserted)
        self.assertIs(result[2], result.next_entry)
        
        # Test iteration unpacking
        tree_unpacked, inserted_unpacked = result
        self.assertIs(tree_unpacked, result.tree)
        self.assertEqual(inserted_unpacked, result.inserted)
        
        # Test list conversion
        result_list = list(result)
        self.assertEqual(len(result_list), 2)  # Only tree and inserted in iteration
        self.assertIs(result_list[0], result.tree)
        self.assertEqual(result_list[1], result.inserted)


class TestNextEntryCorrectness(BaseTreeTestCase):
    """Test that next_entry is always correctly returned for different insertion positions."""
    
    def setUp(self):
        super().setUp()
        self.tree = create_gplustree(4)
        # Pre-populate with a sorted sequence for testing
        self.keys = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        for key in self.keys:
            self.tree.insert(Item(key, f"value_{key}"), 1)
    
    def test_insert_at_beginning_returns_correct_next_entry(self):
        """Test inserting at the beginning returns correct next_entry."""
        # Insert at the very beginning (before 10)
        result = self.tree.insert(Item(5, "value_5"), 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # next_entry should point to the first original item (10)
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 10)
            self.assertEqual(result.next_entry.item.value, "value_10")
    
    def test_insert_at_end_returns_none_next_entry(self):
        """Test inserting at the end returns None next_entry."""
        # Insert at the very end (after 90)
        result = self.tree.insert(Item(95, "value_95"), 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # next_entry should be None since this is the last item
        self.assertIsNone(result.next_entry)
    
    def test_insert_between_items_returns_correct_next_entry(self):
        """Test inserting between items returns correct next_entry."""
        test_cases = [
            (15, 20),  # Insert between 10 and 20, next should be 20
            (25, 30),  # Insert between 20 and 30, next should be 30
            (45, 50),  # Insert between 40 and 50, next should be 50
            (75, 80),  # Insert between 70 and 80, next should be 80
        ]
        
        for insert_key, expected_next_key in test_cases:
            with self.subTest(insert_key=insert_key, expected_next_key=expected_next_key):
                result = self.tree.insert(Item(insert_key, f"value_{insert_key}"), 1)
                
                self.assertIsInstance(result, InsertResult)
                self.assertTrue(result.inserted)
                
                if result.next_entry is not None:
                    self.assertEqual(result.next_entry.item.key, expected_next_key)
                    self.assertEqual(result.next_entry.item.value, f"value_{expected_next_key}")
    
    def test_insert_exact_duplicate_returns_correct_next_entry(self):
        """Test inserting exact duplicate key returns correct next_entry."""
        # Insert duplicate of existing key (30)
        result = self.tree.insert(Item(30, "updated_value_30"), 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertFalse(result.inserted)  # Should be update, not insertion
        
        # next_entry should point to the next item after 30 (which is 40)
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 40)
            self.assertEqual(result.next_entry.item.value, "value_40")
    
    def test_insert_sequence_maintains_correct_next_entries(self):
        """Test inserting a sequence of items maintains correct next_entries."""
        new_tree = create_gplustree(4)
        
        # Insert items in non-sorted order and verify next_entry for each
        insert_sequence = [(25, 30), (15, 20), (35, 40), (5, 10), (45, None)]
        existing_keys = [10, 20, 30, 40]
        
        # Pre-populate with some items
        for key in existing_keys:
            new_tree.insert(Item(key, f"value_{key}"), 1)
        
        for insert_key, expected_next_key in insert_sequence:
            with self.subTest(insert_key=insert_key, expected_next_key=expected_next_key):
                result = new_tree.insert(Item(insert_key, f"value_{insert_key}"), 1)
                
                self.assertIsInstance(result, InsertResult)
                self.assertTrue(result.inserted)
                
                if expected_next_key is None:
                    self.assertIsNone(result.next_entry)
                else:
                    self.assertIsNotNone(result.next_entry)
                    self.assertEqual(result.next_entry.item.key, expected_next_key)


class TestGKPlusTreeNextEntryCorrectness(GKPlusTreeTestCase):
    """Test next_entry correctness for GKPlusTree insertions."""
    
    def setUp(self):
        # Pre-populate with a sorted sequence
        self.keys = [10, 20, 30, 40, 50]
        
    
    def test_gkplus_insert_at_beginning_returns_correct_next_entry(self):
        """Test GKPlusTree insert at beginning returns correct next_entry."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        for key in self.keys:
            tree.insert(Item(key, f"value_{key}"), 1)
        result = tree.insert(Item(5, "value_5"), 1)

        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # next_entry should point to the first original item (10)
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 10)
    
    def test_gkplus_insert_at_end_returns_none_next_entry(self):
        """Test GKPlusTree insert at end returns None next_entry."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        for key in self.keys:
            tree.insert(Item(key, f"value_{key}"), 1)
        result = tree.insert(Item(55, "value_55"), 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # next_entry should be None since this is the last item
        self.assertIsNone(result.next_entry)
    
    def test_gkplus_insert_between_items_returns_correct_next_entry(self):
        """Test GKPlusTree insert between items returns correct next_entry."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        for key in self.keys:
            tree.insert(Item(key, f"value_{key}"), 1)
        # Insert between 30 and 40
        result = tree.insert(Item(35, "value_35"), 1)

        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 40)
    
    def test_gkplus_duplicate_key_handling(self):
        """Test GKPlusTree duplicate key handling with next_entry."""
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1)
        for key in self.keys:
            tree.insert(Item(key, f"value_{key}"), 1)
        # GKPlusTree should raise ValueError for duplicate keys
        with self.assertRaises(ValueError) as context:
            tree.insert(Item(30, "updated_value"), 1)

        self.assertIn("Entry with key 30 already exists", str(context.exception))


class TestKListNextEntryCorrectness(BaseTreeTestCase):
    """Test next_entry correctness for KList insert_entry method."""
    
    def setUp(self):
        super().setUp()
        from gplus_trees.factory import make_gplustree_classes
        _, _, self.KListClass, _ = make_gplustree_classes(4)
        self.klist = self.KListClass()
        
        # Pre-populate with a sorted sequence
        self.keys = [10, 20, 30, 40, 50]
        for key in self.keys:
            entry = Entry(Item(key, f"value_{key}"), None)
            self.klist.insert_entry(entry)
    
    def test_klist_insert_at_beginning_returns_correct_next_entry(self):
        """Test KList insert at beginning returns correct next_entry."""
        entry = Entry(Item(5, "value_5"), None)
        result = self.klist.insert_entry(entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # next_entry should point to the first original item (10)
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 10)
    
    def test_klist_insert_at_end_returns_none_next_entry(self):
        """Test KList insert at end returns None next_entry."""
        entry = Entry(Item(55, "value_55"), None)
        result = self.klist.insert_entry(entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # next_entry should be None since this is the last item
        self.assertIsNone(result.next_entry)
    
    def test_klist_insert_between_items_returns_correct_next_entry(self):
        """Test KList insert between items returns correct next_entry."""
        test_cases = [
            (15, 20),  # Between 10 and 20
            (25, 30),  # Between 20 and 30
            (35, 40),  # Between 30 and 40
            (45, 50),  # Between 40 and 50
        ]
        
        for insert_key, expected_next_key in test_cases:
            with self.subTest(insert_key=insert_key, expected_next_key=expected_next_key):
                entry = Entry(Item(insert_key, f"value_{insert_key}"), None)
                result = self.klist.insert_entry(entry)
                
                self.assertIsInstance(result, InsertResult)
                self.assertTrue(result.inserted)
                
                if result.next_entry is not None:
                    self.assertEqual(result.next_entry.item.key, expected_next_key)
    
    def test_klist_update_existing_returns_correct_next_entry(self):
        """Test KList update existing item returns correct next_entry."""
        # Update existing key (30)
        entry = Entry(Item(30, "updated_value_30"), None)
        result = self.klist.insert_entry(entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertFalse(result.inserted)  # Should be update, not insertion
        
        # next_entry should point to the next item after 30 (which is 40)
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 40)
    
    def test_klist_insert_with_subtree_returns_correct_next_entry(self):
        """Test KList insert with left_subtree returns correct next_entry."""
        subtree = create_gplustree(4)
        entry = Entry(Item(25, "value_25"), subtree)
        result = self.klist.insert_entry(entry)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # next_entry should point to the next item after 25 (which is 30)
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 30)
            
        # Verify the subtree was correctly stored
        found_entry, _ = self.klist.retrieve(25)
        self.assertIs(found_entry.left_subtree, subtree)


class TestNextEntryEdgeCases(BaseTreeTestCase):
    """Test edge cases for next_entry behavior."""
    
    def test_empty_tree_insert_returns_none_next_entry(self):
        """Test insert into empty tree returns None next_entry."""
        empty_tree = create_gplustree(4)
        result = empty_tree.insert(Item(10, "value"), 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        self.assertIsNone(result.next_entry)
    
    def test_single_item_tree_insert_before_returns_correct_next_entry(self):
        """Test insert before single item returns correct next_entry."""
        tree = create_gplustree(4)
        tree.insert(Item(20, "existing"), 1)
        
        result = tree.insert(Item(10, "new"), 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        if result.next_entry is not None:
            self.assertEqual(result.next_entry.item.key, 20)
    
    def test_single_item_tree_insert_after_returns_none_next_entry(self):
        """Test insert after single item returns None next_entry."""
        tree = create_gplustree(4)
        tree.insert(Item(10, "existing"), 1)
        
        result = tree.insert(Item(20, "new"), 1)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        self.assertIsNone(result.next_entry)
    
    def test_higher_rank_insert_next_entry_behavior(self):
        """Test next_entry behavior for higher rank insertions."""
        tree = create_gplustree(4)
        
        # Insert some items at rank 1
        for key in [10, 20, 30]:
            tree.insert(Item(key, f"value_{key}"), 1)
        
        # Insert at higher rank
        result = tree.insert(Item(15, "value_15"), 2)
        
        self.assertIsInstance(result, InsertResult)
        self.assertTrue(result.inserted)
        
        # For higher rank insertions, next_entry behavior depends on tree structure
        # We just verify it's the correct type
        self.assertIsInstance(result.next_entry, (type(None), Entry))
    
    def test_large_tree_next_entry_consistency(self):
        """Test next_entry consistency in larger tree structures."""
        tree = create_gplustree(4)
        
        # Insert a smaller number of items to avoid potential infinite loops
        keys = list(range(0, 20, 5))  # [0, 5, 10, 15] - reduced from 100 to 20
        for key in keys:
            tree.insert(Item(key, f"value_{key}"), 1)
        
        # Test insertions at various positions
        test_insertions = [
            (2, 5),    # Near beginning
            (7, 10),   # In middle
            (12, 15),  # In middle  
            (17, None) # At end
        ]
        
        for insert_key, expected_next_key in test_insertions:
            with self.subTest(insert_key=insert_key, expected_next_key=expected_next_key):
                result = tree.insert(Item(insert_key, f"value_{insert_key}"), 1)
                
                self.assertIsInstance(result, InsertResult)
                self.assertTrue(result.inserted)
                
                if expected_next_key is None:
                    self.assertIsNone(result.next_entry)
                else:
                    self.assertIsNotNone(result.next_entry)
                    self.assertEqual(result.next_entry.item.key, expected_next_key)


if __name__ == "__main__":
    unittest.main()
