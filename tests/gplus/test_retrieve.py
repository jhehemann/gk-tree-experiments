"""Tests for G+-trees retrieve method"""
# pylint: skip-file

import unittest
import logging

# Import factory function instead of concrete classes
from gplus_trees.gplus_tree_base import collect_leaf_keys
from tests.test_base import GPlusTreeTestCase as TreeTestCase

# Configure logging for test
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestRetrieveBase(TreeTestCase):
    """Base class for all retrieve tests"""
    def setUp(self):
        super().setUp()
        # Create some default test items
        self.items = {
            1: self.make_item(1, "value_1"),
            2: self.make_item(2, "value_2"),
            3: self.make_item(3, "value_3"),
            4: self.make_item(4, "value_4"),
            5: self.make_item(5, "value_5"),
            6: self.make_item(6, "value_6"),
            7: self.make_item(7, "value_7"),
            8: self.make_item(8, "value_8"),
            9: self.make_item(9, "value_9"),
            10: self.make_item(10, "value_10"),
        }
    
    def _assert_retrieval_result(self, result, expected_key=None, expected_next_key=None):
        """Helper method to assert retrieval results"""
        found_entry, next_entry = result
        if expected_key is None:
            self.assertIsNone(found_entry, "Expected no found entry")
        else:
            self.assertIsNotNone(found_entry, f"Expected a found entry {expected_key}")
            self.assertEqual(found_entry.item.key, expected_key, 
                             f"Expected found key {expected_key}, got {found_entry.item.key}")

        if expected_next_key is None:
            self.assertIsNone(next_entry, "Expected no next entry")
        else:
            self.assertIsNotNone(next_entry, f"Expected next entry {expected_next_key} for expected key {expected_key}")
            self.assertEqual(next_entry.item.key, expected_next_key, 
                             f"Expected next key {expected_next_key}, got {next_entry.item.key}")


class TestRetrieveEmptyTree(TestRetrieveBase):
    """Test retrieving from an empty tree"""
    def test_retrieve_empty_tree(self):
        """Test retrieving from an empty tree returns None"""
        result = self.tree.retrieve(1)
        self._assert_retrieval_result(result, None, None)


class TestRetrieveSingleNodeTree(TestRetrieveBase):
    """Test retrieving from a tree with a single node"""
    def setUp(self):
        super().setUp()
        # Create a tree with one item
        self.tree.insert(self.items[3], 1)
    
    def test_retrieve_existing_key(self):
        """Test retrieving an existing key"""
        result = self.tree.retrieve(3)
        self._assert_retrieval_result(result, 3, None)
        # Verify the value is correct
        self.assertEqual(result[0].item.value, "value_3")

    def test_retrieve_nonexistent_key_less_than_min(self):
        """Test retrieving a key that doesn't exist (less than min)"""
        result = self.tree.retrieve(1)
        self._assert_retrieval_result(result, None, 3)
    
    def test_retrieve_nonexistent_key_between(self):
        """Test retrieving a key that doesn't exist (between existing keys)"""
        # First add another item to have a range
        self.tree.insert(self.items[5], 1)
        result = self.tree.retrieve(4)
        self._assert_retrieval_result(result, None, 5)
    
    def test_retrieve_nonexistent_key_greater_than_max(self):
        """Test retrieving a key that doesn't exist (greater than max)"""
        result = self.tree.retrieve(5)
        self._assert_retrieval_result(result, None, None)


class TestRetrieveMultiItemLeafNode(TestRetrieveBase):
    """Test retrieving from a tree with multiple items in a single leaf node"""
    def setUp(self):
        super().setUp()
        # Create a tree with multiple items in one leaf node
        keys = [1, 3, 5]
        for key in keys:
            self.tree.insert(self.items[key], 1)
    
    def test_retrieve_first_key(self):
        """Test retrieving the first key"""
        result = self.tree.retrieve(1)
        self._assert_retrieval_result(result, 1, 3)
        self.assertEqual(result[0].item.value, "value_1")

    def test_retrieve_middle_key(self):
        """Test retrieving a middle key"""
        result = self.tree.retrieve(3)
        self._assert_retrieval_result(result, 3, 5)
        self.assertEqual(result[0].item.value, "value_3")
    
    def test_retrieve_last_key(self):
        """Test retrieving the last key"""
        result = self.tree.retrieve(5)
        self._assert_retrieval_result(result, 5, None)
        self.assertEqual(result[0].item.value, "value_5")

    def test_retrieve_nonexistent_key_before_all(self):
        """Test retrieving a key before all existing keys"""
        # Testing for a key before all existing keys
        result = self.tree.retrieve(0)
        self._assert_retrieval_result(result, None, 1)
    
    def test_retrieve_nonexistent_key_between(self):
        """Test retrieving a key between existing keys"""
        # Testing for a key between existing keys
        result = self.tree.retrieve(2)
        self._assert_retrieval_result(result, None, 3)
        
        result = self.tree.retrieve(4)
        self._assert_retrieval_result(result, None, 5)
    
    def test_retrieve_nonexistent_key_after_all(self):
        """Test retrieving a key after all existing keys"""
        # Testing for a key after all existing keys
        result = self.tree.retrieve(6)
        self._assert_retrieval_result(result, None, None)


class TestRetrieveMultiNodeTree(TestRetrieveBase):
    """Test retrieving from a multi-level tree structure"""
    def setUp(self):
        super().setUp()
        # Create a tree with multiple nodes by inserting items with ranks > 1
        keys_ranks = [(1, 1), (2, 2), (3, 1), (5, 3), (7, 1)]
        for key, rank in keys_ranks:
            self.tree.insert(self.items[key], rank)
    
    def test_retrieve_leaf_level_keys(self):
        """Test retrieving keys at leaf level"""
        # Test all existing keys
        for key in [1, 2, 3, 5, 7]:
            result = self.tree.retrieve(key)
            self._assert_retrieval_result(result, key, None if key == 7 else key + (2 if key == 3 or key == 5 else 1))
            self.assertEqual(result[0].item.value, f"value_{key}")
    
    def test_retrieve_nonexistent_keys(self):
        """Test retrieving keys that don't exist"""
        # Test keys that don't exist
        key_next_pairs = [
            (0, 1),    # Before first
            (4, 5),    # Between 3 and 5
            (6, 7),    # Between 5 and 7
            (8, None)  # After last
        ]
        for key, expected_next in key_next_pairs:
            result = self.tree.retrieve(key)
            self._assert_retrieval_result(result, None, expected_next)


class TestRetrieveComplexTree(TestRetrieveBase):
    """Test retrieving from a complex tree structure"""        
    def test_retrieve_all_keys(self):
        """Test retrieving all keys in the complex tree"""        
        keys_ranks = [(1, 4), (2, 1), (3, 3), (4, 2), (5, 1), 
                      (6, 3), (7, 2), (8, 4), (9, 1)]
        
        for key, rank in keys_ranks:
            self.tree.insert(self.items[key], rank)

        keys = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for i, key in enumerate(keys):
            result = self.tree.retrieve(key)
            expected_next = None if i == len(keys) - 1 else keys[i + 1]
            self._assert_retrieval_result(result, key, expected_next)
            self.assertEqual(result[0].item.value, f"value_{key}")
    
    def test_retrieve_nonexistent_keys_complex(self):
        """Test retrieving nonexistent keys in a complex tree"""
        keys_ranks = [(1, 4), (2, 1), (3, 3), (5, 1), 
                      (6, 3), (7, 2), (8, 4), (9, 1)]
        
        for key, rank in keys_ranks:
            self.tree.insert(self.items[key], rank)
        
        nonexistent_keys = [0, 4, 10]
        expected_next = [1, 5, None]
        
        for key, next_key in zip(nonexistent_keys, expected_next):
            result = self.tree.retrieve(key)
            self._assert_retrieval_result(result, None, next_key)


class TestRetrieveEdgeCases(TestRetrieveBase):
    """Test edge cases for retrieve method"""
    def test_retrieve_invalid_key_type(self):
        """Test retrieving with an invalid key type"""
        with self.assertRaises(TypeError):
            self.tree.retrieve("not_an_int")
    
    # def test_retrieve_negative_key(self):
    #     """Test retrieving with a negative key"""
    #     with self.assertRaises(TypeError):
    #         self.tree.retrieve(-1)
    
    def test_retrieve_after_update(self):
        """Test retrieving after updating a value"""
        # Insert initial item
        self.tree.insert(self.items[1], 1)
        
        # Update the value
        updated_item = self.make_item(1, "updated_value")
        self.tree.insert(updated_item, 1)
        
        # Retrieve and verify the updated value
        result = self.tree.retrieve(1)
        self._assert_retrieval_result(result, 1, None)
        self.assertEqual(result[0].item.value, "updated_value")
    
    def test_retrieve_with_duplicate_keys(self):
        """Test retrieving with duplicate keys (should update, not duplicate)"""
        # Insert initial items
        keys = [1, 2, 3]
        for key in keys:
            self.tree.insert(self.items[key], 1)
        
        # Insert a duplicate key
        duplicate_item = self.make_item(2, "duplicate_value")
        self.tree.insert(duplicate_item, 1)
        
        # Verify the item was updated, not duplicated
        result = self.tree.retrieve(2)
        self._assert_retrieval_result(result, 2, 3)
        self.assertEqual(result[0].item.value, "duplicate_value")
        
        # Ensure the total count of items is still 3
        leaf_keys = collect_leaf_keys(self.tree)
        self.assertEqual(len(leaf_keys), 3, "Expected 3 keys in tree")


if __name__ == "__main__":
    unittest.main()