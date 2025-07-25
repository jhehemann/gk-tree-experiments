"""Comprehensive tests for the GKPlusTreeBase insert_min() method."""

import unittest
import copy
from typing import List, Tuple, Optional
from gplus_trees.base import Entry, ItemData, LeafItem
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, get_dummy
from gplus_trees.utils import calc_rank_from_digest_k, get_group_size
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase

from gplus_trees.logging_config import get_test_logger

logger = get_test_logger(__name__)


class TestGKPlusTreeInsertMin(GKPlusTreeTestCase):
    """Comprehensive tests for the insert_min() method in GKPlusTreeBase."""
    
    def setUp(self):
        """Set up test fixtures with different tree configurations."""
        super().setUp()
        self.k = 4  # Default K-list node capacity for tests
    
    def _create_tree(self, keys: List[int], ranks: List[int], k: int = None, tree=None):
        """Helper to create a tree with given keys and ranks."""
        if k is None:
            k = self.k
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
    
    def _calc_rank_for_key(self, key: int, k: int, dim: int = 1) -> int:
        """Calculate rank for a key in given dimension."""
        group_size = get_group_size(k)
        item_data = ItemData(key, f"val_{key}")
        item = LeafItem(item_data)
        digest = item.get_digest_for_dim(dim)
        return calc_rank_from_digest_k(digest, k)
    
    def _create_entry(self, key: int, left_subtree=None) -> Entry:
        """Helper to create an Entry with given key."""
        item = self.make_item(key, f"val_{key}")
        return Entry(item, left_subtree)
    
    def _validate_insert_min_result(self, 
                                   tree: GKPlusTreeBase, 
                                   inserted: bool, 
                                   next_entry: Optional[Entry], 
                                   third_entry: Optional[Entry],
                                   expected_keys: List[int],
                                   expected_inserted: bool = True,
                                   add_dummy: bool = True,
                                   test_name: str = ""):
        """Helper to validate insert_min result."""
        prefix = f"[{test_name}] " if test_name else ""
        
        # Check insertion result
        self.assertEqual(inserted, expected_inserted, 
                        f"{prefix}Expected inserted={expected_inserted}, got {inserted}")
        
        # Validate next_entry
        if next_entry is not None:
            self.assertIsInstance(next_entry, Entry, f"{prefix}next_entry should be Entry")
            self.assertIsNotNone(next_entry.item, f"{prefix}next_entry.item should not be None")
        
        # Validate third_entry  
        if third_entry is not None:
            self.assertIsInstance(third_entry, Entry, f"{prefix}third_entry should be Entry")
            self.assertIsNotNone(third_entry.item, f"{prefix}third_entry.item should not be None")

        # Collect actual keys including dummies
        dummy_keys = sorted(self.get_dummies(tree))
        expected_keys_with_dummies = sorted(expected_keys + dummy_keys)

        # Validate tree structure and invariants
        self.validate_tree(tree, expected_keys_with_dummies, err_msg=f"{prefix}Tree validation failed")
        
        if add_dummy:
            # Check the correct number of dummy keys in relation to the expanded leafs count
            expanded_leaf_count = tree.expanded_count()
            self.assertEqual(len(dummy_keys), expanded_leaf_count + 1,
                            f"{prefix}Expected dummy key count {expanded_leaf_count + 1}, got {len(dummy_keys)}")

    # ==================== EMPTY TREE TESTS ====================
    
    def test_insert_min_empty_tree_rank_1_with_dummy(self):
        """Test insert_min into empty tree with rank 1 and add_dummy=True."""
        tree = create_gkplus_tree(K=self.k)
        entry = self._create_entry(10)
        
        tree, inserted, next_entry, third_entry = tree.insert_min(entry, rank=1, add_dummy=True)
        
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[10], test_name="empty_tree_rank_1_with_dummy"
        )
        
        # Specific assertions for empty tree insertion
        self.assertIsNotNone(tree.node, "Tree should have a node after insertion")
        self.assertEqual(tree.node.rank, 1, "Root node should have rank 1")
        self.assertIsNone(tree.node.right_subtree, "Leaf node should not have right subtree")
    
    def test_insert_min_empty_tree_rank_1_no_dummy(self):
        """Test insert_min into empty tree with rank 1 and add_dummy=False."""
        tree = create_gkplus_tree(K=self.k)
        entry = self._create_entry(10)
        add_dummy = False
        
        tree, inserted, next_entry, third_entry = tree.insert_min(entry, rank=1, add_dummy=add_dummy)
        logger.debug(f"Tree after insert_min with rank 1 and no dummy: {print_pretty(tree)}")
        
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[10], add_dummy=add_dummy, test_name="empty_tree_rank_1_no_dummy"
        )
    
    def test_insert_min_empty_tree_rank_gt_1_with_dummy(self):
        """Test insert_min into empty tree with rank > 1 and add_dummy=True."""
        tree = create_gkplus_tree(K=self.k)
        entry = self._create_entry(10)
        
        tree, inserted, next_entry, third_entry = tree.insert_min(entry, rank=3, add_dummy=True)
        logger.debug(f"Tree after insert_min with rank > 1 and dummy: {print_pretty(tree)}")
        
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[10], test_name="empty_tree_rank_gt_1_with_dummy"
        )
        
        # Specific assertions for higher rank insertion  
        self.assertIsNotNone(tree.node, "Tree should have a node after insertion")
        self.assertEqual(tree.node.rank, 3, "Root node should have rank 3")
        self.assertIsNotNone(tree.node.right_subtree, "Non-leaf node should have right subtree")
    
    def test_insert_min_empty_tree_rank_gt_1_no_dummy(self):
        """Test insert_min into empty tree with rank > 1 and add_dummy=False."""
        tree = create_gkplus_tree(K=self.k)
        entry = self._create_entry(10)
        
        tree, inserted, next_entry, third_entry = tree.insert_min(entry, rank=2)

        
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[10], test_name="empty_tree_rank_gt_1_no_dummy"
        )
    
    # ==================== NON-EMPTY TREE TESTS ====================
    
    def test_insert_min_non_empty_simple_leaf(self):
        """Test insert_min into non-empty tree with simple leaf structure."""
        # Create tree with existing items
        keys = [20, 30]
        ranks = [1, 1]
        base_tree = self._create_tree_via_unzip(keys, ranks, k=self.k)

        entry = self._create_entry(10)

        tree, inserted, next_entry, third_entry = base_tree.insert_min(entry, rank=1, add_dummy=True)

        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[10, 20, 30], test_name="non_empty_simple_leaf"
        )

    def test_insert_min_non_empty_rank_2(self):
        """Test insert_min into non-empty tree with simple leaf structure."""
        # Create tree with existing items
        keys = [100]
        ranks = [2]
        base_tree = self._create_tree_via_unzip(keys, ranks, k=self.k)

        entry = self._create_entry(50)

        tree, inserted, next_entry, third_entry = base_tree.insert_min(entry, rank=2, add_dummy=True)

        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[50, 100], test_name="non_empty_rank_2"
        )
    
    def test_insert_min_non_empty_different_ranks(self):
        """Test insert_min with different rank combinations."""
        # Test various rank combinations
        test_cases = [
            ([20, 30], [1, 2], 10, 1),  # Insert rank 1 into mixed ranks
            # ([20, 30], [2, 2], 10, 2),  # Insert rank 2 into rank 2 tree
            # ([20, 30], [1, 1], 10, 3),  # Insert higher rank into rank 1 tree
            # ([20, 30], [3, 3], 10, 1),  # Insert lower rank into rank 3 tree
        ]
        add_dummy = True
        
        for i, (initial_keys, initial_ranks, insert_key, insert_rank) in enumerate(test_cases):
            with self.subTest(case=i):
                tree = self._create_tree_via_unzip(initial_keys, initial_ranks, k=self.k)
                entry = self._create_entry(insert_key)
                
                tree, inserted, next_entry, third_entry = tree.insert_min(
                    entry, rank=insert_rank, add_dummy=add_dummy
                )
                
                expected_keys = sorted(initial_keys + [insert_key])
                self._validate_insert_min_result(
                    tree, inserted, next_entry, third_entry, 
                    expected_keys, add_dummy=add_dummy,test_name=f"different_ranks_case_{i}"
                )
    
    def test_insert_min_with_existing_minimum(self):
        """Test insert_min when inserting a key smaller than current minimum."""
        tree = self._create_tree_via_unzip([15, 25, 35], [1, 2, 1], k=self.k)
        entry = self._create_entry(5)  # Smaller than all existing keys
        
        tree, inserted, next_entry, third_entry = tree.insert_min(entry, rank=1, add_dummy=True)
        
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[5, 15, 25, 35], test_name="with_existing_minimum"
        )
    
    def test_insert_min_with_complex_tree_structure(self):
        """Test insert_min with a more complex tree structure."""
        # Create a tree with multiple levels and various ranks
        keys = [100, 200, 150, 250, 120, 180]
        ranks = [2, 3, 1, 2, 3, 1]
        tree = self._create_tree_via_unzip(keys, ranks, k=self.k)
        
        # Insert minimum key
        entry = self._create_entry(50)
        tree, inserted, next_entry, third_entry = tree.insert_min(entry, rank=2, add_dummy=True)
        
        expected_keys = sorted(keys + [50])
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys, test_name="complex_tree_structure"
        )
    
    # ==================== UNZIP INTEGRATION TESTS ====================
    
    def test_insert_min_after_unzip_left_result(self):
        """Test insert_min on the left result of unzip operation."""
        # Create initial tree
        original_keys = [10, 20, 30, 40, 50]
        original_ranks = [1, 2, 1, 2, 1]
        tree = self._create_tree_via_unzip(original_keys, original_ranks, k=self.k)
        logger.debug(f"Initial tree: {print_pretty(tree)}")
        
        # Unzip at key 25
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(25)
        logger.debug(f"Left tree after unzip: {print_pretty(left_tree)}")
        logger.debug(f"Right tree after unzip: {print_pretty(right_tree)}")
        
        # Apply insert_min on left result
        min_entry = self._create_entry(5)
        left_tree, inserted, next_entry_after, third_entry = left_tree.insert_min(
            min_entry, rank=1, add_dummy=True
        )
        logger.debug(f"Left tree after insert_min: {print_pretty(left_tree)}")
        
        # Left tree should contain keys < 25 plus the new minimum
        expected_left_keys = [5, 10, 20]
        self._validate_insert_min_result(
            left_tree, inserted, next_entry_after, third_entry, 
            expected_left_keys, test_name="after_unzip_left"
        )
        
        # Validate original unzip results are preserved
        self.assertIsNotNone(right_tree, "Right tree should not be None")
        if not right_tree.is_empty():
            right_keys = self.collect_keys(right_tree)
            # Right tree should contain keys >= 25
            for key in right_keys:
                if key > 0:  # Ignore dummy keys
                    self.assertGreaterEqual(key, 25, f"Right tree key {key} should be >= 25")
    
    def test_insert_min_after_unzip_right_result(self):
        """Test insert_min on the right result of unzip operation."""
        # Create initial tree
        original_keys = [10, 20, 30, 40, 50]
        original_ranks = [1, 2, 1, 2, 1]
        tree = self._create_tree_via_unzip(original_keys, original_ranks, k=self.k)
        
        # Unzip at key 25
        left_tree, key_subtree, right_tree, next_entry = tree.unzip(25)
        
        # Apply insert_min on right result
        min_entry = self._create_entry(28)  # Minimum for right side
        right_tree, inserted, next_entry_after, third_entry = right_tree.insert_min(
            min_entry, rank=1, add_dummy=True
        )
        
        # Right tree should contain keys >= 25 plus the new minimum
        expected_right_keys = [28, 30, 40, 50]
        self._validate_insert_min_result(
            right_tree, inserted, next_entry_after, third_entry, 
            expected_right_keys, test_name="after_unzip_right"
        )
    
    def test_insert_min_after_multiple_unzip_operations(self):
        """Test insert_min after multiple unzip operations."""
        # Create initial tree
        original_keys = [100, 200, 300, 400, 500, 600]
        original_ranks = [2, 1, 3, 1, 2, 1]
        tree = self._create_tree_via_unzip(original_keys, original_ranks, k=self.k)
        
        # First unzip at 350
        left1, key_subtree1, right1, _ = tree.unzip(350)
        
        # Second unzip on left result at 250  
        left2, key_subtree2, right2, _ = left1.unzip(250)
        
        # Apply insert_min on the final left result
        min_entry = self._create_entry(50)
        left2, inserted, next_entry, third_entry = left2.insert_min(
            min_entry, rank=1, add_dummy=True
        )
        
        # Should contain keys < 250 plus new minimum
        expected_keys = [50, 100, 200]
        self._validate_insert_min_result(
            left2, inserted, next_entry, third_entry, 
            expected_keys, test_name="after_multiple_unzip"
        )
    
    # ==================== EDGE CASES AND ERROR CONDITIONS ====================
    
    def test_insert_min_with_none_entry(self):
        """Test insert_min with None entry should raise appropriate error."""
        tree = create_gkplus_tree(K=self.k)
        
        with self.assertRaises((TypeError, AttributeError)):
            tree.insert_min(None, rank=1)
    
    def test_insert_min_with_invalid_rank(self):
        """Test insert_min with invalid rank values."""
        tree = create_gkplus_tree(K=self.k)
        entry = self._create_entry(10)
        
        # Test with zero rank
        with self.assertRaises((ValueError, AssertionError)):
            tree.insert_min(entry, rank=0)
        
        # Test with negative rank
        with self.assertRaises((ValueError, AssertionError)):
            tree.insert_min(entry, rank=-1)
    
    def test_insert_min_preserves_tree_invariants(self):
        """Test that insert_min preserves all tree invariants."""
        # Create trees with different configurations
        test_configs = [
            ([10, 20, 30], [1, 1, 1]),
            ([10, 20, 30], [2, 2, 2]),
            ([10, 20, 30], [1, 2, 3]),
            ([10, 20, 30, 40, 50], [1, 2, 1, 3, 2]),
        ]
        
        for i, (keys, ranks) in enumerate(test_configs):
            with self.subTest(config=i):
                tree = self._create_tree_via_unzip(keys, ranks, k=self.k)
                entry = self._create_entry(5)  # Always minimum
                
                tree, inserted, next_entry, third_entry = tree.insert_min(
                    entry, rank=1, add_dummy=True
                )
                
                # Validate tree structure
                self.validate_tree(tree, err_msg=f"Config {i} failed invariant check")
    
    def test_insert_min_different_k_values(self):
        """Test insert_min with different K values."""
        # k_values = [2, 4, 8, 16]
        k_values = [4]
        
        for k in k_values:
            with self.subTest(k=k):
                tree = create_gkplus_tree(K=k)
                keys = [100, 50, 25, 12, 6]
                
                # Insert multiple items using insert_min
                for i, key in enumerate(keys):
                    logger.debug(f"Inserting key {key} with K={k}, index={i}")
                    entry = self._create_entry(key)
                    rank = self._calc_rank_for_key(key, k)
                    
                    # add_dummy = True if i < len(keys) - 1 else False  # Last one no dummy
                    add_dummy = False
                    if i == len(keys) - 1:
                        logger.debug(f"Last insert_min ({i}) with key {key}, rank={rank}, add_dummy={add_dummy}")
                        add_dummy = True
                    tree, inserted, next_entry, third_entry = tree.insert_min(
                        entry, rank=rank, add_dummy=add_dummy
                    )
                    
                    self.assertTrue(inserted, f"Failed to insert key {key} with K={k}")

                actual_keys = [e.item.key for e in tree]
                logger.debug(f"actual_keys after insertions: {actual_keys}")
                for i, leaf in enumerate(tree.iter_leaf_nodes()):
                    logger.debug(f"Leaf {i}, rank {leaf.rank}: {print_pretty(leaf.set)}")

                # Validate final tree
                expected_keys = keys
                self._validate_insert_min_result(
                    tree, True, None, None, 
                    expected_keys, test_name=f"k_value_{k}"
                )
    
    # ==================== DUMMY HANDLING TESTS ====================
    
    def test_insert_min_add_dummy_flag_behavior(self):
        """Test the add_dummy flag behavior in various scenarios."""
        test_cases = [
            (True, "with_dummy"),
            (False, "without_dummy"),
        ]
        
        for add_dummy, case_name in test_cases:
            with self.subTest(add_dummy=add_dummy):
                tree = create_gkplus_tree(K=self.k)
                entry = self._create_entry(10)
                
                tree, inserted, next_entry, third_entry = tree.insert_min(
                    entry, rank=1, add_dummy=add_dummy
                )
                
                self.assertTrue(inserted, f"Insertion should succeed with add_dummy={add_dummy}")
                
                # Count dummy keys
                dummy_count = len([k for k in self.collect_keys(tree) if k < 0])
                
                if add_dummy:
                    self.assertGreater(dummy_count, 0, "Should have dummy keys when add_dummy=True")
                # Note: add_dummy=False might still create dummies in complex cases
    
    def test_insert_min_dummy_interaction_with_existing_dummies(self):
        """Test insert_min behavior when tree already contains dummies."""
        # Create tree with existing structure (will have dummies)
        tree = self._create_tree_via_unzip([20, 30], [2, 1], k=self.k)
        initial_dummy_count = len([k for k in self.collect_keys(tree) if k < 0])
        
        entry = self._create_entry(10)
        tree, inserted, next_entry, third_entry = tree.insert_min(
            entry, rank=1, add_dummy=True
        )
        
        final_dummy_count = len([k for k in self.collect_keys(tree) if k < 0])
        
        # Should still have dummies (may increase due to new structure)
        self.assertGreaterEqual(final_dummy_count, initial_dummy_count,
                              "Dummy count should not decrease")
        
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys=[10, 20, 30], test_name="dummy_interaction"
        )
    
    # ==================== RETURN VALUE TESTS ====================
    
    def test_insert_min_return_values_consistency(self):
        """Test that insert_min return values are consistent and correct."""
        tree = self._create_tree_via_unzip([20, 30], [1, 2], k=self.k)
        entry = self._create_entry(10)
        
        result = tree.insert_min(entry, rank=1, add_dummy=True)
        
        # Should return a 4-tuple
        self.assertEqual(len(result), 4, "insert_min should return 4-tuple")
        
        tree_result, inserted, next_entry, third_entry = result
        
        # Tree should be the same object (modified in place)
        self.assertIs(tree_result, tree, "Returned tree should be same object")
        
        # Inserted should be boolean
        self.assertIsInstance(inserted, bool, "inserted should be boolean")
        
        # next_entry should be Entry or None
        self.assertTrue(next_entry is None or isinstance(next_entry, Entry),
                       "next_entry should be None or Entry")
        
        # third_entry should be Entry or None  
        self.assertTrue(third_entry is None or isinstance(third_entry, Entry),
                       "third_entry should be None or Entry")
    
    def test_insert_min_next_entry_correctness(self):
        """Test that next_entry returned by insert_min points to correct entry."""
        # Create tree and insert minimum
        tree = self._create_tree_via_unzip([20, 30, 40], [1, 1, 1], k=self.k)
        entry = self._create_entry(10)
        
        tree, inserted, next_entry, third_entry = tree.insert_min(
            entry, rank=1, add_dummy=True
        )
        
        if next_entry is not None:
            # next_entry should point to entry after the inserted minimum
            # In this case, should be related to the tree structure
            self.assertIsInstance(next_entry.item.key, int,
                                "next_entry should have valid key")
    
    # ==================== PERFORMANCE AND STRESS TESTS ====================
    
    def test_insert_min_large_tree(self):
        """Test insert_min with larger tree structures."""
        # Create a tree with many items
        keys = list(range(100, 1000, 50))  # [100, 150, 200, ..., 950]
        ranks = [1 + (i % 3) for i in range(len(keys))]  # Mix of ranks 1, 2, 3
        
        tree = self._create_tree_via_unzip(keys, ranks, k=self.k)
        
        # Insert minimum
        entry = self._create_entry(50)
        tree, inserted, next_entry, third_entry = tree.insert_min(
            entry, rank=1, add_dummy=True
        )
        
        expected_keys = sorted(keys + [50])
        self._validate_insert_min_result(
            tree, inserted, next_entry, third_entry, 
            expected_keys, test_name="large_tree"
        )
    
    def test_insert_min_sequential_insertions(self):
        """Test multiple sequential insert_min operations."""
        tree = create_gkplus_tree(K=self.k)
        
        # Insert items in decreasing order using insert_min
        keys_to_insert = [100, 80, 60, 40, 20, 10, 5]
        inserted_keys = []
        
        for key in keys_to_insert:
            entry = self._create_entry(key)
            rank = self._calc_rank_for_key(key, self.k)
            
            tree, inserted, next_entry, third_entry = tree.insert_min(
                entry, rank=rank, add_dummy=True
            )
            
            self.assertTrue(inserted, f"Failed to insert key {key}")
            inserted_keys.append(key)
            
            # Validate tree after each insertion
            self._validate_insert_min_result(
                tree, True, next_entry, third_entry, 
                inserted_keys, test_name=f"sequential_after_{key}"
            )
    
    # ==================== COMPLEX SCENARIOS ====================
    
    def test_insert_min_mixed_with_regular_operations(self):
        """Test insert_min mixed with regular tree operations."""
        tree = create_gkplus_tree(K=self.k)
        
        # Mix of regular inserts and insert_min
        operations = [
            ('regular', 50, 1),
            ('insert_min', 25, 1),
            ('regular', 75, 2),
            ('insert_min', 10, 1),
            ('regular', 60, 1),
            ('insert_min', 5, 2),
        ]
        
        expected_keys = []
        
        for op_type, key, rank in operations:
            if op_type == 'regular':
                item = self.make_item(key, f"val_{key}")
                tree, inserted, _ = tree.insert(item, rank=rank)
                self.assertTrue(inserted, f"Regular insert of {key} failed")
                expected_keys.append(key)
            else:  # insert_min
                entry = self._create_entry(key)
                
                # always unzip before insert_min
                left_dummy_tree, key_subtree, right, _ = tree.unzip(-1)
                tree, inserted, next_entry, third_entry = right.insert_min(
                    entry, rank=rank, add_dummy=True
                )
                self.assertTrue(inserted, f"insert_min of {key} failed")
                expected_keys.append(key)
        
        # Validate final result
        self._validate_insert_min_result(
            tree, True, None, None, 
            sorted(expected_keys), test_name="mixed_operations"
        )
    
    


if __name__ == "__main__":
    unittest.main()
