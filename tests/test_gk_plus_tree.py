import sys
import os
import unittest
import random
from typing import List, Tuple, Optional, Iterator
from itertools import product, islice
from pprint import pprint
import copy
from tqdm import tqdm
from statistics import median_low


# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from tests.utils import assert_tree_invariants_tc

class TestGKPlusSizeTracking(unittest.TestCase):
    
    def setUp(self):
        # Create trees with different K values for testing
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)
        
    def create_item(self, key, value="val"):
        """Helper to create test items"""
        return Item(key, value)
    
    def test_empty_tree_size(self):
        """Test that an empty tree has a node with size 0"""
        tree = create_gkplus_tree(K=4)
        self.assertTrue(tree.is_empty())
        # Size is a property of nodes, not empty trees
        
    def test_single_insertion_size(self):
        """Test size is 1 after inserting a single item"""
        item = Item(1000, "val")
        tree, inserted = self.tree_k2.insert(item, rank=1)
        self.assertTrue(inserted)
        actual_size = tree.node.get_size()
        self.assertIsNotNone(actual_size)
        self.assertEqual(1, actual_size, "Tree size should be 1 after single insertion")
    
    def test_multiple_insertions_size(self):
        """Test size increases properly with multiple insertions"""
        tree = self.tree_k4
        expected_size = 0
        
        # Insert 10 items sequentially
        for i in range(1, 11):
            item = Item(i * 1000, "val") 
            tree, inserted = tree.insert(item, rank=1)
            expected_size += 1
            actual_size = tree.node.get_size()
            self.assertEqual(expected_size, actual_size, 
                             f"Tree size should be {expected_size} after {i} insertions, got {actual_size}")
    
    def test_duplicate_insertion_size(self):
        """Test size doesn't change when inserting duplicates"""
        tree = self.tree_k2
        
        # First insertion
        item = Item(5000, "val")
        tree, inserted = tree.insert(item, rank=1)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.get_size())
        
        # Duplicate insertion
        item_duplicate = Item(5000, "new_val")
        tree, inserted = tree.insert(item_duplicate, rank=1)
        self.assertFalse(inserted)
        self.assertEqual(1, tree.node.get_size(), "Size should not change after duplicate insertion")
    
    def test_size_with_node_splitting(self):
        """Test size is correctly maintained when nodes are split"""
        tree = self.tree_k2  # Use K=2 to force early splits
        
        # Insert enough items to force node splits
        keys = [100, 200, 300, 400, 500, 600, 700, 800]
        for i, key in enumerate(keys, 1):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=1)
            self.assertEqual(i, tree.node.get_size(), f"Tree should have size {i} after inserting {key}")
        
        tree, _ = tree.insert(Item(450, "val"), rank=2)
        
        # Verify size after all insertions
        self.assertEqual(len(keys) + 1, tree.node.get_size())
        
        # Also verify that node sizes are consistent throughout the tree
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_size_with_different_ranks(self):
        """Test size is correctly tracked with items at different ranks"""
        tree = self.tree_k4
        
        # Insert items with different ranks
        items_and_ranks = [
            (Item(1000, "val"), 1),
            (Item(2000, "val"), 2),
            (Item(3000, "val"), 1),
            (Item(4000, "val"), 3),
            (Item(5000, "val"), 2)
        ]
        
        for i, (item, rank) in enumerate(items_and_ranks, 1):
            tree, _ = tree.insert(item, rank)
            self.assertEqual(i, tree.node.get_size(), f"Tree should have size {i} after inserting item with rank {rank}")
        
        # Verify size after all insertions
        self.assertEqual(len(items_and_ranks), tree.node.size)
        
        # Verify subtree sizes are consistent
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_large_tree_size(self):
        """Test size is correctly maintained in a larger tree with random insertions"""
        tree = self.tree_k4
        keys = random.sample(range(1, 10000), 100)  # 100 unique random keys
        ranks = random.choices(range(1, 6), k=100)  # Random ranks between 1 and 5
        
        # Insert all items
        for i, (key, rank) in enumerate(zip(keys, ranks), 1):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=rank)
            self.assertEqual(i, tree.node.get_size(), f"Tree should have size {i} after inserting {key}")
        
        # Verify size after all insertions
        self.assertEqual(len(keys), tree.node.size)
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_complex_tree_structure(self):
        """Test size in a complex tree with various ranks and splits"""
        tree = self.tree_k4
        
        # Create a mix of items with different ranks
        items = []
        for i in range(1, 51):  # 50 items
            key = i * 100
            rank = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)  # Mix of ranks 1, 2, 3
            items.append((Item(key, "val"), rank))
        
        # Insert all items
        for i, (item, rank) in enumerate(items, 1):
            tree, _ = tree.insert(item, rank)
            actual_size = tree.node.get_size()
            self.assertEqual(i, actual_size, 
                            f"Tree should have size {i} after inserting item {item.key} with rank {rank}")
        
        # Verify size after all insertions
        self.assertEqual(len(items), actual_size)
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))
    
    def test_size_after_insert_empty(self):
        """Test that _insert_empty correctly initializes node size"""
        # Case 1: Leaf node (rank 1)
        tree, inserted = self.tree_k4._insert_empty(Item(1000, "val"), rank=1)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.get_size())
        
        # Case 2: Internal node (rank > 1)
        tree, inserted = self.tree_k4._insert_empty(Item(2000, "val"), rank=3)
        self.assertTrue(inserted)
        self.assertEqual(1, tree.node.get_size())
    
    def test_size_consistency_with_calculate_size(self):
        """Test that node.size matches the result of calculate_size()"""
        tree = self.tree_k4
        
        # Insert several items
        for i in range(1, 21):
            tree, _ = tree.insert(Item(i * 500, "val"), rank=1)
        
        node_size = tree.node.size
        self.assertIsNone(node_size, "Expected node size to be invalidated (none) after insert calculation")
        calculated_size = tree.node.calculate_tree_size()
        
        # Check if the calculated size has been set
        node_size = tree.node.size
        self.assertEqual(node_size, calculated_size, f"Expected tree size to be set to {calculated_size} after calculate_tree_size(); got {node_size}")
        
        # Check size consistency for the root node
        self.assertEqual(node_size, calculated_size, f"Expected size {node_size} to match calculated size {calculated_size}")
        
        # Verify size consistency throughout the tree
        self.verify_calculated_sizes_match(tree)
    
    def verify_subtree_sizes(self, tree):
        """
        Recursive helper to verify that the size of each node equals the sum of its children's sizes
        plus the number of real items in the node itself.
        """
        if tree is None:
            return True
        
        if tree.is_empty():
            print("Empty tree: Use None instead of empty tree")
            
        node = tree.node
        
        # For leaf nodes, size should be the count of non-dummy items
        dummy_key = get_dummy(tree.__class__.DIM).key
        if node.rank == 1:
            calculated_size = sum(1 for entry in node.set if entry.item.key != dummy_key)
            if calculated_size != node.size:
                print(f"Leaf node has size {node.size} but contains {calculated_size} items")
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
            print(f"Node at rank {node.rank} has size {node.size} but calculated size is {calculated_size}")
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
            print("Empty tree: Use None instead of empty tree")
            
        node = tree.node
        node_size = node.size
        
        # Check if node.size matches calculated size
        calculated = node.calculate_tree_size()
        if calculated != node_size:
            print(f"Node size {node_size} doesn't match calculated size {calculated}")
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
        
    def test_rank_mismatch_size_handling(self):
        """Test that size is correctly maintained when handling rank mismatches"""
        tree = self.tree_k4
        
        # Insert an item with rank 1
        tree, _ = tree.insert(Item(1000, "val"), rank=1)
        self.assertEqual(1, tree.node.get_size())
        
        # Insert an item with higher rank, triggering rank mismatch logic
        tree, _ = tree.insert(Item(2000, "val"), rank=3)
        self.assertEqual(2, tree.node.get_size())
        
        # Verify subtree sizes are consistent
        self.assertTrue(self.verify_subtree_sizes(tree))
        
    def test_size_with_internal_node_splitting(self):
        """Test size maintenance during internal node splits"""
        tree = self.tree_k2  # K=2 to force splits quickly
        
        # Insert items with increasing rank to create deep tree with internal nodes
        ranks = [1, 1, 2, 1, 3, 2, 1]
        keys = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        
        for i, (key, rank) in enumerate(zip(keys, ranks), 1):
            tree, _ = tree.insert(Item(key, "val"), rank)
            self.assertEqual(i, tree.node.get_size())
        
        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))

class TestGKPlusItemSlotCount(unittest.TestCase):
    """Tests for the item_slot_count method in GKPlusTreeBase."""
    
    def setUp(self):
        # Create trees with different K values for testing
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)
        
    def test_empty_tree_slot_count(self):
        """Test that an empty tree has 0 item slots."""
        tree = create_gkplus_tree(K=4)
        self.assertTrue(tree.is_empty())
        self.assertEqual(0, tree.item_slot_count())
        
    def test_single_item_tree_slot_count(self):
        """Test slot count for a tree with a single item."""
        # Insert one item
        tree, _ = self.tree_k4.insert(Item(1000, "val"), rank=1)
        
        # A single leaf node will be created with K+1 slots (K=4)
        expected_slots = self.tree_k4.SetClass.KListNodeClass.CAPACITY
        self.assertEqual(expected_slots, tree.item_slot_count(),
                         f"Expected {expected_slots} slots for a single item tree with K=4")
        
    def test_cap_node_slot_count(self):
        """Test slot count for a tree with a node at capacity."""
        tree = self.tree_k4
        cap = 4
        
        # Insert item to fill the node to capacity (beware of dummy item)
        for i in range(1, cap-1):
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        exp_item_count = cap
        expected_slots =  exp_item_count + (exp_item_count % tree.SetClass.KListNodeClass.CAPACITY)
        self.assertEqual(expected_slots, tree.item_slot_count(),
                         f"Expected {expected_slots} slots for a tree with {exp_item_count} items in a single node")

    def test_lt_cap_node_slot_count(self):
        """Test slot count for a tree with multiple items in a single leaf node."""
        tree = self.tree_k8
        
        # Insert items that fit in a single leaf node (K=8)
        for i in range(1, 6):  # Insert 5 items
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        # A single leaf node with K+1 slots
        expected_slots = self.tree_k8.SetClass.KListNodeClass.CAPACITY
        self.assertEqual(expected_slots, tree.item_slot_count(),
                         f"Expected {expected_slots} slots for a tree with 5 items in a single node")
        
    def test_multi_leaf_node_slot_count(self):
        """Test slot count for a tree with multiple leaf nodes."""
        tree = self.tree_k2  # K=2 to force splits quickly
        self.cap = 2
        
        # Insert enough items to cause underlying KlistNode splits
        indices = [i for i in range(1, 2 * self.cap + 1)]
        for i in indices:
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        exp_item_count = len(indices) + 1 # +1 for the dummy item
        expected_slots =  exp_item_count + (exp_item_count % self.tree_k2.SetClass.KListNodeClass.CAPACITY)
        self.assertEqual(expected_slots, tree.item_slot_count(),
                        f"Expected {expected_slots} slots for a tree with {exp_item_count} items and K = {self.cap}")
        
    def test_internal_node_slot_count(self):
        """Test slot count for a tree with internal nodes."""
        tree = self.tree_k2
        
        # Insert items with rank 1 to create leaf nodes
        for i in range(1, 8):
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
            
        # Insert items with rank 2 to create internal nodes
        for i in range(1, 4):
            tree, _ = tree.insert(Item(i * 500, "val"), rank=2)
        
        # Count slots manually by traversing the tree
        total_expected_slots = self._count_slots_manually(tree)
        
        self.assertEqual(total_expected_slots, tree.item_slot_count(),
                        f"Expected {total_expected_slots} slots for a tree with internal nodes")
    
    def test_complex_tree_structure_slot_count(self):
        """Test slot count in a complex tree with various ranks and splits."""
        tree = self.tree_k4
        
        # Create a mix of items with different ranks
        items = []
        for i in range(1, 31):  # 30 items
            key = i * 100
            rank = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)  # Mix of ranks 1, 2, 3
            items.append((Item(key, "val"), rank))
        
        # Insert all items
        for item, rank in items:
            tree, _ = tree.insert(item, rank)
        
        # Count slots manually by traversing the tree
        total_expected_slots = self._count_slots_manually(tree)
        
        self.assertEqual(total_expected_slots, tree.item_slot_count(),
                        f"Expected {total_expected_slots} slots for a complex tree structure")
    
    def test_large_tree_slot_count(self):
        """Test slot count in a larger tree with random insertions."""
        tree = self.tree_k8
        keys = random.sample(range(1, 10000), 50)  # 50 unique random keys
        ranks = [1] * 30 + [2] * 15 + [3] * 5  # Mix of ranks
        random.shuffle(ranks)
        
        # Insert all items
        for key, rank in zip(keys, ranks):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=rank)
        
        # Count slots manually by traversing the tree
        total_expected_slots = self._count_slots_manually(tree)
        
        self.assertEqual(total_expected_slots, tree.item_slot_count(),
                        f"Expected {total_expected_slots} slots for a large tree")
    
    def test_slot_count_after_changing_tree_structure(self):
        """Test that slot count updates correctly when tree structure changes."""
        # Start with a tree with several items
        tree = self.tree_k4
        for i in range(1, 11):
            tree, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
        # Get initial slot count
        initial_slots = tree.item_slot_count()
        
        # Insert an item that causes a node split
        tree, _ = tree.insert(Item(500, "val"), rank=2)
        
        # Get new slot count
        new_slots = tree.item_slot_count()
        
        # The slot count should increase after a split
        self.assertGreaterEqual(new_slots, initial_slots,
                              "Slot count should increase or stay the same after structure changes")
        
        # Count slots manually to verify
        total_expected_slots = self._count_slots_manually(tree)
        self.assertEqual(total_expected_slots, new_slots,
                        f"Expected {total_expected_slots} slots after structure change")
        
    def _count_slots_manually(self, tree):
        """Helper method to count slots by traversing the tree structure."""
        if tree is None:
            return 0
            
        total_slots = 0
        node_queue = [tree.node]
        
        while node_queue:
            current_node = node_queue.pop(0)
            
            # Count slots in this node's set
            total_slots += current_node.set.item_slot_count()
            
            # Add child nodes to the queue
            if current_node.rank > 1:
                # Add left subtrees
                for entry in current_node.set:
                    if entry.left_subtree is not None:
                        node_queue.append(entry.left_subtree.node)
                
                # Add right subtree
                if current_node.right_subtree is not None:
                    node_queue.append(current_node.right_subtree.node)
                    
        return total_slots

class TestGKPlusSplitInplace(unittest.TestCase):
    """Tests for the split_inplace method of the GKPlusTreeBase class."""
    
    ASSERTION_MESSAGE_TEMPLATE = (
        "\n\nSplit result:\n"
        "\nLEFT TREE: {left}\n\n"
        "\nMIDDLE TREE: {middle}\n\n"
        "\nRIGHT TREE: {right}\n"
    )
    
    # Initialize items once to avoid re-creating them in each test
    _KEYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ITEMS = {k: Item(k, "val") for k in _KEYS}

    def _get_split_cases(self, keys: List[int]):
        """Helper method to generate split cases based on keys."""        
        if keys[0] == 0:
            raise ValueError("Smallest key should not be 0 to enable splitting below it.")
        
        if self._find_first_missing(keys) is None:
            raise ValueError("No missing middle key that can be split at.")
        
        return [
                ("smallest key",               min(keys)),
                ("largest key",                max(keys)),
                ("existing middle key",        median_low(keys)),
                ("below smallest",             min(keys) - 1),
                ("above largest",              max(keys) + 1),
                ("non-existing middle key",    self._find_first_missing(keys)),
            ]
    
    def setUp(self):
        # Create trees with different K values for testing
        self.K = 4  # Default capacity
        self.tree_k2 = create_gkplus_tree(K=2)
        self.tree_k4 = create_gkplus_tree(K=4)
        self.tree_k8 = create_gkplus_tree(K=8)

    def create_item(self, key, value="val"):
        """Helper to create test items"""
        return Item(key, value)
    
    def validate_tree(
            self,
            tree,
            expected_keys: Optional[List[int]] = None,
            err_msg: Optional[str] = "",
        ):
        """Validate tree invariants and structure"""
        # Check invariants using stats
        stats = gtree_stats_(tree, {})

        # pprint(stats)
        # print(f"item count: {tree.item_count()}")
        assert_tree_invariants_tc(self, tree, stats)
        
        # Verify expected keys if provided
        if expected_keys:
            self.assertFalse(tree.is_empty(), f"Tree should not be empty.\n{err_msg}")
            actual_keys = sorted(self.collect_keys(tree))
            self.assertEqual(expected_keys, actual_keys, 
                            f"Expected keys {expected_keys} don't match actual keys {actual_keys}\n{err_msg}")
            self.assertEqual(len(expected_keys), tree.item_count(),
                            f"Expected {len(expected_keys)} items in tree, got {tree.item_count()}\n{err_msg}")
    
    def verify_keys_in_order(self, tree):
        """Verify that keys in the tree are in sorted order by traversing leaf nodes."""
        if tree.is_empty():
            return True
            
        dummy_key = get_dummy(tree.__class__.DIM).key
        keys = []
        
        # Collect keys from leaf nodes
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                if entry.item.key != dummy_key:
                    keys.append(entry.item.key)
        
        # Check if the keys are in sorted order
        return keys == sorted(keys)
    
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
    

    # TODO: Somewhere a KList with a single node has head = node
    # and tail = None - Not allowed
    def _run_split_case(self, keys, rank_combo, split_key,
                        exp_left, exp_right, case_name):
        if len(rank_combo) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # build the tree once
        base_tree = create_gkplus_tree(K=2)
        for key, rank in zip(keys, rank_combo):
            base_tree, _ = base_tree.insert(self.ITEMS[key], rank)

        # print(print_pretty(base_tree))
        exit

        msg_head = (
            f"\n\nKey-Rank combo:\n"
            f"K: {keys}\n"
            f"R: {rank_combo}"
            f"\n\nTREE BEFORE SPLIT: {print_pretty(base_tree)}\n"
        )

        # deep-copy and split
        tree_copy = copy.deepcopy(base_tree)
        left, middle, right = tree_copy.split_inplace(split_key)

        msg = f"\n\nSplit at {case_name}" + msg_head
        msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
            left=print_pretty(left),
            middle=print_pretty(middle),
            right=print_pretty(right),
        )

        # assertions
        self.validate_tree(left,  exp_left,  msg)
        self.assertIsNone(middle, msg)
        self.validate_tree(right, exp_right, msg)

    def _find_first_missing(self, lst: list[int]) -> int | None:
        """
        Returns the first integer missing between the min and max of lst,
        or None if there are no gaps.
        """
        if not lst:
            return None

        s = sorted(set(lst))
        for a, b in zip(s, s[1:]):
            if b - a > 1:
                return a + 1
        return None
        
    def test_empty_tree_split(self):
        """Test splitting an empty tree."""
        tree = self.tree_k4
        left, middle, right = tree.split_inplace(500)
        
        # Both trees should be empty and middle should be None
        self.assertTrue(left.is_empty())
        self.assertIsNone(middle)
        self.assertTrue(right.is_empty())
        
        # Validate tree invariants for empty trees
        self.validate_tree(left)
        self.validate_tree(right)
    
    def test_split_single_node_tree(self):
        """Test splitting a tree with a single node."""        
        # Insert a single item
        item = Item(500, "val")
        
        
        with self.subTest("split point > only key"):
            tree = create_gkplus_tree(K=4)
            tree, _ = tree.insert(item, rank=1)
            
            # Split at a key greater than the only key
            left, middle, right = tree.split_inplace(1000)
            
            # Validate left tree with the item
            self.assertIs(tree, left)
            self.validate_tree(left, [500])
            self.assertIsNone(middle)
            self.assertTrue(right.is_empty())

        with self.subTest("split point < only key"):
            # Split at a key less than the only key
            tree = create_gkplus_tree(K=4)

            tree, _ = self.tree_k4.insert(item, rank=1)

            # print("\nSelf tree (tree) before split")
            # print(tree.print_structure())
            
            # print("\nSelf Tree (k4) before split")
            # print(self.tree_k4.print_structure())
            
            left, middle, right = tree.split_inplace(100)

            # Validate right tree with the item
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            self.validate_tree(right, [500])
        
        with self.subTest("split point == only key"):
            tree = create_gkplus_tree(K=4)            
            
            tree, _ = self.tree_k4.insert(item, rank=1)
            left, middle, right = tree.split_inplace(500)
            
            # Validate structure
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)  # The item has no left subtree
            self.validate_tree(right)
    
    def test_split_leaf_node_with_multiple_items(self):
        """Test splitting a leaf node with multiple items."""
        tree = self.tree_k4
        
        # Insert multiple items in increasing order
        keys = [100, 200, 300, 400, 500]
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
        
        # Check that initial tree is valid
        self.validate_tree(tree, keys)

        # Split in the middle (between items)
        left, middle, right = tree.split_inplace(250)

        # Validate the split trees
        self.validate_tree(left, [100, 200])
        self.assertIsNone(middle)
        self.validate_tree(right, [300, 400, 500])
    
    
    def test_split_tree_with_internal_nodes(self):
        """Test splitting a tree with internal nodes."""
        tree = self.tree_k2  # Use K=2 to force more internal nodes
        
        # Insert items with different ranks to create internal nodes
        items_and_ranks = [
            (Item(100, "val_1"), 1),
            (Item(200, "val_2"), 1),
            (Item(300, "val_3"), 2),
            (Item(400, "val_4"), 1),
            (Item(500, "val_5"), 3),
            (Item(600, "val_6"), 2),
            (Item(700, "val_7"), 1)
        ]
        
        all_keys = [item[0].key for item in items_and_ranks]
        
        for item, rank in items_and_ranks:
            tree, _ = tree.insert(item, rank)
        
        # Verify initial tree structure
        self.validate_tree(tree, all_keys)
        
        # Split at a key that requires traversing internal nodes
        left, middle, right = tree.split_inplace(450)
        
        # Validate split trees
        self.validate_tree(left, [100, 200, 300, 400])
        self.assertIsNone(middle)
        self.validate_tree(right, [500, 600, 700])
    
    def test_split_with_node_collapsing(self):
        """Test splitting that causes nodes to collapse."""
        tree = self.tree_k2  # Use K=2 to force node splitting quickly
        
        # Create a tree with specific structure that will force node collapsing during split
        items_and_ranks = [
            (Item(100, "val_1"), 1),
            (Item(200, "val_2"), 2),
            (Item(300, "val_3"), 2),
            (Item(400, "val_4"), 3),
            (Item(500, "val_5"), 1),
            (Item(600, "val_6"), 2)
        ]
        
        all_keys = [item[0].key for item in items_and_ranks]
        
        for item, rank in items_and_ranks:
            tree, _ = tree.insert(item, rank)
        
        # Verify initial tree structure
        self.validate_tree(tree, all_keys)
        
        # Split at a key that will cause node collapsing
        left, middle, right = tree.split_inplace(350)

        # Validate split trees
        self.validate_tree(left, [100, 200, 300])
        self.assertIsNone(middle)
        self.validate_tree(right, [400, 500, 600])
    
    def test_split_with_complex_tree(self):
        """Test splitting a complex tree with many nodes and multiple ranks."""
        tree = self.tree_k4
        
        # Insert many items with varying ranks
        keys = list(range(100, 1100, 100))  # Keys from 100 to 1000
        ranks = [1, 2, 1, 3, 1, 2, 1, 4, 1, 2]  # Mix of ranks
        
        for key, rank in zip(keys, ranks):
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank)
        
        # Verify initial tree structure
        self.validate_tree(tree, keys)

        # Split in the middle
        split_key = 550
        left, middle, right = tree.split_inplace(split_key)

        # Validate split
        expected_left = [k for k in keys if k < split_key]
        expected_right = [k for k in keys if k >= split_key]
        
        self.validate_tree(left, expected_left)
        self.assertIsNone(middle)
        self.validate_tree(right, expected_right)
    
    def test_split_leaf_with_left_subtrees_higher_dim(self):
        """Test splitting a tree where items have left subtrees."""
        tree = create_gkplus_tree(K=4)
        
        # Insert primary items
        keys = [100, 200, 300, 400, 500]
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
        
        # Create and attach left subtrees to some items
        for key in [200, 400]:
            # Create a subtree
            subtree = create_gkplus_tree(K=4, dimension=type(tree).DIM + 1)
            subtree, _ = subtree.insert(Item(key - 50, f"subtree_val_{key}"), rank=1)
            
            # Retrieve the item and set its left subtree
            result = tree.retrieve(key)
            result.found_entry.left_subtree = subtree
        
        # Split between items, not at a key with a left subtree
        left, middle, right = tree.split_inplace(250)
 
        
        # Validate split trees
        self.validate_tree(left, [100, 200])
        self.assertIsNone(middle)
        self.validate_tree(right, [300, 400, 500])
        
        # Check that left subtrees were preserved
        result = left.retrieve(200)
        self.assertIsNotNone(result.found_entry.left_subtree)
        self.assertEqual([150], self.collect_keys(result.found_entry.left_subtree))
        
        result = right.retrieve(400)
        self.assertIsNotNone(result.found_entry.left_subtree)
        self.assertEqual([350], self.collect_keys(result.found_entry.left_subtree))
        
        # Now split at a key with a left subtree
        tree = create_gkplus_tree(K=4)
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
            
        # Create and attach a left subtree to item 300
        subtree = create_gkplus_tree(K=4, dimension=type(tree).DIM + 1)
        subtree, _ = subtree.insert(Item(250, "subtree_val"), rank=1)
        subtree, _ = subtree.insert(Item(275, "subtree_val"), rank=1)
        result = tree.retrieve(300)
        result.found_entry.left_subtree = subtree
        
        # Split exactly at 300
        left, middle, right = tree.split_inplace(300)
        
        # Validate split trees
        self.validate_tree(left, [100, 200])
        self.assertIsNotNone(middle)
        self.validate_tree(middle, [250, 275])
        self.validate_tree(right, [400, 500])
    
    def test_split_at_edge_cases(self):
        """Test splitting at edge case keys (min, max, and beyond)."""
        # Insert keys
        keys = [100, 200, 300, 400, 500]
        
        with self.subTest("Split at key smaller than smallest"):
            tree = create_gkplus_tree(K=4)
            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            # Split at a key smaller than all keys in the tree
            left, middle, right = tree.split_inplace(50)
            
            # Validate split
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            self.validate_tree(right, keys)
        
        with self.subTest("Split at key larger than largest"):
            # Split at a key larger than all keys in the tree
            tree = create_gkplus_tree(K=4)
            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            left, middle, right = tree.split_inplace(600)

            # Validate split
            self.validate_tree(left, keys)
            self.assertIsNone(middle)
            self.assertTrue(right.is_empty())
        
        with self.subTest("Split at minimum key"):
            # Split at the minimum key
            tree = create_gkplus_tree(K=4)

            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            left, middle, right = tree.split_inplace(100)
            
            # Validate split
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            self.validate_tree(right, keys[1:])
        
        with self.subTest("Split at maximum key"):
            # Split at the maximum key
            tree = create_gkplus_tree(K=4)
            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            left, middle, right = tree.split_inplace(500)
            
            # Validate split
            self.validate_tree(left, [100, 200, 300, 400])
            self.assertIsNone(middle)
            self.validate_tree(right)
    
    def test_split_with_random_items(self):
        """Test splitting with randomly generated items and keys."""
        tree = self.tree_k8
        
        # Generate random keys
        num_items = 50
        keys = random.sample(range(1, 1000), num_items)
        
        # Insert items into tree
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
        
        # Verify initial tree structure
        self.validate_tree(tree, sorted(keys))
        
        # Choose a random split point
        split_key = random.choice(range(1, 1000))
        
        # Split the tree
        left, middle, right = tree.split_inplace(split_key)

        # Validate split trees
        expected_left = sorted([k for k in keys if k < split_key])
        expected_right = sorted([k for k in keys if k > split_key])
        
        self.validate_tree(left, expected_left)
        self.assertIsNone(middle)
        self.validate_tree(right, expected_right)
    
    def test_multiple_splits(self):
        """Test performing multiple splits on the same tree."""
        tree = self.tree_k4
        
        # Insert items
        keys = list(range(100, 1100, 100))  # Keys from 100 to 1000
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
        
        # First split
        left1, middle1, right1 = tree.split_inplace(550)
        
        # Verify first split
        expected_left1 = [k for k in keys if k < 550]
        expected_right1 = [k for k in keys if k >= 550]
        
        self.validate_tree(left1, expected_left1)
        self.assertIsNone(middle1)
        self.validate_tree(right1, expected_right1)
        
        # Second split on the left part
        left2, middle2, right2 = left1.split_inplace(350)
        
        # Verify second split
        expected_left2 = [k for k in expected_left1 if k < 350]
        expected_right2 = [k for k in expected_left1 if k >= 350]
        
        self.validate_tree(left2, expected_left2)
        self.assertIsNone(middle2)
        self.validate_tree(right2, expected_right2)
        
        # Third split on the right part
        left3, middle3, right3 = right1.split_inplace(750)
        
        # Verify third split
        expected_left3 = [k for k in expected_right1 if k < 750]
        expected_right3 = [k for k in expected_right1 if k >= 750]
        
        self.validate_tree(left3, expected_left3)
        self.assertIsNone(middle3)
        self.validate_tree(right3, expected_right3)
    
    # def test_large_random_tree_split(self):
    #     """Test splitting a large tree with random data."""
        
    #     # Create trees with different K values for testing
    #     self.K = 4  # Default capacity
    #     tree = self.tree_k2

    #     for rank_combo in tqdm(
    #         product(self.RANKS, repeat=self.NUM_KEYS),
    #         total=self.TOT_COMBINATIONS,
    #         desc="Rank combinations",
    #         unit="combo",
    #     ):
        
    #     # for rank_combo in product(RANKS, repeat=10):
    #         base_tree = create_gkplus_tree(K=2)
            
    #         for key, rank in zip(self.KEYS, rank_combo):
    #             base_tree, _ = base_tree.insert(self.ITEMS[key], rank)

    #         msg_head = f"\n\nKey-Rank combo:\nK: {self.KEYS}\nR: {rank_combo}"
    #         msg_head += f"\n\nTree before split:\n{base_tree.print_structure()}"

    #         # Verify initial tree structure
    #         self.validate_tree(base_tree, self.KEYS, msg_head)

    #         with self.subTest("Split at smallest key"):
    #             split_key = 1
    #             tree_copy = copy.deepcopy(base_tree)
                
    #             msg = msg_head + f"Split at smallest key: {split_key}"

    #             left, middle, right = tree_copy.split_inplace(split_key)
                
    #             msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
    #                 left=left.print_structure(),
    #                 middle=middle,
    #                 right=right.print_structure(),
    #             )

    #             # msg += f"\n\nSplit result:"
    #             # msg += f"\n\nLeft tree:\n{left.print_structure()}"
    #             # msg += f"\n\nMiddle tree:\n{middle}"
    #             # msg += f"\n\nRight tree:\n{right.print_structure()}"

    #             self.validate_tree(left, self.split_1_exp_left, msg)
    #             self.assertIsNone(middle)
    #             self.validate_tree(right, self.split_1_exp_right, msg)

    #         with self.subTest("Split at largest key"):
    #             split_key = 8
    #             tree_copy = copy.deepcopy(base_tree)

    #             msg = msg_head + f"Split at largest key: {split_key}"

    #             left, middle, right = tree_copy.split_inplace(split_key)
                
    #             msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
    #                 left=left.print_structure(),
    #                 middle=middle,
    #                 right=right.print_structure(),
    #             )
    #             # msg += f"\n\nSplit result:"
    #             # msg += f"\n\nLeft tree:\n{left.print_structure()}"
    #             # msg += f"\n\nMiddle tree:\n{middle}"
    #             # msg += f"\n\nRight tree:\n{right.print_structure()}"

    #             self.validate_tree(left, self.split_8_exp_left, msg)
    #             self.assertIsNone(middle)
    #             self.validate_tree(right, self.split_8_exp_right, msg)
            
    #         with self.subTest("Split at existing key in between"):
    #             split_key = 4
    #             tree_copy = copy.deepcopy(base_tree)
                
    #             msg = msg_head + f"Split at existing key in between: {split_key}"

    #             left, middle, right = tree_copy.split_inplace(split_key)
                
    #             msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
    #                 left=left.print_structure(),
    #                 middle=middle,
    #                 right=right.print_structure(),
    #             )
    #             # msg += f"\n\nSplit result:"
    #             # msg += f"\n\nLeft tree:\n{left.print_structure()}"
    #             # msg += f"\n\nMiddle tree:\n{middle}"
    #             # msg += f"\n\nRight tree:\n{right.print_structure()}"

    #             self.validate_tree(left, self.split_4_exp_left, msg)
    #             self.assertIsNone(middle)
    #             self.validate_tree(right, self.split_4_exp_right, msg)
            
    #         with self.subTest("Split at key smaller than smallest"):
    #             split_key = 0
    #             tree_copy = copy.deepcopy(base_tree)
                
    #             msg = msg_head + f"Split at key smaller than smallest: {split_key}"

    #             left, middle, right = tree_copy.split_inplace(split_key)
                
    #             msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
    #                 left=left.print_structure(),
    #                 middle=middle,
    #                 right=right.print_structure(),
    #             )
    #             # msg += f"\nSplit result:"
    #             # msg += f"\nLeft tree: {left.print_structure()}"
    #             # msg += f"\nMiddle tree: {middle}"
    #             # msg += f"\nRight tree: {right.print_structure()}"

    #             self.assertTrue(left.is_empty(), msg)
    #             self.assertIsNone(middle)
    #             self.validate_tree(right, self.split_0_exp_right, msg)
            
    #         with self.subTest("Split at key larger than largest"):
    #             split_key = 9
    #             tree_copy = copy.deepcopy(base_tree)
                
    #             msg = msg_head + f"Split at key larger than largest: {split_key}"

    #             left, middle, right = tree_copy.split_inplace(split_key)
                
    #             msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
    #                 left=left.print_structure(),
    #                 middle=middle,
    #                 right=right.print_structure(),
    #             )
    #             # msg += f"\nSplit result:"
    #             # msg += f"\nLeft tree: {left.print_structure()}"
    #             # msg += f"\nMiddle tree: {middle}"
    #             # msg += f"\nRight tree: {right.print_structure()}"

    #             self.validate_tree(left, self.split_9_exp_left, msg)
    #             self.assertIsNone(middle)
    #             self.assertTrue(right.is_empty(), msg)
            
    #         with self.subTest("Split at non-existing key in between"):
    #             split_key = 5
    #             tree_copy = copy.deepcopy(base_tree)
                
    #             msg = msg_head + f"Split at non-existing key in between: {split_key}"

    #             left, middle, right = tree_copy.split_inplace(split_key)
                
    #             msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
    #                 left=left.print_structure(),
    #                 middle=middle,
    #                 right=right.print_structure(),
    #             )
    #             # msg += f"\nSplit result:"
    #             # msg += f"\nLeft tree: {left.print_structure()}"
    #             # msg += f"\nMiddle tree: {middle}"
    #             # msg += f"\nRight tree: {right.print_structure()}"

    #             self.validate_tree(left, self.split_5_exp_left, msg)
    #             self.assertIsNone(middle)
    #             self.validate_tree(right, self.split_5_exp_right, msg)
            
           
    def test_specific_rank_combo(self):
        keys  =  [1, 2, 3, 5, 6, 7]
        ranks =  (1, 1, 1, 1, 2, 4)

        split_cases = self._get_split_cases(keys)
        # array of tuples with (case_name, split_key)
        split_cases = [
                ("smallest key",               min(keys)),
                # ("largest key",                max(keys)),
                # ("existing middle key",        median_low(keys)),
                # ("below smallest",             min(keys) - 1),
                # ("above largest",              max(keys) + 1),
                # ("non-existing middle key",    self._find_first_missing(keys)),
            ]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name
                )


    ## Test function template for setups that fail when running
    # def test_single_abc(self):
    #     keys  =  [1, 2, 4, 5]
    #     ranks =  [1, 3, 1, 1]

    #     split_cases = self._get_split_cases(keys)
    #     # array of tuples with (case_name, split_key)
    #     split_cases = [("non-existing middle key", 3),
    #         ]

    #     for case_name, split_key in split_cases:
    #         exp_left = [k for k in keys if k < split_key]
    #         exp_right = [k for k in keys if k > split_key]
    #         with self.subTest(case=case_name, split_key=split_key):
    #             self._run_split_case(
    #                 keys, ranks,
    #                 split_key, exp_left,
    #                 exp_right, case_name
    #             )

    def test_split_2_level_tree_at_single_root_key(self):
        keys  =  [1, 3]
        ranks =  [1, 3]

        split_cases = self._get_split_cases(keys)
        # array of tuples with (case_name, split_key)
        split_cases = [("max key", max(keys))]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name
                )

    # def test_all_rank_combinations(self):
    #     """
    #     Exhaustively test every rank-combo and every split-key,
    #     computing the expected left/right key-lists on the fly.
    #     """
    #     keys = [1, 2, 4, 5]
    #     ranks = range(1, 5)
    #     split_keys = [0, 1, 3, 5, 6]
    #     # split_key

    #     num_keys = len(keys)
    #     combinations = len(ranks) ** num_keys

    #     iterations = 10

    #     # combos = islice(product(ranks, repeat=num_keys), iterations)
        
    #     for rank_combo in tqdm(
    #         product(ranks, repeat=num_keys),
    #         total=combinations,
    #         desc="Rank combinations",
    #         unit="combo",
    #     ):
    #         with self.subTest(rank_combo=rank_combo):
    #             # for each possible split_key (including non-existent)
    #             for split_key in split_keys:
    #                 with self.subTest(split_key=split_key):
    #                     # expected keys-to-left and keys-to-right
    #                     exp_left  = [k for k in keys if k < split_key]
    #                     exp_right = [k for k in keys if k > split_key]
    #                     case_name = f"split key: {split_key}"
    #                     self._run_split_case(
    #                         keys,
    #                         rank_combo,
    #                         split_key,
    #                         exp_left,
    #                         exp_right,
    #                         case_name
    #                     )



        # for rank_combo in tqdm(
    #         product(self.RANKS, repeat=self.NUM_KEYS),
    #         total=self.TOT_COMBINATIONS,
    #         desc="Rank combinations",
    #         unit="combo",
    #     ):

    







    # def test_random_splits_with_edge_cases(self):
    #     """Test multiple random splits with edge cases."""
    #     for test_iteration in range(5):  # Run several iterations
    #         tree = self.tree_k4
            
    #         # Generate random data
    #         num_items = random.randint(20, 100)
    #         keys = random.sample(range(1, 5000), num_items)
    #         ranks = [random.randint(1, 3) for _ in range(num_items)]
            
    #         # Insert items
    #         for key, rank in zip(keys, ranks):
    #             tree, _ = tree.insert(Item(key, f"val_{key}"), rank=rank)
                
    #         # Verify initial tree
    #         self.validate_tree(tree, sorted(keys))
            
    #         # Choose several split points including edge cases
    #         split_points = [
    #             min(keys) - random.randint(1, 50),  # Below minimum
    #             min(keys),                          # Exactly minimum
    #             sorted(keys)[len(keys) // 4],       # Lower quartile
    #             sorted(keys)[len(keys) // 2],       # Median
    #             sorted(keys)[3 * len(keys) // 4],   # Upper quartile
    #             max(keys),                          # Exactly maximum
    #             max(keys) + random.randint(1, 50)   # Above maximum
    #         ]
            
    #         for split_point in split_points:
    #             # Split the tree
    #             left, middle, right = tree.split_inplace(split_point)
                
    #             # Validate split
    #             expected_left = sorted([k for k in keys if k < split_point])
    #             expected_right = sorted([k for k in keys if k >= split_point])
                
    #             self.validate_tree(left, expected_left)
    #             self.validate_tree(right, expected_right)
    
    # def test_sequential_random_splits(self):
    #     """Test series of sequential splits on random trees."""
    #     tree = self.tree_k4
        
    #     # Create a large tree
    #     num_items = 150
    #     keys = random.sample(range(1, 10000), num_items)
        
    #     for key in keys:
    #         tree, _ = tree.insert(Item(key, f"val_{key}"), rank=random.randint(1, 3))
        
    #     sorted_keys = sorted(keys)
    #     current_tree = tree
    #     remaining_keys = sorted_keys
        
    #     # Perform sequential splits, each time splitting the right subtree
    #     for i in range(5):  # Do 5 sequential splits
    #         if len(remaining_keys) < 10:  # Stop if not enough keys left
    #             break
                
    #         # Choose a split point at approximately 25% of remaining keys
    #         split_idx = len(remaining_keys) // 4
    #         split_key = remaining_keys[split_idx]
            
    #         # Perform the split
    #         left, middle, right = current_tree.split_inplace(split_key)
            
    #         # Validate the split
    #         expected_left = [k for k in remaining_keys if k < split_key]
    #         expected_right = [k for k in remaining_keys if k >= split_key]
            
    #         self.validate_tree(left, expected_left)
    #         self.validate_tree(right, expected_right)
            
    #         # Continue with right tree for next iteration
    #         current_tree = right
    #         remaining_keys = expected_right

if __name__ == "__main__":
    unittest.main()