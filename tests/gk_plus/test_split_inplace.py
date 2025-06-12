import sys
import os
import unittest
import random
from typing import List, Tuple, Optional, Iterator, TYPE_CHECKING
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
from gplus_trees.g_k_plus.utils import calc_rank
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from tests.gk_plus.base import TreeTestCase
from tests.utils import assert_tree_invariants_tc

from tests.logconfig import logger

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

class TestGKPlusSplitInplace(TreeTestCase):
    """Tests for the split_inplace method of the GKPlusTreeBase class."""
    
    ASSERTION_MESSAGE_TEMPLATE = (
        "\n\nSplit result:\n"
        "\nLEFT TREE: {left}\n\n"
        "\nMIDDLE TREE: {middle}\n\n"
        "\nRIGHT TREE: {right}\n"
    )
    
    # Initialize items once to avoid re-creating them in each test
    _KEYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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

    def create_item(self, key, value="val"):
        """Helper to create test items"""
        return Item(key, value)
    
    
    
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
            
        # dummy_key = get_dummy(type(tree).DIM).key
        keys = []

        # Collect keys from leaf nodes
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                # if entry.item.key != dummy_key:
                    keys.append(entry.item.key)
        
        return sorted(keys)
    
    def _run_split_case(self, keys, rank_combo, split_key,
                        exp_left, exp_right, case_name, gnode_capacity=2):
        if len(rank_combo) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # build the tree once
        base_tree = create_gkplus_tree(K=gnode_capacity)
        for key, rank in zip(keys, rank_combo):
            base_tree, _ = base_tree.insert(self.ITEMS[key], rank)

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

    def _run_split_case_multi_dim(self, keys, rank_combo, split_key,
                        exp_left, exp_right, case_name, gnode_capacity=2, l_factor: float = 1.0):
        if len(rank_combo) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # build the tree once
        base_tree = create_gkplus_tree(K=gnode_capacity, dimension=1, l_factor=l_factor)

        for key, rank in zip(keys, rank_combo):
            base_tree, _ = base_tree.insert(self.ITEMS[key], rank)

        logger.debug(f"Base tree before split: {print_pretty(base_tree)}")
        logger.debug(f"Root node: {print_pretty(base_tree.node.set)}")

        msg_head = (
            f"\n\nKey-Rank combo:\n"
            f"K: {keys}\n"
            f"R: {rank_combo}"
            f"\n\nTREE BEFORE SPLIT: {print_pretty(base_tree)}\n"
        )

        # deep-copy and split
        tree_copy = copy.deepcopy(base_tree)
        left, middle, right = tree_copy.split_inplace(split_key)

        logger.debug(f"Left tree after split: {print_pretty(left)}")
        logger.debug(f"Middle tree after split: {print_pretty(middle)}")
        logger.debug(f"Right tree after split: {print_pretty(right)}")

        msg = f"\n\nSplit at {case_name}" + msg_head
        msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
            left=print_pretty(left),
            middle=print_pretty(middle),
            right=print_pretty(right),
        )

        dummies_left = self.get_dummies(left)
        exp_left = sorted(dummies_left + exp_left)
        dummies_right = self.get_dummies(right)
        exp_right = sorted(dummies_right + exp_right)

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
            self.validate_tree(left, [-1, 500])
            self.assertIsNone(middle)
            self.assertTrue(right.is_empty())

        with self.subTest("split point < only key"):
            # Split at a key less than the only key
            tree = create_gkplus_tree(K=4)
            tree, _ = tree.insert(item, rank=1)
            left, middle, right = tree.split_inplace(100)

            # Validate right tree with the item
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            self.validate_tree(right, [-1, 500])

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
        tree = self.tree_k8
        
        # Insert multiple items in increasing order
        keys = [100, 200, 300, 400, 500]
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
        
        # Check that initial tree is valid
        self.validate_tree(tree, [-1] + keys)

        # Split in the middle (between items)
        left, middle, right = tree.split_inplace(250)

        # Validate the split trees
        self.validate_tree(left, [-1, 100, 200])
        self.assertIsNone(middle)
        self.validate_tree(right, [-1, 300, 400, 500])

    def test_split_tree_with_internal_nodes(self):
        """Test splitting a tree with internal nodes."""
        tree = self.tree_k4  # Use K=2 to force more internal nodes
        
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
        self.validate_tree(tree, [-1] + all_keys)

        # Split at a key that requires traversing internal nodes
        left, middle, right = tree.split_inplace(450)
        
        # Validate split trees
        self.validate_tree(left, [-1, 100, 200, 300, 400])
        self.assertIsNone(middle)
        self.validate_tree(right, [-1, 500, 600, 700])
    
    def test_split_with_complex_tree(self):
        """Test splitting a complex tree with many nodes and multiple ranks."""
        tree = self.tree_k8
        
        # Insert many items with varying ranks
        keys = list(range(100, 1100, 100))  # Keys from 100 to 1000
        ranks = [1, 2, 1, 3, 1, 2, 1, 4, 1, 2]  # Mix of ranks
        
        for key, rank in zip(keys, ranks):
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank)
        
        # Verify initial tree structure
        self.validate_tree(tree, [-1] + keys)

        # Split in the middle
        split_key = 550
        left, middle, right = tree.split_inplace(split_key)

        # Validate split
        expected_left = [-1] + [k for k in keys if k < split_key]
        expected_right = [-1] + [k for k in keys if k > split_key]
        logger.debug(f"Expected left: {expected_left}")
        logger.debug(f"Expected right: {expected_right}")

        self.validate_tree(left, expected_left)
        self.assertIsNone(middle)
        self.validate_tree(right, expected_right)
    
    def test_split_leaf_with_left_subtrees_higher_dim(self):
        """Test splitting a tree where items have left subtrees."""
        cap = 8
        tree = create_gkplus_tree(K=cap)
        
        # Insert primary items
        keys = [100, 200, 300, 400, 500]
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

        # Create and attach left subtrees to some items
        for key in [200, 400]:
            # Create a subtree
            dim = type(tree).DIM + 1  if key == 200 else type(tree).DIM + 2
            rank = 2 if key == 200 else 3
            subtree = create_gkplus_tree(K=4, dimension=dim)
            subtree, _ = subtree.insert(Item(key - 50, f"subtree_val_{key}"), rank=rank)
            
            # Retrieve the item and set its left subtree
            result = tree.retrieve(key)
            result.found_entry.left_subtree = subtree
        
        # Split between items, not at a key with a left subtree
        left, middle, right = tree.split_inplace(250)
 
        # Validate split trees
        self.validate_tree(left, [-1, 100, 200])
        self.assertIsNone(middle)
        self.validate_tree(right, [-1, 300, 400, 500])
        
        # Check that left subtrees were preserved
        result = left.retrieve(200)
        self.assertIsNotNone(result.found_entry.left_subtree)
        self.assertEqual([-2, 150], self.collect_keys(result.found_entry.left_subtree))
        
        result = right.retrieve(400)
        self.assertIsNotNone(result.found_entry.left_subtree)
        self.assertEqual([-3, 350], self.collect_keys(result.found_entry.left_subtree))

        
        # Now split at a key with a left subtree
        tree = create_gkplus_tree(K=cap)
        for key in keys:
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
            
        # Create and attach a left subtree to item 300
        subtree = create_gkplus_tree(K=cap, dimension=type(tree).DIM + 1)
        subtree, _ = subtree.insert(Item(250, "subtree_val"), rank=1)
        subtree, _ = subtree.insert(Item(275, "subtree_val"), rank=1)
        result = tree.retrieve(300)
        result.found_entry.left_subtree = subtree

        # logger.debug(f"Tree before split: {print_pretty(tree)}")
        # logger.debug(f"Tree structure: {tree.print_structure()}")
        
        # Split exactly at 300
        left, middle, right = tree.split_inplace(300)
        
        # Validate split trees
        self.validate_tree(left, [-1, 100, 200])
        self.assertIsNotNone(middle)
        self.validate_tree(middle, [-2, 250, 275])
        self.validate_tree(right, [-1, 400, 500])
    
    def test_split_at_edge_cases(self):
        """Test splitting at edge case keys (min, max, and beyond)."""
        # Insert keys
        keys = [100, 200, 300, 400, 500]
        cap = 8

        with self.subTest("Split at key smaller than smallest"):
            tree = create_gkplus_tree(K=cap)
            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            # Split at a key smaller than all keys in the tree
            left, middle, right = tree.split_inplace(50)
            
            # Validate split
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            self.validate_tree(right, [-1] + keys)
        
        with self.subTest("Split at key larger than largest"):
            # Split at a key larger than all keys in the tree
            tree = create_gkplus_tree(K=cap)
            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            left, middle, right = tree.split_inplace(600)

            # Validate split
            self.validate_tree(left, [-1] + keys)
            self.assertIsNone(middle)
            self.assertTrue(right.is_empty())
        
        with self.subTest("Split at minimum key"):
            # Split at the minimum key
            tree = create_gkplus_tree(K=cap)

            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            left, middle, right = tree.split_inplace(100)
            
            # Validate split
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            self.validate_tree(right, [-1] + keys[1:])
        
        with self.subTest("Split at maximum key"):
            # Split at the maximum key
            tree = create_gkplus_tree(K=cap)
            for key in keys:
                tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)

            left, middle, right = tree.split_inplace(500)
            
            # Validate split
            self.validate_tree(left, [-1, 100, 200, 300, 400])
            self.assertIsNone(middle)
            self.validate_tree(right)
    
    def test_split_with_random_items(self):
        """Test splitting with randomly generated items and keys."""
        tree = self.tree_k16
        k = 8

        # Generate random keys
        num_items = 50
        keys = random.sample(range(1, 1000), num_items)
        # use calc_rank to generate ranks
        ranks = [calc_rank(key=key, k=k, dim=1) for key in keys]
        
        # Insert items into tree
        for key, rank in zip(keys, ranks):
            tree, _ = tree.insert(Item(key, f"val_{key}"), rank=rank)

        # Verify initial tree structure
        dummies = self.get_dummies(tree)
        expected_keys = sorted(dummies + keys)
        self.validate_tree(tree, expected_keys)
        
        # Choose a random split point
        split_key = random.choice(range(1, 1000))
        
        # Split the tree
        left, middle, right = tree.split_inplace(split_key)

        # Validate split trees
        expected_left = [k for k in keys if k < split_key]
        dummies_left = self.get_dummies(left)
        # expected_left = [-1] + expected_left if expected_left else expected_left
        expected_left = sorted(dummies_left + expected_left)

        expected_right = [k for k in keys if k > split_key]
        dummies_right = self.get_dummies(right)
        expected_right = sorted(dummies_right + expected_right)

        self.validate_tree(left, expected_left)
        self.assertIsNone(middle)
        self.validate_tree(right, expected_right)
    
    def test_multiple_splits(self):
        """Test performing multiple splits on the same tree."""
        tree = self.tree_k4
        capacity = 4

        rank_lists = [
            [2, 3, 2, 1, 3],  # Dimension 1
            [1, 2, 3, 4, 2],  # Dimension 2
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, capacity)
        logger.debug(f"Keys: {keys}")
        item_map = { k: self.create_item(k) for k in keys}

        for i in range(len(keys)):
            key = keys[i]
            rank = rank_lists[0][i]
            item = item_map[key]
            tree, _ = tree.insert(item, rank=rank)

        logger.debug(f"Tree after initial insertions: {print_pretty(tree)}")
        
        # Insert items
        # keys = list(range(100, 1100, 100))  # Keys from 100 to 1000
        # for key in keys:
        #     tree, _ = tree.insert(Item(key, f"val_{key}"), rank=1)
        
        # First split
        left1, middle1, right1 = tree.split_inplace(80)
        
        # Verify first split
        expected_left1 = [-1] + [k for k in keys if k < 80]
        expected_right1 = [-1] + [k for k in keys if k > 80]
        
        self.validate_tree(left1, expected_left1)
        self.assertIsNone(middle1)
        self.validate_tree(right1, expected_right1)
        
        # Second split on the left part
        left2, middle2, right2 = left1.split_inplace(7)
        
        # Verify second split
        expected_left2 = [k for k in expected_left1 if k < 7]
        expected_right2 = [-1] + [k for k in expected_left1 if k > 7]
        
        self.validate_tree(left2, expected_left2)
        self.assertIsNone(middle2)
        self.validate_tree(right2, expected_right2)
        
        # Third split on the right part
        left3, middle3, right3 = right1.split_inplace(212)
        
        # Verify third split
        expected_left3 = [k for k in expected_right1 if k < 212]
        expected_right3 = [-1] + [k for k in expected_right1 if k > 212]
        
        self.validate_tree(left3, expected_left3)
        self.assertIsNone(middle3)
        self.validate_tree(right3, expected_right3)
            
    def test_specific_rank_combo(self):
        keys  =  [1, 2, 3, 5, 6, 7]
        ranks =  (1, 1, 1, 1, 2, 4)

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
            exp_left = []
            exp_right =[-1] + [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=8
                )


    def test_split_root_at_max_splitting_leaf(self):
        keys  =  [1, 3, 5, 7]
        ranks =  [1, 3, 1, 1]

        # array of tuples with (case_name, split_key)
        split_cases = [("non-existing middle key", 3)]

        for case_name, split_key in split_cases:
            exp_left = [-1] + [k for k in keys if k < split_key]
            exp_right = [-1] + [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=4
                )

    def test_split_2_level_tree_at_single_root_key(self):
        keys  =  [1, 3]
        ranks =  [1, 3]

        # array of tuples with (case_name, split_key)
        split_cases = [("max key", max(keys))]

        for case_name, split_key in split_cases:
            exp_left = [-1] + [k for k in keys if k < split_key]
            exp_right = []
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=2
                )

    def test_split_root_highest_and_two_consecutive_internal_after_first(self):
        keys  =  [1, 3, 5, 7]
        ranks =  [4, 1, 2, 3]

        # array of tuples with (case_name, split_key)
        split_cases = [("non-existing middle key", 3)]


        for case_name, split_key in split_cases:
            exp_left = [-1] + [k for k in keys if k < split_key]
            exp_right = [-1] + [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name
                )

    def test_split_expanded_root(self):
        keys  =  [1, 3, 5, 7]
        ranks =  [2, 2, 2, 2]

        # array of tuples with (case_name, split_key)
        split_cases = [("existing middle key", 3)]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=4
                )

    def test_split_abc(self):
        keys  =  [1, 3, 5, 7, 9]
        ranks =  [2, 1, 1, 1, 1]

        # array of tuples with (case_name, split_key)
        split_cases = [("existing middle key", 3)]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=4, l_factor=1.0
                )

    def test_split_abcd(self):
        keys  =  [1, 3, 5, 7, 9]
        ranks =  [2, 1, 1, 1, 1]

        # array of tuples with (case_name, split_key)
        split_cases = [("existing middle key", 3)]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=2, l_factor=2.0
                )

    def test_split_abcde(self):
        keys  =  [1, 3, 5, 7, 9]
        ranks =  [4, 4, 4, 4, 3]

        # array of tuples with (case_name, split_key)
        split_cases = [("split before first", 0)]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=2, l_factor=3.0
                )

    def test_split_abcdef(self):
        keys  =  [1, 3, 5, 7, 9]
        ranks =  [1, 2, 2, 2, 2]

        # array of tuples with (case_name, split_key)
        split_cases = [("split before last", 8)]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=4, l_factor=1.0
                )

    def test_split_abcdefg(self):
        keys  =  [1, 3, 5, 7, 9, 11]
        ranks =  [1, 2, 2, 2, 2, 2]

        # array of tuples with (case_name, split_key)
        split_cases = [("split before first", 0)]

        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=4, l_factor=1.0
                )


    def test_all_rank_combinations(self):
        """
        Exhaustively test every rank-combo and every split-key,
        computing the expected left/right key-lists on the fly.
        """
        keys = [1, 3, 5, 7, 9, 11]
        ranks = range(1, 4)
        split_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # split_keys = [0]
        # split_key

        num_keys = len(keys)
        combinations = len(ranks) ** num_keys

        # iterations = 10
        # combos = islice(product(ranks, repeat=num_keys), iterations)
        
        for rank_combo in tqdm(
            product(ranks, repeat=num_keys),
            # combos,
            total=combinations,
            desc="Rank combinations",
            unit="combo",
        ):
            with self.subTest(rank_combo=rank_combo):
                # for each possible split_key (including non-existent)
                for split_key in split_keys:
                    with self.subTest(split_key=split_key):
                        # expected keys-to-left and keys-to-right
                        # do not include dummy items as they are calculated
                        # on the fly in the _run_split_case_multi_dim method
                        exp_left = [k for k in keys if k < split_key]
                        exp_right = [k for k in keys if k > split_key]

                        case_name = f"split key: {split_key}"
                        # self._run_split_case(
                        self._run_split_case_multi_dim(
                            keys,
                            rank_combo,
                            split_key,
                            exp_left,
                            exp_right,
                            case_name,
                            gnode_capacity=4,
                            l_factor=1.0
                        )

    def test_split_with_node_collapsing(self):
        """Test splitting that causes nodes to collapse."""
        tree = self.tree_k4  # Use K=4 to force node splitting quickly

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
        self.validate_tree(tree, [-1] + all_keys)
        
        # Split at a key that will cause node collapsing
        left, middle, right = tree.split_inplace(350)

        # Validate split trees
        self.validate_tree(left, [-1, 100, 200, 300])
        self.assertIsNone(middle)
        self.validate_tree(right, [-1, 400, 500, 600])

if __name__ == "__main__":
    unittest.main()