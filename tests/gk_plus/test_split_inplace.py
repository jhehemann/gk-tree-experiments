import sys
import os
import unittest
import random
from typing import List
from itertools import product
from tqdm import tqdm
import copy
from statistics import median_low



# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.g_k_plus.utils import calc_rank_for_dim
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase
from tests.logconfig import logger
import logging


class TestGKPlusSplitInplace(GKPlusTreeTestCase):
    """Tests for the split_inplace method of the GKPlusTreeBase class."""
    
    ASSERTION_MESSAGE_TEMPLATE = (
        "\n\nSplit result:\n"
        "\nLEFT TREE: {left}\n\n"
        "\nMIDDLE TREE: {middle}\n\n"
        "\nRIGHT TREE: {right}\n"
    )
    
    # Initialize items once to avoid re-creating them in each test
    _KEYS = list(range(1, 1001))
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
    
    # def collect_keys(self, tree):
    #     """Collect all keys from a tree, including dummy keys."""
    #     if tree.is_empty():
    #         return []
        
    #     keys = []
    #     for leaf_node in tree.iter_leaf_nodes():
    #         for entry in leaf_node.set:
    #             keys.append(entry.item.key)

    #     return sorted(keys)
    
    def _run_split_case(self, keys, rank_combo, split_key,
                        exp_left, exp_right, case_name, gnode_capacity=2):
        if len(rank_combo) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # build the tree once
        tree = create_gkplus_tree(K=gnode_capacity)
        for key, rank in zip(keys, rank_combo):
            tree, _, _ = tree.insert(self.ITEMS[key], rank)

        msg_head = (
            f"\n\nKey-Rank combo:\n"
            f"K: {keys}\n"
            f"R: {rank_combo}"
            f"\n\nTREE BEFORE SPLIT: {print_pretty(tree)}\n"
        )

        # deep-copy and split
        left, middle, right = tree.split_inplace(split_key)

        msg = f"\n\nSplit at {case_name}" + msg_head
        msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
            left=print_pretty(left),
            middle=print_pretty(middle),
            right=print_pretty(right),
        )
        exp_left = [-1] + exp_left if exp_left else []
        exp_right = [-1] + exp_right if exp_right else []

        # assertions
        self.assertIs(tree, left, msg)
        self.validate_tree(left,  exp_left,  msg)
        self.assertIsNone(middle, msg)
        self.validate_tree(right, exp_right, msg)

    def _run_split_case_multi_dim(self, keys, rank_combo, split_key,
                        exp_left, exp_right, case_name, gnode_capacity=2, l_factor: float = 1.0):
        if len(rank_combo) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # build the tree once
        tree = create_gkplus_tree(K=gnode_capacity, dimension=1, l_factor=l_factor)
        for key, rank in zip(keys, rank_combo):
            tree, _, _ = tree.insert(self.ITEMS[key], rank)


        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n\n############################################################################################### SPLITTING ###############################################################################################\n\n")

        msg_head = (
            f"\n\nKey-Rank combo:\n"
            f"K: {keys}\n"
            f"R: {rank_combo}"
            f"\n\nTREE BEFORE SPLIT: {print_pretty(tree)}\n"
        )

        left, middle, right = tree.split_inplace(split_key)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Left after split: {print_pretty(left)}")
            logger.debug(f"Middle after split: {print_pretty(middle) if  middle else 'None'}")
            logger.debug(f"Right after split: {print_pretty(right)}")

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

        self.assertIs(tree, left, msg)
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
        tree = create_gkplus_tree(K=4)
        left, middle, right = tree.split_inplace(500)
        self.validate_tree(left)
        self.assertIs(tree, left, "Left tree should be the original empty tree")
        self.validate_tree(right)
        self.assertTrue(left.is_empty())
        self.assertIsNone(middle)
        self.assertTrue(right.is_empty())
        
    def test_split_single_node_tree(self):
        """Test splitting a tree with a single node."""        
        item = Item(500, "val")
        base_tree = create_gkplus_tree(K=4)
        base_tree, _, _ = base_tree.insert(item, rank=1)

        with self.subTest("split point > only key"):
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(1000)
            self.validate_tree(left, [-1, 500])
            self.assertIs(tree, left)
            self.validate_tree(right)
            self.assertIsNone(middle)
        with self.subTest("split point < only key"):
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(100)
            self.validate_tree(right, [-1, 500])
            self.validate_tree(left)
            self.assertIs(tree, left)
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
        with self.subTest("split point == only key"):
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(500)
            self.validate_tree(left)
            self.assertIs(tree, left)
            self.validate_tree(right)
            self.assertTrue(left.is_empty(), "Left should be empty, got: " + print_pretty(left))
            self.assertIsNone(middle)
            self.assertTrue(right.is_empty(), "Right should be empty, got: " + print_pretty(right))

    def test_split_leaf_node_with_multiple_items(self):
        """Test splitting a leaf node with multiple items."""
        tree = create_gkplus_tree(K=8)
        keys = [100, 200, 300, 400, 500]
        for key in keys:
            tree, _, _ = tree.insert(Item(key, "val"), rank=1)
        exp_keys = [-1] + keys  # Include dummy key
        self.validate_tree(tree, exp_keys)

        # Split in the middle (between keys)
        left, middle, right = tree.split_inplace(250)
        self.validate_tree(left, [-1, 100, 200])
        self.assertIs(left, tree, "Left tree should be the original tree")
        self.validate_tree(right, [-1, 300, 400, 500])
        self.assertIsNone(middle)
        
    def test_split_tree_with_internal_nodes(self):
        """Test splitting a tree with internal nodes."""
        tree = create_gkplus_tree(K=4)
        keys = [100, 200, 300, 400, 500, 600, 700]
        ranks = [1, 1, 2, 1, 3, 2, 1]  # Mix of ranks to create internal nodes
        for key, rank in zip(keys, ranks):
            tree, _, _ = tree.insert(Item(key, "val"), rank)
        exp_keys = [-1] + keys  # Include dummy key
        self.validate_tree(tree, exp_keys)

        # Split at a key that requires traversing internal nodes
        left, middle, right = tree.split_inplace(450)
        self.assertIs(tree, left, "Left tree should be the original tree")
        self.validate_tree(left, [-1, 100, 200, 300, 400])
        self.validate_tree(right, [-1, 500, 600, 700])
        self.assertIsNone(middle)
        
    def test_split_with_complex_tree(self):
        """Test splitting a complex tree with many nodes and multiple ranks."""
        tree = create_gkplus_tree(K=8)
        keys = list(range(100, 1100, 100))
        ranks = [1, 2, 1, 3, 1, 2, 1, 4, 1, 2]
        for key, rank in zip(keys, ranks):
            tree, _, _ = tree.insert(Item(key, "val"), rank)
        exp_keys = [-1] + keys  # Include dummy key
        self.validate_tree(tree, exp_keys)

        # Split in the middle and validate the split
        split_key = 550
        left, middle, right = tree.split_inplace(split_key)
        exp_left = [-1] + [k for k in keys if k < split_key]
        exp_right = [-1] + [k for k in keys if k > split_key]        
        self.validate_tree(left, exp_left)
        self.assertIs(tree, left, "Left tree should be the original tree")
        self.validate_tree(right, exp_right)
        self.assertIsNone(middle)
        
    def test_split_leaf_with_left_subtrees_higher_dim(self):
        """Test splitting a tree where items have left subtrees."""
        k = 8
        base_tree = create_gkplus_tree(K=k)
        keys = [100, 200, 300, 400, 500]
        for key in keys:
            base_tree, _, _ = base_tree.insert(Item(key, "val"), rank=1)

        with self.subTest("Split at non-existent key"):
            # Create and attach left subtrees of different dimensions to some items
            tree = copy.deepcopy(base_tree)
            for key in [200, 400]:
                dim = type(tree).DIM + 1  if key == 200 else type(tree).DIM + 2
                rank = 2 if key == 200 else 3
                subtree = create_gkplus_tree(K=4, dimension=dim)
                subtree, _, _ = subtree.insert(Item(key - 50, "subtree_val"), rank=rank)
                entry = tree.retrieve(key)[0]
                entry.left_subtree = subtree
                self.validate_tree(tree)
            
            # Split tree and check results
            left, middle, right = tree.split_inplace(250)
            self.validate_tree(left, [-1, 100, 200])
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertIsNone(middle)
            self.validate_tree(right, [-1, 300, 400, 500])
            
            # Check that left subtrees were preserved
            subtree_200 = left.retrieve(200)[0].left_subtree
            # self.assertIsNotNone(subtree_200, "Left subtree for key 200 should not be None")
            self.assertFalse(subtree_200.is_empty(), "Left subtree for key 200 should not be empty")
            exp_keys_200 = [-2, 150]
            self.validate_tree(subtree_200, exp_keys_200)
            subtree_400 = right.retrieve(400)[0].left_subtree
            self.assertIsNotNone(subtree_400, "Left subtree for key 400 should not be None")
            self.assertFalse(subtree_400.is_empty(), "Left subtree for key 400 should not be empty")
            exp_keys_400 = [-3, 350]
            self.validate_tree(subtree_400, exp_keys_400)
        with self.subTest("Split at existing key with left subtree"):
            # Create and attach left subtree of different dimension to an item
            tree = copy.deepcopy(base_tree)
            subtree = create_gkplus_tree(K=k, dimension=type(tree).DIM + 1)
            subtree, _, _ = subtree.insert(Item(250, "subtree_val"), rank=1)
            subtree, _, _ = subtree.insert(Item(275, "subtree_val"), rank=1)
            entry = tree.retrieve(300)[0]
            entry.left_subtree = subtree
            self.validate_tree(tree)
            
            # Split at key 300 with left subtree
            left, middle, right = tree.split_inplace(300)
            self.validate_tree(left, [-1, 100, 200])
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.validate_tree(middle, [-2, 250, 275])
            self.validate_tree(right, [-1, 400, 500])
    
    def test_split_at_edge_cases_rank_1(self):
        """Test splitting at edge case keys (min, max, and beyond)."""
        k = 8
        base_tree = create_gkplus_tree(K=k)
        keys = [100, 200, 300, 400, 500]
        for key in keys:
            base_tree, _, _ = base_tree.insert(Item(key, "val"), rank=1)

        with self.subTest("Split at key smaller than smallest"):
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(50)
            self.validate_tree(left)
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            exp_right = [-1] + keys
            self.validate_tree(right, exp_right)
        with self.subTest("Split at key larger than largest"):
            # Split at a key larger than all keys in the tree
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(600)
            exp_left = [-1] + keys
            self.validate_tree(left, exp_left)
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertIsNone(middle)
            self.validate_tree(right)
            self.assertTrue(right.is_empty())
        with self.subTest("Split at minimum key"):
            # Split at the minimum key
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(100)
            self.validate_tree(left)
            self.assertTrue(left.is_empty())
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertIsNone(middle)
            exp_right = [-1] + keys[1:]  # Exclude the minimum key
            self.validate_tree(right, exp_right)
        with self.subTest("Split at maximum key"):
            # Split at the maximum key
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(500)
            exp_left = [-1] + keys[:-1]  # Exclude the maximum key
            self.validate_tree(left, exp_left)
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertIsNone(middle)
            self.validate_tree(right)
            self.assertTrue(right.is_empty())

    def test_split_at_edge_cases_rank_gt_1(self):
        """Test splitting at edge case keys (min, max, and beyond)."""
        k = 8
        base_tree = create_gkplus_tree(K=k)
        keys = [100, 200, 300, 400, 500]
        ranks = [2, 1, 2, 3, 1]
        for key, rank in zip(keys, ranks):
            base_tree, _, _ = base_tree.insert(Item(key, "val"), rank=rank)

        with self.subTest("Split at key smaller than smallest"):
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(50)
            self.validate_tree(left)
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertTrue(left.is_empty())
            self.assertIsNone(middle)
            exp_right = [-1] + keys
            self.validate_tree(right, exp_right)
        with self.subTest("Split at key larger than largest"):
            # Split at a key larger than all keys in the tree
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(600)
            exp_left = [-1] + keys
            self.validate_tree(left, exp_left)
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertIsNone(middle)
            self.validate_tree(right)
            self.assertTrue(right.is_empty())
        with self.subTest("Split at minimum key"):
            # Split at the minimum key
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(100)
            self.validate_tree(left)
            self.assertTrue(left.is_empty())
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertIsNone(middle)
            exp_right = [-1] + keys[1:]  # Exclude the minimum key
            self.validate_tree(right, exp_right)
        with self.subTest("Split at maximum key"):
            # Split at the maximum key
            tree = copy.deepcopy(base_tree)
            left, middle, right = tree.split_inplace(500)
            exp_left = [-1] + keys[:-1]  # Exclude the maximum key
            self.validate_tree(left, exp_left)
            self.assertIs(tree, left, "Left tree should be the original tree")
            self.assertIsNone(middle)
            self.validate_tree(right)
            self.assertTrue(right.is_empty())


    def test_split_with_node_collapsing(self):
        """Test splitting at a keythat causes nodes to collapse."""
        k = 4
        tree = create_gkplus_tree(K=k)
        keys = [100, 200, 300, 400, 500, 600]
        ranks = [1, 2, 2, 3, 1, 2]
        for key, rank in zip(keys, ranks):
            tree, _, _ = tree.insert(Item(key, "val"), rank=rank)
        left, middle, right = tree.split_inplace(350)
        self.validate_tree(left, [-1, 100, 200, 300])
        self.assertIs(tree, left, "Left tree should be the original tree")
        self.assertIsNone(middle)
        self.validate_tree(right, [-1, 400, 500, 600])
    
    def test_split_with_random_items(self):
        """Test splitting with randomly generated items and keys."""
        # Create a tree with random items
        k = 4
        tree = create_gkplus_tree(K=k)
        num_items = 50
        keys = random.sample(range(1, 1000), num_items)
        ranks = [calc_rank_for_dim(key=key, k=k, dim=1) for key in keys]
        for key, rank in zip(keys, ranks):
            tree, _, _ = tree.insert(Item(key, f"val_{key}"), rank=rank)

        # Verify initial tree structure
        dummies = self.get_dummies(tree)
        expected_keys = sorted(dummies + keys) # sort to handle dummy keys in other dimensions
        self.validate_tree(tree, expected_keys)
        
        # Choose a random split point and split the tree
        split_key = random.choice(range(1, 1000))
        left, middle, right = tree.split_inplace(split_key)
        exp_left = [k for k in keys if k < split_key]
        dummies_left = self.get_dummies(left)
        exp_left = sorted(dummies_left + exp_left)
        self.validate_tree(left, exp_left)
        self.assertIs(tree, left, "Left tree should be the original tree")
        exp_right = [k for k in keys if k > split_key]
        dummies_right = self.get_dummies(right)
        exp_right = sorted(dummies_right + exp_right) # sort to handle dummies in other dimensions
        self.validate_tree(right, exp_right)
        self.assertIsNone(middle)

    def test_multiple_splits(self):
        """Test performing multiple splits on the same tree."""
        k = 4
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [2, 3, 2, 1, 3],  # Dimension 1
            [1, 2, 3, 4, 2],  # Dimension 2
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        item_map = { k: self.create_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _, _ = tree.insert(item, rank=rank)

        with self.subTest("First split at 80"):
            left1, middle1, right1 = tree.split_inplace(80)
            exp_left1 = [-1] + [k for k in keys if k < 80]
            self.validate_tree(left1, exp_left1)
            self.assertIs(tree, left1, "Left tree should be the original tree")
            exp_right1 = [-1] + [k for k in keys if k > 80]
            self.validate_tree(right1, exp_right1)
            self.assertIsNone(middle1)
        with self.subTest("Second split at 7 on left part"):
            # Second split on the left part
            left2, middle2, right2 = left1.split_inplace(7)
            exp_left2 = [k for k in exp_left1 if k < 7]
            self.validate_tree(left2, exp_left2)
            self.assertIs(left1, left2, "Left tree should be the original tree")
            exp_right2 = [-1] + [k for k in exp_left1 if k > 7]
            self.validate_tree(right2, exp_right2)
            self.assertIsNone(middle2)
        with self.subTest("Third split at 212 on right part"):
            left3, middle3, right3 = right1.split_inplace(212)
            exp_left3 = [k for k in exp_right1 if k < 212]
            self.validate_tree(left3, exp_left3)
            self.assertIs(right1, left3, "Left tree should be the original tree")
            exp_right3 = [-1] + [k for k in exp_right1 if k > 212]
            self.validate_tree(right3, exp_right3)
            self.validate_tree(left3, exp_left3)
            self.assertIsNone(middle3)
            
    def test_split_root_at_max_splitting_leaf(self):
        keys  =  [1, 3, 5, 7]
        ranks =  [1, 3, 1, 1]
        split_cases = [("split at second key", 3)]
        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
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
        split_cases = [("split at second key", 3)]
        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
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
        split_cases = [("split at second key", 3)]
        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name
                )

    def test_split_expanded_root(self):
        keys  =  [1, 3, 5, 7]
        ranks =  [2, 2, 2, 2]
        split_cases = [("split at second key", 3)]
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

    def test_split_at_first_item_having_rank_2(self):
        keys  =  [1, 3, 5, 7, 9, 11]
        ranks =  [2, 1, 2, 2, 2, 2]
        split_cases = [("split at first key", 1)]
        for case_name, split_key in split_cases:
            exp_left = []
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=8,
                )

    # Tests for complex rank combinations used to cause errors in test_all_rank_combinations()
    def test_split_ab(self):
        keys  =  [1, 2, 3, 5, 6, 7]
        ranks =  (1, 1, 1, 1, 2, 4)
        split_cases = [("split at first key", 1)]
        for case_name, split_key in split_cases:
            exp_left = []
            exp_right =[k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=8
                )

    def test_split_abc(self):
        keys  =  [1, 3, 5, 7, 9]
        ranks =  [2, 1, 1, 1, 1]
        split_cases = [("split at second key", 3)]
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
        split_cases = [("split at second key", 3)]
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
        split_cases = [("split before first key", 0)]
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
        split_cases = [("split before last key", 8)]
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
        split_cases = [("split before first key", 0)]
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

    def test_split_abcdefgh(self):
        keys  =  [1, 3, 5, 7, 9, 11]
        ranks =  [1, 2, 2, 2, 2, 2]
        split_cases = [("split at second key", 3)]
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

    def test_split_abcdefghi(self):
        keys  =  [1, 3, 5, 7, 9, 11]
        ranks =  [1, 1, 2, 2, 2, 2]
        split_cases = [("split at second key", 3)]
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

    def test_split_abcdefghij(self):
        keys  =  [651, 704, 654, 473, 517, 904, 268, 26, 453, 398, 114, 14, 962, 801, 83, 459, 393, 513, 810, 34, 221, 279, 29, 540, 570, 909, 498, 998, 90, 36, 107, 24, 74, 597, 389, 97, 88, 762, 374, 596, 898, 599, 826, 875, 55, 624, 639, 583, 718, 497]
        ranks =  [2, 1, 1, 2, 3, 2, 1, 2, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2]
        split_cases = [("split at key 389", 389)]
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

    def test_split_abcdefghijk(self):
        keys  =  [851, 90, 775, 902, 973, 76, 754]
        ranks =  [1, 1, 1, 1, 1, 3, 1]
        k = 4
        ranks = [calc_rank_for_dim(key=key, k=k, dim=1) for key in keys]
        split_cases = [("split at smallest key", 76)]
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
        
    def test_split_abcdefghijkl(self):
        keys  =  [1, 3, 5, 7, 9, 11]
        ranks =  [3, 1, 1, 1, 1, 3]
        split_cases = [("split before last", 10)]
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

    def test_split_abcdefghijklm(self):
        keys  =  [1, 3, 5, 7, 9, 11]
        ranks =  [2, 1, 2, 2, 2, 2]
        split_cases = [("split at first key", 1)]
        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=8, l_factor=1.0
                )

    def test_split_abcdefghijklmn(self):
        keys  =  [761, 346, 990, 874, 340, 250]
        # keys  =  [4, 3, 6, 5, 2, 1]
        ranks =  [1, 1, 1, 1, 1, 2]
        items = [Item(k, f"val_{k}") for k in keys]

        # build the tree once
        base_tree = create_gkplus_tree(K=4, l_factor=1.0)
        msg = ""
        for item, rank in zip(items, ranks):
            base_tree, _, _ = base_tree.insert(item, rank)
            msg += f"Tree after inserting {item.key}: {print_pretty(base_tree)}"

        exp_keys = keys
        dummies_left = self.get_dummies(base_tree)
        exp_keys = sorted(dummies_left + exp_keys)
        self.validate_tree(base_tree, exp_keys, msg)

    def test_split_abcdefghijklmno(self):
        keys  =  [1, 3, 5, 7, 9, 11]
        ranks =  [3, 3, 3, 3, 3, 3]
        split_cases = [("split at max key", 11)]
        for case_name, split_key in split_cases:
            exp_left = [k for k in keys if k < split_key]
            exp_right = [k for k in keys if k > split_key]
            with self.subTest(case=case_name, split_key=split_key):
                self._run_split_case_multi_dim(
                    keys, ranks,
                    split_key, exp_left,
                    exp_right, case_name,
                    gnode_capacity=8, l_factor=1.0
                )

    # Uncomment to run exhaustive tests of all rank combinations and split keys.
    def test_all_rank_combinations_specific_keys(self):
        """
        Exhaustively test every rank-combo and every split-key,
        computing the expected left/right key-lists on the fly.
        """
        keys = [1, 3, 5, 7, 9, 11]
        ranks = range(1, 4)
        split_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        num_keys = len(keys)
        combinations = len(ranks) ** num_keys
        # iterations = 1
        # combos = islice(product(ranks, repeat=num_keys), iterations)
        
        for rank_combo in tqdm(
            product(ranks, repeat=num_keys),
            # combos,
            total=combinations,
            desc="Split specific key-rank combos",
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
                        self._run_split_case(
                            keys,
                            rank_combo,
                            split_key,
                            exp_left,
                            exp_right,
                            case_name,
                            gnode_capacity=8,
                            # l_factor=1.0
                        )

            
    # Uncomment to run exhaustive tests of all rank combinations and split keys.
    def test_all_rank_combinations_specific_keys_multi_dim(self):
        """
        Exhaustively test every rank-combo and every split-key,
        computing the expected left/right key-lists on the fly.
        """
        keys = [1, 3, 5, 7, 9, 11]
        ranks = range(1, 4)
        split_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        num_keys = len(keys)
        combinations = len(ranks) ** num_keys
        # iterations = 10
        # combos = islice(product(ranks, repeat=num_keys), iterations)
        
        for rank_combo in tqdm(
            product(ranks, repeat=num_keys),
            # combos,
            total=combinations,
            desc="Split specific key-rank combos multi dim",
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

    def test_rank_combinations_random_keys_and_split_points(self):
        """Test splitting with randomly generated items and keys for multiple split points."""
        k = 4
        num_items = 50
        repetitions = 100  # Number of subtests with different random seeds/split keys

        for i in tqdm(range(repetitions), desc="Split with random items", unit="trial"):
            with self.subTest(run=i):
                # Create a new tree for each subtest
                
                tree = create_gkplus_tree(K=k)
                keys = random.sample(range(1, 1000), num_items)
                ranks = [calc_rank_for_dim(key=key, k=k, dim=1) for key in keys]
                msg = f"Keys:  {keys}"
                msg += f"\nRanks: {ranks}"
                for key, rank in zip(keys, ranks):
                    tree, _, _ = tree.insert(Item(key, f"val_{key}"), rank=rank)
                msg += f"\n\nTree before split: {print_pretty(tree)}"

                # Verify initial tree structure
                dummies = self.get_dummies(tree)
                msg += f"\nDummies: {dummies}"
                expected_keys = sorted(dummies + keys)  # sort to handle dummy keys in other dimensions
                msg += f"\nExpected keys: {expected_keys}"
                self.validate_tree(tree, expected_keys, msg)

                split_cases = self._get_split_cases(keys)
                for split_case, split_key in split_cases:
                    with self.subTest(split_case=split_case, split_key=split_key, keys=keys, ranks=ranks):
                        # expected keys-to-left and keys-to-right
                        # do not include dummy items as they are calculated
                        # on the fly in the _run_split_case_multi_dim method
                        exp_left = [k for k in keys if k < split_key]
                        exp_right = [k for k in keys if k > split_key]
                        case_name = f"split key: {split_key}"
                        self._run_split_case_multi_dim(
                            keys,
                            ranks,
                            split_key,
                            exp_left,
                            exp_right,
                            case_name,
                            gnode_capacity=4,
                            l_factor=1.0
                        )

    def get_split_key(self, keys):
        """
        Returns a random split key that is on average equally likely to be before all keys, after all keys or between two successive keys.

        This is useful for testing the split_inplace method with various scenarios.
        """
        if not keys:
            raise ValueError("Key list is empty")

        # Calculate the range from which to choose the split key
        # Add the average distance between keys before the first key and after the last key
        min_key = min(keys)
        max_key = max(keys)
        avg_distance = int((max_key - min_key) / (len(keys) + 1))
        split_key = random.choice(range(max(0, min_key - avg_distance), max_key + avg_distance))

        # logger.debug(f"min_key: {min_key}, max_key: {max_key}, avg_distance: {avg_distance}, min_range: {min_key - avg_distance}, max_range: {max_key + avg_distance}, split_key: {split_key}")

        return split_key


if __name__ == "__main__":
    unittest.main()