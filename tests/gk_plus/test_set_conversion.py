"""Tests for GK+ tree utility functions"""
import unittest
import random
from typing import List, Optional

from gplus_trees.g_k_plus.factory import make_gkplustree_classes

from gplus_trees.base import Entry
from gplus_trees.g_k_plus.factory import create_gkplus_tree
# from gplus_trees.g_k_plus.utils import _tree_to_klist, _klist_to_tree
from gplus_trees.klist_base import KListBase
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, get_dummy, _tree_to_klist, _klist_to_tree
from gplus_trees.g_k_plus.g_k_plus_base import print_pretty
from gplus_trees.g_k_plus.utils import calc_rank
from tests.gk_plus.base import TreeTestCase as GKPlusTreeTestCase
import logging

logger = logging.getLogger(__name__)

class TestSetConversion(GKPlusTreeTestCase):
    def create_entries(self, keys: List[int], with_subtrees: Optional[bool] = False) -> List[Entry]:
        """
        Create a list of Entry objects from a list of keys each with value "val_{key}" and empty left subtrees.
        """
        if not with_subtrees:
            # Create entries without subtrees
            return [Entry(self.make_item(key, f"val_{key}"), None) for key in keys]
        
        items = [self.make_item(key, f"val_{key}") for key in keys]
        subtrees = [create_gkplus_tree(K=self.k, dimension=1) for _ in items]
        # for i, subtree in enumerate(subtrees):
        #     # Set the dummy item for the subtree
        #     subtrees[i] = subtree.insert(self.make_item(1000 + i, "subtree_val"), 1)[0]
        return [Entry(item, subtree) for item, subtree in zip(items, subtrees)]
    
    def assert_entries_present_same_instance(
            self,
            exp_entries: List[Entry],
            act_entries: List[Entry],
        ) -> None:
        """
        Assert that the expected entries are present in the actual entries and refer to the same instances. There may be more actual entries than expected due to dummies.
        """
        for exp_entry in exp_entries:
            act_entry = next((e for e in act_entries if e.item.key == exp_entry.item.key), None)
            self.assertIsNotNone(act_entry, f"Expected entry {exp_entry} not found in actual entries")
            self.assertIs(exp_entry, act_entry, 
                          f"Expected entry {exp_entry} does not refer to the same instance as actual entry {act_entry}")
            self.assertIs(exp_entry.left_subtree, act_entry.left_subtree)

class TestKListToTree(TestSetConversion):
    """Test the conversion from a KList to a GKPlusTree"""
    def setUp(self):
        """Set up test fixtures"""
        self.k = 4
        _, _, KListClass, _ = make_gkplustree_classes(self.k)
        self.klist = KListClass()

    def test_type_validation(self):
        """Test that _klist_to_tree validates the input type"""
        # Test with a non-KListBase object
        with self.assertRaises(TypeError):
            _klist_to_tree("not a klist", K=2, DIM=1)

    def test_empty_klist(self):
        """Test that _klist_to_tree works with an empty KList"""
        tree = _klist_to_tree(self.klist, K=4, DIM=1)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.validate_tree(tree, [])
        self.assertTrue(tree.is_empty())

    def test_no_expansion_rank_1(self):
        """Test that _klist_to_tree correctly converts a non-empty KList to a tree"""
        rank_lists = [
            [1, 1], # Dimension 1 (Test 1: conversion and stopping)
            [1, 1], # Dimension 2
            [1, 1], # Dimension 3 (Test 2: conversion and stopping)
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        items = [self.make_item(key, f"val_{i}") for i, key in enumerate(keys)]
        entries = [Entry(item, None) for item in items]
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)

        with self.subTest("Conversion to DIM 1"):
            tree = _klist_to_tree(self.klist, self.k, DIM=1)
            self.assertIsInstance(tree, GKPlusTreeBase)
            self.assertEqual(tree.DIM, 1)
            self.validate_tree(tree, [-1] + keys)
            self.assert_entries_present_same_instance(entries, list(tree))
            self.assertIsInstance(tree.node.set, KListBase)
        with self.subTest("Conversion to DIM 3 (with other dimensions dummy as a normal item)"):
            # Use -2 as dummy representative as it has rank 1 in DIM 3 for k = 4
            self.klist, _, _ = self.klist.insert_entry(Entry(get_dummy(2), None))
            tree = _klist_to_tree(self.klist, self.k, DIM=3)
            self.assertIsInstance(tree, GKPlusTreeBase)
            self.assertEqual(tree.DIM, 3)
            self.validate_tree(tree, [-3, -2] + keys)
            self.assert_entries_present_same_instance(entries, list(tree))
            self.assertIsInstance(tree.node.set, KListBase)

    def test_no_expansion_rank_gt_1(self):
        """Test that _klist_to_tree correctly converts a non-empty KList to a tree"""
        rank_lists = [
            [2, 1], # Dimension 1 (Test 1: conversion and stopping)
            [2, 1], # Dimension 2 
            [2, 1], # Dimension 3 (Test 2: conversion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, 4)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)

        with self.subTest("Conversion to DIM 1"):
            tree = _klist_to_tree(self.klist, self.k, DIM=1)
            self.assertIsInstance(tree, GKPlusTreeBase)
            self.assertEqual(tree.DIM, 1)
            self.validate_tree(tree, [-1] + keys)
            self.assert_entries_present_same_instance(entries, list(tree))
            self.assertIsInstance(tree.node.set, KListBase)
        with self.subTest("Conversion to DIM 3 (with other dimensions dummy as a normal item)"):
            # Use -2 as dummy representative as it has rank 1 in DIM 3 -> no expansion
            self.klist, _, _ = self.klist.insert_entry(Entry(get_dummy(2), None))
            tree = _klist_to_tree(self.klist, self.k, DIM=3)
            self.assertIsInstance(tree, GKPlusTreeBase)
            self.assertEqual(tree.DIM, 3)
            self.validate_tree(tree, [-3, -2] + keys)
            self.assert_entries_present_same_instance(entries, list(tree))
            self.assertIsInstance(tree.node.set, KListBase)

    def test_expansion_rank_1_dim_1(self):
        """Test that the conversion from a Klist to a tree with expansion at rank 1 in DIM 1"""
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 (conversion)
            [2, 1, 1, 1], # Dimension 2 (expansion and stopping)
        ]  
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)
        tree = _klist_to_tree(self.klist, self.k, DIM=1)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 1)
        self.validate_tree(tree, [-2, -1] + keys)
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.DIM, 2)
        self.validate_tree(tree.node.set, [-2, -1] + keys)
        self.assertIsInstance(tree.node.set.node.set, KListBase)
        
    def test_expansion_rank_1_dim_gt_1(self):
        rank_lists = [
            [1, 1, 1], # Dimension 1
            [1, 1, 1], # Dimension 2
            [1, 1, 1], # Dimension 3 (conversion)
            [2, 1, 1], # Dimension 4 (expansion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)

        # Use -2 as dummy representative as it has rank 1 in DIM 3
        self.klist, _, _ = self.klist.insert_entry(Entry(get_dummy(2), None))
        tree = _klist_to_tree(self.klist, self.k, DIM=3)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 3)
        self.validate_tree(tree, [-4, -3, -2] + keys)
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.DIM, 4)
        self.validate_tree(tree.node.set, [-4, -3, -2] + keys)
        self.assertIsInstance(tree.node.set.node.set, KListBase)

    def test_expansion_rank_gt_1_dim_1(self):
        rank_lists = [
            [2, 2, 2, 2], # Dimension 1 (conversion)
            [2, 1, 1, 1], # Dimension 2 (expansion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)
        tree = _klist_to_tree(self.klist, self.k, DIM=1)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 1)
        self.validate_tree(tree, [-1] + keys) # expansion only in root, no additional entry in leaf
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.DIM, 2)
        self.validate_tree(tree.node.set, [-2, -1] + keys)
        self.assertIsInstance(tree.node.set.node.set, KListBase)

    def test_expansion_rank_gt_1_dim_gt_1(self):
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 
            [2, 2, 2, 2], # Dimension 2 (conversion)
            [2, 1, 1, 1], # Dimension 3 (expansion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)
        tree = _klist_to_tree(self.klist, self.k, DIM=2)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 2)
        self.validate_tree(tree, [-2] + keys) # expansion only in root, no additional entry in leaf
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.DIM, 3)
        self.validate_tree(tree.node.set, [-3, -2] + keys)
        self.assertIsInstance(tree.node.set.node.set, KListBase)

    def test_multi_expansion_rank_1_dim_1(self):
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 (conversion)
            [1, 1, 1, 1], # Dimension 2 (expansion)
            [2, 1, 1, 1], # Dimension 3 (expansion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        self.validate_key_ranks(keys, rank_lists, self.k)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)
        tree = _klist_to_tree(self.klist, self.k, DIM=1)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 1)
        dummy_keys = self.get_dummies(tree)
        self.validate_tree(tree, dummy_keys + keys)
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertEqual(tree.node.set.DIM, 2)
        self.validate_tree(tree.node.set, dummy_keys + keys)

    def test_multi_expansion_rank_1_dim_gt_1(self):
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 
            [1, 1, 1, 1], # Dimension 2 (conversion)
            [1, 1, 1, 1], # Dimension 3 (expansion)
            [2, 1, 1, 1], # Dimension 4 (expansion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)
        tree = _klist_to_tree(self.klist, self.k, DIM=2)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 2)
        dummy_keys = self.get_dummies(tree)
        self.validate_tree(tree, dummy_keys + keys)
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertEqual(tree.node.set.DIM, 3)
        self.validate_tree(tree.node.set, dummy_keys + keys)
        self.assertEqual(tree.node.set.node.set.DIM, 4)
        self.validate_tree(tree.node.set.node.set, dummy_keys + keys)


    def test_multi_expansion_rank_gt_1_dim_1(self):
        rank_lists = [
            [2, 2, 2, 2], # Dimension 1 (conversion)
            [2, 2, 2, 2], # Dimension 2 (expansion)
            [2, 1, 1, 1], # Dimension 3 (expansion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        self.validate_key_ranks(keys, rank_lists, self.k)

        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)
        tree = _klist_to_tree(self.klist, self.k, DIM=1)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 1)
        self.validate_tree(tree, [-1] + keys)
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.DIM, 2)
        self.validate_tree(tree.node.set, [-2, -1] + keys)
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.node.set.DIM, 3)
        self.validate_tree(tree.node.set.node.set, [-3, -2] + keys)
        self.assertIsInstance(tree.node.set.node.set.node.set, KListBase)

    def test_multi_expansion_rank_gt_1_dim_gt_1(self):
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1
            [2, 2, 2, 2], # Dimension 2 (conversion)
            [2, 2, 2, 2], # Dimension 3 (expansion)
            [2, 1, 1, 1], # Dimension 4 (expansion and stopping)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        for entry in entries:
            self.klist, _, _ = self.klist.insert_entry(entry)
        tree = _klist_to_tree(self.klist, self.k, DIM=2)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 2)
        self.validate_tree(tree, [-2] + keys)
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.DIM, 3)
        self.validate_tree(tree.node.set, [-3, -2] + keys)
        self.assertIsInstance(tree.node.set.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.node.set.DIM, 4)
        self.validate_tree(tree.node.set.node.set, [-4, -3] + keys)
        self.assertIsInstance(tree.node.set.node.set.node.set, KListBase)

    def test_lfactor_multi_expansion_rank_gt_1_dim_gt_1(self):
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1
            [2, 2, 2, 2], # Dimension 2 (conversion)
            [2, 2, 2, 2], # Dimension 3 (expansion)
            [2, 1, 1, 1], # Dimension 4 (expansion and stopping)
        ]
        k = 2  # Use a smaller K for this test
        l_factor = 2
        _, _, KListClass, _ = make_gkplustree_classes(k)
        klist = KListClass()
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        entries = self.create_entries(keys)
        for entry in entries:
            klist, _, _ = klist.insert_entry(entry)
        tree = _klist_to_tree(klist, k, DIM=2, l_factor=l_factor)
        self.assertIsInstance(tree, GKPlusTreeBase)
        self.assertEqual(tree.DIM, 2)
        self.assertEqual(tree.l_factor, l_factor)
        self.validate_tree(tree, [-2] + keys)
        self.assert_entries_present_same_instance(entries, list(tree))
        self.assertIsInstance(tree.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.DIM, 3)
        self.assertEqual(tree.node.set.l_factor, l_factor)
        self.validate_tree(tree.node.set, [-3, -2] + keys)
        self.assertIsInstance(tree.node.set.node.set, GKPlusTreeBase)
        self.assertEqual(tree.node.set.node.set.DIM, 4)
        self.assertEqual(tree.node.set.node.set.l_factor, l_factor)
        self.validate_tree(tree.node.set.node.set, [-4, -3] + keys)
        self.assertIsInstance(tree.node.set.node.set.node.set, KListBase)
    

class TestTreeToKList(TestSetConversion):
    """Test the utility functions for GKPlus trees"""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        # Create trees with different K values for testing
        self.k = 4
        self.empty_tree = create_gkplus_tree(K=self.k)
        self.klist: KListBase = None

    def create_tree_from_entries(
            self,
            entries: List[Entry],
            ranks: List[int],
            DIM: int
        ) -> GKPlusTreeBase:
        """Create a GKPlusTree for a given dimension from a list of entries"""
        tree = create_gkplus_tree(K=self.k, dimension=DIM)
        for i, entry in enumerate(entries):
            tree, _, _ = tree.insert_entry(entry, ranks[i])
        return tree

    def test_tree_to_klist_type_validation(self):
        """Test that _tree_to_klist validates the input type"""
        # Test with a non-GKPlusTreeBase object
        with self.assertRaises(TypeError):
            _tree_to_klist("not a tree")

    def test_tree_to_klist_empty_tree(self):
        """Test that _tree_to_klist works with an empty tree"""
        empty_tree = self.empty_tree
        self.klist = _tree_to_klist(empty_tree)
        self.assertTrue(self.klist.is_empty())
        self.validate_klist(self.klist, [])

    def test_unexpanded_tree_to_klist_rank_1_dim_1(self):
        """Test that _tree_to_klist works with an unexpanded tree (rank 1, dim 1)"""
        col_dim = 1
        rank_lists = [[1, 1, 1]]  # Dimension 1 (collapse)
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_unexpanded_tree_to_klist_rank_1_dim_gt_1(self):
        """Test that _tree_to_klist works with an unexpanded tree (rank 1, dim > 1)"""
        col_dim = 2
        rank_lists = [
            [1, 1, 1],  # Dimension 1
            [1, 1, 1],  # Dimension 2 (collapse)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_unexpanded_tree_to_klist_rank_gt_1_dim_1(self):
        """Test that _tree_to_klist works with an unexpanded tree (rank > 1, dim 1)"""
        col_dim = 1
        rank_lists = [[1, 2, 1]]  # Dimension 1 (collapse)
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_unexpanded_tree_to_klist_rank_gt_1_dim_gt_1(self):
        """Test that _tree_to_klist works with an unexpanded tree (rank > 1, dim > 1)"""
        col_dim = 2
        rank_lists = [
            [1, 1, 1], # Dimension 1
            [1, 2, 1], # Dimension 2 (collapse)
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_expanded_tree_to_klist_rank_1_dim_1(self):
        """Test that _tree_to_klist works with an expanded tree (rank 1, dim 1)"""
        col_dim = 1
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 (collapse expanded)
            [2, 1, 1, 1], # Dimension 2
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_expanded_tree_to_klist_rank_1_dim_gt_1(self):
        """Test that _tree_to_klist works with an expanded tree (rank 1, dim > 1)"""
        col_dim = 2
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1
            [1, 1, 1, 1], # Dimension 2 (collapse expanded)
            [2, 1, 1, 1], # Dimension 3
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_expanded_tree_to_klist_rank_gt_1_dim_1(self):
        """Test that _tree_to_klist works with an expanded tree (rank > 1, dim 1)"""
        col_dim = 1
        rank_lists = [
            [2, 2, 2, 2], # Dimension 1 (collapse expanded)
            [2, 1, 1, 1], # Dimension 2 
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_expanded_tree_to_klist_rank_gt_1_dim_gt_1(self):
        """Test that _tree_to_klist works with an expanded tree (rank > 1, dim > 1)"""
        col_dim = 2
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1
            [2, 2, 2, 2], # Dimension 2 (collapse expanded)
            [2, 1, 1, 1], # Dimension 3 
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_multi_expanded_tree_to_klist_rank_1_dim_1(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 1
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 (collapse multiple expanded)
            [1, 1, 1, 1], # Dimension 2 
            [2, 1, 1, 1], # Dimension 3 
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_multi_expanded_tree_to_klist_rank_1_dim_gt_1(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 2 
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 
            [1, 1, 1, 1], # Dimension 2 (collapse multiple expanded)
            [1, 1, 1, 1], # Dimension 3
            [2, 1, 1, 1], # Dimension 4
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_multi_expanded_tree_to_klist_rank_gt_1_dim_1(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 1
        rank_lists = [
            [2, 2, 2, 2], # Dimension 1 (collapse multiple expanded)
            [2, 2, 2, 2], # Dimension 2 
            [2, 1, 1, 1], # Dimension 3
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_multi_expanded_tree_to_klist_rank_gt_1_dim_gt_1(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 2
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 
            [2, 2, 2, 2], # Dimension 2 (collapse multiple expanded)
            [2, 2, 2, 2], # Dimension 3
            [2, 1, 1, 1], # Dimension 4
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_multi_expanded_tree_to_klist_different_ranks(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 2
        rank_lists = [
            [1, 1, 1, 1], # Dimension 1 
            [2, 2, 2, 2], # Dimension 2 (collapse multiple expanded)
            [1, 1, 1, 1], # Dimension 3 (additional expansion in leaf)
            [2, 1, 1, 1], # Dimension 4
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_problematic_v1(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 5
        rank_lists = [
            [1, 2, 1, 1, 1], # Dimension 1 
            [1, 2, 1, 1, 1], # Dimension 2 
            [1, 1, 1, 2, 1], # Dimension 3 
            [3, 2, 1, 1, 1], # Dimension 4
            [1, 1, 1, 1, 2], # Dimension 5 (collapse multiple expanded)
            [1, 1, 1, 1, 2], # Dimension 6 (additional expansion in leaf)
            [1, 1, 1, 1, 2], # Dimension 7
            [1, 2, 1, 2, 1], # Dimension 8
            [1, 2, 1, 5, 1], # Dimension 9
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_problematic_v2(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 1
        rank_lists = [
            [1, 1, 1, 1, 2], # Dimension 1
            [1, 1, 1, 1, 2], # Dimension 2
            [1, 1, 1, 1, 2], # Dimension 3
            [1, 2, 1, 2, 1], # Dimension 4
            [1, 2, 1, 5, 1], # Dimension 5
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_multi_expanded_tree_to_klist_multi_nodes(self):
        """Test that _tree_to_klist works with a multiple expanded tree (rank > 1, dim > 1)"""
        col_dim = 2  # Dimension to collapse
        rank_lists = [
            [1, 1, 1, 1, 2, 1, 1, 1, 1], # Dimension 1 
            [2, 2, 2, 2, 1, 1, 1, 1, 1], # Dimension 2 (collapse multiple expanded)
            [1, 1, 1, 1, 2, 1, 1, 1, 1], # Dimension 3 (additional expansion in leaf)
            [2, 1, 1, 1, 2, 1, 1, 1, 1], # Dimension 4
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, self.k)
        entries = self.create_entries(keys, with_subtrees=True)
        tree = self.create_tree_from_entries(entries, rank_lists[col_dim - 1], DIM=col_dim)
        self.klist = _tree_to_klist(tree)
        self.validate_klist(self.klist, entries)

    def test_roundtrip_tree_to_klist_to_tree_dim_5_old(self):
        """Test the round trip from tree to KList and back to tree"""
        # Create a random tree
        iterations = 10
        initial_tree_dim = 5
        self.k = 4
        num_entries = 50
        for _ in range(iterations):
            with self.subTest(iteration=_):
                keys = random.sample(range(1, 10000), num_entries)
                keys = sorted(keys)
                entries = self.create_entries(keys)
                ranks = [calc_rank(key=key, k=self.k, dim=initial_tree_dim) for key in keys]
                
                
                msg = f"\n\n\nIteration {_}: dim = {initial_tree_dim}"
                msg += f"\n\nkeys = {keys}"
                msg += f"\nranks = {ranks}"
                
                # Log rank information for debugging
                self._log_ranks(self.k, keys, 10)
                self._log_ranks(self.k, list(range(-11, 0, -1)), 11)

                # Create original tree
                original_tree = self.create_tree_from_entries(entries, ranks, DIM=initial_tree_dim)
                # msg += f"\n\noriginal_tree = {print_pretty(original_tree)}"
                
                # Get dummies and validate original tree
                dummies = self.get_dummies(original_tree)
                keys_sorted = sorted(keys)
                exp_keys = sorted(dummies + keys_sorted)
                self.validate_tree(original_tree, exp_keys, msg)
                
                # Convert tree to klist
                klist = _tree_to_klist(original_tree)
                # msg += f"\n\nklist = {print_pretty(klist)}"
                
                # Validate klist (entries should match original entries, not including dummies)
                entries_sorted = sorted(entries, key=lambda e: e.item.key)
                self.validate_klist(klist, entries_sorted, msg)
                
                # Convert klist back to tree
                new_tree = _klist_to_tree(klist, K=self.k, DIM=initial_tree_dim)
                # msg += f"\n\nnew_tree = {print_pretty(new_tree)}"
                self.validate_tree(new_tree, exp_keys, msg)
                
                # Check that original entries are present in new tree
                # Note: We compare the original entries (not sorted) with the new tree
                self.assert_entries_present_same_instance(entries, list(new_tree))

    def test_roundtrip_tree_to_klist_to_tree_dim_5_problematic(self):
        """Test the round trip from tree to KList and back to tree"""
        # Create a random tree
        iterations = 10
        initial_tree_dim = 5
        self.k = 4
        num_entries = 50
        keys = [3337, 1519, 7882, 9415, 9604]
        logger.debug(f"Keys: {keys}")
        # while len(keys) < num_entries:
        #     keys.add(random.randint(1, 10000))
        keys = sorted(list(keys))
        logger.debug(f"Keys sorted: {keys}")
        entries = self.create_entries(keys)
        logger.debug(f"Entries: {[entry.item.key for entry in entries]}")

        ranks = [calc_rank(key=key, k=self.k, dim=initial_tree_dim) for key in keys]
        logger.debug(f"Ranks: {ranks}")
        
        
        msg = f"\n\n\nIteration 1: dim = {initial_tree_dim}"
        msg += f"\n\nkeys = {keys}"
        msg += f"\nranks = {ranks}"
        
        # Log rank information for debugging
        self._log_ranks(self.k, keys, 10)
        dummy_keys_for_printing = list(range(-1, -11, -1))
        self._log_ranks(self.k, dummy_keys_for_printing, 10)

        # Create original tree
        original_tree = self.create_tree_from_entries(entries, ranks, DIM=initial_tree_dim)
        # msg += f"\n\noriginal_tree = {print_pretty(original_tree)}"
        
        # Get dummies and validate original tree
        dummies = self.get_dummies(original_tree)
        keys_sorted = sorted(keys)
        exp_keys = sorted(dummies + keys_sorted)
        self.validate_tree(original_tree, exp_keys, msg)
        
        # Convert tree to klist
        klist = _tree_to_klist(original_tree)
        # msg += f"\n\nklist = {print_pretty(klist)}"
        
        # Validate klist (entries should match original entries, not including dummies)
        entries_sorted = sorted(entries, key=lambda e: e.item.key)
        self.validate_klist(klist, entries_sorted, msg)
        
        # Convert klist back to tree
        new_tree = _klist_to_tree(klist, K=self.k, DIM=initial_tree_dim)
        # msg += f"\n\nnew_tree = {print_pretty(new_tree)}"
        self.validate_tree(new_tree, exp_keys, msg)
        
        # Check that original entries are present in new tree
        # Note: We compare the original entries (not sorted) with the new tree
        self.assert_entries_present_same_instance(entries, list(new_tree))

if __name__ == '__main__':
    unittest.main()
