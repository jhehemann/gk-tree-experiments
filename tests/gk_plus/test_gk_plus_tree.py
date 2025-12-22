# import sys
# import os
# import unittest
# import random
# from typing import List, Tuple, Optional, Iterator, TYPE_CHECKING
# from itertools import product, islice
# from pprint import pprint
# import copy
# from tqdm import tqdm
# from statistics import median_low

# # Add the src directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from gplus_trees.base import Item
# from tests.gk_plus.base import TreeTestCase
# from gplus_trees.g_k_plus.factory import create_gkplus_tree
# from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
# from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
# from tests.utils import assert_tree_invariants_tc

# if TYPE_CHECKING:
#     from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

# from tests.logconfig import logger        

# class TestGKPlusItemSlotCount(TreeTestCase):
#     """Tests for the item_slot_count method in GKPlusTreeBase."""
    
#     def setUp(self):
#         # Create trees with different K values for testing
#         self.tree_k2 = create_gkplus_tree(K=2)
#         self.tree_k4 = create_gkplus_tree(K=4)
#         self.tree_k8 = create_gkplus_tree(K=8)
        
#     def test_empty_tree_slot_count(self):
#         """Test that an empty tree has 0 item slots."""
#         tree = create_gkplus_tree(K=4)
#         self.assertTrue(tree.is_empty())
#         self.assertEqual(0, tree.item_slot_count())
        
#     def test_single_item_tree_slot_count(self):
#         """Test slot count for a tree with a single item."""
#         # Insert one item
#         tree, _, _ = self.tree_k4.insert(Item(1000, "val"), rank=1)
        
#         # A single leaf node will be created with K+1 slots (K=4)
#         expected_slots = self.tree_k4.SetClass.KListNodeClass.CAPACITY
#         self.assertEqual(expected_slots, tree.item_slot_count(),
#                          f"Expected {expected_slots} slots for a single item tree with K=4")
        
#     def test_cap_node_slot_count(self):
#         """Test slot count for a tree with a node at capacity."""
#         tree = self.tree_k4
#         cap = 4
        
#         # Insert item to fill the node to capacity (beware of dummy item)
#         for i in range(1, cap-1):
#             tree, _, _ = tree.insert(Item(i * 1000, "val"), rank=1)
        
#         exp_item_count = cap
#         expected_slots =  exp_item_count + (exp_item_count % tree.SetClass.KListNodeClass.CAPACITY)
#         self.assertEqual(expected_slots, tree.item_slot_count(),
#                          f"Expected {expected_slots} slots for a tree with {exp_item_count} items in a single node")

#     def test_lt_cap_node_slot_count(self):
#         """Test slot count for a tree with multiple items in a single leaf node."""
#         tree = self.tree_k8
        
#         # Insert items that fit in a single leaf node (K=8)
#         for i in range(1, 6):  # Insert 5 items
#             tree, _, _ = tree.insert(Item(i * 1000, "val"), rank=1)
#
#         # A single leaf node with K+1 slots
#         expected_slots = self.tree_k8.SetClass.KListNodeClass.CAPACITY
#         self.assertEqual(expected_slots, tree.item_slot_count(),
#                          f"Expected {expected_slots} slots for a tree with 5 items in a single node")
#
#     def test_multi_leaf_node_slot_count(self):
#         """Test slot count for a tree with multiple leaf nodes."""
#         tree = self.tree_k2  # K=2 to force splits quickly
#         self.cap = 2
        
#         # Insert enough items to cause underlying KlistNode splits
#         indices = [i for i in range(1, 2 * self.cap + 1)]
#         for i in indices:
#             tree, _, _ = tree.insert(Item(i * 1000, "val"), rank=1)
#
#         exp_item_count = len(indices) + 1 # +1 for the dummy item
#         expected_slots =  exp_item_count + (exp_item_count % self.tree_k2.SetClass.KListNodeClass.CAPACITY)
#         self.assertEqual(expected_slots, tree.item_slot_count(),
#                         f"Expected {expected_slots} slots for a tree with {exp_item_count} items and K = {self.cap}")
        
#     def test_internal_node_slot_count(self):
#         """Test slot count for a tree with internal nodes."""
#         tree = self.tree_k2
        
#         # Insert items with rank 1 to create leaf nodes
#         for i in range(1, 8):
#             tree, _, _ = tree.insert(Item(i * 1000, "val"), rank=1)
#
#         # Insert items with rank 2 to create internal nodes
#         for i in range(1, 4):
#             tree, _, _ = tree.insert(Item(i * 500, "val"), rank=2)
#
#         # Count slots manually by traversing the tree
#         total_expected_slots = self._count_slots_manually(tree)
#
#         self.assertEqual(total_expected_slots, tree.item_slot_count(),
#                         f"Expected {total_expected_slots} slots for a tree with internal nodes")
    
#     def test_complex_tree_structure_slot_count(self):
#         """Test slot count in a complex tree with various ranks and splits."""
#         tree = self.tree_k4
        
#         # Create a mix of items with different ranks
#         items = []
#         for i in range(1, 31):  # 30 items
#             key = i * 100
#             rank = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)  # Mix of ranks 1, 2, 3
#             items.append((Item(key, "val"), rank))
        
#         # Insert all items
#         for item, rank in items:
#             tree, _, _ = tree.insert(item, rank)
#
#         # Count slots manually by traversing the tree
#         total_expected_slots = self._count_slots_manually(tree)
#
#         self.assertEqual(total_expected_slots, tree.item_slot_count(),
#                         f"Expected {total_expected_slots} slots for a complex tree structure")
#
#     def test_large_tree_slot_count(self):
#         """Test slot count in a larger tree with random insertions."""
#         tree = self.tree_k8
#         keys = random.sample(range(1, 10000), 50)  # 50 unique random keys
#         ranks = [1] * 30 + [2] * 15 + [3] * 5  # Mix of ranks
#         random.shuffle(ranks)
        
#         # Insert all items
#         for key, rank in zip(keys, ranks):
#             item = Item(key, "val")
#             tree, _, _ = tree.insert(item, rank=rank)
#
#         # Count slots manually by traversing the tree
#         total_expected_slots = self._count_slots_manually(tree)
#
#         self.assertEqual(total_expected_slots, tree.item_slot_count(),
#                         f"Expected {total_expected_slots} slots for a large tree")
#
#     def test_slot_count_after_changing_tree_structure(self):
#         """Test that slot count updates correctly when tree structure changes."""
#         # Start with a tree with several items
#         tree = self.tree_k4
#         for i in range(1, 11):
#             tree, _, _ = tree.insert(Item(i * 1000, "val"), rank=1)
#
#         # Get initial slot count
#         initial_slots = tree.item_slot_count()
#   
#         # Insert an item that causes a node split
#         tree, _, _ = tree.insert(Item(500, "val"), rank=2)
#
#         # Get new slot count
#         new_slots = tree.item_slot_count()
#
#         # The slot count should increase after a split
#         self.assertGreaterEqual(new_slots, initial_slots,
#                               "Slot count should increase or stay the same after structure changes")
        
#         # Count slots manually to verify
#         total_expected_slots = self._count_slots_manually(tree)
#         self.assertEqual(total_expected_slots, new_slots,
#                         f"Expected {total_expected_slots} slots after structure change")
        
#     def _count_slots_manually(self, tree):
#         """Helper method to count slots by traversing the tree structure."""
#         if tree is None:
#             return 0
            
#         total_slots = 0
#         node_queue = [tree.node]
        
#         while node_queue:
#             current_node = node_queue.pop(0)
            
#             # Count slots in this node's set
#             total_slots += current_node.set.item_slot_count()
            
#             # Add child nodes to the queue
#             if current_node.rank > 1:
#                 # Add left subtrees
#                 for entry in current_node.set:
#                     if entry.left_subtree is not None:
#                         node_queue.append(entry.left_subtree.node)
                
#                 # Add right subtree
#                 if current_node.right_subtree is not None:
#                     node_queue.append(current_node.right_subtree.node)
                    
#         return total_slots

