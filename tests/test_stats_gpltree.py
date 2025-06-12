import unittest
from unittest.mock import patch
from pprint import pprint

# Import from factory and base modules instead of archive
from gplus_trees.factory import make_gplustree_classes, create_gplustree
from gplus_trees.base import Item
from gplus_trees.gplus_tree_base import DUMMY_KEY, DUMMY_ITEM, gtree_stats_, print_pretty

# Constants for test configuration
K_VALUE = 4  # Capacity value for KList nodes

from tests.logconfig import logger

class TestGTreeStatsInvalidProperties(unittest.TestCase):
    def setUp(self):
        """Set up test trees of various sizes and structures"""
        # Get classes for the specified capacity K
        self.GPlusTreeK, self.GPlusNodeK, self.KListK, self.KListNodeK = make_gplustree_classes(K_VALUE)
        
        # Test items
        self.item1 = Item(1, "val_1")
        self.item2 = Item(2, "val_2")
        self.item3 = Item(3, "val_3")
        self.item4 = Item(4, "val_4")

        self.replica1 = Item(1, None)
        self.replica2 = Item(2, None)
        self.replica3 = Item(3, None)
        self.replica4 = Item(4, None)
        
        # Single-node tree
        single_set1 = (
            self.KListK()
            .insert(DUMMY_ITEM, None)
            .insert(self.item1, None)
        )
        self.single_node_tree = self.GPlusTreeK(
            self.GPlusNodeK(1, single_set1, None)
        )

        # 3-node tree (1 root, 2 leaves)
        leaf_set2 = (
            self.KListK()
            .insert(self.item3, None)
            .insert(self.item4, None)
        )
        self.leaf_tree2 = self.GPlusTreeK(self.GPlusNodeK(1, leaf_set2, self.GPlusTreeK()))

        leaf_set1 = (
            self.KListK()
            .insert(DUMMY_ITEM, None)
            .insert(self.item1, None)
            .insert(self.item2, None)
        )
        self.leaf_tree1 = self.GPlusTreeK(self.GPlusNodeK(1, leaf_set1, self.GPlusTreeK()))
        self.leaf_tree1.node.next = self.leaf_tree2

        root_set = (
            self.KListK()
            .insert(DUMMY_ITEM, None)
            .insert(self.replica3, self.leaf_tree1)
        )
        self.tree_3_nodes = self.GPlusTreeK(self.GPlusNodeK(3, root_set, self.leaf_tree2))

    def test_single_node_tree_item_order(self):
        """Test stats computed for an empty tree"""
        # Swap first item (DUMMY_ITEM) with item 3
        self.single_node_tree.node.set.head.entries[0].item = self.item3
        logger.debug(f"Single node tree: {print_pretty(self.single_node_tree)}")

        stats = gtree_stats_(self.single_node_tree)
        
        self.assertEqual(stats.least_item.key, 3,
                            "Expected least item key to be 3")
        self.assertFalse(stats.is_search_tree, "Expected NO search tree")
        self.assertFalse(stats.leaf_keys_in_order,
                            "Expected unordered leaf keys")
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertEqual(stats.real_item_count, 2)
        
    def test_single_node_tree_linked_leafs(self):
        """Test stats computed for an empty tree"""
        # Add an unexpected leaf node by manipulating next pointer
        self.single_node_tree.node.next = self.leaf_tree2
        
        stats = gtree_stats_(self.single_node_tree)
        
        self.assertEqual(stats.least_item.key, DUMMY_KEY,
                            "Expected least item key to be DUMMY_KEY")
        self.assertEqual(stats.greatest_item.key, 1,
                            "Expected greatest item key to be 1")
        
        self.assertFalse(stats.linked_leaf_nodes)
        self.assertEqual(stats.real_item_count, 3)


class TestGTreeStatsInvalidLargeRandomTree(unittest.TestCase):
    def setUp(self):
        """Test stats computed for a large random tree"""
        # Get classes for the specified capacity K
        self.GPlusTreeK, self.GPlusNodeK, self.KListK, self.KListNodeK = make_gplustree_classes(K_VALUE)
        
        # Create a large random tree
        self.tree = create_gplustree(K_VALUE)
        keys = [1, 2, 3, 4, 5]
        ranks = [2, 3, 2, 1, 4]
        self.item_map = { k: (Item(k, f"val_{k}")) for k in keys}
        self.rank_map = { key: rank for key, rank in zip(keys, ranks) }
        for k in keys:
            item, rank = self.item_map[k], self.rank_map[k]
            self.tree.insert(item, rank)
    
    def test_heap_internal_eq_child_rank(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root_entries = list(root.set)
        internal_rank_3 = root_entries[1].left_subtree
        internal_rank_2_r = internal_rank_3.node.right_subtree
        internal_rank_2_r.node.rank = 1 # Equal to child rank

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.is_heap, "Expected NO heap property")

    def test_heap_root_lt_child_rank(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root.rank = 2

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.is_heap, "Expected NO heap property")

    def test_internal_packed(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root_entries = list(root.set)
        internal_rank_3 = root_entries[1].left_subtree
        internal_rank_2_r = internal_rank_3.node.right_subtree
        internal_rank_2_r.node.set.delete(3)

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.internal_packed, "Node has less than 2 entries")

    def test_internal_has_replicas(self):
        # Manipulate ranks to create a non-heap property
        root = self.tree.node
        root_entries = list(root.set)
        internal_rank_3 = root_entries[1].left_subtree
        internal_rank_2_r = internal_rank_3.node.right_subtree
        node_set = list(internal_rank_2_r.node.set)
        node_set[1].item.value = "val_3"    # Set value to a non-replica

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.internal_has_replicas, "Expected item that is not a replica")