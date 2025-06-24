import sys
import os
import random

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase
from gplus_trees.logging_config import get_test_logger
import logging

logger = get_test_logger(__name__)

class TestGKPlusTreeItemCountTracking(GKPlusTreeTestCase):
    def test_empty_tree(self):
        """Test that an empty tree has no node"""
        tree = create_gkplus_tree(K=4)
        self.assertIsNone(tree.item_cnt, 
                          "item_cnt should be None before triggering item_count()")
        self.assertEqual(tree.item_count(), 0,
                         "Item count should be 0 after triggering item_count()")
        self.assertEqual(tree.item_cnt, 0,
                         "item_cnt should have changed to 0 after triggering item_count()")
        
    def test_empty_tree_insertion_rank_1(self):
        """Test count is 1 after inserting a single item"""
        item = self.create_item(1)
        tree, _ = self.tree_k2.insert(item, rank=1)
        self.assertIsNone(tree.item_cnt, 
                          "item_cnt should be None before triggering item_count()")
        self.assertEqual(tree.item_count(), 2,
                            "Item count should be 2 after inserting one item (1 dummy + 1 item)")
        self.assertEqual(tree.item_cnt, 2,
                          "item_cnt should be 2 after item_count() is called")
        
    def test_empty_tree_insertion_rank_gt_1(self):
        """Test count is 1 after inserting a single item"""
        item = self.create_item(1)
        tree, _ = self.tree_k2.insert(item, rank=3)
        self.assertIsNone(tree.item_cnt, 
                          "item_cnt should be None before triggering item_count()")
        self.assertEqual(tree.item_count(), 2,
                            "Item count should be 2 after inserting one item (1 dummy + 1 item)")
        self.assertEqual(tree.item_cnt, 2,
                          "item_cnt should be 2 after item_count() is called")
        
    def test_leaf_insertion(self):
        """Test count is 3 after inserting an item between two existing items"""
        # Insert two items first
        tree = self.tree_k4
        item1 = self.create_item(1)
        item2 = self.create_item(3)
        tree, _ = tree.insert(item1, rank=1)
        tree, _ = tree.insert(item2, rank=1)
        item_between = self.create_item(2)
        logger.debug(f"Tree after initial insertions: {print_pretty(tree)}")
        tree, _ = tree.insert(item_between, rank=1)
        logger.debug(f"Tree after initial insertions + key 310: {print_pretty(tree)}")

        self.assertIsNone(tree.item_cnt, 
                          "item_cnt should be None before triggering item_count()")
        self.assertEqual(tree.item_count(), 4,
                            "Item count should be 4 after inserting three items (1 dummy + 3 items)")
        self.assertEqual(tree.item_cnt, 4,
                          "item_cnt should be 4 after item_count() is called")
    
    def test_root_insertion(self):
        """Test count is 3 after inserting an item in the root"""
        tree = self.tree_k4
        item1 = self.create_item(1)
        item2 = self.create_item(3)
        tree, _ = tree.insert(item1, rank=2)
        tree, _ = tree.insert(item2, rank=2)
        item_root = self.create_item(2)
        logger.debug(f"Tree after initial insertions: {print_pretty(tree)}")
        tree, _ = tree.insert(item_root, rank=1)
        logger.debug(f"Tree after initial insertions + key 310: {print_pretty(tree)}")

        self.assertIsNone(tree.item_cnt, 
                          "item_cnt should be None before triggering item_count()")
        self.assertEqual(tree.item_count(), 4,
                            "Item count should be 4 after inserting three items (1 dummy + 3 items)")
        self.assertEqual(tree.item_cnt, 4,
                          "item_cnt should be 4 after item_count() is called")   

    def test_internal_insertion_long_path(self):
        """Test count increases properly with multiple insertions"""
        tree = self.tree_k4
        rank_lists = [
            [1, 4, 2, 1, 3, 2],
            [1, 2, 1, 1, 1, 1]
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=4)
        logger.debug(f"Keys: {keys}")
        item_map = { k: self.create_item(k) for k in keys}

        for i in range(len(keys)):
            key = keys[i]
            rank = rank_lists[0][i]
            item = item_map[key]
            tree, _ = tree.insert(item, rank=rank)
        
        insert_item = self.create_item(30)
        logger.debug(f"Tree after initial insertions: {print_pretty(tree)}")
        tree, _ = tree.insert(insert_item, rank=3)
        logger.debug(f"Tree after initial insertions + key 310: {print_pretty(tree)}")
        
        cur = tree
        while cur.node.right_subtree is not None:
            self.assertIsNone(cur.item_cnt,
                                f"item_cnt should be None on path to inserted item at node {print_pretty(cur.node.set)}")
            for entry in cur.node.set:
                if entry.item.key == insert_item.key:
                    if entry.left_subtree is not None:
                        self.assertIsNone(entry.left_subtree.item_cnt,
                                        f"item_cnt should be None for insert keys left subtree (left split), got {entry.left_subtree.item_cnt} for: {print_pretty(entry.left_subtree)}")
                if entry.item.key > insert_item.key:
                    cur = entry.left_subtree
                    break
            else:
                cur = cur.node.right_subtree
        
        exp_item_count = 1 + len(keys) + 1  # 1 dummy, len(keys) and 310
        self.assertEqual(tree.item_count(), exp_item_count,
                        f"Item count should be {exp_item_count} after inserting item {insert_item.key}; tree: {print_pretty(tree)}")

        cur = tree
        while True:
            leaf_keys = list(cur)
            self.assertEqual(cur.item_cnt, len(leaf_keys),
                            f"item_cnt should be {len(leaf_keys)} on path to inserted item at node {print_pretty(cur.node.set)}")
            for entry in cur.node.set:
                if entry.item.key > insert_item.key:
                    cur = entry.left_subtree
                    break
            else:
                cur = cur.node.right_subtree
            if cur.node.right_subtree is None:
                break

        self.assertTrue(self.verify_subtree_sizes(tree))

    def test_insertion_triggering_klist_to_tree(self):
        """Test count increases properly with multiple insertions"""
        tree = self.tree_k4
        rank_lists = [
            [1, 1, 1, 1],
            [2, 1, 1, 2]
        ]
        
        keys = self.find_keys_for_rank_lists(rank_lists, k=4)
        logger.debug(f"Keys: {keys}")
        item_map = { k: self.create_item(k) for k in keys}

        for i in range(len(keys)):
            key = keys[i]
            rank = rank_lists[0][i]
            item = item_map[key]
            tree, _ = tree.insert(item, rank=rank)

        max_dim = tree.get_max_dim()
        expanded_leafs = tree.get_expanded_leaf_count()
        expected_keys = [entry.item.key for entry in tree]
        logger.debug(f"Tree after inserting {len(keys)} items: {print_pretty(tree)}")
        logger.debug(f"Tree size should be {len(keys) + expanded_leafs + 1} after inserting {len(keys)} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}")

        self.assertEqual(len(keys) + expanded_leafs + 1, tree.item_count(), f"Tree size should be {len(keys) + expanded_leafs + 1} after inserting {len(keys)} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}")

        # # Verify structure at each step
        # expected_real_keys = [j * 500 for j in range(1, len(keys)+1)]
        # real_keys = [entry.item.key for entry in tree.iter_real_entries()]
        # self.assertEqual(expected_real_keys, real_keys, 
        #                 f"Tree should contain keys {expected_real_keys} after {i} insertions")

        self.assertTrue(self.verify_subtree_sizes(tree))

    def test_duplicate_insertion_size(self):
        """Test size doesn't change when inserting duplicates"""
        tree = self.tree_k2
        
        # First insertion
        item = Item(1, "val")
        tree, inserted = tree.insert(item, rank=1)
        self.assertTrue(inserted)
        self.assertEqual(2, tree.node.get_node_item_count())
        
        # Duplicate insertion
        item_duplicate = Item(1, "new_val")
        tree, inserted = tree.insert(item_duplicate, rank=1)
        self.assertIsNone(tree.item_cnt,
                          "item_cnt should be None before triggering item_count()")
        self.assertEqual(2, tree.item_count(), 
                         "item_cnt should remain 2 after duplicate insertion")
        self.assertEqual(tree.item_cnt, 2)
    

    def test_large_tree_size(self):
        """Test size is correctly maintained in a larger tree with random insertions"""
        # TODO: Test with k=2 when all other tests are passing (Currently k=2 is not working due to recursion depth issues)
        tree = self.tree_k4
        # self.print_hash_info(key=15, k=2, num_levels=3)
        # exit()
        # Generate 1000 unique random keys between 1 and 1000000
        random.seed(42)  # For reproducibility
        unique_keys = random.sample(range(1, 100), 99)
        # Insert all items
        inserted_count = 0
        for i, key in enumerate(unique_keys, 1):
        # for i in range(1, 1000):
            item = Item(key, "val")
            tree, _ = tree.insert(item, rank=1)

            inserted_count += 1
            max_dim = tree.get_max_dim()
            dummy_cnt = self.get_dummy_count(tree)
            expanded_leafs = tree.get_expanded_leaf_count()
            expected_keys = [entry.item.key for entry in tree]
            expected_item_count = inserted_count + dummy_cnt

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tree after inserting {inserted_count} items: {print_pretty(tree)}")

            if inserted_count == 6:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"tree: {print_pretty(tree.node.set.node.right_subtree)}")
                    logger.debug(f"Tree structure at insertion {inserted_count}: {tree.node.set.node.right_subtree.print_structure()}")

            self.assertEqual(expected_item_count, tree.item_count(), f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}, tree: {print_pretty(tree)}, node_set: {print_pretty(tree.node.set)}, tree structure: {tree.print_structure()}")

        self.validate_tree(tree)
    
    def test_size_consistency_with_calculate_size(self):
        """Test that node.size matches the result of calculate_size()"""
        tree = self.tree_k4
        
        # Use subtests to track individual insertions
        # r = range(4, 11)  # Range of keys to insert
        for i, key in enumerate(range(4, 11), start=1):
            
            with self.subTest(f"Insert item with key {key * 500}"):
                tree, inserted = tree.insert(Item(key * 500, "val"), rank=1)
                self.assertTrue(inserted, f"Item with key {key * 500} should be inserted successfully")

                max_dim = tree.get_max_dim()
                expanded_leafs = tree.get_expanded_leaf_count()
                expected_keys = [entry.item.key for entry in tree]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Tree after inserting {i} items: {print_pretty(tree)}")
                    logger.debug(f"Tree size should be {i + expanded_leafs + 1} after inserting {i} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}")
                
                self.assertEqual(i + expanded_leafs + 1, tree.item_count(), f"Tree size should be {i + expanded_leafs + 1} after inserting {i} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}")

                # Verify structure at each step
                expected_real_keys = [j * 500 for j in range(4, key+1)]
                real_keys = [entry.item.key for entry in tree.iter_real_entries()]
                self.assertEqual(expected_real_keys, real_keys, 
                                f"Tree should contain keys {expected_real_keys} after {i} insertions")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tree after all insertions evaluated after individual subtest: {print_pretty(tree)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after all insertions evaluated after for loop: {print_pretty(tree)}")
    
    def test_rank_mismatch_size_handling(self):
        """Test that size is correctly maintained when handling rank mismatches"""
        tree = self.tree_k4
        tree, _ = tree.insert(Item(1000, "val"), rank=1)        
        
        # Insert an item with higher rank, triggering rank mismatch logic
        tree, _ = tree.insert(Item(2000, "val"), rank=3)
        self.assertEqual(3, tree.item_count())

        # Verify subtree sizes are consistent
        self.assertTrue(self.verify_subtree_sizes(tree))

    def test_split_tree(self):
        """Test that size is correctly maintained after splitting a tree"""
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
            tree, _ = tree.insert(item, rank=rank)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Initial tree before splits: {print_pretty(tree)}")
        with self.subTest("First split at 80"):
            left1, _, right1 = tree.split_inplace(80)
            self.assertIsNone(left1.item_cnt, 
                          "item_cnt should be None before triggering item_count()")
            self.assertEqual(left1.item_count(), 3,
                                "Item count should be 3 after inserting three items (1 dummy + 3 items)")
            self.assertEqual(left1.item_cnt, 3,
                            "item_cnt should be 3 after item_count() is called")
            
            self.assertIsNone(right1.item_cnt, 
                          "item_cnt should be None before triggering item_count()")
            self.assertEqual(right1.item_count(), 4,
                                "Item count should be 4 after inserting three items (1 dummy + 3 items)")
            self.assertEqual(right1.item_cnt, 4,
                            "item_cnt should be 4 after item_count() is called") 
        with self.subTest("Second split at 7 on left part"):
            # Second split on the left part
            left2, middle2, right2 = left1.split_inplace(7)
            self.assertIsNone(left2.item_cnt,
                          "item_cnt should be None before triggering item_count()")
            self.assertEqual(left2.item_count(), 2,
                                "Item count should be 2 after inserting two items (1 dummy + 1 item)")
            self.assertEqual(left2.item_cnt, 2,
                            "item_cnt should be 2 after item_count() is called")
            self.assertIsNone(right2.item_cnt,
                          "item_cnt should be None before triggering item_count()")
            self.assertEqual(right2.item_count(), 2,
                                "Item count should be 2 after inserting two items (1 dummy + 1 item)")
            self.assertEqual(right2.item_cnt, 2,
                            "item_cnt should be 2 after item_count() is called")

        with self.subTest("Third split at 212 on right part"):
            left3, _, right3 = right1.split_inplace(212)
            self.assertIsNone(left3.item_cnt,
                          "item_cnt should be None before triggering item_count()")
            self.assertEqual(left3.item_count(), 3,
                                "Item count should be 3 after inserting three items (1 dummy + 2 items)")
            self.assertEqual(left3.item_cnt, 3,
                            "item_cnt should be 3 after item_count() is called")
            self.assertIsNone(right3.item_cnt,
                          "item_cnt should be None before triggering item_count()")
            self.assertEqual(right3.item_count(), 2,
                                "Item count should be 2 after inserting two items (1 dummy + 1 item)")
            self.assertEqual(right3.item_cnt, 2,
                            "item_cnt should be 2 after item_count() is called")