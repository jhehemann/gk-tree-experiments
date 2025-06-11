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
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from tests.gk_plus.base import TreeTestCase
from tests.utils import assert_tree_invariants_tc

from tests.logconfig import logger

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

class TestInsertMultipleDimensions(TreeTestCase):
    def test_early_return_dim_2(self):
        """Test size is correctly maintained in a larger tree with random insertions"""
        tree = self.tree_k2
        # self.print_hash_info(key=15, k=2, num_levels=3)
        # exit()
        
        rank_lists = [
            [2, 2],
            [1, 2],
        ]

        # rank_lists = [
        #     [1, 2, 2],
        #     [1, 2, 1],
        # ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=2)
        logger.debug(f"Keys: {keys}")
        item_map = { k: self.create_item(k) for k in keys}

        # rank_lists_insert_item =[[2], [2]]
        # insert_key = self.find_keys_for_rank_lists(rank_lists_insert_item, k=2)
        insert_key_idx = 0
        logger.debug(f"Insert key: {keys[insert_key_idx]}")
        insert_item = self.create_item(keys[insert_key_idx])

        # for i in range(len(keys)):
        #     # Skip the insert key
        #     if keys[i] == keys[insert_key_idx]:
        #         logger.debug(f"Skipping key {keys[i]} as it is the insert key")
        #         continue
        #     key = keys[i]
        #     item = item_map[key]
        #     rank = rank_lists[0][i]
        #     tree, _ = tree.insert(item, rank=rank)
        
        # Insert all items
        inserted_count = 0
        for i, key in enumerate(keys):
        # for i in range(1, 1000):
            if keys[i] == keys[insert_key_idx]:
                logger.debug(f"Skipping key {keys[i]} as it is the insert key")
                continue
            item = Item(key, "val")
            rank = rank_lists[0][i]
            tree, _ = tree.insert(item, rank=rank)
            inserted_count += 1
            max_dim = tree.get_max_dim()
            dummy_cnt = self.get_dummy_count(tree)
            expanded_leafs = tree.get_expanded_leaf_count()
            expected_keys = [entry.item.key for entry in tree]
            expected_item_count = inserted_count + dummy_cnt
            logger.debug(f"Tree after inserting {inserted_count} items: {print_pretty(tree)}")
            logger.debug(f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}")


            self.assertEqual(expected_item_count, tree.item_count(), f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs} (dummy count {dummy_cnt}). Leaf keys: {expected_keys}, tree: {print_pretty(tree)}, node_set: {print_pretty(tree.node.set)}, tree structure: {tree.print_structure()}")

        logger.debug(f"Tree after initial insertions: {print_pretty(tree)}")
        tree, _ = tree.insert(insert_item, rank=rank_lists[0][insert_key_idx])
        logger.debug(f"Tree after initial insertions + {insert_item.key}: {print_pretty(tree)}")

        max_dim = tree.get_max_dim()
        expanded_leafs = tree.get_expanded_leaf_count()
        dummy_cnt = self.get_dummy_count(tree)
        inserted_count += 1
        expected_keys = [entry.item.key for entry in tree]
        logger.debug(f"Tree after inserting {inserted_count} items: {print_pretty(tree)}")
        logger.debug(f"Tree structure: {tree.print_structure()}")
        expected_item_count = inserted_count + dummy_cnt

        logger.debug(f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}")


        self.assertEqual(expected_item_count, tree.item_count(), f"Tree size should be {expected_item_count} after inserting {inserted_count} items with max dimension {max_dim} and expanded leaf count {expanded_leafs} (dummy count {dummy_cnt}). Leaf keys: {expected_keys}, tree: {print_pretty(tree)}, node_set: {print_pretty(tree.node.set)}, tree structure: {tree.print_structure()}")


        text = " | ".join(str(e.item.key) for e in tree.node.set)
        logger.debug(f"Root set keys after all inserts: {text}")

        for e in tree:
            logger.debug(f"Entry: {e.item.key}, value: {e.item.value}, left_subtree: {e.left_subtree}")

            # self.assertEqual(i + 1, tree.item_count(), f"Tree should have size {i + 1} after inserting {key}")

        # Verify subtree sizes
        self.assertTrue(self.verify_subtree_sizes(tree))