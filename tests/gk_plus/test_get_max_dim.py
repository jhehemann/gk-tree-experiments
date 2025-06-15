import sys
import os
import unittest


# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
from gplus_trees.base import Item
from tests.gk_plus.base import TreeTestCase as GKPlusTreeTestCase
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.gplus_tree_base import gtree_stats_, print_pretty
from tests.utils import assert_tree_invariants_tc
from gplus_trees.base import calculate_group_size, count_trailing_zero_bits
from gplus_trees.g_k_plus.utils import calc_rank_from_digest, calc_rank, calc_ranks_for_multiple_dimensions
from tests.logconfig import logger


class TestGKPlusDimensionTracking(GKPlusTreeTestCase):

    # def setUp(self):
        # Create dummy ranks for each of the dimensions
        # dummies = [get_dummy(i).key for i in range(1, 5)]
        # # Convert dummy key to positive integer to calculate a rank from its hash
        # pos_dummies = [abs(dummies[i]) for i in range(len(dummies))]
        # self.dummy_ranks = calc_ranks_for_multiple_dimensions(pos_dummies, k, dimensions=5)
        
        # print(f"Calculated ranks for dummies {dummies}:")
        # for dim_idx, ranks in enumerate(self.dummy_ranks):
        #     print(f"  Dim {dim_idx+1}: {ranks}")
        
    
    # def validate_keys(self, keys, rank_lists, k):
    #     """
    #     Validate whether each key in the list produces the correct ranks through repeated SHA-256 hashing.
        
    #     Parameters:
    #         keys (List[int]): List of integer keys to validate.
    #         rank_lists (List[List[int]]): List of rank lists per hashing level.
    #         k (int): The group size parameter (must be power of 2).
            
    #     Returns:
    #         bool: True if all keys match the expected rank sequences, False otherwise.
    #     """
    #     group_size = calculate_group_size(k)
    #     num_hashes = len(rank_lists)
        
    #     self.assertTrue(num_hashes > 0, "Rank lists must have at least one level")
    #     self.assertTrue(len(rank_lists[0]) == len(keys), "Rank lists and keys must have the same length")
        
    #     for i, key in enumerate(keys):
    #         current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
    #         for level in range(num_hashes):
    #             expected_rank = rank_lists[level][i]
    #             actual_rank = calc_rank_from_digest(current_hash, group_size)
    #             self.assertEqual(actual_rank, expected_rank, f"Rank mismatch for key {key} at level {level}")
    #             current_hash = hashlib.sha256(current_hash).digest()

    #     return True
    
    def test_tree_k4(self):
        """Test tree with K=4"""
        k = 4
        tree = create_gkplus_tree(K=k)
        
        rank_lists = [
            [1, 1, 1, 1, 1],  # Dimension 1
            [1, 1, 2, 1, 1],  # Dimension 2
        ]
        
        keys = self.find_keys_for_rank_lists(rank_lists, k)
        logger.debug(f"Keys: {keys}")
        self.validate_key_ranks(keys, rank_lists, k)
        
        with self.subTest(f"Insert items below expansion threshold 4"):
            # Insert the first 3 items
            for i in range(3):
                key = keys[i]
                rank = rank_lists[0][i]
                item = self.create_item(key)
                tree, inserted = tree.insert(item, rank=rank)
                self.assertTrue(inserted, 
                                f"Item with key {key} should be inserted successfully")
            max_dim = tree.get_max_dim()
            expanded_leafs = tree.get_expanded_leaf_count()
            expected_keys = [entry.item.key for entry in tree]
            self.assertEqual(tree.DIM, 1)
            self.assertEqual(tree.item_count(), 4, 
                             f"Tree size should be 4 after inserting 3 items with max dimension {max_dim} and expanded leaf count {expanded_leafs}. Leaf keys: {expected_keys}")
            # Verify structure at each step
            expected_real_keys = [key for key in keys[:3]]
            real_keys = [entry.item.key for entry in tree.iter_real_entries()]

            self.assertEqual(expected_real_keys, real_keys, 
                            f"Tree should contain keys {expected_real_keys} after {i} insertions")

            self.assertEqual(tree.get_max_dim(), 1, "Max dimension should be 1 after inserting 4 items")


        with self.subTest(f"Insert item to trigger expansion"):
            # Insert one more item to trigger expansion
            tree, inserted = tree.insert(self.create_item(keys[3]), rank=rank_lists[0][3])
            self.assertTrue(inserted,
                            f"Item with key {keys[3]} should be inserted successfully")
            max_dim = tree.get_max_dim()
            self.assertEqual(max_dim, 2,
                             "Maximum tree dimension should increase to 2 after expansion")
            expanded_leafs = tree.get_expanded_leaf_count()
            expected_keys = [entry.item.key for entry in tree]
            self.assertEqual(tree.item_count(), 6, 
                             f"Tree size should be 6. 4 Real items + 2 dummy items for each of the 2 dimensions.")

            # Verify structure after expansion
            expected_real_keys = keys[:4]  # First 4 keys should be present
            real_keys = [entry.item.key for entry in tree.iter_real_entries()]
            self.assertEqual(expected_real_keys, real_keys,
                            f"Tree should contain keys {expected_real_keys} after expansion")

            self.assertEqual(tree.get_max_dim(), 2, 
                             "Max dimension should be 2 after inserting 5 items")
