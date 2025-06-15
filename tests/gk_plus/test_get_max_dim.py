import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.gk_plus.base import TreeTestCase as GKPlusTreeTestCase
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from tests.logconfig import logger


class TestGKPlusDimensionTracking(GKPlusTreeTestCase):    
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
