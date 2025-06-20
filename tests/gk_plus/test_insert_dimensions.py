import sys
import os
import copy

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.utils import calc_rank_for_dim
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase as TreeTestCase

from tests.logconfig import logger

class TestInsertMultipleDimensions(TreeTestCase):
    ASSERTION_MESSAGE_TEMPLATE = (
        "TREE RESULT"
        "\nTREE: {tree}\n\n"
        "\nROOT SET: {root}\n"
    )
    
    # Initialize items once to avoid re-creating them in each test
    _KEYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ITEMS = {k: Item(k, "val") for k in _KEYS}
    
    def _run_insert_case_multi_dim(self, keys, rank_combo, insert_pair,
                        exp_keys, case_name, gnode_capacity=2, l_factor: float = 1.0):
        if len(rank_combo) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # build the tree once
        base_tree = create_gkplus_tree(K=gnode_capacity, dimension=1, l_factor=l_factor)

        for key, rank in zip(keys, rank_combo):
            base_tree, _ = base_tree.insert(self.ITEMS[key], rank)

        logger.debug(f"Tree after initial insertions: {print_pretty(base_tree)}")
        logger.debug(f"Root node: {print_pretty(base_tree.node.set)}")

        msg_head = (
            f"\n\nKey-Rank combo:\n"
            f"K: {keys}\n"
            f"R: {rank_combo}"
            f"\n\nTREE AFTER INITIAL INSERTIONS: {print_pretty(base_tree)}\n"
        )

        # deep-copy and split
        tree_copy = copy.deepcopy(base_tree)
        tree, _ = tree_copy.insert(self.ITEMS[insert_pair[0]], rank=insert_pair[1])

        logger.debug(f"Tree after inserting {insert_pair[0]} with rank {insert_pair[1]}: {print_pretty(tree)}")

        msg = f"\n\nInsert {case_name}" + msg_head
        msg += self.ASSERTION_MESSAGE_TEMPLATE.format(
            tree=print_pretty(tree),
            root=print_pretty(tree.node.set),
        )

        dummies_tree = self.get_dummies(tree)
        exp_tree_keys = sorted(dummies_tree + exp_keys)

        # assertions
        self.validate_tree(tree,  exp_tree_keys,  msg)

    def test_early_return_dim_2(self):
        """Test size is correctly maintained in a larger tree with random insertions"""
        k = 2
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [2, 2],
            [1, 2],
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=2)
        logger.debug(f"Keys: {keys}")
        insert_key_idx = 0
        logger.debug(f"Insert key: {keys[insert_key_idx]}")
        insert_item = self.create_item(keys[insert_key_idx])
        
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

        self.assertTrue(self.verify_subtree_sizes(tree))

    def test_specific_keys(self):
        """Test inserting specific keys into a tree with multiple dimensions"""
        k = 2
        tree = create_gkplus_tree(K=k)
        keys = [419, 533, 555, 719, 883, 120, 181, 389]
        ranks = [calc_rank_for_dim(key=key, k=k, dim=1) for key in keys]

        # Insert items into the tree
        for i, key in enumerate(keys):
            item = self.create_item(key)
            rank = ranks[i]
            tree, _ = tree.insert(item, rank=rank)        

        dum_keys = self.get_dummies(tree)
        logger.debug(f"Keys ({len(keys)}): {keys}")
        logger.debug(f"Dummies ({len(dum_keys)}): {dum_keys}")
        exp_keys = sorted(dum_keys + keys)
        logger.debug(f"Expected keys after insertions ({len(exp_keys)}): {exp_keys}")
        logger.debug(f"Tree after inserting items: {print_pretty(tree)}")
        logger.debug(f"Root node: {print_pretty(tree.node.set)}")

        self.validate_tree(tree, exp_keys)

    def test_insert_middle(self):
        """Test inserting specific keys into a tree with multiple dimensions"""
        keys  =  [1, 3, 7, 9, 11]
        ranks =  [1, 1, 2, 2, 2]
        
        insert_key = 5
        insert_rank = 2
        insert_pair = (insert_key, insert_rank)

        # array of tuples with (case_name, split_key)
        insert_cases = [(f"Insert key {insert_key} rank {insert_rank}",  insert_pair)]

        for case_name, insert_pair in insert_cases:
            exp_keys = sorted([k for k in keys] + [insert_key])  
            with self.subTest(case=case_name, insert_pair=insert_pair):
                self._run_insert_case_multi_dim(
                    keys, ranks,
                    insert_pair, exp_keys, case_name,
                    gnode_capacity=4, l_factor=1.0
                )