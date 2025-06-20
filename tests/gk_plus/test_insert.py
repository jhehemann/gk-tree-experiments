import sys
import os
import unittest
import random
from typing import List, TYPE_CHECKING
from itertools import product
from tqdm import tqdm
import copy
from statistics import median_low

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gplus_trees.base import Item, Entry
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase
from tests.logconfig import logger

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase



class TestGKPlusInsert(GKPlusTreeTestCase):
    
    def _find_non_existing_middle(self, keys: List[int]) -> int:
        """Find a non-existing middle key that can be inserted."""
        # use median_low to find a middle point
        if not keys:
            raise ValueError("The list of keys is empty.")
        if len(keys) == 1:
            return keys[0] + 1 if keys[0] > 0 else 1
        middle = median_low(keys)
        if middle in keys:
            # If the median is already in keys, find the next non-existing key
            for offset in range(1, len(keys) + 1):
                if middle - offset not in keys:
                    return middle - offset
                if middle + offset not in keys:
                    return middle + offset
        return middle

    # def _get_insert_cases(self, keys: List[int], rank_lists: List[List[int]]):
    #     """Helper method to generate insert cases based on keys."""
    #     if keys[0] == 0:
    #         raise ValueError("Smallest key should not be 0 to enable splitting below it.")

    #     insert_cases = []
    #     insert_positions = [
    #         "before smallest",
    #         "after smallest", 
    #         "before largest",
    #         "after largest",
    #         "non-existing middle"
    #     ]

    #     logger.debug(f"Rank lists for insert cases: {rank_lists}")
    #     # Get max ranks for each dimension
    #     # Exclude the last dimension as it is not relevant as an insert case (fixed to not extend)
    #     max_ranks = [max(rank_list) for rank_list in rank_lists[:-1]]  
        
    #     # Generate all possible rank combinations
    #     # For each max_rank, create range(1, max_rank + 2) to include a rank higher than max_rank
    #     logger.debug(f"Max ranks for each dimension: {max_ranks}")
    #     insert_ranks = [list(range(1, max_rank + 2)) for max_rank in max_ranks]
    #     logger.debug(f"Insert ranks: {insert_ranks}")
    #     all_rank_combinations = list(product(*insert_ranks))
    #     logger.debug(f"All rank combinations: {all_rank_combinations}")
    #     exit()
        
    #     # Define insert keys for each position
    #     insert_keys = {}
    #     insert_keys["before smallest"] = min(keys) - 1
    #     insert_keys["after smallest"] = min(keys) + 1  
    #     insert_keys["before largest"] = max(keys) - 1
    #     insert_keys["after largest"] = max(keys) + 1
        
    #     non_existing_middle = self._find_non_existing_middle(keys)
    #     if non_existing_middle is None:
    #         raise ValueError("No missing middle key that can be split at.")
    #     insert_keys["non-existing middle"] = non_existing_middle
        
    #     # Generate all combinations of positions and ranks
    #     for pos in insert_positions:
    #         insert_key = insert_keys[pos]
    #         for rank_combo in all_rank_combinations:
    #             insert_cases.append((pos, insert_key, rank_combo))

    #     return insert_cases
    
    def _get_insert_cases(self, keys: List[int], num_cases_per_position=10):
        """Helper method to generate multiple random insert cases based on keys.
        Randomness is applied to vary ranks in extended dimensions.
        Args:
            keys: List of existing keys in sorted order
            num_cases_per_position: Number of cases to generate per position
            
        Returns:
            List[(case_name, insert_key)]: List of tuples with case name and insert key
        """
        if min(keys) <= 0:
            raise ValueError("All keys should be > 0 to enable splitting below them.")
        
        sorted_keys = sorted(keys)
        insert_cases = []
        
        # 1. "before smallest" - keys before the smallest existing key
        min_key = min(sorted_keys)
        if min_key > 1:
            # Generate keys from 1 to min_key-1
            possible_keys = list(range(max(0, min_key - num_cases_per_position), min_key))
            for key in possible_keys[:num_cases_per_position]:
                insert_cases.append(("before smallest", key))
        
        # 2. "after smallest" - keys between smallest and its successor
        if len(sorted_keys) > 1:
            # Between first and second key
            start_key = sorted_keys[0] + 1
            end_key = sorted_keys[1]
            possible_keys = [k for k in range(start_key, end_key) if k not in keys]
            for key in possible_keys[:num_cases_per_position]:
                insert_cases.append(("after smallest", key))
        else:
            # Only one key exists, insert after it
            for i in range(num_cases_per_position):
                insert_key = sorted_keys[0] + 1 + i
                insert_cases.append(("after smallest", insert_key))
        
        # 3. "before largest" - keys between largest's predecessor and largest
        if len(sorted_keys) > 1:
            # Between second-to-last and last key
            start_key = sorted_keys[-2] + 1
            end_key = sorted_keys[-1]
            possible_keys = [k for k in range(start_key, end_key) if k not in keys]
            for key in possible_keys[:num_cases_per_position]:
                insert_cases.append(("before largest", key))
        
        # 4. "after largest" - keys after the largest existing key
        max_key = max(sorted_keys)
        for i in range(num_cases_per_position):
            insert_key = max_key + 1 + i
            insert_cases.append(("after largest", insert_key))
        
        # 5. "non-existing middle" - keys in gaps between consecutive keys
        middle_cases = []
        for i in range(len(sorted_keys) - 1):
            start_key = sorted_keys[i] + 1
            end_key = sorted_keys[i + 1]
            gap_keys = [k for k in range(start_key, end_key) if k not in keys]
            
            # Add some keys from this gap
            for key in gap_keys:
                if len(middle_cases) < num_cases_per_position:
                    middle_cases.append(key)
        
        # If we don't have enough middle keys from gaps, add some after the largest
        while len(middle_cases) < num_cases_per_position:
            additional_key = max_key + 1 + len(middle_cases)
            middle_cases.append(additional_key)
        
        for key in middle_cases[:num_cases_per_position]:
            insert_cases.append(("non-existing middle", key))
        
        return insert_cases

class TestInternalMethodsWithEntryInsert(TestGKPlusInsert):
    """Test cases for internal methods insert_entry of GKPlusTreeBase."""
    # Initialize items once to avoid re-creating them in each test
    # _KEYS = list(range(1, 1001))
    # ITEMS = {k: Item(k, "val") for k in _KEYS}
    
    def _run_insert_case(self, keys, rank_combo, insert_entry, insert_rank, exp_keys, case_name, gnode_capacity=2, l_factor: float = 1.0):
        if len(rank_combo[0]) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # build the tree once
        tree = create_gkplus_tree(K=gnode_capacity, dimension=1, l_factor=l_factor)
        for key, rank in zip(keys, rank_combo[0]):
            tree, _ = tree.insert(Item(key, "val"), rank)

        msg_head = (
            f"\n\nKey-Rank combo:\n"
            f"K: {keys}\n"
            f"R: {rank_combo}"
            f"\n\nTREE BEFORE INSERT: {print_pretty(tree)}\n"
        )

        new_tree, inserted = tree.insert_entry(insert_entry, insert_rank)
        self.assertTrue(inserted, "Inserted entry should be True")
        
        insert_key = insert_entry.item.key
        msg = msg_head + f"\n\nInsert {case_name}: {insert_key}\n"
        msg += f"Tree after insert: {print_pretty(new_tree)}\n" 
        self.assertIs(new_tree, tree, msg)

        dummies = self.get_dummies(new_tree)
        exp_keys = sorted(exp_keys + dummies)
        self.validate_tree(new_tree,  exp_keys,  msg)
        inserted_entry = new_tree.retrieve(insert_key).found_entry
        self.assertIs(inserted_entry, insert_entry,
                      f"Inserted entry should match the original entry: {insert_entry}")
        self.assertEqual(inserted_entry, insert_entry,
                         f"Inserted entry should match the original entry: {insert_entry}")
        self.assertEqual(inserted_entry.item.key, insert_key,
                         f"Inserted entry's key should match the insert key: {insert_key}")


    def test_insert_entry_empty_tree(self):
        with self.subTest("Insert rank 1"):
            tree = create_gkplus_tree(K=4)
            key, rank = 1, 1
            item = Item(key, f"val_{key}")
            entry = Entry(item, None)
            tree, _ = tree.insert_entry(entry, rank)
            inserted_entry = tree.retrieve(key).found_entry
            self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
            self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
            self.assertIs(inserted_entry.item, item,
                            "Inserted entry's item should match the original item")
        with self.subTest("Insert rank > 1"):
            tree = create_gkplus_tree(K=4)
            key, rank = 1, 3
            item = Item(key, f"val_{key}")
            entry = Entry(item, None)
            tree, _ = tree.insert_entry(entry, rank)
            inserted_entry = tree.retrieve(key).found_entry
            self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
            self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
            self.assertIs(inserted_entry.item, item,
                            "Inserted entry's item should match the original item")
    
    def test_insert_entry_non_empty_no_extension(self):
        base_tree = create_gkplus_tree(K=8)
        keys = [1, 3, 5, 6, 7]
        ranks = [2, 3, 1, 2, 4]
        for i, k in enumerate(keys):
            rank = ranks[i]
            base_tree.insert(Item(k, f"val_{k}"), rank=rank)
        logger.debug(f"Base tree: {print_pretty(base_tree)}")

        insert_key = 4
        test_ranks = [1, 2, 3, 4]
        
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with None left subtree"):
                tree = copy.deepcopy(base_tree)
                item = Item(insert_key, f"val")
                entry = Entry(item, None)
                tree, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key).found_entry
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item,
                        "Inserted entry's item should match the original item")
                self.assertIsNone(inserted_entry.left_subtree,
                        "Inserted entry's left subtree should be None")
        
        left_subtree = create_gkplus_tree(K=8)
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with existing left subtree (empty)"):
                tree = copy.deepcopy(base_tree)
                item = Item(insert_key, f"val")
                entry = Entry(item, left_subtree)
                tree, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key).found_entry
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item,
                        "Inserted entry's item should match the original item")
                self.assertIsNotNone(inserted_entry.left_subtree,
                        "Inserted entry's left subtree should not be None")
                self.assertIs(inserted_entry.left_subtree, left_subtree)

    def test_insert_entry_non_empty_with_leaf_extension(self):
        k = 2
        l_factor = 1.0
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [2, 2, 2],  # Dimension 1
            [1, 2, 1],  # Dimension 2
        ]
        insert_rank = 1
        keys = self.find_keys_for_rank_lists(rank_lists, k=k, spacing=True)
        # self.validate_key_ranks(keys, rank_lists)

        item_map = { k: self.create_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _ = tree.insert(item, rank=rank)

        insert_cases = self._get_insert_cases(keys)
        # for case_name, insert_key in insert_cases:
        #     logger.debug(f"  {case_name:20} | {insert_key:8}")
        for case_name, insert_key in insert_cases:
            insert_entry = Entry(Item(insert_key, value="val"), type(tree)(l_factor=l_factor))
            exp_keys = sorted(keys + [insert_key])
            with self.subTest(case=case_name, insert_key=insert_key):
                self._run_insert_case(
                    keys, rank_lists,
                    insert_entry, insert_rank,
                    exp_keys, case_name,
                    gnode_capacity=k, l_factor=l_factor
                )

    # def test_many_rank_combinations_specific_keys(self):
    #     """
    #     Exhaustively test many rank-combo and insert_key combinations,
    #     computing the expected key lists on the fly.
    #     """
    #     k = 4
    #     l_factor = 1.0
    #     ranks       = range(1,4)
    #     insert_rank = 1

    #     num_keys      = len(range(1,4))
    #     # Precompute all possible per‐key rank‐tuples for each free dimension:
    #     dim1_choices = list(product(ranks, repeat=num_keys))  # 3^3 = 27
    #     dim2_choices = list(product(ranks, repeat=num_keys))  # 3^3 = 27
    #     # dim3_choices = list(product(ranks, repeat=num_keys))  # 3^3 = 27
    #     fixed_dim   = tuple([4,1,1])                         # 1 choice

    #     total = len(dim1_choices) * len(dim2_choices) * 1  # 27 * 27 * 1 = 729

    #     for dim1, dim2 in tqdm(
    #         product(dim1_choices, dim2_choices),
    #         total=total,
    #         desc="Insert with specific key-rank combinations",
    #         unit="combo",
    #     ):
    #         # pack into a single "rank_combo" of shape (3 dimensions × 3 keys)
    #         rank_combo = [dim1, dim2, fixed_dim]
    #         keys = self.find_keys_for_rank_lists(rank_combo, k=k, spacing=True)
    #         insert_cases = self._get_insert_cases(keys)
    #         # logger.debug(f"Insert cases for keys {keys}: {insert_cases}")

    #         with self.subTest(rank_combo=rank_combo, keys=keys):
    #             # for each possible insert_key (including non-existent)
    #             for case_name, insert_key in insert_cases:
    #                 insert_entry = Entry(
    #                     Item(insert_key, "val"),
    #                     create_gkplus_tree(K=k, l_factor=l_factor)
    #                 )
    #                 with self.subTest(insert_key=insert_key):
    #                     exp_keys = sorted(keys + [insert_key])
    #                     case_name = f"insert key: {insert_key}"
    #                     self._run_insert_case(
    #                         keys,
    #                         rank_combo,
    #                         insert_entry,
    #                         insert_rank,
    #                         exp_keys,
    #                         case_name,
    #                         gnode_capacity=k,
    #                         l_factor=l_factor
    #                     )

    def test_insert_entry(self):
        base_tree = create_gkplus_tree(K=8, l_factor=1.0)
        keys = [1, 3, 5, 6, 7]
        ranks = [2, 3, 1, 2, 4]
        for i, k in enumerate(keys):
            rank = ranks[i]
            base_tree.insert(Item(k, f"val_{k}"), rank=rank)
        logger.debug(f"Base tree: {print_pretty(base_tree)}")

        insert_key = 4
        test_ranks = [1, 2, 3, 4]
        
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with None left subtree"):
                tree = copy.deepcopy(base_tree)
                item = Item(insert_key, f"val")
                entry = Entry(item, None)
                tree, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key).found_entry
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item,
                        "Inserted entry's item should match the original item")
                self.assertIsNone(inserted_entry.left_subtree,
                        "Inserted entry's left subtree should be None")
        
        left_subtree = type(base_tree)()
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with existing left subtree (empty)"):
                tree = copy.deepcopy(base_tree)
                item = Item(insert_key, f"val")
                entry = Entry(item, left_subtree)
                tree, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key).found_entry
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item,
                        "Inserted entry's item should match the original item")
                self.assertIsNotNone(inserted_entry.left_subtree,
                        "Inserted entry's left subtree should not be None")
                self.assertIs(inserted_entry.left_subtree, left_subtree)
    
    
    
            
