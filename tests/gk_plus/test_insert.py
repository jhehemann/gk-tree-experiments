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
from gplus_trees.gplus_tree_base import print_pretty
from tests.test_base import GKPlusTreeTestCase
from tests.logconfig import logger
import logging

class TestGKPlusInsert(GKPlusTreeTestCase):
    # Class-level cache for frequently used items to avoid repeated creation
    # Extended range to cover all test cases in child classes
    ITEMS = {}

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
    
    def _get_insert_cases(self, keys: List[int], num_cases_per_position=1):
        """Helper method to generate multiple random insert cases based on keys.
        Optimized to reduce unnecessary computations.
        Args:
            keys: List of existing keys in sorted order
            num_cases_per_position: Number of cases to generate per position
            
        Returns:
            List[(case_name, insert_key)]: List of tuples with case name and insert key
        """
        if min(keys) <= 0:
            raise ValueError("All keys should be > 0 to enable splitting below them.")
        
        sorted_keys = sorted(keys)
        keys_set = set(keys)  # Use set for O(1) lookup instead of list
        insert_cases = []
        
        # 1. "before smallest" - keys before the smallest existing key
        min_key = min(sorted_keys)
        if min_key > 1:
            # Generate keys from max(1, min_key - num_cases_per_position) to min_key-1
            start = max(1, min_key - num_cases_per_position)
            for key in range(start, min_key):
                insert_cases.append(("before smallest", key))
        
        # 2. "after largest" - keys after the largest existing key
        max_key = max(sorted_keys)
        for i in range(num_cases_per_position):
            insert_key = max_key + 1 + i
            insert_cases.append(("after largest", insert_key))
        
        # 3. "middle gaps" - keys between existing keys (optimized)
        gap_count = 0
        for i in range(len(sorted_keys) - 1):
            if gap_count >= num_cases_per_position:
                break
            start_key = sorted_keys[i] + 1
            end_key = sorted_keys[i + 1]
            
            # Only process if there's a gap and we haven't reached the limit
            for key in range(start_key, min(end_key, start_key + num_cases_per_position)):
                if key not in keys_set:
                    insert_cases.append(("in middle gap", key))
                    gap_count += 1
                    if gap_count >= num_cases_per_position:
                        break
        
        # 6. Don't include existing keys to avoid duplicate insertion issues
        # (Duplicate insertion testing should be handled separately if needed)
        
        return insert_cases

class TestInternalMethodsWithEntryInsert(TestGKPlusInsert):
    """Test cases for internal methods insert_entry of GKPlusTreeBase."""
    # ITEMS inherited from parent class
    
    def _run_insert_case(self, keys, rank_combo, insert_entry, insert_rank, exp_keys, case_name, gnode_capacity=2, l_factor: float = 1.0):
        if len(rank_combo[0]) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")
        
        # Build the tree fresh each time to avoid state contamination
        tree = create_gkplus_tree(K=gnode_capacity, dimension=1, l_factor=l_factor)
        
        # Pair up keys with ranks
        pairs = list(zip(keys, rank_combo[0]))

        random.seed(42)  # For reproducibility in tests
        # Shuffle in place
        random.shuffle(pairs)

        for key, rank in pairs:
            if key not in self.ITEMS:
                self.ITEMS[key] = Item(key, "val")
            tree, _, _ = tree.insert(self.ITEMS[key], rank)

        msg_head = (
            f"\n\nKey-Rank combo:\n"
            f"K: {keys}\n"
            f"R: {rank_combo}"
            # f"\n\nTREE BEFORE INSERT: {print_pretty(tree)}\n"
        )

        insert_key = insert_entry.item.key
        
        new_tree, inserted, _ = tree.insert_entry(insert_entry, insert_rank)
        self.assertTrue(inserted, f"Inserted entry should be True for new key {insert_key}")
        
        msg = msg_head + f"\n\nInsert {case_name}: {insert_key}\n"
        # msg += f"Tree after insert: {print_pretty(new_tree)}\n" 
        self.assertIs(new_tree, tree, msg)

        # Validate tree
        dummies = self.get_dummies(new_tree)
        full_exp_keys = sorted(exp_keys + dummies)
        self.validate_tree(new_tree, full_exp_keys, msg)
        
        # Verify the entry was properly inserted
        retrieved_entry = new_tree.retrieve(insert_key)[0]
        self.assertIs(retrieved_entry, insert_entry,
                      f"Inserted entry should match the original entry: {insert_entry}")
        self.assertEqual(retrieved_entry, insert_entry,
                         f"Inserted entry should match the original entry: {insert_entry}")
        self.assertEqual(retrieved_entry.item.key, insert_key,
                         f"Inserted entry's key should match the insert key: {insert_key}")

    def test_insert_entry_empty_tree(self):
        with self.subTest("Insert rank 1"):
            tree = create_gkplus_tree(K=4)
            key, rank = 1, 1
            item = Item(key, f"val_{key}")
            entry = Entry(item, None)
            tree, _, _ = tree.insert_entry(entry, rank)
            inserted_entry = tree.retrieve(key)[0]
            self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
            self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
            self.assertIs(inserted_entry.item, item,
                            "Inserted entry's item should match the original item")
        with self.subTest("Insert rank > 1"):
            tree = create_gkplus_tree(K=4)
            key, rank = 1, 3
            item = Item(key, f"val_{key}")
            entry = Entry(item, None)
            tree, _, _ = tree.insert_entry(entry, rank)
            inserted_entry = tree.retrieve(key)[0]
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Base tree: {print_pretty(base_tree)}")

        insert_key = 4
        test_ranks = [1, 2, 3, 4]
        
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with None left subtree"):
                tree = copy.deepcopy(base_tree)
                item = Item(insert_key, f"val")
                entry = Entry(item, None)
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
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
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item,
                        "Inserted entry's item should match the original item")
                self.assertIsNotNone(inserted_entry.left_subtree,
                        "Inserted entry's left subtree should not be None")
                self.assertIs(inserted_entry.left_subtree, left_subtree)

    def test_insert_no_order(self):
        tree = create_gkplus_tree(K=4)
        keys = [1519, 3337, 7882, 9415, 9604]
        ranks = [1, 1, 1, 1, 2]
        for i, k in enumerate(keys):
            rank = ranks[i]
            tree.insert(Item(k, f"val_{k}"), rank=rank)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Base tree: {print_pretty(tree)}")

        expected_keys = [-2, -1] + keys
        self.validate_tree(tree, expected_keys)

    def test_insert_non_empty_no_extension(self):
        k = 2
        col_dim = 1
        rank_lists = [
            [1, 1, 1, 1, 2], # Dimension 1
            [1, 1, 1, 1, 2], # Dimension 2
            [1, 1, 1, 1, 2], # Dimension 3
            [1, 2, 1, 2, 1], # Dimension 4
            [1, 2, 1, 5, 1], # Dimension 5
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        tree = create_gkplus_tree(K=k, dimension=col_dim)
        for i, k in enumerate(keys):
            rank = rank_lists[i]
            tree.insert(Item(k, f"val_{k}"), rank=rank_lists[col_dim][i])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f" tree: {print_pretty(tree)}")
        self.validate_tree(tree)

        

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
            tree, _, _ = tree.insert(item, rank=rank)

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

    def test_insert_a(self):
        k = 2
        l_factor = 1.0
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [1, 2, 2],  # Dimension 1
            [1, 2, 1],  # Dimension 2
            [3, 1, 1],  # Dimension 3
        ]
        insert_rank = 2
        keys = self.find_keys_for_rank_lists(rank_lists, k=k, spacing=True)

        item_map = { k: self.create_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _, _ = tree.insert(item, rank=rank)

        insert_cases = [("before smallest", 66)]
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


    def test_failing_insert(self):
        """
        Test case for inserting an entry with a rank that exceeds the maximum allowed rank.
        This should raise a ValueError.
        """
        k = 4
        l_factor = 1.0
        tree = create_gkplus_tree(K=k, l_factor=l_factor)
        keys = [5460821, 16086335, 213272, 247004, 3041202, 575234, 12638198, 10383222, 4772687, 7317390, 9287529, 13824389, 5478587, 13768818, 6189859, 3817678, 15664487, 1008346, 14259099, 15081134, 12812636, 1591264, 12666734, 1949470, 8933946, 13688810, 9122911, 1225412, 1556988, 2589184, 14291852, 6113659, 8813389, 800201, 3862326, 267231, 7792995, 9015323, 16521524, 8066855, 3126492, 14606199, 4635690, 9253709, 8925209, 6839744, 4578265, 10734784, 9419945, 9890292, 5041851, 1804287, 3958286, 8177004, 1789510, 2095402, 1076454, 9373975, 14256775, 6304075, 5254915, 5319031, 13033595, 6829512, 5469475, 15061886, 10164120, 12331686, 9736860, 11666581, 1016913, 12721221, 14717825, 16402064, 4521660, 14688291, 7313804, 2368214, 1332039, 12961412, 8641813, 14096768, 11987075, 15538809, 16200977, 2886661, 14174391, 759185, 12959440, 7847742, 10358304, 10805639, 847040, 5753948, 9384891, 14591460, 10835, 6301154, 4731887, 6191909, 11226834, 13565805, 11027459, 16552137, 15798640, 12909217, 10175684, 3965614, 12256852, 7879374, 7876479, 16064655, 659305, 5393998, 14198126, 7168335, 5083056, 2603402, 11195521, 16046047, 3057873, 16291615, 5242727, 29903, 14935514, 7487849, 13334490, 15550935, 10100381, 2731642, 10461090, 3161399, 8720726, 4482169, 3255297, 6928521, 12246037, 11287974, 12627827, 15177440, 569995, 3865892, 16470588, 10167654, 762373, 3989761, 321941, 4885825, 7738968, 8859255, 3333919, 4245644, 7773770, 5890162, 11977530, 9947413, 5931158, 639338, 6150932, 2932038, 3281523, 11728674, 8602883, 7801016, 2117382, 2334944, 8171961, 13744565, 7902034, 10978236, 11434538, 3320947, 16766367, 16767706, 15107444, 8695902, 15979983, 12060904, 16136101, 7170975, 7942796, 15344614, 15438789, 7993140, 2346674, 7688789, 14889652, 9019231, 8260255, 15277387, 15597923, 6584319, 2809844, 6957671, 5258208, 13859784, 11192246, 11710795, 12723382, 6777766, 4372347, 5685851, 15068673, 11830763, 16676938, 7859700, 7090453, 7440336, 2310171, 6842642, 12520795, 8571664, 5435906, 6219662, 9303190, 4027075, 6480641, 6798664, 12256436, 11830882, 2221761, 4890433, 786839, 15315513, 8535198, 2239876, 8903371, 390293, 9974638, 5969951, 7227182, 12914990, 14793723, 6176398, 1368685, 13600690, 12169967, 11097352, 12948020, 6674111, 594923, 631974, 236657, 3085487, 8107994, 13382832, 9051754, 4674814, 13056957, 622083, 12008170, 3661486, 4001322, 1727895, 8410100, 10887961, 12372104, 12423111, 1476634, 12053597, 10068748, 12771125, 15476383, 10370395, 213777, 6058757, 15002813, 391514, 8263452, 4487145, 4024001, 6874389, 504901, 8321226, 2037461, 9548636, 13084323, 12149658, 9022733, 394073, 15721178, 2908499, 10382614, 10581855, 218073, 8188131, 5844086, 7314434, 9174963, 11010463, 11775811, 7682147, 12986244, 13468312, 5767218, 10466590, 1130640, 1199371, 14451370, 8201048, 5898044, 2909918, 16457691, 5860875, 6205805, 1738748, 13822497, 15160128, 7470298, 908806, 4289171, 16627084, 4983448, 8069803, 6187977, 15725921, 7235312, 4913230, 10422546, 4461305, 10540975, 6168203, 12690157, 4302897, 3148690, 13112494, 10930283, 14751201, 9085071, 8837656, 3651466, 2176706, 3226894, 6396724, 5151700, 2880930, 14254987, 11693739, 5371381, 14636839, 4698541, 9141604, 3288858, 9171166, 15995801, 8551912, 3952638, 3189847, 5854316, 12491691, 226371, 8180614, 10941814, 8819402, 5963754, 7798038, 14681553, 513315, 10611744, 16325618, 14716087, 15820917, 9857219]

        ranks = [1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 4, 2, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 3, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 3]
        
        for key, rank in zip(keys, ranks):
            if key not in self.ITEMS:
                self.ITEMS[key] = Item(key, f"val_{key}")
                tree, _, _ = tree.insert(self.ITEMS[key], rank=rank)
        self.validate_tree(tree)
       
    def test_many_rank_combinations_specific_keys(self):
        """
        Exhaustively test many rank-combo and insert_key combinations,
        computing the expected key lists on the fly.
        """
        k = 2
        l_factor = 1.0
        ranks = range(1, 4)
        insert_rank = 2

        num_keys = len(range(1, 4))
        # Precompute all possible per‐key rank‐tuples for each free dimension:
        
        dim1_choices = list(product(ranks, repeat=num_keys))
        dim2_choices = list(product(ranks, repeat=num_keys))
        fixed_dim = tuple([3, 1, 1, 2, 1, 1, 3, 1])

        total = len(dim1_choices) * len(dim2_choices) * 1

        # Cache computations to avoid redundant calculations
        insert_cases_cache = {}
        keys_cache = {}
        empty_subtree = create_gkplus_tree(K=k, l_factor=l_factor)
        entry_cache = {}

        for dim1, dim2 in tqdm(
            product(dim1_choices, dim2_choices),
            total=total,
            desc="Insert with specific key-rank combinations",
            unit="combo",
        ):
            rank_combo = [dim1, dim2, fixed_dim]

            # Cache keys computation
            rank_combo_tuple = tuple(tuple(dim) for dim in rank_combo)
            if rank_combo_tuple not in keys_cache:
                keys_cache[rank_combo_tuple] = self.find_keys_for_rank_lists(rank_combo, k=k, spacing=True)
            keys = keys_cache[rank_combo_tuple]
            
            # Cache insert cases for this key set
            keys_tuple = tuple(keys)
            if keys_tuple not in insert_cases_cache:
                insert_cases_cache[keys_tuple] = self._get_insert_cases(keys)
            insert_cases = insert_cases_cache[keys_tuple]

            with self.subTest(rank_combo=rank_combo, keys=keys):
                # logger.info(f"Testing rank combo: {rank_combo} with keys: {keys}")
                # for each possible insert_key (including non-existent)
                for case_name, insert_key in insert_cases:
                    # Cache Entry objects to avoid repeated creation
                    if insert_key not in entry_cache:
                        if insert_key in self.ITEMS:
                            item = self.ITEMS[insert_key]
                        else:
                            item = Item(insert_key, "val")
                            self.ITEMS[insert_key] = item
                            
                        entry_cache[insert_key] = Entry(item, empty_subtree)
                    insert_entry = entry_cache[insert_key]
                    
                    with self.subTest(insert_key=insert_key):
                        # Pre-compute expected keys once
                        exp_keys = keys + [insert_key]
                        case_name_str = f"insert key {insert_key} {case_name}"
                        self._run_insert_case(
                            keys,
                            rank_combo,
                            insert_entry,
                            insert_rank,
                            exp_keys,
                            case_name_str,
                            gnode_capacity=k,
                            l_factor=l_factor
                        )

    def test_insert_entry(self):
        base_tree = create_gkplus_tree(K=8, l_factor=1.0)
        keys = [1, 3, 5, 6, 7]
        ranks = [2, 3, 1, 2, 4]
        for i, k in enumerate(keys):
            rank = ranks[i]
            base_tree.insert(Item(k, f"val_{k}"), rank=rank)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Base tree: {print_pretty(base_tree)}")

        insert_key = 4
        test_ranks = [1, 2, 3, 4]
        
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with None left subtree"):
                tree = copy.deepcopy(base_tree)
                item = Item(insert_key, f"val")
                entry = Entry(item, None)
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
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
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item,
                        "Inserted entry's item should match the original item")
                self.assertIsNotNone(inserted_entry.left_subtree,
                        "Inserted entry's left subtree should not be None")
                self.assertIs(inserted_entry.left_subtree, left_subtree)
    