import copy
import logging
import random
from statistics import median_low
from typing import ClassVar

from gplus_trees.base import DummyItem, Entry, ItemData
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.utils import calc_ranks_multi_dims
from gplus_trees.gplus_tree_base import get_dummy, print_pretty
from gplus_trees.invariants import check_leaf_keys_and_values
from tests.test_base import GKPlusTreeTestCase

logger = logging.getLogger(__name__)


class TestGKPlusInsert(GKPlusTreeTestCase):
    # Class-level cache for frequently used items to avoid repeated creation
    # Extended range to cover all test cases in child classes
    ITEMS: ClassVar[dict] = {}

    def _find_non_existing_middle(self, keys: list[int]) -> int:
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

    def _get_insert_cases(self, keys: list[int], num_cases_per_position=1):
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

    def _run_insert_case(
        self, keys, rank_combo, insert_entry, insert_rank, exp_keys, case_name, gnode_capacity=2, l_factor: float = 1.0
    ):
        if len(rank_combo[0]) != len(keys):
            raise ValueError("Rank combo length must match number of keys.")

        # Build the tree fresh each time to avoid state contamination
        tree = create_gkplus_tree(K=gnode_capacity, dimension=1, l_factor=l_factor)

        # Pair up keys with ranks
        pairs = list(zip(keys, rank_combo[0], strict=False))

        random.seed(42)  # For reproducibility in tests
        # Shuffle in place
        random.shuffle(pairs)

        for key, rank in pairs:
            if key not in self.ITEMS:
                self.ITEMS[key] = self.make_item(key, "val")
            tree, _, _ = tree.insert(self.ITEMS[key], rank)

        msg_head = (
            f"\n\nKey-Rank combo:\nK: {keys}\nR: {rank_combo}"
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
        self.assertIs(retrieved_entry, insert_entry, f"Inserted entry should match the original entry: {insert_entry}")
        self.assertEqual(
            retrieved_entry, insert_entry, f"Inserted entry should match the original entry: {insert_entry}"
        )
        self.assertEqual(
            retrieved_entry.item.key, insert_key, f"Inserted entry's key should match the insert key: {insert_key}"
        )

    def test_insert_entry_empty_tree(self):
        with self.subTest("Insert rank 1"):
            tree = create_gkplus_tree(K=4)
            key, rank = 1, 1
            item = self.make_item(key, f"val_{key}")
            entry = Entry(item, None)
            tree, _, _ = tree.insert_entry(entry, rank)
            inserted_entry = tree.retrieve(key)[0]
            self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
            self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
            self.assertIs(inserted_entry.item, item, "Inserted entry's item should match the original item")
        with self.subTest("Insert rank > 1"):
            tree = create_gkplus_tree(K=4)
            key, rank = 1, 3
            item = self.make_item(key, f"val_{key}")
            entry = Entry(item, None)
            tree, _, _ = tree.insert_entry(entry, rank)
            inserted_entry = tree.retrieve(key)[0]
            self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
            self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
            self.assertIs(inserted_entry.item, item, "Inserted entry's item should match the original item")

    def test_insert_entry_non_empty_no_extension(self):
        base_tree = create_gkplus_tree(K=8)
        keys = [1, 3, 5, 6, 7]
        ranks = [2, 3, 1, 2, 4]
        for i, k in enumerate(keys):
            rank = ranks[i]
            base_tree.insert(self.make_item(k, f"val_{k}"), rank=rank)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Base tree: {print_pretty(base_tree)}")

        insert_key = 4
        test_ranks = [1, 2, 3, 4]

        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with None left subtree"):
                tree = copy.deepcopy(base_tree)
                item = self.make_item(insert_key, "val")
                entry = Entry(item, None)
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item, "Inserted entry's item should match the original item")
                self.assertIsNone(inserted_entry.left_subtree, "Inserted entry's left subtree should be None")

        left_subtree = create_gkplus_tree(K=8)
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with existing left subtree (empty)"):
                tree = copy.deepcopy(base_tree)
                item = self.make_item(insert_key, "val")
                entry = Entry(item, left_subtree)
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item, "Inserted entry's item should match the original item")
                self.assertIsNotNone(inserted_entry.left_subtree, "Inserted entry's left subtree should not be None")
                self.assertIs(inserted_entry.left_subtree, left_subtree)

    def test_insert_no_order(self):
        tree = create_gkplus_tree(K=4)
        keys = [1519, 3337, 7882, 9415, 9604]
        ranks = [1, 1, 1, 1, 2]
        for i, k in enumerate(keys):
            rank = ranks[i]
            tree.insert(self.make_item(k, f"val_{k}"), rank=rank)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Base tree: {print_pretty(tree)}")

        dummy_keys = self.get_dummies(tree)
        expected_keys = dummy_keys + keys
        self.validate_tree(tree, expected_keys)

    def test_insert_non_empty_no_extension(self):
        k = 2
        col_dim = 1
        rank_lists = [
            [1, 1, 1, 1, 2],  # Dimension 1
            [1, 1, 1, 1, 2],  # Dimension 2
            [1, 1, 1, 1, 2],  # Dimension 3
            [1, 2, 1, 2, 1],  # Dimension 4
            [1, 2, 1, 5, 1],  # Dimension 5
        ]
        keys = self.find_keys_for_rank_lists(rank_lists, k=k)
        tree = create_gkplus_tree(K=k, dimension=col_dim)
        for i, k in enumerate(keys):
            rank_lists[i]
            tree.insert(self.make_item(k, f"val_{k}"), rank=rank_lists[col_dim][i])
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

        item_map = {k: self.make_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _, _ = tree.insert(item, rank=rank)

        insert_cases = self._get_insert_cases(keys)
        # for case_name, insert_key in insert_cases:
        #     logger.debug(f"  {case_name:20} | {insert_key:8}")
        for case_name, insert_key in insert_cases:
            insert_entry = Entry(self.make_item(insert_key, value="val"), type(tree)(l_factor=l_factor))
            exp_keys = sorted([*keys, insert_key])
            with self.subTest(case=case_name, insert_key=insert_key):
                self._run_insert_case(
                    keys,
                    rank_lists,
                    insert_entry,
                    insert_rank,
                    exp_keys,
                    case_name,
                    gnode_capacity=k,
                    l_factor=l_factor,
                )

    def test_insert_entry_case_a(self):
        base_tree = create_gkplus_tree(K=8, l_factor=1.0)
        keys = [1, 3, 5, 6, 7]
        ranks = [2, 3, 1, 2, 4]
        for i, k in enumerate(keys):
            rank = ranks[i]
            base_tree.insert(self.make_item(k, f"val_{k}"), rank=rank)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Base tree: {print_pretty(base_tree)}")

        insert_key = 4
        test_ranks = [1, 2, 3, 4]

        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with None left subtree"):
                tree = copy.deepcopy(base_tree)
                item = self.make_item(insert_key, "val")
                entry = Entry(item, None)
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item, "Inserted entry's item should match the original item")
                self.assertIsNone(inserted_entry.left_subtree, "Inserted entry's left subtree should be None")

        left_subtree = type(base_tree)()
        for rank in test_ranks:
            with self.subTest(f"Insert rank {rank} with existing left subtree (empty)"):
                tree = copy.deepcopy(base_tree)
                item = self.make_item(insert_key, "val")
                entry = Entry(item, left_subtree)
                tree, _, _ = tree.insert_entry(entry, rank)
                inserted_entry = tree.retrieve(insert_key)[0]
                self.assertIsNotNone(inserted_entry, "Inserted entry should not be None")
                self.assertIs(inserted_entry, entry, "Inserted entry should match the original entry")
                self.assertIs(inserted_entry.item, item, "Inserted entry's item should match the original item")
                self.assertIsNotNone(inserted_entry.left_subtree, "Inserted entry's left subtree should not be None")
                self.assertIs(inserted_entry.left_subtree, left_subtree)

    def test_insert_case_a(self):
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

        item_map = {k: self.make_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _, _ = tree.insert(item, rank=rank)

        insert_cases = [("before smallest", 66)]
        for case_name, insert_key in insert_cases:
            insert_entry = Entry(self.make_item(insert_key, value="val"), type(tree)(l_factor=l_factor))
            exp_keys = sorted([*keys, insert_key])
            with self.subTest(case=case_name, insert_key=insert_key):
                self._run_insert_case(
                    keys,
                    rank_lists,
                    insert_entry,
                    insert_rank,
                    exp_keys,
                    case_name,
                    gnode_capacity=k,
                    l_factor=l_factor,
                )

    def test_insert_case_b(self):
        k = 2
        l_factor = 1.0
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [3, 2, 3, 2, 3],  # Dimension 1
            [2, 1, 2, 1, 1],  # Dimension 2
            [3, 1, 1, 2, 1, 1, 3, 1],  # Dimension 3
        ]

        insert_rank = 2
        keys = self.find_keys_for_rank_lists(rank_lists, k=k, spacing=True)

        item_map = {k: self.make_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _, _ = tree.insert(item, rank=rank)

        insert_cases = [("before smallest", 66)]
        for case_name, insert_key in insert_cases:
            insert_entry = Entry(self.make_item(insert_key, value="val"), type(tree)(l_factor=l_factor))
            exp_keys = sorted([*keys, insert_key])
            with self.subTest(case=case_name, insert_key=insert_key):
                self._run_insert_case(
                    keys,
                    rank_lists,
                    insert_entry,
                    insert_rank,
                    exp_keys,
                    case_name,
                    gnode_capacity=k,
                    l_factor=l_factor,
                )

    def test_insert_case_c(self):
        k = 2
        l_factor = 1.0
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [2, 1, 1, 3],  # Dimension 1
            [3, 1, 2, 1],  # Dimension 2
            [3, 1, 1, 2, 1, 1, 3, 1],  # Dimension 3
        ]

        insert_rank = 2
        keys = self.find_keys_for_rank_lists(rank_lists, k=k, spacing=True)

        item_map = {k: self.make_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _, _ = tree.insert(item, rank=rank)

        insert_cases = [("after largest", 704)]
        for case_name, insert_key in insert_cases:
            insert_entry = Entry(self.make_item(insert_key, value="val"), type(tree)(l_factor=l_factor))
            exp_keys = sorted([*keys, insert_key])
            with self.subTest(case=case_name, insert_key=insert_key):
                self._run_insert_case(
                    keys,
                    rank_lists,
                    insert_entry,
                    insert_rank,
                    exp_keys,
                    case_name,
                    gnode_capacity=k,
                    l_factor=l_factor,
                )

    def test_insert_case_d(self):
        k = 2
        l_factor = 1.0
        tree = create_gkplus_tree(K=k)
        rank_lists = [
            [1, 1, 1, 1],  # Dimension 1
            [1, 2, 1, 2],  # Dimension 2
            [3, 1, 1, 2, 1, 1, 3, 1],  # Dimension 3
        ]

        insert_rank = 2
        keys = self.find_keys_for_rank_lists(rank_lists, k=k, spacing=True)

        item_map = {k: self.make_item(k) for k in keys}
        for idx, item in enumerate(item_map.values()):
            rank = rank_lists[0][idx]
            tree, _, _ = tree.insert(item, rank=rank)

        insert_cases = [("after largest", 25)]
        for case_name, insert_key in insert_cases:
            insert_entry = Entry(self.make_item(insert_key, value="val"), type(tree)(l_factor=l_factor))
            exp_keys = sorted([*keys, insert_key])
            with self.subTest(case=case_name, insert_key=insert_key):
                self._run_insert_case(
                    keys,
                    rank_lists,
                    insert_entry,
                    insert_rank,
                    exp_keys,
                    case_name,
                    gnode_capacity=k,
                    l_factor=l_factor,
                )

    # def test_many_rank_combinations_specific_keys(self):
    #     """
    #     Exhaustively test many rank-combo and insert_key combinations,
    #     computing the expected key lists on the fly.
    #     """
    #     k = 2
    #     l_factor = 1.0
    #     ranks = range(1, 4)
    #     insert_rank = 2

    #     num_keys = len(range(1, 5))
    #     # Precompute all possible per-key rank-tuples for each free dimension:

    #     dim1_choices = list(product(ranks, repeat=num_keys))
    #     dim2_choices = list(product(ranks, repeat=num_keys))
    #     fixed_dim = tuple([3, 1, 1, 2, 1, 1, 3, 1])

    #     total = len(dim1_choices) * len(dim2_choices) * 1

    #     # Cache computations to avoid redundant calculations
    #     insert_cases_cache = {}
    #     keys_cache = {}
    #     empty_subtree = create_gkplus_tree(K=k, l_factor=l_factor)
    #     entry_cache = {}

    #     for dim1, dim2 in tqdm(
    #         product(dim1_choices, dim2_choices),
    #         total=total,
    #         desc="Insert with specific key-rank combinations",
    #         unit="combo",
    #     ):
    #         rank_combo = [dim1, dim2, fixed_dim]

    #         # Cache keys computation
    #         rank_combo_tuple = tuple(tuple(dim) for dim in rank_combo)
    #         if rank_combo_tuple not in keys_cache:
    #             keys_cache[rank_combo_tuple] = self.find_keys_for_rank_lists(rank_combo, k=k, spacing=True)
    #         keys = keys_cache[rank_combo_tuple]

    #         # Cache insert cases for this key set
    #         keys_tuple = tuple(keys)
    #         if keys_tuple not in insert_cases_cache:
    #             insert_cases_cache[keys_tuple] = self._get_insert_cases(keys)
    #         insert_cases = insert_cases_cache[keys_tuple]

    #         with self.subTest(rank_combo=rank_combo, keys=keys):
    #             # logger.info(f"Testing rank combo: {rank_combo} with keys: {keys}")
    #             # for each possible insert_key (including non-existent)
    #             for case_name, insert_key in insert_cases:
    #                 # Cache Entry objects to avoid repeated creation
    #                 if insert_key not in entry_cache:
    #                     if insert_key in self.ITEMS:
    #                         item = self.ITEMS[insert_key]
    #                     else:
    #                         item = self.make_item(insert_key, "val")
    #                         self.ITEMS[insert_key] = item

    #                     entry_cache[insert_key] = Entry(item, empty_subtree)
    #                 insert_entry = entry_cache[insert_key]

    #                 with self.subTest(insert_key=insert_key):
    #                     # Pre-compute expected keys once
    #                     exp_keys = keys + [insert_key]
    #                     case_name_str = f"insert key {insert_key} {case_name}"
    #                     self._run_insert_case(
    #                         keys,
    #                         rank_combo,
    #                         insert_entry,
    #                         insert_rank,
    #                         exp_keys,
    #                         case_name_str,
    #                         gnode_capacity=k,
    #                         l_factor=l_factor
    #                     )


class TestGKPlusNegativeKeyGuard(GKPlusTreeTestCase):
    """Verify GK+ insert paths reject negative keys reserved for dummies.

    Negative keys are reserved for internal dummy items (key ``-dim``, see
    :func:`gplus_trees.g_k_plus.g_k_plus_base.get_dummy`).  Without a guard,
    a negative user key silently collides with a dummy: the real item is
    swallowed and dropped from the real-item accounting
    (``real_item_count`` reports 0, ``retrieve`` returns the ``DummyItem``).
    Both public insert entry points must reject it — the inherited
    ``insert`` and the GK+-specific ``insert_entry``.
    """

    def test_insert_negative_key_rejected(self):
        # Inherited GPlusTreeBase.insert path (item + rank).
        tree = create_gkplus_tree(K=4)
        with self.assertRaises(ValueError):
            tree.insert(self.make_item(-1, "v"), 1)

    def test_insert_entry_negative_key_rejected(self):
        # GK+-specific insert_entry path (Entry + rank), key -1 == dim-1 dummy.
        tree = create_gkplus_tree(K=4)
        with self.assertRaises(ValueError):
            tree.insert_entry(Entry(self.make_item(-1, "v"), None), 1)

    def test_insert_entry_other_negative_key_rejected(self):
        # Any negative key is reserved (dummy key is -dim), not just -1.
        tree = create_gkplus_tree(K=4)
        with self.assertRaises(ValueError):
            tree.insert_entry(Entry(self.make_item(-5, "v"), None), 2)

    def test_zero_key_is_allowed(self):
        # 0 is the smallest valid real key and must remain insertable.
        tree = create_gkplus_tree(K=4)
        tree.insert_entry(Entry(self.make_item(0, "v"), None), 1)
        self.assertEqual(tree.real_item_count(), 1)
        found, _ = tree.retrieve(0)
        self.assertIsNotNone(found)
        self.assertEqual(found.item.value, "v")


class TestGKPlusLeafDummyIdentity(GKPlusTreeTestCase):
    """Leaf-identity contract for GK+-trees: every dummy circulating in leaf
    sets must be *the* cached sentinel of its dimension, because invariant
    checks compare by identity (``item is dummy``).

    Regression: ``get_dummy`` was cached with ``@functools.cache`` directly,
    which keys positional and keyword calls separately.  Leaf construction
    (positional calls) and :func:`check_leaf_keys_and_values` (keyword call)
    therefore held two distinct sentinels per dimension, and ``order_ok``
    was misreported as ``False`` for every GK+-tree.
    """

    def test_check_leaf_keys_reports_order_ok(self):
        # Small tree without leaf expansion: the only dummy is the tree's
        # dim-1 sentinel, so order_ok isolates the identity contract.
        tree = create_gkplus_tree(K=4)
        for key, rank in [(3, 1), (7, 2), (12, 1)]:
            tree, _, _ = tree.insert(self.make_item(key, f"v{key}"), rank)
        self.assertEqual(tree.get_expanded_leaf_count(), 0, "premise: no expanded leaf sets")

        keys, presence_ok, _, order_ok = check_leaf_keys_and_values(tree, [3, 7, 12])
        self.assertTrue(order_ok, f"order_ok misreported for sorted leaf keys {keys}")
        self.assertTrue(presence_ok)

    def test_expanded_leaf_dummies_are_canonical_sentinels(self):
        # Dimension expansion creates dummies for dims > 1; each must be the
        # cached sentinel of its dimension, no matter which call style the
        # constructing code path used.
        keys = [5, 2, 9, 1, 7, 3, 8, 4, 6, 10, 15, 12, 20]
        ranks = calc_ranks_multi_dims(keys, 4, dimensions=1)[0]
        tree = create_gkplus_tree(K=4)
        for key, rank in zip(keys, ranks, strict=True):
            tree, _, _ = tree.insert(self.make_item(key, f"v{key}"), rank)
        self.assertGreater(tree.get_expanded_leaf_count(), 0, "premise: expanded leaf sets exist")

        first_item = next(iter(next(iter(tree.iter_leaf_nodes())).set)).item
        self.assertIs(
            first_item,
            tree._get_dummy(),
            "premise of this fixture: stream opens with the tree's own sentinel "
            "(an expanded first leaf may legitimately open with a deeper dimension's sentinel)",
        )

        for leaf in tree.iter_leaf_nodes():
            for entry in leaf.set:
                if entry.item.key < 0:
                    self.assertIs(
                        entry.item,
                        get_dummy(-entry.item.key),
                        f"dummy with key {entry.item.key} is not the cached sentinel of its dimension",
                    )


class TestCheckLeafKeysExpansionAware(GKPlusTreeTestCase):
    """``check_leaf_keys_and_values`` must treat dimension sentinels in
    expanded leaf sets as sentinels, not as real keys.

    Regression: every entry after the first was treated as a real key, so
    each dimension sentinel (key -2, -3, …) broke the monotonicity check
    and leaked into the returned key list — ``order_ok``, ``presence_ok``
    and ``all_have_values`` were misreported as ``False`` for every
    GK+-tree with expanded leaf sets.
    """

    def _build_expanded_tree(self):
        keys = [5, 2, 9, 1, 7, 3, 8, 4, 6, 10, 15, 12, 20]
        ranks = calc_ranks_multi_dims(keys, 4, dimensions=1)[0]
        tree = create_gkplus_tree(K=4)
        for key, rank in zip(keys, ranks, strict=True):
            tree, _, _ = tree.insert(self.make_item(key, f"v{key}"), rank)
        self.assertGreater(tree.get_expanded_leaf_count(), 0, "premise: expanded leaf sets exist")
        return tree, keys

    def test_expanded_tree_reports_all_ok(self):
        tree, keys = self._build_expanded_tree()

        got, presence_ok, all_have_values, order_ok = check_leaf_keys_and_values(tree, keys)
        self.assertEqual(got, sorted(keys), "sentinels must not leak into the key list")
        self.assertTrue(presence_ok)
        self.assertTrue(all_have_values)
        self.assertTrue(order_ok)

    def test_forged_sentinel_flips_order_ok(self):
        # A sentinel is only legitimate as the cached per-dimension object;
        # an equal-keyed impostor must still be reported.
        tree, keys = self._build_expanded_tree()

        for leaf in tree.iter_leaf_nodes():
            for entry in leaf.set:
                if entry.item.key < -1:
                    entry.item = DummyItem(ItemData(key=entry.item.key))
                    break
            else:
                continue
            break

        _, _, _, order_ok = check_leaf_keys_and_values(tree, keys)
        self.assertFalse(order_ok, "forged sentinel must flip order_ok")

    def test_expanded_first_leaf_reports_all_ok(self):
        # Rank collisions expand the *first* leaf set, so the stream opens
        # with a deeper dimension's sentinel instead of the tree's own dummy.
        keys = list(range(1, 11))
        tree = create_gkplus_tree(K=2)
        for key in keys:
            tree, _, _ = tree.insert(self.make_item(key, f"v{key}"), 1)
        self.assertGreater(tree.get_expanded_leaf_count(), 0, "premise: expanded leaf sets exist")
        first_item = next(iter(next(iter(tree.iter_leaf_nodes())).set)).item
        self.assertIsNot(first_item, tree._get_dummy(), "premise: stream opens with a deeper sentinel")

        got, presence_ok, all_have_values, order_ok = check_leaf_keys_and_values(tree, keys)
        self.assertEqual(got, keys)
        self.assertTrue(presence_ok)
        self.assertTrue(all_have_values)
        self.assertTrue(order_ok)

    def test_disorder_across_sentinel_flips_order_ok(self):
        # Real-key order must be checked *across* sentinels: a real key that
        # regresses right after a mid-stream sentinel has to flip order_ok.
        tree = create_gkplus_tree(K=4)
        for key in range(1, 31):
            tree, _, _ = tree.insert(self.make_item(key, f"v{key}"), 1)

        entries = [entry for leaf in tree.iter_leaf_nodes() for entry in leaf.set]
        target = None
        last_real = None
        for i, entry in enumerate(entries):
            if entry.item.key >= 0:
                last_real = entry.item.key
            elif last_real is not None:
                target = next((e for e in entries[i + 1 :] if e.item.key >= 0), None)
                break
        self.assertIsNotNone(target, "premise: a sentinel sits between real keys")
        self.assertGreater(last_real, 0, "premise: forged key 0 regresses below the key before the sentinel")

        target.item = self.make_item(0, "forged")
        _, _, _, order_ok = check_leaf_keys_and_values(tree, None)
        self.assertFalse(order_ok, "disorder across a sentinel must flip order_ok")

    def test_mid_chain_leaf_opening_with_sentinel_reports_all_ok(self):
        # Rank-2 separators split the leaf chain; rank-1 collisions between
        # them expand mid-chain leaf sets, whose streams then *open* with
        # higher-dimension sentinels (structurally distinct from a single
        # expanded leaf: sentinel placement varies per leaf node).
        keys = list(range(1, 25))
        tree = create_gkplus_tree(K=2)
        for key in keys:
            rank = 2 if key in (8, 16) else 1
            tree, _, _ = tree.insert(self.make_item(key, f"v{key}"), rank)

        leaves = list(tree.iter_leaf_nodes())
        self.assertGreater(len(leaves), 1, "premise: leaf chain has multiple nodes")
        mid_chain_sentinel_openers = [leaf for leaf in leaves[1:] if next(iter(leaf.set)).item.key < 0]
        self.assertTrue(mid_chain_sentinel_openers, "premise: a mid-chain leaf set opens with a sentinel")

        got, presence_ok, all_have_values, order_ok = check_leaf_keys_and_values(tree, keys)
        self.assertEqual(got, keys)
        self.assertTrue(presence_ok)
        self.assertTrue(all_have_values)
        self.assertTrue(order_ok)
