"""Tests for the gtree_stats_ function and Stats dataclass in tree_stats.py"""

import logging
import unittest

from gplus_trees.base import Entry, InternalItem, ItemData, LeafItem
from gplus_trees.factory import create_gplustree, make_gplustree_classes
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
from gplus_trees.gplus_tree_base import DUMMY_KEY, gtree_stats_
from gplus_trees.klist_base import KListBase
from gplus_trees.tree_stats import Stats, _get_capacity

logger = logging.getLogger(__name__)

# Default capacity for tests
K_VALUE = 4


def _make_item(key, value=None):
    """Create a LeafItem with the given key and value."""
    return LeafItem(ItemData(key, value if value is not None else f"val_{key}"))


def _make_replica(key):
    """Create an InternalItem (replica) with the given key."""
    return InternalItem(ItemData(key))


def _build_tree_with_items(keys, ranks):
    """Build a G+-tree by inserting items with given keys and ranks.

    Returns (tree, item_map) where *item_map* maps key → LeafItem.
    """
    tree = create_gplustree(K_VALUE)
    item_map = {}
    for k, r in zip(keys, ranks, strict=False):
        item = _make_item(k)
        item_map[k] = item
        tree.insert(item, r)
    return tree, item_map


# ─── Tests on empty / trivially-small trees ────────────────────────


class TestStatsEmptyTree(unittest.TestCase):
    """gtree_stats_ on an empty tree should return neutral / zero stats."""

    def test_none_input(self):
        stats = gtree_stats_(None)
        self._assert_empty(stats)

    def test_empty_tree(self):
        tree = create_gplustree(K_VALUE)
        stats = gtree_stats_(tree)
        self._assert_empty(stats)

    # ── helpers ──
    def _assert_empty(self, stats: Stats):
        self.assertEqual(stats.gnode_height, 0)
        self.assertEqual(stats.gnode_count, 0)
        self.assertEqual(stats.item_count, 0)
        self.assertEqual(stats.real_item_count, 0)
        self.assertEqual(stats.item_slot_count, 0)
        self.assertEqual(stats.leaf_count, 0)
        self.assertEqual(stats.rank, -1)
        self.assertTrue(stats.is_heap)
        self.assertIsNone(stats.least_item)
        self.assertIsNone(stats.greatest_item)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.internal_has_replicas)
        self.assertTrue(stats.internal_packed)
        self.assertTrue(stats.set_thresholds_met)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.all_leaf_values_present)
        self.assertTrue(stats.leaf_keys_in_order)


class TestStatsSingleItemTree(unittest.TestCase):
    """Tree containing exactly one real item (rank 1 → single leaf node)."""

    def setUp(self):
        self.tree, self.items = _build_tree_with_items([42], [1])

    def test_counts(self):
        stats = gtree_stats_(self.tree)
        self.assertEqual(stats.gnode_count, 1)
        self.assertEqual(stats.gnode_height, 1)
        self.assertEqual(stats.leaf_count, 1)
        self.assertEqual(stats.real_item_count, 1)
        # item_count includes the dummy item
        self.assertEqual(stats.item_count, 2)
        self.assertEqual(stats.rank, 1)

    def test_boolean_flags_all_true(self):
        stats = gtree_stats_(self.tree)
        self.assertTrue(stats.is_heap)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.internal_has_replicas)
        self.assertTrue(stats.internal_packed)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.all_leaf_values_present)
        self.assertTrue(stats.leaf_keys_in_order)

    def test_least_and_greatest(self):
        stats = gtree_stats_(self.tree)
        self.assertEqual(stats.least_item.key, DUMMY_KEY)
        self.assertEqual(stats.greatest_item.key, 42)


class TestStatsSingleHighRankItem(unittest.TestCase):
    """Tree with a single item inserted at rank > 1 (root + two leaves)."""

    def setUp(self):
        self.tree, self.items = _build_tree_with_items([10], [3])

    def test_structure(self):
        stats = gtree_stats_(self.tree)
        self.assertEqual(stats.rank, 3)
        self.assertEqual(stats.leaf_count, 2)
        self.assertEqual(stats.real_item_count, 1)
        self.assertGreaterEqual(stats.gnode_count, 3)  # root + 2 leaves

    def test_all_invariants_hold(self):
        stats = gtree_stats_(self.tree)
        self.assertTrue(stats.is_heap)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.internal_has_replicas)
        self.assertTrue(stats.internal_packed)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.all_leaf_values_present)
        self.assertTrue(stats.leaf_keys_in_order)


# ─── Multi-item trees with valid structure ─────────────────────────


class TestStatsMultipleItems(unittest.TestCase):
    """Build a tree with several items and verify aggregated stats."""

    def setUp(self):
        self.keys = [1, 2, 3, 4, 5]
        self.ranks = [2, 3, 2, 1, 4]
        self.tree, self.items = _build_tree_with_items(self.keys, self.ranks)

    def test_real_item_count(self):
        stats = gtree_stats_(self.tree)
        self.assertEqual(stats.real_item_count, len(self.keys))

    def test_leaf_keys_in_order(self):
        stats = gtree_stats_(self.tree)
        self.assertTrue(stats.leaf_keys_in_order)

    def test_all_invariants_hold(self):
        stats = gtree_stats_(self.tree)
        self.assertTrue(stats.is_heap)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.internal_has_replicas)
        self.assertTrue(stats.internal_packed)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.all_leaf_values_present)
        self.assertTrue(stats.leaf_keys_in_order)

    def test_least_and_greatest(self):
        stats = gtree_stats_(self.tree)
        # Least item is the dummy
        self.assertLessEqual(stats.least_item.key, min(self.keys))
        self.assertEqual(stats.greatest_item.key, max(self.keys))

    def test_rank_histogram_populated(self):
        hist = {}
        gtree_stats_(self.tree, rank_hist=hist)
        self.assertTrue(len(hist) > 0, "rank_hist should be populated")


class TestStatsLargerTree(unittest.TestCase):
    """Bigger tree to stress the recursive aggregation."""

    def setUp(self):
        self.keys = list(range(1, 21))
        self.ranks = [1, 3, 2, 1, 4, 1, 2, 3, 1, 2, 1, 5, 1, 2, 3, 1, 1, 2, 1, 6]
        self.tree, _ = _build_tree_with_items(self.keys, self.ranks)

    def test_real_item_count(self):
        stats = gtree_stats_(self.tree)
        self.assertEqual(stats.real_item_count, 20)

    def test_leaf_keys_in_order(self):
        stats = gtree_stats_(self.tree)
        self.assertTrue(stats.leaf_keys_in_order)

    def test_invariants_hold(self):
        stats = gtree_stats_(self.tree)
        self.assertTrue(stats.is_heap)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.internal_has_replicas)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.all_leaf_values_present)


# ─── Deliberate violations ─────────────────────────────────────────


class TestStatsHeapViolation(unittest.TestCase):
    """Manually break the heap property and verify stats detect it."""

    def setUp(self):
        self.keys = [1, 2, 3, 4, 5]
        self.ranks = [2, 3, 2, 1, 4]
        self.tree, _ = _build_tree_with_items(self.keys, self.ranks)

    def test_heap_root_lt_child_rank(self):
        """Root rank set below a child's rank → not a heap."""
        self.tree.node.rank = 2  # lower than max child rank
        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.is_heap, "Expected is_heap=False")

    def test_heap_internal_eq_child_rank(self):
        """An internal node whose rank equals a child's rank → not a heap."""
        root = self.tree.node
        root_entries = list(root.set)
        # Find an internal child to tamper with
        for entry in root_entries:
            if entry.left_subtree and not entry.left_subtree.is_empty():
                child_tree = entry.left_subtree
                if child_tree.node.right_subtree and not child_tree.node.right_subtree.is_empty():
                    # Set child rank equal to its grandchild's rank
                    grandchild = child_tree.node.right_subtree
                    child_tree.node.rank = grandchild.node.rank
                    break

        stats = gtree_stats_(self.tree)
        self.assertFalse(stats.is_heap, "Expected is_heap=False when internal rank equals child rank")


class TestStatsSearchTreeViolation(unittest.TestCase):
    """Break the search tree ordering property."""

    def setUp(self):
        GPlusTreeK, GPlusNodeK, KListK, _ = make_gplustree_classes(K_VALUE)
        self.GPlusTreeK = GPlusTreeK
        self.GPlusNodeK = GPlusNodeK
        self.KListK = KListK

    def test_swapped_first_item(self):
        """Put a large key in the first slot of a single-node tree."""
        item1 = _make_item(1)
        item3 = _make_item(3)
        tree = create_gplustree(K_VALUE)
        tree.insert(item1, 1)

        # Swap the DUMMY_ITEM slot with item3 → key order violated
        tree.node.set.head.entries[0].item = item3
        stats = gtree_stats_(tree)
        self.assertFalse(stats.is_search_tree, "Expected is_search_tree=False after swapping first key")


class TestStatsLinkedLeafViolation(unittest.TestCase):
    """Break the leaf-chain linkage and verify detection."""

    def setUp(self):
        self.GPlusTreeK, self.GPlusNodeK, self.KListK, _ = make_gplustree_classes(K_VALUE)

    def test_extra_leaf_via_next(self):
        """Attach an extra leaf node via `.next` on a single-node tree."""
        tree = create_gplustree(K_VALUE)
        item1 = _make_item(1)
        item3 = _make_item(3)
        item4 = _make_item(4)

        tree.insert(item1, 1)

        # Build an extra leaf and hook it up
        extra_set = self.KListK()
        extra_set, _, _ = extra_set.insert_entry(Entry(item3, None))
        extra_set, _, _ = extra_set.insert_entry(Entry(item4, None))
        extra_leaf = self.GPlusTreeK(self.GPlusNodeK(1, extra_set, None))
        tree.node.next = extra_leaf

        stats = gtree_stats_(tree)
        self.assertFalse(stats.linked_leaf_nodes, "Expected linked_leaf_nodes=False with rogue leaf attached")
        # The extra leaf items should be counted
        self.assertEqual(stats.real_item_count, 3)


class TestStatsInternalPackedViolation(unittest.TestCase):
    """Remove an entry from an internal node so it has fewer than 2 entries."""

    def setUp(self):
        self.keys = [1, 2, 3, 4, 5]
        self.ranks = [2, 3, 2, 1, 4]
        self.tree, _ = _build_tree_with_items(self.keys, self.ranks)

    def test_internal_packed_violation(self):
        """Delete an entry from an internal node → internal_packed=False."""
        root = self.tree.node
        root_entries = list(root.set)

        # Walk into a child subtree that is internal (rank >= 2)
        for entry in root_entries:
            if entry.left_subtree and not entry.left_subtree.is_empty():
                child = entry.left_subtree
                if child.node.rank >= 2 and child.node.right_subtree:
                    target = child.node.right_subtree
                    if not target.is_empty() and target.node.rank >= 2 and target.node.set.item_count() >= 2:
                        # Remove the last real entry
                        entries_list = list(target.node.set)
                        target.node.set.delete(entries_list[-1].item.key)
                        stats = gtree_stats_(self.tree)
                        self.assertFalse(
                            stats.internal_packed,
                            "Expected internal_packed=False after deleting entry from internal node",
                        )
                        return

        # If no suitable internal node found, skip
        self.skipTest("Could not find a suitable internal node to violate packing")


class TestStatsInternalReplicaViolation(unittest.TestCase):
    """Set value on a replica inside an internal node → internal_has_replicas=False."""

    def setUp(self):
        self.keys = [1, 2, 3, 4, 5]
        self.ranks = [2, 3, 2, 1, 4]
        self.tree, _ = _build_tree_with_items(self.keys, self.ranks)

    def test_non_replica_in_internal_node(self):
        """Replace an InternalItem replica with a LeafItem (which exposes a
        real value) on an internal (rank >= 2) node.

        ``InternalItem.value`` always returns ``None`` by design, so the
        only way to trigger the ``entry.item.value is not None`` check
        inside ``gtree_stats_`` is to swap the item for a ``LeafItem``.
        """

        def _tamper_internal(tree):
            """Walk the tree, find first suitable replica on a rank>=2 node
            and swap it for a LeafItem with a non-None value."""
            if tree is None or tree.is_empty():
                return False
            node = tree.node
            if node.rank >= 2:
                for entry in node.set:
                    if entry.item.key >= 0 and isinstance(entry.item, InternalItem):
                        # Replace with a LeafItem that has a real value
                        entry.item = LeafItem(entry.item._item)
                        entry.item.value = "not_a_replica"
                        return True
                # Recurse into left subtrees
                for entry in node.set:
                    if entry.left_subtree and _tamper_internal(entry.left_subtree):
                        return True
                if node.right_subtree and _tamper_internal(node.right_subtree):
                    return True
            return False

        if not _tamper_internal(self.tree):
            self.skipTest("Could not find an internal replica to tamper with")

        stats = gtree_stats_(self.tree)
        self.assertFalse(
            stats.internal_has_replicas,
            "Expected internal_has_replicas=False after replacing a replica with a LeafItem carrying a value",
        )


class TestStatsLeafKeysOrder(unittest.TestCase):
    """Verify leaf_keys_in_order detects out-of-order leaf keys."""

    def test_ordered_keys(self):
        tree, _ = _build_tree_with_items([1, 2, 3], [1, 1, 1])
        stats = gtree_stats_(tree)
        self.assertTrue(stats.leaf_keys_in_order)

    def test_disordered_keys_via_swap(self):
        """Swap two leaf entries' items so leaf keys are out of order."""
        tree, _items = _build_tree_with_items([1, 2, 3], [1, 1, 1])
        # Direct swap on the single leaf node
        entries = list(tree.node.set)
        real_entries = [e for e in entries if e.item.key >= 0]
        if len(real_entries) >= 2:
            # Swap keys by swapping Item objects
            real_entries[0].item, real_entries[-1].item = (real_entries[-1].item, real_entries[0].item)
            stats = gtree_stats_(tree)
            self.assertFalse(stats.leaf_keys_in_order, "Expected leaf_keys_in_order=False after swapping leaf keys")


class TestStatsAllLeafValuesPresent(unittest.TestCase):
    """Verify detection of missing leaf values."""

    def test_all_values_present(self):
        tree, _ = _build_tree_with_items([1, 2, 3], [1, 1, 1])
        stats = gtree_stats_(tree)
        self.assertTrue(stats.all_leaf_values_present)

    def test_missing_leaf_value(self):
        """Set a leaf item's value to None → all_leaf_values_present=False."""
        tree, _items = _build_tree_with_items([1, 2, 3], [1, 1, 1])
        # Null out the value of the first real item at the leaf
        for entry in tree.node.set:
            if entry.item.key >= 0:
                entry.item.value = None
                break

        stats = gtree_stats_(tree)
        self.assertFalse(
            stats.all_leaf_values_present, "Expected all_leaf_values_present=False after nulling a leaf value"
        )


# ─── set_thresholds_met violation ──────────────────────────────────


class TestStatsSetThresholdsMetViolation(unittest.TestCase):
    """Test ``set_thresholds_met`` using GKPlus-trees.

    ``set_thresholds_met`` is only meaningful for GKPlusTreeBase instances
    (see ``invariants.py`` and ``tests/utils.py``).  In a standard
    GPlusTree there is no recursive-instantiation mechanism to prevent
    KList overflow, so the flag is irrelevant there.

    For a GKPlusTree, ``threshold = l_factor * K``.
    ``set_thresholds_met`` becomes ``False`` when a KList node-set has
    ``item_count() > threshold``.
    """

    def _build_gkplus_tree(self, keys, ranks, K=K_VALUE, l_factor=1.0):
        """Build a GK+-tree by inserting items with given keys and ranks."""
        tree = create_gkplus_tree(K=K, dimension=1, l_factor=l_factor)
        item_map = {}
        for k, r in zip(keys, ranks, strict=False):
            item = _make_item(k)
            item_map[k] = item
            tree.insert(item, r)
        return tree, item_map

    def test_valid_gkplus_tree_thresholds_met(self):
        """A well-formed GKPlus-tree should have set_thresholds_met=True."""
        tree, _ = self._build_gkplus_tree([1, 2, 3, 4, 5], [2, 3, 2, 1, 4])
        stats = gtree_stats_(tree)
        self.assertTrue(stats.set_thresholds_met)

    def test_leaf_klist_exceeds_threshold(self):
        """Manually insert extra entries into a GKPlus leaf KList so
        item_count > l_factor * K → set_thresholds_met=False."""
        K = K_VALUE
        l_factor = 1.0
        threshold = int(l_factor * K)

        # Insert only rank-1 items so the tree stays a single leaf node
        # With K=4 and l_factor=1.0, threshold=4; DUMMY + 3 items = 4 = threshold.
        tree, _ = self._build_gkplus_tree([1, 2, 3], [1, 1, 1], K=K, l_factor=l_factor)

        # Adding one more entry pushes count > threshold
        extra_item = _make_item(99)
        tree.node.set.insert_entry(Entry(extra_item, None))

        self.assertGreater(tree.node.set.item_count(), threshold, "Precondition: KList count must exceed threshold")

        stats = gtree_stats_(tree)
        self.assertFalse(
            stats.set_thresholds_met, "Expected set_thresholds_met=False when GKPlus KList exceeds threshold"
        )

    def test_internal_klist_exceeds_threshold(self):
        """Overflow an internal node's KList in a GKPlus-tree
        → set_thresholds_met=False."""
        K = K_VALUE
        l_factor = 1.0
        tree, _ = self._build_gkplus_tree([1, 2, 3, 4, 5], [2, 3, 2, 1, 4], K=K, l_factor=l_factor)
        threshold = int(l_factor * K)

        def _overflow_internal(t):
            """Find an internal (rank >= 2) node and stuff extra entries."""
            if t is None or t.is_empty():
                return False
            node = t.node
            if node.rank >= 2:
                current = node.set.item_count()
                # Insert enough extra replicas to exceed threshold
                for i in range(threshold - current + 1):
                    extra_key = 10000 + i
                    replica = _make_replica(extra_key)
                    node.set.insert_entry(Entry(replica, None))
                if node.set.item_count() > threshold:
                    return True
            # Recurse into children
            for entry in node.set:
                if entry.left_subtree and _overflow_internal(entry.left_subtree):
                    return True
            return bool(node.right_subtree and _overflow_internal(node.right_subtree))

        if not _overflow_internal(tree):
            self.skipTest("Could not overflow any internal node's KList")

        stats = gtree_stats_(tree)
        self.assertFalse(
            stats.set_thresholds_met, "Expected set_thresholds_met=False after overflowing GKPlus internal KList"
        )

    def test_standard_gplus_tree_overflow_is_irrelevant(self):
        """In a standard GPlusTree, set_thresholds_met may fire False
        on overflow, but the invariant checkers intentionally skip it.

        This test documents that the flag IS NOT enforced for GPlusTree
        and that a well-formed GPlusTree has set_thresholds_met=True.
        """
        tree, _ = _build_tree_with_items([1, 2, 3, 4, 5], [2, 3, 2, 1, 4])
        stats = gtree_stats_(tree)
        # For a well-formed GPlusTree the flag should still be True
        self.assertTrue(stats.set_thresholds_met)


# ─── set_thresholds_met regression (random tree collapse) ─────────


class TestSetThresholdsMetRandomTreeRegression(unittest.TestCase):
    """Regression tests for ``set_thresholds_met`` invariant violations
    that appeared when ``check_and_collapse_tree`` failed to collapse
    undersized GKPlusTree inner sets back to KLists after unzip splits.

    Two root causes were identified:

    1. Empty GKPlusTree sets were returned as-is instead of being
       converted to empty KLists.  An empty GKPlusTree has
       ``item_count() == 0 <= threshold``, violating the invariant
       that every GKPlusTree inner set has ``item_count > threshold``.

    2. A rank-based fast-reject (``2^(rank-1) > threshold``) incorrectly
       skipped trees that had high rank but very few items — an artefact
       of unzip splitting a tree without lowering the root rank.

    These tests build random GK+-trees with deterministic seeds that
    previously triggered the violation.
    """

    def _build_random_tree(self, n, K, l_factor, seed):
        """Build a random GK+-tree replicating the stats script logic."""
        import random as _random

        import numpy as _np

        rng = _random.Random(seed)
        np_rng = _np.random.RandomState(seed)

        space = 1 << 24
        indices = rng.sample(range(1, space), k=n)
        p = 1.0 - (1.0 / K)
        ranks = np_rng.geometric(p, size=n)

        tree = create_gkplus_tree(K=K, l_factor=l_factor)
        for idx, rank in zip(indices, ranks, strict=False):
            item = LeafItem(ItemData(idx, "val"))
            tree.insert(item, int(rank))

        return tree

    def test_seed42_k4_n200_invariants(self):
        """Full invariant check on the original failure scenario (seed=42, K=4).

        n=200 is sufficient to trigger dimensional nesting with K=4
        (threshold=4) and reproduce the collapse bug.
        """
        from gplus_trees.invariants import assert_tree_invariants_raise

        tree = self._build_random_tree(n=200, K=4, l_factor=1.0, seed=42)
        stats = gtree_stats_(tree)
        # Should not raise
        assert_tree_invariants_raise(tree, stats)

    def test_multiple_seeds_k4(self):
        """Several seeds with K=4, n=200 — broader coverage."""
        for seed in (42, 43, 100):
            with self.subTest(seed=seed):
                tree = self._build_random_tree(n=200, K=4, l_factor=1.0, seed=seed)
                stats = gtree_stats_(tree)
                self.assertTrue(stats.set_thresholds_met, f"set_thresholds_met violated for seed={seed}")


# ─── _get_capacity helper ──────────────────────────────────────────


class TestGetCapacity(unittest.TestCase):
    """Test the _get_capacity helper function."""

    def test_returns_correct_K(self):
        for K in (2, 4, 8, 16):
            TreeK, _, _, _ = make_gplustree_classes(K)
            cap = _get_capacity(TreeK)
            self.assertEqual(cap, K, f"Expected _get_capacity to return {K}, got {cap}")


# ─── rank_hist parameter ──────────────────────────────────────────


class TestStatsRankHistogram(unittest.TestCase):
    """Verify the rank histogram is correctly populated."""

    def test_single_item_rank_1(self):
        tree, _ = _build_tree_with_items([5], [1])
        hist = {}
        gtree_stats_(tree, rank_hist=hist)
        self.assertIn(1, hist)
        # Rank 1 should count the dummy + the real item
        self.assertEqual(hist[1], 2)

    def test_multi_rank_tree(self):
        keys = [1, 2, 3, 4, 5]
        ranks = [2, 3, 2, 1, 4]
        tree, _ = _build_tree_with_items(keys, ranks)
        hist = {}
        gtree_stats_(tree, rank_hist=hist)
        total = sum(hist.values())
        stats = gtree_stats_(tree)
        self.assertEqual(total, stats.item_count, "Sum of rank_hist values should equal total item_count")


# ─── Edge cases ────────────────────────────────────────────────────


class TestStatsRankConsistency(unittest.TestCase):
    """Stats.rank should equal the root node's rank."""

    def test_root_rank(self):
        tree, _ = _build_tree_with_items([1, 2, 3, 4, 5], [2, 3, 2, 1, 4])
        stats = gtree_stats_(tree)
        self.assertEqual(stats.rank, tree.node.rank)


class TestStatsHeightLeafCountConsistency(unittest.TestCase):
    """Basic consistency between height, leaf_count, and gnode_count."""

    def test_single_leaf(self):
        tree, _ = _build_tree_with_items([1], [1])
        stats = gtree_stats_(tree)
        self.assertEqual(stats.gnode_height, 1)
        self.assertEqual(stats.leaf_count, 1)
        self.assertEqual(stats.gnode_count, 1)

    def test_multi_node(self):
        tree, _ = _build_tree_with_items([1, 2, 3, 4, 5], [2, 3, 2, 1, 4])
        stats = gtree_stats_(tree)
        self.assertGreater(stats.gnode_height, 1)
        self.assertGreater(stats.leaf_count, 1)
        self.assertGreater(
            stats.gnode_count, stats.leaf_count, "gnode_count should exceed leaf_count when internal nodes exist"
        )


class TestStatsItemSlotCount(unittest.TestCase):
    """item_slot_count should be >= item_count (slots include unused capacity)."""

    def test_slot_count(self):
        tree, _ = _build_tree_with_items([1, 2, 3, 4, 5], [2, 3, 2, 1, 4])
        stats = gtree_stats_(tree)
        self.assertGreaterEqual(stats.item_slot_count, stats.item_count, "item_slot_count must be >= item_count")


class TestStatsMultipleKValues(unittest.TestCase):
    """Verify stats work across different K capacities."""

    def _run_for_k(self, K):
        TreeK, _, _, _ = make_gplustree_classes(K)
        tree = TreeK()
        for i in range(1, 11):
            tree.insert(_make_item(i), (i % 3) + 1)
        stats = gtree_stats_(tree)
        self.assertEqual(stats.real_item_count, 10)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.is_heap)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.leaf_keys_in_order)
        self.assertTrue(stats.all_leaf_values_present)

    def test_k2(self):
        self._run_for_k(2)

    def test_k4(self):
        self._run_for_k(4)

    def test_k8(self):
        self._run_for_k(8)


class TestStatsIdempotent(unittest.TestCase):
    """Calling gtree_stats_ twice on the same tree should yield identical results."""

    def test_idempotent(self):
        tree, _ = _build_tree_with_items([1, 2, 3, 4, 5], [2, 3, 2, 1, 4])
        s1 = gtree_stats_(tree)
        s2 = gtree_stats_(tree)
        for field in Stats.__dataclass_fields__:
            self.assertEqual(
                getattr(s1, field), getattr(s2, field), f"Stats.{field} differs between two consecutive calls"
            )


# ─── Inner (higher-dimension) tree stats ───────────────────────────


class TestStatsInnerTreeBasic(unittest.TestCase):
    """Verify that inner_stats is populated when a GKPlus leaf's KList
    overflows and is replaced by a recursively instantiated GKPlusTree.

    With K=4 and l_factor=1.0, threshold=4.  Inserting 5 rank-1 items
    produces DUMMY + 5 items = 6 entries → exceeds threshold → inner
    tree at dimension 2 is created.
    """

    def _build_expanded_tree(self, n_items=5, K=4, l_factor=1.0):
        tree = create_gkplus_tree(K=K, dimension=1, l_factor=l_factor)
        for i in range(1, n_items + 1):
            tree.insert(_make_item(i * 10), rank=1)
        return tree

    def test_inner_stats_populated(self):
        """A tree whose leaf KList exceeded the threshold should have
        inner_stats after calling gtree_stats_."""
        tree = self._build_expanded_tree()
        # Precondition: node.set must be a GKPlusTreeBase
        self.assertIsInstance(
            tree.node.set, GKPlusTreeBase, "Expected node.set to be a recursively instantiated GKPlusTreeBase"
        )
        stats = gtree_stats_(tree)
        self.assertIsNotNone(
            stats.inner_stats, "Expected inner_stats to be populated when node_set is a GKPlusTreeBase"
        )
        self.assertGreater(len(stats.inner_stats), 0)

    def test_inner_stats_is_valid_stats(self):
        """Each element in inner_stats should be a Stats instance with
        sensible values."""
        tree = self._build_expanded_tree()
        stats = gtree_stats_(tree)
        for i, inner in enumerate(stats.inner_stats):
            self.assertIsInstance(inner, Stats, f"inner_stats[{i}] should be a Stats")
            self.assertGreater(inner.gnode_count, 0, f"inner_stats[{i}].gnode_count should be > 0")
            self.assertGreater(inner.item_count, 0, f"inner_stats[{i}].item_count should be > 0")

    def test_inner_tree_all_flags_true(self):
        """A well-formed expanded tree should have all inner invariant
        flags True."""
        tree = self._build_expanded_tree()
        stats = gtree_stats_(tree)
        for inner in stats.inner_stats:
            self.assertTrue(inner.is_heap)
            self.assertTrue(inner.is_search_tree)
            self.assertTrue(inner.internal_has_replicas)
            self.assertTrue(inner.internal_packed)
            self.assertTrue(inner.set_thresholds_met)
            self.assertTrue(inner.linked_leaf_nodes)
            self.assertTrue(inner.all_leaf_values_present)
            self.assertTrue(inner.leaf_keys_in_order)

    def test_outer_flags_all_true(self):
        """The outer (dimension 1) stats should also have all flags True
        when both dimensions are healthy."""
        tree = self._build_expanded_tree()
        stats = gtree_stats_(tree)
        self.assertTrue(stats.is_heap)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.internal_has_replicas)
        self.assertTrue(stats.set_thresholds_met)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.all_leaf_values_present)
        self.assertTrue(stats.leaf_keys_in_order)


class TestStatsInnerTreeViolationPropagation(unittest.TestCase):
    """Verify that an invariant violation inside a higher-dimension tree
    is propagated to the outer tree's stats.
    """

    def _build_expanded_tree(self, n_items=5, K=4, l_factor=1.0):
        tree = create_gkplus_tree(K=K, dimension=1, l_factor=l_factor)
        for i in range(1, n_items + 1):
            tree.insert(_make_item(i * 10), rank=1)
        return tree

    def test_inner_search_tree_violation_propagates(self):
        """Swap keys in the inner tree's leaf to break is_search_tree,
        and verify the outer stats also report is_search_tree=False."""
        tree = self._build_expanded_tree()
        inner_tree = tree.node.set
        self.assertIsInstance(inner_tree, GKPlusTreeBase)

        # Find the first leaf in the inner tree and swap two real entries
        for leaf in inner_tree.iter_leaf_nodes():
            entries = list(leaf.set)
            real = [e for e in entries if e.item.key >= 0]
            if len(real) >= 2:
                real[0].item, real[-1].item = real[-1].item, real[0].item
                break

        stats = gtree_stats_(tree)
        # Inner should report the violation
        self.assertIsNotNone(stats.inner_stats)
        inner_search_ok = all(s.is_search_tree for s in stats.inner_stats)
        self.assertFalse(inner_search_ok, "Expected at least one inner_stats to have is_search_tree=False")
        # Outer should inherit the violation
        self.assertFalse(stats.is_search_tree, "Inner search-tree violation must propagate to outer stats")

    def test_inner_leaf_value_violation_propagates(self):
        """Null a leaf value in the inner tree → all_leaf_values_present
        should propagate False to outer."""
        tree = self._build_expanded_tree()
        inner_tree = tree.node.set
        self.assertIsInstance(inner_tree, GKPlusTreeBase)

        # Null out a real item's value in the inner tree's leaf
        for leaf in inner_tree.iter_leaf_nodes():
            for entry in leaf.set:
                if entry.item.key >= 0:
                    entry.item.value = None
                    break
            break

        stats = gtree_stats_(tree)
        self.assertFalse(stats.all_leaf_values_present, "Inner all_leaf_values_present=False must propagate to outer")

    def test_inner_heap_violation_propagates(self):
        """Break the heap property in the inner tree and verify it
        propagates to the outer stats."""
        tree = self._build_expanded_tree()
        inner_tree = tree.node.set
        self.assertIsInstance(inner_tree, GKPlusTreeBase)

        # If the inner tree has rank > 1, we can break the heap
        if inner_tree.node.rank > 1:
            inner_tree.node.rank = 1  # lower root rank below children
            stats = gtree_stats_(tree)
            self.assertFalse(stats.is_heap, "Inner heap violation must propagate to outer stats")
        else:
            # Inner tree is a single leaf → can't violate heap
            self.skipTest("Inner tree has rank 1, cannot create heap violation")


class TestStatsNoInnerStatsForKList(unittest.TestCase):
    """Trees whose node sets remain KLists should have inner_stats=None."""

    def test_standard_gplus_tree_no_inner_stats(self):
        """Standard GPlusTree never has inner trees."""
        tree, _ = _build_tree_with_items([1, 2, 3, 4, 5], [2, 3, 2, 1, 4])
        stats = gtree_stats_(tree)
        self.assertIsNone(stats.inner_stats, "Standard GPlusTree should not have inner_stats")

    def test_small_gkplus_tree_no_inner_stats(self):
        """A GKPlus tree with few items (no KList overflow) should have
        inner_stats=None."""
        # K=4, l_factor=1.0: threshold=4. 3 items + dummy = 4 ≤ threshold
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1.0)
        for key in [10, 20, 30]:
            tree.insert(_make_item(key), rank=1)
        self.assertIsInstance(tree.node.set, KListBase, "Precondition: node.set should still be a KList")
        stats = gtree_stats_(tree)
        self.assertIsNone(stats.inner_stats, "Non-expanded GKPlus tree should not have inner_stats")

    def test_empty_tree_no_inner_stats(self):
        stats = gtree_stats_(None)
        self.assertIsNone(stats.inner_stats)


class TestStatsInnerStatsIdempotent(unittest.TestCase):
    """inner_stats should be identical across multiple calls."""

    def test_idempotent_with_inner_stats(self):
        tree = create_gkplus_tree(K=4, dimension=1, l_factor=1.0)
        for i in range(1, 6):
            tree.insert(_make_item(i * 10), rank=1)

        s1 = gtree_stats_(tree)
        s2 = gtree_stats_(tree)

        self.assertEqual(len(s1.inner_stats or []), len(s2.inner_stats or []))
        for i, (a, b) in enumerate(zip(s1.inner_stats or [], s2.inner_stats or [], strict=False)):
            for field in Stats.__dataclass_fields__:
                if field == "inner_stats":
                    continue  # Skip recursive comparison
                self.assertEqual(
                    getattr(a, field), getattr(b, field), f"inner_stats[{i}].{field} differs between consecutive calls"
                )


class TestStatsInnerMultiNodeTree(unittest.TestCase):
    """Test inner stats on a GKPlus tree with mixed ranks producing
    multiple gnodes, some with inner trees at various levels."""

    def test_multi_rank_gkplus_tree(self):
        """Insert items with varied ranks and enough volume to trigger
        expansion at some nodes.  All invariants should hold."""
        tree = create_gkplus_tree(K=2, dimension=1, l_factor=1.0)
        # K=2 means threshold=2; any leaf with > 2 entries triggers expansion
        keys = list(range(10, 110, 10))
        ranks = [2, 3, 2, 1, 4, 1, 2, 3, 1, 2]
        for k, r in zip(keys, ranks, strict=False):
            tree.insert(_make_item(k), r)

        stats = gtree_stats_(tree)
        self.assertTrue(stats.is_heap)
        self.assertTrue(stats.is_search_tree)
        self.assertTrue(stats.set_thresholds_met)
        self.assertTrue(stats.linked_leaf_nodes)
        self.assertTrue(stats.all_leaf_values_present)
        self.assertTrue(stats.leaf_keys_in_order)

        # With K=2 and 10 items, there should be inner trees
        if stats.inner_stats:
            for inner in stats.inner_stats:
                self.assertTrue(inner.is_heap)
                self.assertTrue(inner.is_search_tree)
                self.assertTrue(inner.set_thresholds_met)


class TestStatsInnerStatsField(unittest.TestCase):
    """Verify the inner_stats field default and presence in Stats."""

    def test_inner_stats_default_none(self):
        """Stats created manually should default inner_stats to None."""
        s = Stats(
            gnode_height=0,
            gnode_count=0,
            item_count=0,
            real_item_count=0,
            item_slot_count=0,
            leaf_count=0,
            rank=-1,
            is_heap=True,
            least_item=None,
            greatest_item=None,
            is_search_tree=True,
            internal_has_replicas=True,
            internal_packed=True,
            set_thresholds_met=True,
            linked_leaf_nodes=True,
            all_leaf_values_present=True,
            leaf_keys_in_order=True,
        )
        self.assertIsNone(s.inner_stats)

    def test_inner_stats_in_dataclass_fields(self):
        """inner_stats should be a recognized dataclass field."""
        self.assertIn("inner_stats", Stats.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
