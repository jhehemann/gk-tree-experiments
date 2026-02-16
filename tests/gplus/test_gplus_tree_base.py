"""Tests for GPlusTreeBase helpers, hooks, and refactored methods.

Covers functionality that is either newly extracted or previously
untested:

- ``retrieve(with_next=False)``
- ``_descend_to_leaf`` / ``_find_in_leaf``
- ``iter_leaf_nodes``
- ``physical_height`` (tree-level)
- ``get_dummy`` / ``_get_dummy`` hook
- ``_make_empty_tree`` hook
- ``_SplitContext`` construction
"""

import unittest

from gplus_trees.base import (
    DummyItem,
    Entry,
    InternalItem,
    ItemData,
    LeafItem,
)
from gplus_trees.factory import create_gplustree, make_gplustree_classes
from gplus_trees.gplus_tree_base import (
    DUMMY_ITEM,
    DUMMY_KEY,
    GPlusTreeBase,
    _SplitContext,
    get_dummy,
)

from tests.test_base import GPlusTreeTestCase


# ── retrieve(with_next=False) ──────────────────────────────────────

class TestRetrieveWithNextFalse(GPlusTreeTestCase):
    """Exercise the ``with_next=False`` code path through GPlusTreeBase."""

    def setUp(self):
        super().setUp()
        self.items = {k: self.make_item(k, f"v{k}") for k in range(1, 8)}

    # -- empty / single-item trees --

    def test_empty_tree(self):
        found, nxt = self.tree.retrieve(1, with_next=False)
        self.assertIsNone(found)
        self.assertIsNone(nxt)

    def test_single_leaf_found(self):
        self.tree.insert(self.items[3], 1)
        found, nxt = self.tree.retrieve(3, with_next=False)
        self.assertIsNotNone(found)
        self.assertEqual(found.item.key, 3)
        self.assertIsNone(nxt, "with_next=False must return None for next")

    def test_single_leaf_not_found(self):
        self.tree.insert(self.items[3], 1)
        found, nxt = self.tree.retrieve(99, with_next=False)
        self.assertIsNone(found)
        self.assertIsNone(nxt)

    # -- multi-item leaf --

    def test_multi_leaf_all_keys(self):
        for k in [1, 3, 5]:
            self.tree.insert(self.items[k], 1)
        for k in [1, 3, 5]:
            found, nxt = self.tree.retrieve(k, with_next=False)
            self.assertIsNotNone(found, f"Key {k} must be found")
            self.assertEqual(found.item.key, k)
            self.assertIsNone(nxt)

    def test_multi_leaf_missing_key(self):
        for k in [1, 3, 5]:
            self.tree.insert(self.items[k], 1)
        found, nxt = self.tree.retrieve(4, with_next=False)
        self.assertIsNone(found)
        self.assertIsNone(nxt)

    # -- multi-level tree --

    def test_multi_level_tree(self):
        """Insert with varied ranks, then retrieve every key with with_next=False."""
        keys_ranks = [(1, 1), (2, 2), (3, 1), (5, 3), (7, 1)]
        for k, r in keys_ranks:
            self.tree.insert(self.items[k], r)
        for k, _ in keys_ranks:
            found, nxt = self.tree.retrieve(k, with_next=False)
            self.assertIsNotNone(found, f"Key {k} must be found")
            self.assertEqual(found.item.key, k)
            self.assertIsNone(nxt, f"with_next=False must return None next for key {k}")

    def test_with_next_true_vs_false_consistency(self):
        """For any found entry, with_next=True and with_next=False agree on 'found'."""
        keys_ranks = [(1, 4), (2, 1), (3, 3), (5, 1), (7, 2)]
        for k, r in keys_ranks:
            self.tree.insert(self.items[k], r)

        for k, _ in keys_ranks:
            f_true, _ = self.tree.retrieve(k, with_next=True)
            f_false, _ = self.tree.retrieve(k, with_next=False)
            self.assertIs(f_true, f_false,
                          f"Found entries for key {k} must be identical objects")


# ── _descend_to_leaf / _find_in_leaf ──────────────────────────────

class TestDescendToLeaf(GPlusTreeTestCase):
    """Direct tests for the split-out ``_descend_to_leaf`` method."""

    def test_single_leaf_node(self):
        item = self.make_item(5, "v5")
        self.tree.insert(item, 1)
        leaf = self.tree._descend_to_leaf(5)
        self.assertEqual(leaf.rank, 1)

    def test_multi_level_descends_to_rank_1(self):
        items = {k: self.make_item(k, f"v{k}") for k in [1, 3, 5, 7]}
        for k, r in [(1, 1), (3, 3), (5, 1), (7, 2)]:
            self.tree.insert(items[k], r)
        for k in [1, 3, 5, 7]:
            leaf = self.tree._descend_to_leaf(k)
            self.assertEqual(leaf.rank, 1,
                             f"_descend_to_leaf({k}) must reach a rank-1 node")

    def test_nonexistent_key_still_reaches_leaf(self):
        """Even for a missing key, descent should reach a rank-1 node."""
        items = {k: self.make_item(k, f"v{k}") for k in [1, 3, 5]}
        for k, r in [(1, 1), (3, 3), (5, 1)]:
            self.tree.insert(items[k], r)
        leaf = self.tree._descend_to_leaf(99)
        self.assertEqual(leaf.rank, 1)


class TestFindInLeaf(GPlusTreeTestCase):
    """Direct tests for the static ``_find_in_leaf`` method."""

    def _make_leaf_node(self, keys):
        """Build a standalone leaf node with the given keys."""
        tree = create_gplustree(4)
        for k in keys:
            tree.insert(self.make_item(k, f"v{k}"), 1)
        return tree.node  # rank-1 node

    def test_found_with_next(self):
        leaf = self._make_leaf_node([1, 3, 5])
        found, nxt = GPlusTreeBase._find_in_leaf(leaf, 3, with_next=True)
        self.assertEqual(found.item.key, 3)
        self.assertEqual(nxt.item.key, 5)

    def test_found_last_key_no_next_link(self):
        leaf = self._make_leaf_node([1, 3, 5])
        found, nxt = GPlusTreeBase._find_in_leaf(leaf, 5, with_next=True)
        self.assertEqual(found.item.key, 5)
        # No next pointer linked, so next_entry is None
        self.assertIsNone(nxt)

    def test_not_found_returns_none(self):
        leaf = self._make_leaf_node([1, 3, 5])
        found, nxt = GPlusTreeBase._find_in_leaf(leaf, 99, with_next=False)
        self.assertIsNone(found)
        self.assertIsNone(nxt)

    def test_with_next_false(self):
        leaf = self._make_leaf_node([1, 3, 5])
        found, nxt = GPlusTreeBase._find_in_leaf(leaf, 3, with_next=False)
        self.assertIsNotNone(found)
        self.assertIsNone(nxt)


# ── iter_leaf_nodes ────────────────────────────────────────────────

class TestIterLeafNodes(GPlusTreeTestCase):
    """Tests for ``GPlusTreeBase.iter_leaf_nodes``."""

    def test_empty_tree_yields_nothing(self):
        leaves = list(self.tree.iter_leaf_nodes())
        self.assertEqual(leaves, [])

    def test_single_leaf_yields_one(self):
        self.tree.insert(self.make_item(1, "v1"), 1)
        leaves = list(self.tree.iter_leaf_nodes())
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].rank, 1)

    def test_multi_level_yields_all_leaves_in_order(self):
        """All leaf keys collected via iter_leaf_nodes should be sorted."""
        keys_ranks = [(10, 1), (20, 2), (30, 1), (40, 3), (50, 1)]
        for k, r in keys_ranks:
            self.tree.insert(self.make_item(k, f"v{k}"), r)

        all_keys = []
        for leaf in self.tree.iter_leaf_nodes():
            self.assertEqual(leaf.rank, 1)
            for entry in leaf.set:
                if entry.item.key >= 0:
                    all_keys.append(entry.item.key)

        self.assertEqual(all_keys, sorted(all_keys),
                         "Leaf keys must be in sorted order")
        # All inserted keys should be present
        for k, _ in keys_ranks:
            self.assertIn(k, all_keys, f"Key {k} missing from iter_leaf_nodes")

    def test_leaf_next_pointers_are_consistent(self):
        """Each leaf's `next` pointer should match the next yielded leaf."""
        keys_ranks = [(1, 1), (2, 3), (3, 1), (4, 2), (5, 1)]
        for k, r in keys_ranks:
            self.tree.insert(self.make_item(k, f"v{k}"), r)

        leaves = list(self.tree.iter_leaf_nodes())
        for i, leaf in enumerate(leaves[:-1]):
            self.assertIsNotNone(leaf.next,
                                 f"Leaf {i} should have a next pointer")
            self.assertIs(leaf.next.node, leaves[i + 1],
                          f"Leaf {i}.next.node should be leaf {i+1}")
        # Last leaf should have no next
        if leaves:
            self.assertIsNone(leaves[-1].next,
                              "Last leaf should have no next pointer")


# ── physical_height (tree-level) ──────────────────────────────────

class TestPhysicalHeight(GPlusTreeTestCase):
    """Tree-level ``physical_height`` tests (KList-level tests are in test_klist.py)."""

    def test_single_leaf(self):
        self.tree.insert(self.make_item(1, "v1"), 1)
        h = self.tree.physical_height()
        self.assertGreaterEqual(h, 1, "Single leaf must have height >= 1")

    def test_grows_with_tree_depth(self):
        """A taller tree should have >= physical_height of a shorter one."""
        # rank-1-only tree (flat)
        flat = create_gplustree(4)
        for k in [1, 2, 3]:
            flat.insert(self.make_item(k, f"v{k}"), 1)

        # multi-rank tree (deeper)
        deep = create_gplustree(4)
        keys_ranks = [(1, 1), (2, 3), (3, 1), (4, 2), (5, 4)]
        for k, r in keys_ranks:
            deep.insert(self.make_item(k, f"v{k}"), r)

        self.assertGreaterEqual(deep.physical_height(), flat.physical_height(),
                                "Deeper tree should have >= physical_height")

    def test_consistent_across_calls(self):
        keys_ranks = [(1, 2), (3, 3), (5, 1)]
        for k, r in keys_ranks:
            self.tree.insert(self.make_item(k, f"v{k}"), r)
        h1 = self.tree.physical_height()
        h2 = self.tree.physical_height()
        self.assertEqual(h1, h2, "physical_height must be deterministic")


# ── get_dummy / _get_dummy hook ───────────────────────────────────

class TestGetDummy(unittest.TestCase):
    """Tests for the module-level ``get_dummy`` function and
    the ``_get_dummy`` hook on GPlusTreeBase."""

    def test_returns_dummy_item(self):
        d = get_dummy(dim=1)
        self.assertIsInstance(d, DummyItem)
        self.assertEqual(d.key, -1)

    def test_higher_dim_has_lower_key(self):
        d2 = get_dummy(dim=2)
        self.assertEqual(d2.key, -2)
        d5 = get_dummy(dim=5)
        self.assertEqual(d5.key, -5)

    def test_caching_returns_same_object(self):
        a = get_dummy(dim=3)
        b = get_dummy(dim=3)
        self.assertIs(a, b, "lru_cache should return the identical object")

    def test_different_dims_return_different_objects(self):
        a = get_dummy(dim=1)
        b = get_dummy(dim=2)
        self.assertIsNot(a, b)

    def test_base_tree_hook_returns_module_dummy(self):
        tree = create_gplustree(4)
        d = tree._get_dummy()
        self.assertIs(d, DUMMY_ITEM)


# ── _make_empty_tree hook ─────────────────────────────────────────

class TestMakeEmptyTree(unittest.TestCase):
    """Verify ``_make_empty_tree`` returns the correct concrete type."""

    def test_returns_same_class(self):
        TreeClass, _, _, _ = make_gplustree_classes(4)
        tree = TreeClass()
        empty = tree._make_empty_tree()
        self.assertIsInstance(empty, TreeClass)
        self.assertTrue(empty.is_empty())

    def test_two_calls_are_distinct_instances(self):
        tree = create_gplustree(4)
        a = tree._make_empty_tree()
        b = tree._make_empty_tree()
        self.assertIsNot(a, b)


# ── _SplitContext ─────────────────────────────────────────────────

class TestSplitContext(unittest.TestCase):
    """Basic construction and attribute access of ``_SplitContext``."""

    def test_slots_assigned(self):
        tree = create_gplustree(4)
        item = LeafItem(ItemData(key=10, value="v"))
        entry = Entry(item, None)
        ctx = _SplitContext(
            right_parent=tree,
            right_entry=entry,
            left_parent=tree,
            left_x_entry=None,
        )
        self.assertIs(ctx.right_parent, tree)
        self.assertIs(ctx.right_entry, entry)
        self.assertIs(ctx.left_parent, tree)
        self.assertIsNone(ctx.left_x_entry)

    def test_mutation(self):
        tree1 = create_gplustree(4)
        tree2 = create_gplustree(4)
        ctx = _SplitContext(
            right_parent=tree1, right_entry=None,
            left_parent=tree1, left_x_entry=None,
        )
        ctx.right_parent = tree2
        self.assertIs(ctx.right_parent, tree2)


# ── print_structure delegation ────────────────────────────────────

class TestPrintStructure(GPlusTreeTestCase):
    """Ensure ``print_structure`` returns a non-empty string."""

    def test_empty_tree(self):
        result = self.tree.print_structure()
        self.assertIsInstance(result, str)
        self.assertIn("Empty", result)

    def test_non_empty_tree(self):
        self.tree.insert(self.make_item(1, "v1"), 1)
        result = self.tree.print_structure()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ── Type-error edge cases ────────────────────────────────────────

class TestInsertTypeErrors(GPlusTreeTestCase):
    """Verify ``insert`` raises TypeError correctly."""

    def test_non_item_raises(self):
        with self.assertRaises(TypeError):
            self.tree.insert("not_an_item", 1)

    def test_non_int_rank_raises(self):
        item = self.make_item(1, "v")
        with self.assertRaises(TypeError):
            self.tree.insert(item, 0)

    def test_float_rank_raises(self):
        item = self.make_item(1, "v")
        with self.assertRaises(TypeError):
            self.tree.insert(item, 1.5)

    def test_negative_rank_raises(self):
        item = self.make_item(1, "v")
        with self.assertRaises(TypeError):
            self.tree.insert(item, -1)


class TestRetrieveTypeErrors(GPlusTreeTestCase):
    """Verify ``retrieve`` raises TypeError for non-int keys."""

    def test_string_key_raises(self):
        with self.assertRaises(TypeError):
            self.tree.retrieve("abc")

    def test_float_key_raises(self):
        with self.assertRaises(TypeError):
            self.tree.retrieve(3.14)


# ── InsertResult backward compatibility ──────────────────────────

class TestInsertResultUnpacking(GPlusTreeTestCase):
    """Ensure InsertResult can be unpacked as a plain tuple."""

    def test_three_element_unpack(self):
        item = self.make_item(1, "v1")
        tree, inserted, nxt = self.tree.insert(item, 1)
        self.assertIs(tree, self.tree)
        self.assertTrue(inserted)
        self.assertIsNone(nxt)

    def test_named_access(self):
        item = self.make_item(1, "v1")
        result = self.tree.insert(item, 1)
        self.assertIs(result.tree, self.tree)
        self.assertTrue(result.inserted)
        self.assertIsNone(result.next_entry)


if __name__ == "__main__":
    unittest.main()
