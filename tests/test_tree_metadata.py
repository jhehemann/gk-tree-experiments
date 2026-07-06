"""Contract tests for the observer metadata interface.

The observer modules (``tree_stats``, ``display``, ``invariants``) rely on
a narrow metadata surface defined on :class:`GPlusTreeBase` and overridden
by :class:`GKPlusTreeBase`:

- ``dimension``
- ``dummy_item``
- ``set_conversion_threshold``
- ``inner_tree_of``
- ``tracked_size``

Every tree variant must satisfy the same contract, so the assertions live
in a shared mixin and run against both variants.
"""

import random
import unittest

from gplus_trees.base import DummyItem, ItemData, LeafItem
from gplus_trees.factory import create_gplustree
from gplus_trees.g_k_plus.factory import create_gkplus_tree, make_gkplustree_classes
from gplus_trees.gplus_tree_base import DUMMY_ITEM, get_dummy
from gplus_trees.klist_base import KListBase


def _make_item(key: int) -> LeafItem:
    return LeafItem(ItemData(key, f"val_{key}"))


def _iter_nodes(tree):
    """Yield every G-node of *tree* in the tree's own dimension.

    Children are only visited for internal nodes (right subtree present):
    a leaf's entries link back to shallower-dimension subtrees when the
    leaf belongs to an inner tree, so descending there would change
    dimension mid-traversal.
    """
    if tree is None or tree.is_empty():
        return
    node = tree.node
    yield node
    if node.right_subtree is not None:
        for entry in node.set:
            yield from _iter_nodes(entry.left_subtree)
        yield from _iter_nodes(node.right_subtree)


class MetadataContractMixin:
    """Assertions every tree variant's metadata interface must satisfy.

    Subclasses provide ``make_tree()`` (empty tree), ``fill_tree(tree)``
    (returns the list of inserted keys), and the ``expected_*`` values.
    """

    expected_dimension: int
    expected_threshold: float | None
    tracks_size: bool

    def make_tree(self):
        raise NotImplementedError

    def fill_tree(self, tree) -> list[int]:
        raise NotImplementedError

    # -- dimension ----------------------------------------------------
    def test_dimension_is_positive_int(self):
        tree = self.make_tree()
        self.assertIsInstance(tree.dimension, int)
        self.assertGreaterEqual(tree.dimension, 1)

    def test_dimension_matches_expected(self):
        self.assertEqual(self.make_tree().dimension, self.expected_dimension)

    # -- dummy_item ---------------------------------------------------
    def test_dummy_item_is_dummy_with_negative_dimension_key(self):
        tree = self.make_tree()
        self.assertIsInstance(tree.dummy_item, DummyItem)
        self.assertEqual(tree.dummy_item.key, -tree.dimension)

    def test_dummy_item_is_identity_stable(self):
        tree = self.make_tree()
        self.assertIs(tree.dummy_item, tree.dummy_item)
        self.assertIs(tree.dummy_item, self.make_tree().dummy_item)

    def test_leaf_items_include_dummy_key(self):
        """The filled tree's leaves contain the tree's own dummy key."""
        tree = self.make_tree()
        self.fill_tree(tree)
        leaf_keys = [entry.item.key for leaf in tree.iter_leaf_nodes() for entry in leaf.set]
        self.assertIn(tree.dummy_item.key, leaf_keys)

    # -- set_conversion_threshold --------------------------------------
    def test_set_conversion_threshold_matches_expected(self):
        self.assertEqual(self.make_tree().set_conversion_threshold, self.expected_threshold)

    # -- inner_tree_of --------------------------------------------------
    def test_inner_tree_of_klist_is_none(self):
        tree = self.make_tree()
        self.assertIsNone(tree.inner_tree_of(tree.SetClass()))

    def test_inner_tree_of_node_sets_is_none_or_deeper_tree(self):
        tree = self.make_tree()
        self.fill_tree(tree)

        def check(subject):
            for node in _iter_nodes(subject):
                inner = subject.inner_tree_of(node.set)
                if isinstance(node.set, KListBase):
                    self.assertIsNone(inner)
                else:
                    self.assertIs(inner, node.set)
                    self.assertEqual(inner.dimension, subject.dimension + 1)
                    check(inner)

        check(tree)

    # -- tracked_size ----------------------------------------------------
    def test_tracked_size_of_filled_tree(self):
        tree = self.make_tree()
        keys = self.fill_tree(tree)
        size = tree.tracked_size()
        if self.tracks_size:
            self.assertEqual(size, len(keys))
        else:
            self.assertIsNone(size)


class TestGPlusTreeMetadataContract(MetadataContractMixin, unittest.TestCase):
    """G⁺-trees: single-dimension, no set conversion, no size accounting."""

    expected_dimension = 1
    expected_threshold = None
    tracks_size = False

    def make_tree(self):
        return create_gplustree(K=4)

    def fill_tree(self, tree) -> list[int]:
        keys = list(range(1, 41))
        for i, key in enumerate(keys):
            tree.insert(_make_item(key), rank=(i % 3) + 1)
        return keys

    def test_dummy_item_is_the_shared_sentinel(self):
        self.assertIs(self.make_tree().dummy_item, DUMMY_ITEM)

    def test_leaves_contain_dummy_item_identically(self):
        """G⁺-trees store the exact ``dummy_item`` object in their leaves,
        which is what ``invariants.check_leaf_keys_and_values`` relies on."""
        tree = self.make_tree()
        self.fill_tree(tree)
        leaf_items = [entry.item for leaf in tree.iter_leaf_nodes() for entry in leaf.set]
        self.assertTrue(
            any(item is tree.dummy_item for item in leaf_items),
            "tree.dummy_item must appear (by identity) among the leaf items",
        )


class TestGKPlusTreeMetadataContract(MetadataContractMixin, unittest.TestCase):
    """Gᵏ⁺-trees: dimensioned, converting node sets, tracking their size."""

    K = 2
    L_FACTOR = 1.0
    expected_dimension = 1
    expected_threshold = 2.0  # l_factor * K
    tracks_size = True

    def make_tree(self):
        return create_gkplus_tree(K=self.K, l_factor=self.L_FACTOR)

    def fill_tree(self, tree) -> list[int]:
        # Deterministic fill (fixed seed / rank cycle) that drives several
        # node sets over the conversion threshold (l_factor * K = 2), so
        # the tree contains expanded, deeper-dimension inner sets.
        keys = random.Random(2080).sample(range(1, 100_000), 80)
        for i, key in enumerate(keys):
            tree.insert(_make_item(key), rank=1 + (i % 5))
        return keys

    def test_dummy_item_is_the_dimension_sentinel(self):
        self.assertIs(self.make_tree().dummy_item, get_dummy(dim=1))

    def test_filled_tree_has_an_expanded_inner_set(self):
        """The fill must actually exercise the inner-tree branch of the contract."""
        tree = self.make_tree()
        self.fill_tree(tree)
        inner_trees = [
            tree.inner_tree_of(node.set) for node in _iter_nodes(tree) if not isinstance(node.set, KListBase)
        ]
        self.assertTrue(inner_trees, "expected at least one expanded node set")

    def test_threshold_scales_with_l_factor(self):
        tree = create_gkplus_tree(K=4, l_factor=1.5)
        self.assertEqual(tree.set_conversion_threshold, 6.0)


class TestGKPlusTreeDeeperDimensionMetadata(unittest.TestCase):
    """A dimension-3 Gᵏ⁺-tree class reports its own dimension and dummy."""

    def make_tree(self):
        GKPlusTreeK, _, _, _ = make_gkplustree_classes(K=4, dimension=3)
        return GKPlusTreeK()

    def test_dimension(self):
        self.assertEqual(self.make_tree().dimension, 3)

    def test_dummy_item(self):
        tree = self.make_tree()
        self.assertIs(tree.dummy_item, get_dummy(dim=3))
        self.assertEqual(tree.dummy_item.key, -3)


if __name__ == "__main__":
    unittest.main()
