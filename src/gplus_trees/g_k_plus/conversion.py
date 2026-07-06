"""KList / GKPlusTree conversion logic for GK+-trees.

Provides :class:`GKPlusConversionMixin`, a mixin class that adds
``_convert_node_set`` and ``_check_and_convert_set`` to
:class:`GKPlusTreeBase`.

Conversions are triggered when a KList exceeds ``k · l_factor``
threshold-counting items (expand) or a GKPlusTree falls to or below
that threshold (collapse).  Only real items and the owning tree's own
dummy count toward the threshold; dummies carried in from other
dimensions do not (see ``bulk_create._threshold_item_count``).
These conversions are what create recursive dimensional nesting:
an expanded KList becomes a GK+-tree of the next dimension.

Everything here is an internal seam of the growth path: callers outside
the ``g_k_plus`` package reach conversions only implicitly, through
``GKPlusTreeBase.insert_entry`` (and the other tree-level operations
``unzip`` / ``zip``).

The two package-internal entry points split by cache ownership:

- ``_convert_node_set(node)`` converts a set attached to a node of a
  live tree.  The receiver must be the tree owning *node*; its cached
  counts are invalidated internally, so callers carry no follow-up
  contract.
- ``_check_and_convert_set(set)`` is the pure set→set function for
  free-standing sets (split halves) that are placed into freshly
  constructed trees, whose caches start empty.

Complexity summary (n = items in set, k = KList capacity,
threshold = k · l_factor):

+------------------------------+--------------------------------------------+
| Operation                    | Time                                       |
+==============================+============================================+
| ``_convert_node_set``        | O(1) dispatch + cache invalidation         |
| ``_check_and_convert_set``   | O(1) dispatch                              |
| ``_check_and_expand_klist``  | O(n · h_{d+1}) when conversion triggers    |
|                              | (bulk-creates a GK+-tree of dim d+1)       |
| ``_check_and_collapse_tree`` | O(threshold) early-exit counting;          |
|                              | O(n_{d+1}) only when collapse happens      |
|                              | (n_{d+1} ≤ threshold, so bounded by O(k))  |
+------------------------------+--------------------------------------------+
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gplus_trees.base import AbstractSetDataStructure
from gplus_trees.g_k_plus.bulk_create import _klist_to_tree, _threshold_item_count, _tree_to_klist
from gplus_trees.klist_base import KListBase

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusNodeBase, GKPlusTreeBase


class GKPlusConversionMixin:
    """Mixin that contributes KList↔GKPlusTree conversion methods to *GKPlusTreeBase*."""

    def _convert_node_set(self, node: GKPlusNodeBase) -> None:
        """Convert *node*'s set between KList and GKPlusTree based on thresholds.

        The receiver must be the tree that owns *node*: after replacing
        ``node.set``, the receiver's cached counts (``item_cnt``, ``size``,
        ``expanded_cnt``) are invalidated so the next ``item_count()`` /
        ``real_item_count()`` recomputes from the converted set.  The
        invalidation happens unconditionally — callsites reach this point
        only after mutating the node, so the caches are stale either way.
        """
        node.set = self._check_and_convert_set(node.set)
        self._invalidate_tree_size()

    def _check_and_convert_set(self, set: AbstractSetDataStructure) -> AbstractSetDataStructure:
        """Check and convert a free-standing set between KList and GKPlusTree.

        Pure set→set function: no tree cache is touched.  Use this for
        split halves that are placed into freshly constructed trees
        (whose caches start empty).  For a set attached to a node of a
        live tree, use :meth:`_convert_node_set` instead, which also
        invalidates the owning tree's cached counts.
        """
        # Avoid circular import at module level
        from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

        if isinstance(set, KListBase):
            return self._check_and_expand_klist(set)
        elif isinstance(set, GKPlusTreeBase):
            return self._check_and_collapse_tree(set)
        else:
            raise TypeError(f"Unsupported set type: {type(set).__name__}. Expected KListBase or GKPlusTreeBase.")

    def _check_and_expand_klist(self, klist: KListBase) -> AbstractSetDataStructure:
        """
        Check if a KList exceeds the threshold and should be converted to a GKPlusTree.

        Args:
            klist: The KList to check

        Returns:
            Either the original KList or a new GKPlusTree based on the threshold
        """
        # Check if the item count exceeds l_factor * CAPACITY
        k = klist.KListNodeClass.CAPACITY
        threshold = int(k * self.l_factor)
        if klist.item_count() <= threshold:
            return klist

        # The raw count exceeds the threshold, but only entries of this
        # tree's own dimension weigh against it (see _threshold_item_count:
        # counting carried dummies would make the threshold unsatisfiable).
        if _threshold_item_count(klist, self.dummy_item.key) > threshold:
            # Convert to GKPlusTree with increased dimension
            new_dim = type(self).DIM + 1
            new_tree = _klist_to_tree(klist, k, new_dim, self.l_factor)
            return new_tree

        return klist

    def _check_and_collapse_tree(self, tree: GKPlusTreeBase) -> AbstractSetDataStructure:
        """
        Check if a GKPlusTree has few enough items to be collapsed into a KList.

        Args:
            tree: The GKPlusTree to check

        Returns:
            Either the original tree or a new KList based on the threshold

        Implementation note:
            An early-exit loop counts the entries that weigh against the
            threshold — real items (key ≥ 0) and the receiver's own
            dummy — and aborts as soon as the count exceeds the collapse
            threshold.  This is O(threshold) = O(k) instead of O(n),
            because we stop after seeing threshold + 1 counting entries.
            Dummies carried in from other dimensions are excluded, using
            the same counting rule as the expansion check — the shared
            rule is what keeps the conversion hysteresis-free (ADR-0002).

            Empty GKPlusTree sets (artifacts of unzip splits) are
            converted to empty KLists unconditionally, since an empty
            GKPlusTree would violate the ``set_thresholds_met`` invariant
            (every GKPlusTree inner set must exceed the threshold).
        """
        k = self.KListClass.KListNodeClass.CAPACITY
        threshold = int(k * self.l_factor)

        if tree.DIM == 1:
            # We want to keep the GKPlusTree structure for dimension 1
            return tree

        if tree.is_empty():
            # Convert to an empty KList rather than keeping an empty
            # GKPlusTree, which would violate the set_thresholds_met
            # invariant (a GKPlusTree set must exceed the threshold).
            return _tree_to_klist(tree)

        # Early-exit counting: iterate the tree's entries and stop as
        # soon as the count exceeds the collapse threshold.  Iterating
        # *tree* also yields dummies of dimensions above the receiver's
        # (the inner tree's own sentinel and deeper ones); the equality
        # check excludes those alongside the carried lower-dimension ones.
        own_dummy_key = self.dummy_item.key
        counted = 0
        for entry in tree:
            key = entry.item.key
            if key >= 0 or key == own_dummy_key:
                counted += 1
                if counted > threshold:
                    return tree

        return _tree_to_klist(tree)
