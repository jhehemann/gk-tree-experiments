"""KList / GKPlusTree conversion logic for GK+-trees.

Provides :class:`GKPlusConversionMixin`, a mixin class that adds
``check_and_convert_set``, ``check_and_expand_klist`` and
``check_and_collapse_tree`` to :class:`GKPlusTreeBase`.

Conversions are triggered when a KList exceeds ``k · l_factor`` items
(expand) or a GKPlusTree shrinks below that threshold (collapse).
These conversions are what create recursive dimensional nesting:
an expanded KList becomes a GK+-tree of the next dimension.

Complexity summary (n = items in set, k = KList capacity,
threshold = k · l_factor):

+-----------------------------+--------------------------------------------+
| Operation                   | Time                                       |
+=============================+============================================+
| ``check_and_convert_set``   | O(1) dispatch                              |
| ``check_and_expand_klist``  | O(n · h_{d+1}) when conversion triggers    |
|                             | (bulk-creates a GK+-tree of dim d+1)       |
| ``check_and_collapse_tree`` | O(threshold) early-exit counting;          |
|                             | O(n_{d+1}) only when collapse happens      |
|                             | (n_{d+1} ≤ threshold, so bounded by O(k))  |
+-----------------------------+--------------------------------------------+
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gplus_trees.base import AbstractSetDataStructure
from gplus_trees.g_k_plus.bulk_create import _klist_to_tree, _tree_to_klist
from gplus_trees.klist_base import KListBase

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase


class GKPlusConversionMixin:
    """Mixin that contributes KList↔GKPlusTree conversion methods to *GKPlusTreeBase*."""

    def check_and_convert_set(self, set: AbstractSetDataStructure) -> AbstractSetDataStructure:
        """Check and convert set between KList and GKPlusTree based on thresholds.

        TODO(#1): After calling this method and replacing node.set, the caller MUST
        call _invalidate_tree_size() to ensure cached counts are invalidated.
        Otherwise, item_count() will return stale values leading to incorrect conversions.
        """
        # Avoid circular import at module level
        from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

        if isinstance(set, KListBase):
            return self.check_and_expand_klist(set)
        elif isinstance(set, GKPlusTreeBase):
            return self.check_and_collapse_tree(set)
        else:
            raise TypeError(f"Unsupported set type: {type(set).__name__}. Expected KListBase or GKPlusTreeBase.")

    def check_and_expand_klist(self, klist: KListBase) -> AbstractSetDataStructure:
        """
        Check if a KList exceeds the threshold and should be converted to a GKPlusTree.

        Args:
            klist: The KList to check

        Returns:
            Either the original KList or a new GKPlusTree based on the threshold

        Note (Issue #1):
            If conversion occurs, caller must call _invalidate_tree_size() on the parent tree.
        """
        # Check if the item count exceeds l_factor * CAPACITY
        k = klist.KListNodeClass.CAPACITY
        threshold = int(k * self.l_factor)
        if klist.item_count() > threshold:
            # Convert to GKPlusTree with increased dimension
            new_dim = type(self).DIM + 1
            new_tree = _klist_to_tree(klist, k, new_dim, self.l_factor)
            return new_tree

        return klist

    def check_and_collapse_tree(self, tree: GKPlusTreeBase) -> AbstractSetDataStructure:
        """
        Check if a GKPlusTree has few enough items to be collapsed into a KList.

        Args:
            tree: The GKPlusTree to check

        Returns:
            Either the original tree or a new KList based on the threshold

        Note (Issue #1):
            If conversion occurs, caller must call _invalidate_tree_size() on the parent tree.

        Implementation note:
            An early-exit loop counts real items (key ≥ 0) and aborts as
            soon as the count exceeds the collapse threshold.  This is
            O(threshold) = O(k) instead of O(n), because we stop after
            seeing threshold + 1 real items.

            When the count stays ≤ threshold the tree is small enough to
            speculatively convert via ``_tree_to_klist`` and verify the
            actual KList size (which may be slightly larger due to
            lower-dimension dummies that are preserved).

            Empty GKPlusTree sets (artifacts of unzip splits) are
            converted to empty KLists unconditionally, since an empty
            GKPlusTree would violate the ``set_thresholds_met`` invariant
            (every GKPlusTree inner set must have item_count > threshold).
        """
        k = self.KListClass.KListNodeClass.CAPACITY
        threshold = int(k * self.l_factor)

        if tree.DIM == 1:
            # We want to keep the GKPlusTree structure for dimension 1
            return tree

        if tree.is_empty():
            # Convert to an empty KList rather than keeping an empty
            # GKPlusTree, which would violate the set_thresholds_met
            # invariant (a GKPlusTree set must have item_count > threshold).
            return _tree_to_klist(tree)

        # Early-exit real-item count: iterate the tree's entries and
        # count items with key >= 0, stopping as soon as the count
        # exceeds the collapse threshold.  This is O(threshold) instead
        # of O(n) because we abort after seeing threshold+1 real items.
        real_count = 0
        for entry in tree:
            if entry.item.key >= 0:
                real_count += 1
                if real_count > threshold:
                    return tree

        # The tree has few enough real items that it may fit in a KList.
        # Convert and verify the resulting size (which also includes
        # lower-dimension dummies that _tree_to_klist preserves).
        new_klist = _tree_to_klist(tree)
        klist_size = new_klist.item_count()

        if klist_size <= threshold:
            return new_klist

        # Resulting KList exceeds the threshold (due to preserved
        # lower-dimension dummies).  Keep the tree structure.
        return tree
