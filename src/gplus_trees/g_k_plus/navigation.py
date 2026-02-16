"""Navigation helpers for GK+-trees.

Provides :class:`GKPlusNavigationMixin`, a mixin class that adds
``find_pivot``, ``get_min``, ``get_max``, ``get_min_leaf_tree`` and
``get_max_leaf`` to :class:`GKPlusTreeBase`.

Single-dimension complexity (n = items, h = height ≈ log n):

+------------------------+--------------------------------------------+
| Operation              | Time (per dimension)                       |
+========================+============================================+
| ``find_pivot``         | O(h_d) descent + inner-set iteration cost  |
| ``get_min``            | O(h_d) + inner-set ``get_min`` cost        |
| ``get_max``            | O(h_d) + inner-set ``get_max`` cost        |
| ``get_min_leaf_tree``  | O(h_d) — same-dimension descent only       |
| ``get_max_leaf``       | O(h_d) — same-dimension descent only       |
+------------------------+--------------------------------------------+

Note: ``get_min_leaf_tree`` and ``get_max_leaf`` follow left/right
subtrees within the *same* dimension, so they do not recurse into inner
GK+-trees. However, ``find_pivot``, ``get_min``, and ``get_max`` iterate
over entries from leaf node sets. When a leaf's ``node.set`` is a
GK+-tree of dimension *d+1*, that iteration triggers ``__iter__`` on the
inner tree, adding O(h_{d+1}) setup + O(n_{d+1}) scan per inner tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gplus_trees.base import Entry
from gplus_trees.gplus_tree_base import get_dummy, print_pretty

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusNodeBase, GKPlusTreeBase


class GKPlusNavigationMixin:
    """Mixin that contributes navigation / query methods to *GKPlusTreeBase*."""

    def find_pivot(self) -> tuple[Entry | None, Entry | None]:
        """
        Returns the pivot entry of a node in the next lower dimension.

        This method is always called from a node of the next lower dimension.
        The pivot entry is the first entry that is either the dummy entry of
        the lower dimension itself or the next larger entry in the current tree.

        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (pivot_entry, successor_entry)
            where successor_entry is the first entry with key > pivot.key, or None if pivot
            is the last entry in the tree.

        Complexity:
            O(h_d) for the leftmost-leaf descent via ``get_min_leaf_tree``
            (same-dimension traversal only). Then scans at most a few
            entries from the leaf's set to locate the pivot. Unlike the
            previous implementation that called ``__iter__`` (which also
            computed ``get_max_leaf``), this avoids an unnecessary O(h_d)
            rightmost-path descent.

            When the leaf's ``node.set`` is an inner GK+-tree of
            dimension d+1, the ``for entry in node.set`` triggers
            ``__iter__`` on that tree, adding O(h_{d+1}) setup cost.
            This compounds recursively through all active dimensions.
        """

        if self.is_empty():
            return None, None

        dummy_key = get_dummy(self.DIM - 1).key

        # Descend directly to the first leaf instead of using __iter__
        # which first computes get_max_leaf() — an unnecessary O(log n) cost.
        cur = self.get_min_leaf_tree()
        pivot = None
        while cur is not None:
            node = cur.node
            for entry in node.set:
                if pivot is not None:
                    if entry.item.key > pivot.item.key:
                        return pivot, entry
                elif entry.item.key >= dummy_key:
                    pivot = entry
            cur = node.next  # follow leaf chain

        if pivot is None:
            raise ValueError(f"No pivot entry in tree {print_pretty(self)}")

        return pivot, None

    # TODO: Check indifference: This may return an entry with dummy key y, although a subsequent
    # leaf may have been expanded to a higher dimension with a dummy key x < y.
    # However, y is the first entry yielded when iterating over the tree.
    def get_min(self) -> tuple[Entry | None, Entry | None]:
        """
        Get the minimum entry in the tree. This corresponds to the entry with the dummy item
        of the maximum dimension in successive first leaf nodes.

        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (minimum_entry, next_entry)

        Complexity:
            O(h_d) + inner-set ``get_min`` cost. The ``iter_leaf_nodes``
            descent is O(h_d); the subsequent ``get_min`` on a KList
            is O(1), on an inner GK+-tree it recurses into dim d+1.
        """
        if self.is_empty():
            return None, None

        first_leaf = next(self.iter_leaf_nodes(), None)
        if first_leaf is None:
            raise ValueError("Tree is empty, cannot retrieve minimum.")

        return first_leaf.set.get_min()

    def get_max(self) -> tuple[Entry | None, Entry | None]:
        """
        Get the maximum entry in the tree.

        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (maximum_entry, next_entry)

        Complexity:
            O(h_d) + inner-set ``get_max`` cost. Descends via ``get_max_leaf``
            (rightmost path, O(h_d)); ``get_max`` on a KList is O(1), on
            an inner GK+-tree it recurses into dim d+1.
        """
        max_leaf = self.get_max_leaf()
        return max_leaf.set.get_max()

    def get_min_leaf_tree(self: GKPlusTreeBase) -> GKPlusTreeBase:
        """
        Get the minimum leaf tree in the current dimension.

        Descends the leftmost path of the tree to find the first leaf-level
        tree (rank 1).

        Returns:
            GKPlusTreeBase: The leftmost leaf-level tree, or ``None`` if empty.

        Complexity:
            O(h_d) — descends through left subtrees of the *same* dimension.
            Does not recurse into inner GK+-tree sets.
        """
        if self.is_empty():
            return None

        cur = self
        while cur.node.rank > 1:
            for entry in cur.node.set:
                if entry.left_subtree is not None:
                    cur = entry.left_subtree
                    break
            else:
                cur = cur.node.right_subtree
        return cur

    def get_max_leaf(self: GKPlusTreeBase) -> GKPlusNodeBase:
        """
        Get the rightmost leaf node in the current dimension.

        Descends via ``right_subtree`` at each level.  Returns a
        :class:`GKPlusNodeBase` (not a tree) because callers typically
        need to access ``node.next`` or ``node.set`` directly.

        Returns:
            GKPlusNodeBase: The rightmost leaf node, or ``None`` if empty.

        Complexity:
            O(h_d) — follows the rightmost path within the same dimension.
            Does not recurse into inner GK+-tree sets.
        """
        if self.is_empty():
            return None

        node = self.node
        while node.rank > 1:
            node = node.right_subtree.node

        return node
