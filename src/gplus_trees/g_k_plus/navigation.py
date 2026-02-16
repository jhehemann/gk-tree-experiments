"""Navigation helpers for GK+-trees.

Provides :class:`GKPlusNavigationMixin`, a mixin class that adds
``find_pivot``, ``get_min``, ``get_max``, ``get_min_leaf_tree`` and
``get_max_leaf`` to :class:`GKPlusTreeBase`.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

from gplus_trees.base import Entry
from gplus_trees.gplus_tree_base import print_pretty, get_dummy

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, GKPlusNodeBase


class GKPlusNavigationMixin:
    """Mixin that contributes navigation / query methods to *GKPlusTreeBase*."""

    def find_pivot(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Returns the pivot entry of a node in the next lower dimension.

        This method is always called from a node of the next lower dimension.
        The pivot entry is the first entry that is either the dummy entry of
        the lower dimension itself or the next larger entry in the current tree.

        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (pivot_entry, successor_entry)
            where successor_entry is the first entry with key > pivot.key, or None if pivot
            is the last entry in the tree.
        """

        if self.is_empty():
            return None, None

        dummy = get_dummy(self.DIM - 1).key

        pivot = None
        successor = None
        for entry in self:
            if pivot is not None:
                if entry.item.key > pivot.item.key:
                    successor = entry
                    break
            else:
                if entry.item.key == dummy or entry.item.key > dummy:
                    pivot = entry

        if pivot is None:
            raise ValueError(f"No pivot entry in tree {print_pretty(self)}")

        return pivot, successor

    # TODO: Check indifference: This may return an entry with dummy key y, although a subsequent
    # leaf may have been expanded to a higher dimension with a dummy key x < y.
    # However, y is the first entry yielded when iterating over the tree.
    def get_min(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Get the minimum entry in the tree. This corresponds to the entry with the dummy item of the maximum dimension in successive first leaf nodes.
        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (minimum_entry, next_entry)
        """
        if self.is_empty():
            return None, None

        first_leaf = next(self.iter_leaf_nodes(), None)
        if first_leaf is None:
            raise ValueError("Tree is empty, cannot retrieve minimum.")

        return first_leaf.set.get_min()

    def get_max(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Get the maximum entry in the tree.
        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (maximum_entry, next_entry)
        """
        max_leaf = self.get_max_leaf()
        return max_leaf.set.get_max()

    def get_min_leaf_tree(self: 'GKPlusTreeBase') -> 'GKPlusNodeBase':
        """
        Get the minimum leaf node tree in the tree in the current dimension.
        Returns:
            GKPlusNodeBase: The minimum node in the tree.
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

    def get_max_leaf(self: 'GKPlusTreeBase') -> 'GKPlusNodeBase':
        """
        Get the maximum node in the tree in the current dimension.
        Returns:
            GKPlusNodeBase: The maximum node in the tree.
        """
        if self.is_empty():
            return None

        node = self.node
        while node.rank > 1:
            node = node.right_subtree.node

        return node
