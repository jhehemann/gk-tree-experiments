"""Unzip (split) operation for GK+-trees.

Provides :class:`GKPlusUnzipMixin`, a mixin class that adds
``unzip``, ``lift``, and their private helpers to :class:`GKPlusTreeBase``.

The unzip operation splits a GK+-tree at a given key into a left part
(keys < key) and a right part (keys ≥ key), preserving search-tree
invariants and re-linking leaf chains.

Single-dimension complexity (n = items, h = height ≈ log n,
k = KList capacity, l = KList nodes):

+-------------------------------+------------------------------------------+
| Operation                     | Time (per dimension)                     |
+===============================+==========================================+
| ``unzip``                     | O(h · (log l + k))                       |
| ``lift``                      | O(1) (node construction)                 |
+-------------------------------+------------------------------------------+

When node sets are inner GK+-trees, the ``split_inplace`` / ``unzip``
call on ``node.set`` recurses into dimension d+1. The
``check_and_convert_set`` calls after splitting use early-exit
counting (O(k) per call) and only trigger collapse when the split
half is small enough (≤ threshold items).
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from gplus_trees.base import AbstractSetDataStructure, Entry
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import get_dummy

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, GKPlusNodeBase


class GKPlusUnzipMixin:
    """Mixin that contributes ``unzip``, ``lift`` and helpers to *GKPlusTreeBase*."""

    def unzip(self, key: int):
        """
        Unzip the tree at the given key, splitting it into two parts.

        Args:
            key: The key value to unzip at

        Returns:
            A tuple of (left_tree, right_tree) where:
            - left_tree: A tree containing all entries with keys < key
            - right_tree: A tree with entries with keys > key
        """
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")

        # Early return for empty tree
        if self.is_empty():
            TreeClass = type(self)
            return self, None, TreeClass(l_factor=self.l_factor), None

        if key < get_dummy(self.DIM).key:
            raise ValueError(f"Cannot unzip at key {key} less than dimension dummy key {get_dummy(self.DIM).key}")

        node = self.node
        rank = node.rank
        is_leaf = rank == 1
        right_subtree = node.right_subtree

        self._invalidate_tree_size()


        # Unzip the node set
        if isinstance(node.set, KListBase):
            left_set, left_subtree_of_key, right_set, next_entry = node.set.split_inplace(key)
        else:
            left_set, left_subtree_of_key, right_set, next_entry = node.set.unzip(key)

        # After splitting an inner-tree set, either half may have shrunk
        # below the collapse threshold.  Re-check both so that undersized
        # GKPlusTrees are converted back to KLists.  For KList splits this
        # is a no-op (a KList never needs expansion right after a split).
        left_set = self.check_and_convert_set(left_set)
        right_set = self.check_and_convert_set(right_set)


        # Update the current node's set
        node.set = left_set
        left_next = node.next

        # Handle leaf nodes
        if is_leaf:
            return self._handle_leaf_unzip(left_set, left_subtree_of_key, right_set,
                                         next_entry, left_next)

        # Handle non-leaf nodes

        return self._handle_nonleaf_unzip(key, left_set, left_subtree_of_key, right_set,
                                        next_entry, right_subtree, rank)

    def _retrieve_next(self, key: int) -> Optional['Entry']:
        """Retrieve the next entry in the tree after the given key."""
        pivot, pivot_next = self.find_pivot()
        dummy_key = get_dummy(self.DIM).key

        if pivot and pivot.item.key > dummy_key:
            return pivot
        else:
            return pivot_next

    def _handle_leaf_unzip(self, left_set, left_subtree_of_key, right_set, next_entry, left_next):
        """Handle unzipping at leaf level."""
        if left_set.is_empty():
            # Return right subtree directly
            right_return = self.lift(right_set, None, 1)
            if not right_return.is_empty():
                right_return.node.next = left_next

                if next_entry is None:
                    next_entry = right_return._retrieve_next(get_dummy(self.DIM).key)
            else:
                if next_entry is None:
                    next_entry = left_next._retrieve_next(get_dummy(self.DIM).key) if left_next else None


            self.node = None
            return self, left_subtree_of_key, right_return, next_entry

        right_return = self.lift(right_set, None, 1)
        if not right_return.is_empty():

            right_return.node.next = left_next
            if next_entry is None:
                next_entry = right_return._retrieve_next(get_dummy(self.DIM).key)
        else:
            if next_entry is None:
                next_entry = left_next._retrieve_next(get_dummy(self.DIM).key) if left_next else None

        self.node.set = left_set
        self.node.next = None


        return self, left_subtree_of_key, right_return, next_entry

    def _handle_nonleaf_unzip(self, key, left_set, left_subtree_of_key, right_set, next_entry, right_subtree, rank):
        """Handle unzipping at non-leaf level."""
        # Case 1: Found the split key with its left subtree
        if left_subtree_of_key:
            return self._handle_key_subtree_case(key, left_set, left_subtree_of_key, right_set,
                                               next_entry, right_subtree, rank)

        # Cases 2: Unified handling for left set with multiple items or single/empty
        # O(1) check: after conversion, GKPlusTree has > threshold > 1 items;
        # for KList, item_count() is O(1).  Sentinel value 2 means "multiple".
        left_count = left_set.item_count() if isinstance(left_set, KListBase) else 2
        return self._handle_no_key_subtree_case(key, left_set, left_count, right_set,
                                              next_entry, right_subtree, rank)

    def _handle_key_subtree_case(self, key, left_set, left_subtree_of_key, right_set, next_entry, right_subtree, rank):
        """Handle the case where the split key has a left subtree."""

        # Update current node based on left set size
        # O(1) check: after conversion, GKPlusTree has > threshold > 1
        # items; for KList, item_count() is O(1).
        if not isinstance(left_set, KListBase) or left_set.item_count() > 1:
            self.node.right_subtree = left_subtree_of_key
        else:
            self.node = left_subtree_of_key.node

        # Handle next entry and right subtree unzipping
        key_subtree, right_return, updated_next_entry = self._unzip_right_side(key, next_entry, right_subtree, right_set, rank)

        # Use updated next_entry if provided, otherwise keep original
        if updated_next_entry is not None:
            next_entry = updated_next_entry

        # Unlink leaf nodes
        l_max_leaf = left_subtree_of_key.get_max_leaf()
        if l_max_leaf:
            l_max_leaf.next = None

        return self, key_subtree, right_return, next_entry

    def _handle_no_key_subtree_case(self, key, left_set, left_count, right_set, next_entry, right_subtree, rank):
        """Unified handler for cases where left set has multiple items or single/empty items."""
        key_subtree = None
        right_return = None

        if next_entry:
            if next_entry.left_subtree:
                # Unzip the next entry's left subtree
                l_right_subtree, key_subtree, r_leftmost_subtree, new_next = next_entry.left_subtree.unzip(key)

                if left_count > 1:
                    # Multi-item case: use left subtree as right subtree
                    self.node.right_subtree = l_right_subtree
                else:
                    # Single/empty case: use left subtree as current node
                    self.node = l_right_subtree.node if l_right_subtree else None

                # Update next_entry's left subtree with safe empty check
                next_entry.left_subtree = r_leftmost_subtree if r_leftmost_subtree and not r_leftmost_subtree.is_empty() else None
                right_return = self.lift(right_set, right_subtree, rank)

                # Update next_entry if we got a valid new one
                if new_next is not None:
                    next_entry = new_next
            else:
                # No left subtree to unzip
                if left_count > 1:
                    self.node.right_subtree = None
                else:
                    self.node = None
                right_return = self.lift(right_set, right_subtree, rank)

            # Break leaf links if at leaf level and no right subtree
            if rank == 1 and self.node and self.node.right_subtree is None:
                self.node.next = None
        else:
            # No next entry - unzip right subtree directly
            if right_subtree:
                l_right_subtree, key_subtree, r_leftmost_subtree, new_next = right_subtree.unzip(key)
                next_entry = new_next
            else:
                l_right_subtree, r_leftmost_subtree = None, None

            if left_count > 1:
                # Multi-item case
                self.node.right_subtree = l_right_subtree
                right_return = r_leftmost_subtree
            else:
                # Single/empty case
                self.node.right_subtree = l_right_subtree
                right_return = r_leftmost_subtree

        return self, key_subtree, right_return, next_entry

    def _unzip_right_side(self, key, next_entry, right_subtree, right_set, rank):
        """Helper method to handle right-side unzipping logic."""
        if next_entry:
            left, key_subtree, r_leftmost_subtree, new_next = next_entry.left_subtree.unzip(key)
            next_entry.left_subtree = r_leftmost_subtree if r_leftmost_subtree and not r_leftmost_subtree.is_empty() else None
            right_return = self.lift(right_set, right_subtree, rank)
            updated_next_entry = new_next if new_next is not None else next_entry
        else:
            if right_subtree:
                left, key_subtree, r_leftmost_subtree, new_next = right_subtree.unzip(key)
                updated_next_entry = new_next
            else:
                key_subtree, r_leftmost_subtree, new_next = None, None, None
                updated_next_entry = None
            right_return = self.lift(right_set, r_leftmost_subtree, rank)

        return key_subtree, right_return, updated_next_entry


    def lift(self, set: Optional[AbstractSetDataStructure], right: 'GKPlusTreeBase', rank: int) -> 'GKPlusTreeBase':
        """
        Create a GKPlusTree from a set, a right subtree at the specified rank.

        Args:
            set: The tree node set. If empty, return the right subtree.
            right: The tree node's right subtree
            rank: The rank of the tree

        Returns:
            A new GKPlusTreeBase instance containing the merged entries
        """

        TreeClass = type(self)
        if right is not None and right.is_empty():
            right = None

        if set.is_empty():
            if right is None:
                # If both set and right subtree are empty, return an empty tree
                return TreeClass(l_factor=self.l_factor)
            return right

        new_node = self.NodeClass(rank, set, right)

        return TreeClass(new_node, self.l_factor)
