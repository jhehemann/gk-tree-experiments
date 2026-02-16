"""Zip (merge) operation for GK+-trees.

Provides :class:`GKPlusZipMixin`, a mixin class that adds
``check_convert_same_sets`` and ``zip`` to :class:`GKPlusTreeBase`.

The zip operation merges two GK+-trees that cover disjoint key ranges
into a single tree, preserving search-tree invariants and leaf-chain
linkage.

Single-dimension complexity (n = items in both trees, h = height ≈ log n):

+-------------------------------+------------------------------------------+
| Operation                     | Time (per dimension)                     |
+===============================+==========================================+
| ``zip``                       | O(h · (log l + k))  amortised            |
| ``check_convert_same_sets``   | O(n) when type conversion needed         |
+-------------------------------+------------------------------------------+

When node sets are inner GK+-trees, ``zip`` recurses into dimension d+1
for set merging. The ``find_pivot`` calls at each level also pay the
inner-tree traversal cost (see navigation.py).
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from gplus_trees.base import (
    Entry,
    _get_replica,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import get_dummy
from gplus_trees.g_k_plus.bulk_create import bulk_create_gkplus_tree

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase


class GKPlusZipMixin:
    """Mixin that contributes ``check_convert_same_sets`` and ``zip`` to *GKPlusTreeBase*."""

    def check_convert_same_sets(self, left: 'GKPlusTreeBase', other: 'GKPlusTreeBase') -> Tuple['GKPlusTreeBase', 'GKPlusTreeBase']:
        """
        Check and convert the sets of left and right trees to appropriate types.

        Args:
            left: The left GKPlusTreeBase instance.
            right: The right GKPlusTreeBase instance.

        """
        # Normalize sets to same type before merging
        l_set_is_klist = isinstance(left.node.set, KListBase)
        o_set_is_klist = isinstance(other.node.set, KListBase)

        # Convert to same type if needed
        if l_set_is_klist != o_set_is_klist:
            if l_set_is_klist:
                # Convert left from KList to GKPlusTree
                left.node.set = bulk_create_gkplus_tree(left.node.set, other.node.set.DIM, other.l_factor, type(left.node.set))
            else:
                # Convert other from KList to GKPlusTree
                other.node.set = bulk_create_gkplus_tree(other.node.set, left.node.set.DIM, left.l_factor, type(other.node.set))
                _, _, other.node.set, _ = other.node.set.unzip(-1)

        return left, other

    # TODO: Return r_pivot properly so find_pivot calls are minimized
    def zip(self, left: 'GKPlusTreeBase', other: 'GKPlusTreeBase', is_root: bool = True) -> 'GKPlusTreeBase':
        """
        Zip two GKPlusTreeBase instances together, merging their entries.

        Args:
            other: Another GKPlusTreeBase instance to zip with. The other must not have the dimensions dummy item. It will be included during the zipping process.

        Returns:
            A new GKPlusTreeBase instance containing the merged entries
        """
        from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

        if not isinstance(other, GKPlusTreeBase):
            raise TypeError(f"other must be an instance of GKPlusTreeBase, got {type(other).__name__}")

        if other.DIM != left.DIM:
            raise ValueError(f"Dimension mismatch: left.DIM={left.DIM}, other.DIM={other.DIM}")


        # Early returns for empty trees
        if left.is_empty():
            if other.is_empty():
                return left, None, None

            if is_root:
                is_root = False
                other_rank = other.node.rank
                dummy_tree = self._create_dummy_singleton_tree(other_rank, other.DIM, other.l_factor, left.KListClass)
                left.node = dummy_tree.node
                left._invalidate_tree_size()
                left, r_pivot, r_pivot_next = self.zip(left, other)
                left.node.set = self.check_and_convert_set(left.node.set)
                left._invalidate_tree_size()  # Correctly invalidates after conversion

                return left, r_pivot, r_pivot_next
            r_pivot, r_pivot_next = other.find_pivot()
            return other, r_pivot, r_pivot_next

        if other.is_empty():
            return left, None, None

        left_rank = left.node.rank

        # Case 1: Left rank < other rank - insert left into other's leftmost position
        if left_rank < other.node.rank:
            if is_root:
                is_root = False  # Mark that we're no longer at root level

            l_pivot = left.node.set.find_pivot()[0]

            # Insert replica of left pivot into other's set
            replica_entry = Entry(_get_replica(l_pivot.item), None)

            if isinstance(other.node.set, KListBase):
                r_pivot, r_pivot_next = other.node.set.find_pivot()
                other.node.set, _, _ = other.node.set.insert_entry(replica_entry, other.node.rank)
            else:
                singleton_tree = bulk_create_gkplus_tree([replica_entry], left.DIM + 1, l_factor=left.l_factor, KListClass=left.SetClass)
                other.node.set, r_pivot, r_pivot_next = singleton_tree.zip(singleton_tree, other.node.set)


            other.node.set = other.check_and_convert_set(other.node.set)
            other._invalidate_tree_size()  # Correctly invalidates after conversion

            # Zip r_pivot's left subtree into left
            if r_pivot and r_pivot.left_subtree:
                r_pivot.left_subtree, r_pivot_sub, r_pivot_sub_next = left.zip(left, r_pivot.left_subtree)
                # r_pivot.left_subtree = left
                r_pivot.left_subtree._invalidate_tree_size()
                other._invalidate_tree_size()
                r_pivot = r_pivot_sub
                r_pivot_next = r_pivot_sub_next

            else:

                if r_pivot_next is not None:
                    other_min_leaf_tree = r_pivot_next.left_subtree.get_min_leaf_tree()
                else:
                    other_min_leaf_tree = other.node.right_subtree.get_min_leaf_tree()


                if left.node.rank == 1:
                    left.node.next = other_min_leaf_tree
                else:
                    max_left_leaf = left.get_max_leaf()
                    max_left_leaf.next = other_min_leaf_tree

                r_pivot.left_subtree = left
                r_pivot, r_pivot_next = other_min_leaf_tree.node.set.find_pivot()  # Update r_pivot after reassignment
                if r_pivot_next is None and other_min_leaf_tree.node.next is not None:
                    r_pivot_next = other_min_leaf_tree.node.next.node.set.find_pivot()[0]

                other._invalidate_tree_size()

            return other, r_pivot, r_pivot_next

        # Case 2: Left rank > other rank - recursively zip into right subtree
        elif left.node.rank > other.node.rank:
            left.node.right_subtree, r_pivot, r_pivot_next = left.node.right_subtree.zip(left.node.right_subtree, other, is_root=False)

            left._invalidate_tree_size()
            return left, r_pivot, r_pivot_next

        # Case 3: Same rank - merge sets
        else:
            left, other = self.check_convert_same_sets(left, other)

            # Merge the sets based on their types
            if isinstance(left.node.set, KListBase) and isinstance(other.node.set, KListBase):
                # Both KLists - merge entries directly
                dummy_lower_dim = get_dummy(left.DIM - 1)
                for entry in other.node.set:
                    if entry.item.key > dummy_lower_dim.key:
                        left.node.set, _, _ = left.node.set.insert_entry(entry)
                left.node.set = left.check_and_convert_set(left.node.set)
                # TODO(#1): Add left._invalidate_tree_size() after set conversion
                left._invalidate_tree_size()
                r_pivot, r_pivot_next = other.node.set.find_pivot()


            elif isinstance(left.node.set, GKPlusTreeBase) and isinstance(other.node.set, GKPlusTreeBase):
                # Both are GKPlusTreeBase (after conversion) - recursively zip
                left.node.set, r_pivot, r_pivot_next = left.node.set.zip(left.node.set, other.node.set)
                left.node.set = self.check_and_convert_set(left.node.set)
                # TODO(#1): Add left._invalidate_tree_size() after set conversion
                left._invalidate_tree_size()
            else:
                raise TypeError(f"Set types should match after conversion, got {type(left.node.set).__name__} and {type(other.node.set).__name__}")


            # Link leaf nodes
            if left.node.rank == 1:
                left.node.next = other.node.next
                left._invalidate_tree_size()
                if r_pivot_next is None and other.node.next is not None:
                    r_pivot_next = other.node.next.node.set.find_pivot()[0]
                return left, r_pivot, r_pivot_next

            else:

                if r_pivot.left_subtree is None:

                    if r_pivot_next is not None:
                        other_min_leaf_tree = r_pivot_next.left_subtree.get_min_leaf_tree()
                    else:
                        other_min_leaf_tree = other.node.right_subtree.get_min_leaf_tree()

                    max_left_leaf = left.node.right_subtree.get_max_leaf()
                    max_left_leaf.next = other_min_leaf_tree

                    r_pivot.left_subtree = left.node.right_subtree
                    r_pivot, r_pivot_next = other_min_leaf_tree.node.set.find_pivot()
                    if r_pivot_next is None and other_min_leaf_tree.node.next is not None:
                        r_pivot_next = other_min_leaf_tree.node.next.node.set.find_pivot()[0]
                else:
                    r_pivot.left_subtree, r_pivot_sub, r_pivot_sub_next = left.node.right_subtree.zip(left.node.right_subtree, r_pivot.left_subtree)

                    r_pivot.left_subtree._invalidate_tree_size()
                    other._invalidate_tree_size()
                    r_pivot = r_pivot_sub
                    r_pivot_next = r_pivot_sub_next

                left.node.right_subtree = other.node.right_subtree


            return left, r_pivot, r_pivot_next
