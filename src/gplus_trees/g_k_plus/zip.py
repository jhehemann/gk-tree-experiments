"""Zip (merge) operation for GK+-trees.

Provides :class:`GKPlusZipMixin`, a mixin class that adds
``check_convert_same_sets`` and ``zip`` to :class:`GKPlusTreeBase`.
"""

from __future__ import annotations
import logging
from typing import Tuple, TYPE_CHECKING

from gplus_trees.base import (
    Entry,
    _get_replica,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import print_pretty, get_dummy
from gplus_trees.logging_config import get_logger
from gplus_trees.g_k_plus.bulk_create import bulk_create_gkplus_tree

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

logger = get_logger(__name__)


def IS_DEBUG():
    """Check if debug logging is enabled."""
    return logger.isEnabledFor(logging.DEBUG)


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
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Converted left set to GKPlusTreeBase: {print_pretty(left.node.set)}")
            else:
                # Convert other from KList to GKPlusTree
                other.node.set = bulk_create_gkplus_tree(other.node.set, left.node.set.DIM, left.l_factor, type(other.node.set))
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Converted other set to GKPlusTreeBase: {print_pretty(other.node.set)}")
                _, _, other.node.set, _ = other.node.set.unzip(-1)
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] unzipped other.node.set at {-1}: {print_pretty(other.node.set)}")

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

        if IS_DEBUG():
            logger.debug(f"[DIM {left.DIM}] Zipping trees: ")
            logger.debug(f"[DIM {left.DIM}] Left (item_count: {left.item_count()}, real_item_count: {left.real_item_count()}): {print_pretty(left)}")
            logger.debug(f"[DIM {left.DIM}] Other (item_count: {other.item_count()}, real_item_count: {other.real_item_count()}): {print_pretty(other)}")
            logger.debug(f"[DIM {left.DIM}] Left items: {[e.item.key for e in left]}")
            logger.debug(f"[DIM {left.DIM}] Other items: {[e.item.key for e in other]}")

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
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Created dummy tree: {print_pretty(left)}")
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
            if IS_DEBUG():
                logger.debug(f"[DIM {left.DIM}] Found l_pivot: {l_pivot.item.key if l_pivot else None}")

            # Insert replica of left pivot into other's set
            replica_entry = Entry(_get_replica(l_pivot.item), None)
            
            if isinstance(other.node.set, KListBase):
                r_pivot, r_pivot_next = other.node.set.find_pivot()
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Inserting replica entry {replica_entry.item.key} into other's KList set")
                other.node.set, _, _ = other.node.set.insert_entry(replica_entry, other.node.rank)
            else:
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Zipping replica entry {replica_entry.item.key} into other's GKPlusTreeBase set")
                singleton_tree = bulk_create_gkplus_tree([replica_entry], left.DIM + 1, l_factor=left.l_factor, KListClass=left.SetClass)
                other.node.set, r_pivot, r_pivot_next = singleton_tree.zip(singleton_tree, other.node.set)
            
            if IS_DEBUG():
                logger.debug(f"[DIM {left.DIM}] Found r_pivot: {r_pivot.item.key if r_pivot else None}; Next: {r_pivot_next.item.key if r_pivot_next else None}")
            
            other.node.set = other.check_and_convert_set(other.node.set)
            other._invalidate_tree_size()  # Correctly invalidates after conversion

            # Zip r_pivot's left subtree into left
            if r_pivot and r_pivot.left_subtree:
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Found r_pivot {r_pivot.item.key} with left subtree: {print_pretty(r_pivot.left_subtree)}")
                    logger.debug(f"[DIM {left.DIM}] Left tree before zipping (item_count: {left.item_count()}, real_item_count: {left.real_item_count()}): {print_pretty(left)}")
                r_pivot.left_subtree, r_pivot_sub, r_pivot_sub_next = left.zip(left, r_pivot.left_subtree)
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Left tree after zipping (item_count: {left.item_count()}, real_item_count: {left.real_item_count()}): {print_pretty(left)}")
                    logger.debug(f"[DIM {left.DIM}] r_pivot.left_subtree after zip (item_count: {r_pivot.left_subtree.item_count()}, real_item_count: {r_pivot.left_subtree.real_item_count()}): {print_pretty(r_pivot.left_subtree)}")
                    logger.debug(f"[DIM {left.DIM}] r_pivot_sub ({r_pivot_sub.item.key}) after zip")
                    logger.debug(f"[DIM {left.DIM}] r_pivot_sub_next ({r_pivot_sub_next.item.key if r_pivot_sub_next else None}) after zip")
                    logger.debug(f"[DIM {left.DIM}] Left Items: {[e.item.key for e in left]}")
                    logger.debug(f"[DIM {left.DIM}] Left next: {print_pretty(left.node.next)}")
                # r_pivot.left_subtree = left
                r_pivot.left_subtree._invalidate_tree_size()
                other._invalidate_tree_size()
                r_pivot = r_pivot_sub
                r_pivot_next = r_pivot_sub_next

            else:
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] No left subtree found for r_pivot {r_pivot.item.key if r_pivot else None}, using left as its left subtree")
                
                if r_pivot_next is not None:
                    other_min_leaf_tree = r_pivot_next.left_subtree.get_min_leaf_tree()
                else:
                    other_min_leaf_tree = other.node.right_subtree.get_min_leaf_tree()
                
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Found other min leaf tree set: {print_pretty(other_min_leaf_tree.node.set)}")

                if left.node.rank == 1:
                    left.node.next = other_min_leaf_tree
                else:
                    max_left_leaf = left.get_max_leaf()
                    max_left_leaf.next = other_min_leaf_tree

                r_pivot.left_subtree = left
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Set r_pivot {r_pivot.item.key}'s left_subtree to left: {print_pretty(left)}")
                    logger.debug(f"[DIM {left.DIM}] Other after reassignment: {print_pretty(other)}")
                r_pivot, r_pivot_next = other_min_leaf_tree.node.set.find_pivot()  # Update r_pivot after reassignment
                if r_pivot_next is None and other_min_leaf_tree.node.next is not None:
                    if IS_DEBUG():
                        logger.debug(f"[DIM {left.DIM}] Updating r_pivot_next from other_min_leaf_tree's successor: {print_pretty(other_min_leaf_tree.node.next)}")
                    r_pivot_next = other_min_leaf_tree.node.next.node.set.find_pivot()[0]

                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Updated r_pivot after reassignment: {r_pivot.item.key if r_pivot else None}, Next: {r_pivot_next.item.key if r_pivot_next else None}")
                    logger.debug(f"[DIM {left.DIM}] r_pivot left subtree: {print_pretty(r_pivot.left_subtree) if r_pivot and r_pivot.left_subtree else None}")
                    logger.debug(f"[DIM {left.DIM}] r_pivot next left subtree: {print_pretty(r_pivot_next.left_subtree) if r_pivot_next and r_pivot_next.left_subtree else None}")
                other._invalidate_tree_size()

            

            if IS_DEBUG():
                logger.debug(f"[DIM {left.DIM}] new_other (item_count: {other.item_count()}, real_item_count: {other.real_item_count()}): {print_pretty(other)}")
                logger.debug(f"[DIM {left.DIM}] Other Items: {[e.item.key for e in other]}")
                for i, leaf in enumerate(other.iter_leaf_nodes()):
                    logger.debug(f"[DIM {left.DIM}] Other Leaf {i}: {print_pretty(leaf.set)}")
                    logger.debug(f"[DIM {left.DIM}] Leaf next: {print_pretty(leaf.next)}")
                logger.debug(f"[DIM {left.DIM}] Left node next: {print_pretty(left.node.next)}")
                logger.debug(f"[DIM {left.DIM}] new_left (item_count: {left.item_count()}, real_item_count: {left.real_item_count()}): {print_pretty(left)}")
                logger.debug(f"[DIM {left.DIM}] Left Items: {[e.item.key for e in left]}")

            return other, r_pivot, r_pivot_next
        
        # Case 2: Left rank > other rank - recursively zip into right subtree
        elif left.node.rank > other.node.rank:
            # if other.node.rank == 1:
            # new_right = type(left)(l_factor=left.l_factor)

            left.node.right_subtree, r_pivot, r_pivot_next = left.node.right_subtree.zip(left.node.right_subtree, other, is_root=False)

            if IS_DEBUG():
                logger.debug(f"[DIM {left.DIM}] Zipped right subtree (item_count: {left.node.right_subtree.item_count()}, real_item_count: {left.node.right_subtree.real_item_count()}): {print_pretty(left.node.right_subtree)}")
                logger.debug(f"[DIM {left.DIM}] Items right subtree: {[e.item.key for e in left.node.right_subtree]}")
                logger.debug(f"[DIM {left.DIM}] r_pivot: {r_pivot.item.key if r_pivot else None}, r_pivot_next: {r_pivot_next.item.key if r_pivot_next else None}")
                logger.debug(f"[DIM {left.DIM}] r_pivot left subtree: {print_pretty(r_pivot.left_subtree) if r_pivot and r_pivot.left_subtree else None}")
                logger.debug(f"[DIM {left.DIM}] r_pivot next left subtree: {print_pretty(r_pivot_next.left_subtree) if r_pivot_next and r_pivot_next.left_subtree else None}")

            left._invalidate_tree_size()
            if IS_DEBUG():
                logger.debug(f"[DIM {left.DIM}] Left items: {[e.item.key for e in left]}")
                logger.debug(f"[DIM {left.DIM}] Left: (item_count: {left.item_count()}, real_item_count: {left.real_item_count()}): {print_pretty(left)}")
            return left, r_pivot, r_pivot_next

        # Case 3: Same rank - merge sets
        else:
            left, other = self.check_convert_same_sets(left, other)

            # Merge the sets based on their types
            if isinstance(left.node.set, KListBase) and isinstance(other.node.set, KListBase):
                # Both KLists - merge entries directly
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Merging KLists: {print_pretty(left.node.set)} and {print_pretty(other.node.set)}")
                dummy_lower_dim = get_dummy(left.DIM - 1)
                for entry in other.node.set:
                    if entry.item.key > dummy_lower_dim.key:
                        left.node.set, _, _ = left.node.set.insert_entry(entry)
                left.node.set = left.check_and_convert_set(left.node.set)
                # TODO(#1): Add left._invalidate_tree_size() after set conversion
                left._invalidate_tree_size()
                r_pivot, r_pivot_next = other.node.set.find_pivot()
                
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Found r_pivot: {r_pivot.item.key if r_pivot else None}. Pivot left subtree: {print_pretty(r_pivot.left_subtree) if r_pivot and r_pivot.left_subtree else None}")
                    logger.debug(f"[DIM {left.DIM}] Found r_pivot_next: {r_pivot_next.item.key if r_pivot_next else None}. Pivot next left subtree: {print_pretty(r_pivot_next.left_subtree) if r_pivot_next and r_pivot_next.left_subtree else None}")
                
            elif isinstance(left.node.set, GKPlusTreeBase) and isinstance(other.node.set, GKPlusTreeBase):
                # Both are GKPlusTreeBase (after conversion) - recursively zip
                # if IS_DEBUG():
                #     logger.debug(f"[DIM {left.DIM}] Merging GKPlusTrees: {print_pretty(left.node.set)} and {print_pretty(other.node.set)}")
                left.node.set, r_pivot, r_pivot_next = left.node.set.zip(left.node.set, other.node.set)
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Found r_pivot: {r_pivot.item.key if r_pivot else None}. Pivot left subtree: {print_pretty(r_pivot.left_subtree) if r_pivot and r_pivot.left_subtree else None}")
                    logger.debug(f"[DIM {left.DIM}] Found r_pivot_next: {r_pivot_next.item.key if r_pivot_next else None}. Pivot next left subtree: {print_pretty(r_pivot_next.left_subtree) if r_pivot_next and r_pivot_next.left_subtree else None}")
                left.node.set = self.check_and_convert_set(left.node.set)
                # TODO(#1): Add left._invalidate_tree_size() after set conversion
                left._invalidate_tree_size()
            else:
                raise TypeError(f"Set types should match after conversion, got {type(left.node.set).__name__} and {type(other.node.set).__name__}")


            # Link leaf nodes
            if left.node.rank == 1:
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Linking leaf leaf nodes.")
                    logger.debug(f"[DIM {left.DIM}] Leaf to be zipped into: {print_pretty(left)}")
                    logger.debug(f"[DIM {left.DIM}] Linked leaf: {print_pretty(other.node.next)}")
                left.node.next = other.node.next
                left._invalidate_tree_size()
                if r_pivot_next is None and other.node.next is not None:
                    if IS_DEBUG():
                        logger.debug(f"[DIM {left.DIM}] Updating r_pivot_next from other's successor: {print_pretty(other.node.next)}")
                    r_pivot_next = other.node.next.node.set.find_pivot()[0]
                # left.node.set = left.check_and_convert_set(left.node.set)
                return left, r_pivot, r_pivot_next
            
            
            else:
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] r_pivot.left_subtree: {print_pretty(r_pivot.left_subtree)}")
                    
                if r_pivot.left_subtree is None:
                    if IS_DEBUG():
                        logger.debug(f"[DIM {left.DIM}] No left subtree found for r_pivot {r_pivot.item.key}, using left.node.right_subtree as its left subtree")
                    
                    if r_pivot_next is not None:
                        other_min_leaf_tree = r_pivot_next.left_subtree.get_min_leaf_tree()
                    else:
                        other_min_leaf_tree = other.node.right_subtree.get_min_leaf_tree()

                    # other_min_leaf_tree = other.get_min_leaf_tree()
                    if IS_DEBUG():
                        logger.debug(f"[DIM {left.DIM}] Other: {print_pretty(other)}")
                        logger.debug(f"[DIM {left.DIM}] Found other min leaf tree: {print_pretty(other_min_leaf_tree)}")
                        

                    max_left_leaf = left.node.right_subtree.get_max_leaf()
                    if IS_DEBUG():
                        logger.debug(f"[DIM {left.DIM}] Found max left leaf tree: {max_left_leaf}")
                        logger.debug(f"node with set: {print_pretty(max_left_leaf.set)}")
                        logger.debug(f"[DIM {left.DIM}] Linking left to right")
                    max_left_leaf.next = other_min_leaf_tree

                    r_pivot.left_subtree = left.node.right_subtree
                    r_pivot, r_pivot_next = other_min_leaf_tree.node.set.find_pivot()
                    if r_pivot_next is None and other_min_leaf_tree.node.next is not None:
                        if IS_DEBUG():
                            logger.debug(f"[DIM {left.DIM}] Updating r_pivot_next from other_min_leaf_tree's successor: {print_pretty(other_min_leaf_tree.node.next)}")
                        r_pivot_next = other_min_leaf_tree.node.next.node.set.find_pivot()[0]
                else:
                    if IS_DEBUG():
                        logger.debug(f"[DIM {left.DIM}] Found left subtree for r_pivot {r_pivot.item.key}: {print_pretty(r_pivot.left_subtree)}")
                        logger.debug(f"[DIM {left.DIM}] Left right subtree before zip: {print_pretty(left.node.right_subtree)}")
                    r_pivot.left_subtree, r_pivot_sub, r_pivot_sub_next = left.node.right_subtree.zip(left.node.right_subtree, r_pivot.left_subtree)
                    if IS_DEBUG():
                        logger.debug(f"[DIM {left.DIM}] New left after zip: {print_pretty(r_pivot.left_subtree)}")
                        logger.debug(f"[DIM {left.DIM}] r_pivot after zip, before assignment: {r_pivot.item.key if r_pivot else None}")
                        logger.debug(f"[DIM {left.DIM}] r_pivot left subtree after zip, before assignment: {print_pretty(r_pivot.left_subtree)}")

                    r_pivot.left_subtree._invalidate_tree_size()
                    other._invalidate_tree_size()
                    r_pivot = r_pivot_sub
                    r_pivot_next = r_pivot_sub_next

                left.node.right_subtree = other.node.right_subtree
                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] r_pivot after handling left subtree: {r_pivot.item.key if r_pivot else None}")
                    logger.debug(f"[DIM {left.DIM}] r_pivot.left_subtree: {print_pretty(r_pivot.left_subtree)}")
                    logger.debug(f"[DIM {left.DIM}] Left: {print_pretty(left)}")
                    logger.debug(f"[DIM {left.DIM}] Other: {print_pretty(other)}")
                    logger.debug(f"[DIM {left.DIM}] root node next: {print_pretty(left.node.next)}")

                if IS_DEBUG():
                    logger.debug(f"[DIM {left.DIM}] Zipped sets: {print_pretty(left)}")

            return left, r_pivot, r_pivot_next
