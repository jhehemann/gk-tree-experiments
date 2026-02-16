"""Insertion logic for GK+-trees.

Provides :class:`GKPlusInsertMixin`, a mixin class that adds the core
insertion helpers (``_insert_empty``, ``_insert_non_empty``,
``_insert_new_item``, etc.) to :class:`GKPlusTreeBase`.

Single-dimension complexity (n = items, k = KList capacity,
h = height ≈ log n, l = KList nodes):

+---------------------------+----------------------------------------------+
| Operation                 | Time (per dimension)                         |
+===========================+==============================================+
| ``_insert_empty``         | O(k)  (builds leaf KList)                    |
| ``_insert_non_empty``     | O(h · (log l + k))  amortised                |
| ``_insert_new_item``      | O(h · (log l + k))  amortised                |
| ``_insert_first_iteration``| O(log l + k) per node                       |
| ``_insert_with_split``    | O(log l + k) per node                        |
| ``_handle_rank_mismatch`` | O(k)  (set creation)                         |
+---------------------------+----------------------------------------------+

When a leaf's ``node.set`` is a GK+-tree of dimension d+1 (after KList
expansion), set-level operations (``insert_entry``, ``split_inplace`` /
``unzip``) recurse into that inner tree. The total worst-case cost
across D active dimensions is
O(h_1 · h_2 · … · h_D · (log l + k)).
In practice, inner trees are bounded by ~k · l_factor items, keeping
h_{d+1} small.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

from gplus_trees.utils import calc_rank_from_digest_k
from gplus_trees.base import (
    AbstractSetDataStructure,
    Entry,
    _get_replica,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import get_dummy
from gplus_trees.g_k_plus.bulk_create import bulk_create_gkplus_tree

if TYPE_CHECKING:
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, GKPlusNodeBase


class GKPlusInsertMixin:
    """Mixin that contributes insertion methods to *GKPlusTreeBase*."""

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update(self, cur, x_entry) -> Tuple['GKPlusTreeBase', bool]:
        """Placeholder for future update logic (e.g. value replacement)."""
        raise NotImplementedError(
            f"Entry with key {x_entry.item.key} already exists in the "
            "tree. Updates are not yet implemented."
        )

    # ------------------------------------------------------------------
    # Leaf construction helpers
    # ------------------------------------------------------------------

    def _make_leaf_klist(self, x_entry: Entry) -> AbstractSetDataStructure:
        """Builds a KList for a single leaf node containing the dummy and x_item."""
        SetClass = self.SetClass

        # start with a fresh empty set of entries
        leaf_set = SetClass()

        # insert the dummy entry, pointing at an empty subtree
        leaf_set, _, _ = leaf_set.insert_entry(Entry(get_dummy(type(self).DIM), None))

        # now insert the real item, also pointing at an empty subtree
        leaf_set, _, _ = leaf_set.insert_entry(x_entry)

        return leaf_set

    def _make_leaf_trees(self, x_entry: Entry) -> Tuple['GKPlusTreeBase', 'GKPlusTreeBase']:
        """
        Builds two linked leaf-level GKPlusTreeBase nodes for x_item insertion
        and returns the corresponding G+-trees.
        """
        TreeK = type(self)
        NodeK = self.NodeClass
        SetK = self.SetClass

        # Build right leaf
        right_set = SetK()
        right_set, _, _ = right_set.insert_entry(x_entry)
        right_node = NodeK(1, right_set, None)
        right_leaf = TreeK(right_node, self.l_factor)

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set, _, _ = left_set.insert_entry(Entry(get_dummy(type(self).DIM), None))
        left_node = NodeK(1, left_set, None)
        left_leaf = TreeK(left_node, self.l_factor)

        # Link leaves
        left_leaf.node.next = right_leaf
        return left_leaf, right_leaf

    # ------------------------------------------------------------------
    # Core insertion methods
    # ------------------------------------------------------------------

    def _insert_empty(self, x_entry: Entry, rank: int) -> 'GKPlusTreeBase':
        """Build the initial tree structure depending on rank."""
        # Single-level leaf
        inserted = True
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_entry)
            self.node = self.NodeClass(rank, leaf_set, None)
            return self, inserted, None

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_entry)
        root_set, _, _ = self.SetClass().insert_entry(Entry(get_dummy(dim=self.DIM), None))
        root_set, _, _ = root_set.insert_entry(Entry(_get_replica(x_entry.item), l_leaf_t))
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        return self, inserted, None

    def _insert_non_empty(self, x_entry: Entry, rank: int) -> 'GKPlusTreeBase':
        """Optimized version for inserting into a non-empty tree."""
        x_item = x_entry.item
        x_key = x_item.key

        cur = self
        parent = None
        p_next_entry = None

        # Loop until we find where to insert
        while True:
            cur._invalidate_tree_size()
            node = cur.node
            node_rank = node.rank  # Cache attribute access

            # Case 1: Found node with matching rank - ready to insert
            if node_rank == rank:
                return self._insert_new_item(cur, x_entry)

            # Case 2: Current rank too small - handle rank mismatch
            if node_rank < rank:
                cur = self._handle_rank_mismatch(cur, parent, p_next_entry, rank)
                continue

            # Case 3: Descend to next level (current rank > rank)
            parent = cur
            next_entry = node.set.retrieve(x_key)[1]
            if next_entry:
                cur = next_entry.left_subtree
            else:
                cur = node.right_subtree
            p_next_entry = next_entry

    def _handle_rank_mismatch(
        self,
        cur: 'GKPlusTreeBase',
        parent: 'GKPlusTreeBase',
        p_next: Entry,
        rank: int,
        left_parent: Optional['GKPlusTreeBase'] = None,
    ) -> 'GKPlusTreeBase':
        """
        If the current node's rank < rank, we need to create or unfold a
        node to match the new rank.
        This is done by creating a new G+-node and linking it to the parent.
        Attributes:
            cur (GKPlusTreeBase): The current G+-tree.
            parent (GKPlusTreeBase): The parent G+-tree.
            p_next (tuple): The next entry in the parent tree.
            rank (int): The rank to match.
        Returns:
            GKPlusTreeBase: The updated G+-tree.
        """
        TreeClass = type(self)

        if parent is None and left_parent is None:

            # create a new root node
            old_node = self.node
            dummy = get_dummy(dim=TreeClass.DIM)
            root_set, _, _ = self.SetClass().insert_entry(Entry(dummy, None))
            self.node = self.NodeClass(
                rank,
                root_set,
                TreeClass(old_node, self.l_factor)
            )
            return self

        # Unfold intermediate node between parent and current
        # Locate the current node's pivot and place its replica first in the intermediate node.

        pivot = cur.node.set.find_pivot()[0]
        pivot_replica = _get_replica(pivot.item)
        new_set, _, _ = self.SetClass().insert_entry(Entry(pivot_replica, None))
        new_tree = TreeClass(l_factor=self.l_factor)
        new_tree.node = self.NodeClass(rank, new_set, cur)

        if p_next:
            p_next.left_subtree = new_tree
        else:
            parent.node.right_subtree = new_tree

        if left_parent is not None:
            # If we have a left parent, we need to link the new tree to it
            left_parent.node.right_subtree = new_tree

        return new_tree

    def _insert_min(
        self,
        x_entry: Entry,
        rank: int,
        add_dummy: bool = True,
    ) -> Tuple['GKPlusTreeBase', bool, Optional[Entry], Optional[Entry]]:
        """Insert an entry that is the new minimum into the tree.

        Args:
            x_entry: The entry to insert.
            rank: The rank for the entry.
            add_dummy: Whether to add a dummy entry for the dimension.

        Returns:
            Tuple of (tree, inserted, pivot_entry, next_entry).

        Raises:
            TypeError: If *x_entry* is not an :class:`Entry`.
            NotImplementedError: When the tree is non-empty (not yet implemented).
        """
        if not isinstance(x_entry, Entry):
            raise TypeError(f"insert_min(): expected Entry, got {type(x_entry).__name__}")
        if self.is_empty():
            return self._insert_empty(x_entry, rank)
        raise NotImplementedError("_insert_min into non-empty tree is not yet implemented")

    # ------------------------------------------------------------------
    # _insert_new_item – broken into first-iteration and split phases
    # ------------------------------------------------------------------

    def _insert_new_item(
        self,
        cur: 'GKPlusTreeBase',
        x_entry: Entry,
    ) -> 'GKPlusTreeBase':
        """
        Insert a new item key. For internal nodes, we only store the key.
        For leaf nodes, we store the full item.

        Args:
            cur: The current G+-tree node where insertion starts
            x_entry: The entry to be inserted

        Returns:
            The updated G+-tree
        """
        # Avoid circular import at module level
        from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

        # Pre-cache all frequently used values to minimize overhead
        x_item = x_entry.item
        x_key = x_item.key
        replica = _get_replica(x_item)
        TreeClass = type(self)
        NodeClass = self.NodeClass
        check_and_convert_set = self.check_and_convert_set
        l_factor = self.l_factor
        capacity = self.KListClass.KListNodeClass.CAPACITY

        # Parent tracking variables
        right_parent = None
        right_entry = None
        left_parent = None
        left_x_entry = None

        while True:
            cur._invalidate_tree_size()
            node = cur.node
            is_leaf = node.rank == 1

            # ── First iteration (no split required) ──────────────────
            if right_parent is None:
                result = self._insert_first_iteration(
                    cur, node, x_entry, x_item, x_key, replica,
                    is_leaf, check_and_convert_set, capacity,
                    GKPlusTreeBase,
                )
                # result is either a final return tuple or a continuation tuple
                if result[0] == "_continue":
                    _, right_parent, left_parent, right_entry, left_x_entry, cur = result
                    continue
                return result

            # ── Subsequent iterations (split + reconstruct) ──────────
            result = self._insert_with_split(
                cur, node, x_entry, x_item, x_key, replica,
                is_leaf, check_and_convert_set, capacity,
                right_parent, right_entry, left_parent, left_x_entry,
                TreeClass, NodeClass, l_factor,
                GKPlusTreeBase,
            )
            if result[0] == "_continue":
                _, right_parent, right_entry, left_parent, left_x_entry, cur = result
                continue
            return result

    def _insert_first_iteration(
        self,
        cur, node, x_entry, x_item, x_key, replica,
        is_leaf, check_and_convert_set, capacity,
        GKPlusTreeBase,
    ):
        """Handle the first iteration of ``_insert_new_item`` (no split).

        Returns either a final ``(tree, inserted, next_entry)`` tuple or a
        continuation sentinel ``("_continue", right_parent, left_parent,
        right_entry, left_x_entry, cur)`` so the caller can proceed to the
        split phase.
        """
        is_gkplus_type = isinstance(node.set, GKPlusTreeBase)
        insert_entry = x_entry if is_leaf else Entry(replica, None)
        if not is_gkplus_type:
            res = node.set.insert_entry(insert_entry)
            node.set, inserted, next_entry = res[0], res[1], res[2]
            if not inserted:
                return self.update(cur, x_entry)
            node.set = check_and_convert_set(node.set)  # only KLists can be extended
        else:
            digest = x_item.get_digest_for_dim(self.DIM + 1)
            new_rank = calc_rank_from_digest_k(digest, capacity)
            res = node.set.insert_entry(insert_entry, rank=new_rank)
            node.set, inserted, next_entry = res[0], res[1], res[2]
            if not inserted:
                return self.update(cur, x_entry)

        subtree = next_entry.left_subtree if next_entry else node.right_subtree
        insert_entry.left_subtree = subtree if not is_leaf else insert_entry.left_subtree

        # Fastest path for leaf nodes - direct return
        if is_leaf:
            self._invalidate_tree_size()
            if next_entry is None and node.next is not None:
                next_entry = node.next.find_pivot()[0]
            return self, True, next_entry

        # Signal to caller to continue into the split phase
        return (
            "_continue",
            cur,       # right_parent
            cur,       # left_parent
            next_entry,  # right_entry
            insert_entry,  # left_x_entry
            subtree,   # new cur
        )

    def _insert_with_split(
        self,
        cur, node, x_entry, x_item, x_key, replica,
        is_leaf, check_and_convert_set, capacity,
        right_parent, right_entry, left_parent, left_x_entry,
        TreeClass, NodeClass, l_factor,
        GKPlusTreeBase,
    ):
        """Handle subsequent iterations of ``_insert_new_item`` (split phase).

        Returns either a final ``(tree, inserted, next_entry)`` tuple or a
        continuation sentinel ``("_continue", right_parent, right_entry,
        left_parent, left_x_entry, cur)`` so the caller loops again.
        """

        # Perform split operation and cache converted results
        if isinstance(node.set, KListBase):
            left_split, _, right_split, next_entry = node.set.split_inplace(x_key)
        else:
            left_split, _, right_split, next_entry = node.set.unzip(x_key)

        left_split = check_and_convert_set(left_split)

        # Cache item counts early to avoid repeated method calls in conditionals
        right_item_count = right_split.item_count()
        left_item_count = left_split.item_count()

        # Handle right side creation
        new_tree = None
        if right_item_count > 0 or is_leaf:
            insert_entry = x_entry if is_leaf else Entry(replica, None)
            if isinstance(right_split, GKPlusTreeBase):
                digest = x_item.get_digest_for_dim(self.DIM + 1)
                new_rank = calc_rank_from_digest_k(digest, capacity)
                tree_insert = bulk_create_gkplus_tree([insert_entry], self.DIM + 1, l_factor=self.l_factor, KListClass=self.SetClass)

                right_split, r_pivot, r_pivot_next = tree_insert.zip(tree_insert, right_split)
                right_split._invalidate_tree_size()
            else:
                right_split, _, _ = right_split.insert_entry(insert_entry)

            # Create new tree node
            right_split = check_and_convert_set(right_split)
            new_tree = TreeClass(l_factor=l_factor)

            new_tree.node = NodeClass(node.rank, right_split, node.right_subtree)

        # Optimized parent reference updates with minimal branching
        if new_tree:
            if right_entry is not None:
                right_entry.left_subtree = new_tree
            else:
                right_parent.node.right_subtree = new_tree
            next_right_parent = new_tree
            next_right_entry = next_entry
        else:
            next_right_parent = right_parent
            next_right_entry = right_entry

        right_parent = next_right_parent
        right_entry = next_right_entry

        # Handle left side with optimized control flow
        if left_item_count > 1 or is_leaf:
            cur.node.set = left_split
            if next_entry:
                cur.node.right_subtree = next_entry.left_subtree

            if left_x_entry is not None:
                # Only True once: When we have a left split and not yet updated the x_entry
                # inserted (without split) into the initial node
                left_x_entry.left_subtree = cur

            next_left_parent = cur
            next_left_x_entry = None  # Left split never contains x_item
            next_cur = cur.node.right_subtree
        else:
            # Collapse single-item nodes for non-leaves
            new_subtree = (
                next_entry.left_subtree if next_entry is not None else cur.node.right_subtree
            )

            if left_x_entry is not None:
                left_x_entry.left_subtree = new_subtree
            else:
                left_parent.node.right_subtree = new_subtree

            next_left_parent = left_parent
            next_left_x_entry = left_x_entry
            next_cur = new_subtree

        left_parent = next_left_parent
        left_x_entry = next_left_x_entry

        # Handle leaf level with early return optimization
        if is_leaf:
            if next_entry is None and node.next is not None:
                    next_entry = node.next.find_pivot()[0]
            if new_tree and cur:
                new_tree.node.next = cur.node.next
                cur.node.next = new_tree

                # Right subtrees of left splits are not reset in higher dimensional leaf nodes.
                # Do it here to maintain search tree structure across dimensions.
                cur.node.right_subtree = None  # No right subtree at leaf level
            self._invalidate_tree_size()
            return self, True, next_entry

        # Signal to caller to continue
        return ("_continue", right_parent, right_entry, left_parent, left_x_entry, next_cur)

    # ------------------------------------------------------------------
    # Dummy singleton tree
    # ------------------------------------------------------------------

    def _create_dummy_singleton_tree(self, rank: int, dim: int, l_factor: float, KListClass) -> 'GKPlusTreeBase':
        """Create a dummy singleton tree with a single entry containing the dummy item.

        Args:
            rank: The rank for the new tree node.
            dim: The dimension of the tree.
            l_factor: The load factor for the tree.
            KListClass: The KList class to use for creating the leaf set.

        Returns:
            A GKPlusTreeBase instance with a single dummy entry.
        """
        TreeK = type(self)
        NodeK = self.NodeClass
        SetK = self.SetClass
        dummy_item = get_dummy(dim)
        dummy_entry = Entry(dummy_item, None)

        # Create a leaf node with a single dummy entry
        leaf_set = KListClass()
        leaf_set, _, _ = leaf_set.insert_entry(dummy_entry)
        node = NodeK(1, leaf_set, None)
        leaf_tree = TreeK(node, l_factor)

        if rank == 1:
            return leaf_tree

        # Create a root node with the dummy entry and the leaf tree as right subtree
        root_set = SetK()
        root_set, _, _ = root_set.insert_entry(Entry(dummy_item, None))
        root_node = NodeK(rank, root_set, leaf_tree)
        return TreeK(root_node, l_factor)
