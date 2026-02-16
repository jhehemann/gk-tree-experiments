"""G+-tree base implementation.

Provides :class:`GPlusNodeBase` (a single tree node) and
:class:`GPlusTreeBase` (the recursively-defined tree), both intended to
be subclassed via factory-created specialisations.

Time-complexity summary (n = items, k = KList capacity, l = KList nodes):

+-----------------------+---------------------------------------+
| Operation             | Time                                  |
+=======================+=======================================+
| ``insert``            | O(log n · (log l + k))  amortised      |
| ``retrieve``          | O(log n · (log l + log k))             |
| ``physical_height``   | O(n)  (recursive)                     |
+-----------------------+---------------------------------------+
"""

from __future__ import annotations
import functools
from typing import Optional, Tuple, Type, Union

from gplus_trees.base import (
    AbstractSetDataStructure,
    InsertResult,
    ItemData,
    LeafItem,
    InternalItem,
    DummyItem,
    Entry,
    _get_replica,
)
from gplus_trees.klist_base import KListBase
import logging
from gplus_trees.logging_config import get_logger

logger = get_logger(__name__)

# ── Backward-compatible re-exports ──────────────────────────────────
# These symbols used to live here; they now live in dedicated modules
# but are re-exported so existing ``from gplus_trees.gplus_tree_base
# import …`` statements keep working.
from gplus_trees.tree_stats import Stats, gtree_stats_, _get_capacity  # noqa: F401
from gplus_trees.display import print_pretty, collect_leaf_keys, print_structure as _print_structure_fn  # noqa: F401

# Constants
DUMMY_KEY = -1
DUMMY_ITEM = DummyItem(ItemData(key=DUMMY_KEY))


@functools.lru_cache(maxsize=None)
def get_dummy(dim: int) -> DummyItem:
    """Return a cached :class:`DummyItem` for the given dimension.

    The key is ``-dim`` (e.g. dimension 1 → key ``-1``, dimension 2 → key ``-2``).
    Results are cached so successive calls for the same *dim* return the
    identical object.
    """
    return DummyItem(ItemData(key=-(dim)))


class _SplitContext:
    """Mutable context that tracks parent references during the
    split-and-relink phase of :pymeth:`GPlusTreeBase._insert_new_item`.
    """
    __slots__ = ('right_parent', 'right_entry', 'left_parent', 'left_x_entry')

    def __init__(
        self,
        right_parent: GPlusTreeBase,
        right_entry: Optional[Entry],
        left_parent: GPlusTreeBase,
        left_x_entry: Optional[Entry],
    ) -> None:
        self.right_parent = right_parent
        self.right_entry = right_entry
        self.left_parent = left_parent
        self.left_x_entry = left_x_entry

class GPlusNodeBase:
    """
    Base class for G+-tree nodes. Factory will set:
      - SetClass  : which AbstractSetDataStructure to use for entries
      - TreeClass : which GPlusTree to build for child/subtree pointers
    """
    __slots__ = ("rank", "set", "right_subtree", "next")

    # set by factory
    SetClass: Type[AbstractSetDataStructure]
    TreeClass: Type[GPlusTreeBase]

    def __init__(
        self,
        rank: int,
        set: AbstractSetDataStructure,
        right: Optional[GPlusTreeBase] = None
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.rank = rank
        self.set = set
        self.right_subtree = right
        self.next = None    # leaf‐chain pointer
    
class GPlusTreeBase(AbstractSetDataStructure):
    """
    A G+-tree is a recursively defined structure that is either empty or contains a single G+-node.
    Attributes:
        node (Optional[GPlusNode]): The G+-node that the tree contains. If None, the tree is empty.
    
    TODO (Issue #5): Add first_leaf pointer to enable O(1) access to first leaf node:
        - Add 'first_leaf' to __slots__ tuple
        - Initialize in __init__ method
        - Maintain during insertions and tree modifications
        - This will significantly speed up sequential access operations
    """
    __slots__ = ("node",)  # TODO(#5): Add "first_leaf" here
    
    # set by factory
    NodeClass: Type[GPlusNodeBase]
    SetClass: Type[AbstractSetDataStructure]
    
    def __init__(self, node: Optional[GPlusNodeBase] = None):
        self.node: Optional[GPlusNodeBase] = node

    def is_empty(self) -> bool:
        return self.node is None
    
    def __str__(self):
        return "Empty GPlusTree" if self.is_empty() else f"GPlusTree(node={self.node})"

    __repr__ = __str__

    # ── Hooks for subclass customisation ──────────────────────────────
    def _get_dummy(self) -> DummyItem:
        """Return the dummy item for this tree type.

        Override in subclasses for dimension-specific dummies.
        """
        return DUMMY_ITEM

    def _on_node_visited(self, cur: GPlusTreeBase) -> None:
        """Hook called when a node is visited during insertion.

        Override to invalidate caches, bump counters, etc.
        """

    def _make_empty_tree(self) -> GPlusTreeBase:
        """Create a new empty tree of the same concrete type.

        Override in subclasses if the constructor requires extra
        parameters (e.g. ``l_factor``).
        """
        return type(self)()

    # ── Public API ───────────────────────────────────────────────────
    def insert(self, x: Union[InternalItem, LeafItem], rank: int, x_left: Optional[GPlusTreeBase] = None) -> InsertResult:
        """Insert an item into the G+-tree (amortised O(log n)).

        If an item with the same key already exists, its value is updated
        at the leaf level and ``InsertResult.inserted`` is ``False``.

        Args:
            x: The item (key, value) to insert.
            rank: The rank of the item.  Must be a positive integer.
            x_left: Optional left subtree to attach to the new entry.

        Returns:
            InsertResult: ``(tree, inserted, next_entry)`` where *tree*
            is the (mutated) root, *inserted* indicates whether a new key
            was added, and *next_entry* is the in-order successor.

        Raises:
            TypeError: If *x* is not an ``InternalItem`` / ``LeafItem``
                or *rank* is not a positive ``int``.
        """
        if not isinstance(x, Union[InternalItem, LeafItem]):
            raise TypeError(f"insert(): expected InternalItem or LeafItem, got {type(x).__name__}")
        if not isinstance(rank, int) or rank <= 0:
            raise TypeError(f"insert(): rank must be a positive int, got {rank!r}")
        insert_entry = Entry(x, x_left)
        if self.is_empty():
            return self._insert_empty(insert_entry, rank)
        return self._insert_non_empty(insert_entry, rank)

    def retrieve(
        self, key: int, with_next: bool = True
    ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Search for *key* in the G+-tree.

        Iteratively descends the tree (O(log n) levels, each doing
        O(log l + log k) work in the KList), then performs a final
        lookup in the leaf node's KList.

        Args:
            key: The integer key to search for.
            with_next: If ``True`` (default) the in-order successor is
                also returned.  Pass ``False`` to skip the successor
                lookup when only existence / value matters.

        Returns:
            A ``(found_entry, next_entry)`` tuple.  *found_entry* is the
            matching :class:`Entry` or ``None``; *next_entry* is the
            in-order successor or ``None``.
        """
        if not isinstance(key, int):
            raise TypeError(f"retrieve(): key must be an int, got {type(key).__name__}")

        if self.is_empty():
            return None, None

        leaf_node = self._descend_to_leaf(key)
        return self._find_in_leaf(leaf_node, key, with_next)

    def _descend_to_leaf(self, key: int) -> GPlusNodeBase:
        """Navigate from the root to the leaf node that should contain *key*.

        At each internal level, ``set.retrieve`` (O(log l + log k))
        determines whether to follow a ``left_subtree`` or the
        ``right_subtree``.  Total: O(height · (log l + log k)).
        """
        cur = self
        while True:
            node = cur.node
            if node.rank == 1:
                return node
            _, next_entry = node.set.retrieve(key, with_next=True)
            cur = next_entry.left_subtree if next_entry else node.right_subtree

    @staticmethod
    def _find_in_leaf(
        leaf_node: GPlusNodeBase, key: int, with_next: bool
    ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """Search for *key* in a leaf node and optionally find the next entry."""
        found_entry, next_entry = leaf_node.set.retrieve(key, with_next=with_next)
        if not with_next:
            return found_entry, None
        if next_entry is None and leaf_node.next is not None:
            next_node = leaf_node.next.node
            for entry in next_node.set:
                if entry.item.key > key:
                    next_entry = entry
                    break
        return found_entry, next_entry

    def delete(self, item):
        raise NotImplementedError("delete not implemented yet")

    # Private Methods
    def _make_leaf_klist(self, x_entry: Entry) -> AbstractSetDataStructure:
        """Builds a KList for a single leaf node containing the dummy and x_item."""
        SetClass = self.SetClass
        dummy = self._get_dummy()

        leaf_set = SetClass()
        leaf_set, _, _ = leaf_set.insert_entry(Entry(dummy, None))
        leaf_set, _, _ = leaf_set.insert_entry(x_entry)
        return leaf_set

    def _make_leaf_trees(self, x_entry: Entry) -> Tuple[GPlusTreeBase, GPlusTreeBase]:
        """
        Builds two linked leaf-level GPlusTreeBase nodes for x_item insertion
        and returns the corresponding G+-trees.
        """
        NodeK = self.NodeClass
        SetK = self.SetClass
        dummy = self._get_dummy()

        # Build right leaf
        right_set = SetK()
        right_set, _, _ = right_set.insert_entry(x_entry)
        right_node = NodeK(1, right_set, None)
        right_leaf = self._make_empty_tree()
        right_leaf.node = right_node

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set, _, _ = left_set.insert_entry(Entry(dummy, None))
        left_node = NodeK(1, left_set, None)
        left_leaf = self._make_empty_tree()
        left_leaf.node = left_node

        # Link leaves
        left_leaf.node.next = right_leaf
        # TODO(#5): Set first_leaf pointer here
        return left_leaf, right_leaf

    def _insert_empty(self, insert_entry: Entry, rank: int) -> InsertResult:
        """Build the initial tree structure depending on rank."""
        dummy = self._get_dummy()
        # Single-level leaf
        if rank == 1:
            leaf_set = self._make_leaf_klist(insert_entry)
            self.node = self.NodeClass(rank, leaf_set, None)
            return InsertResult(self, True, None)

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(insert_entry)
        root_set, _, _ = self.SetClass().insert_entry(Entry(dummy, None))
        root_set, _, _ = root_set.insert_entry(Entry(_get_replica(insert_entry.item), l_leaf_t))
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        return InsertResult(self, True, None)

    def _insert_non_empty(self, x_entry: Entry, rank: int) -> InsertResult:
        """Insert into a non-empty tree.

        Walks down the tree until a node whose rank matches *rank* is
        found, potentially inserting intermediate nodes via
        :meth:`_handle_rank_mismatch`.  Delegates to
        :meth:`_update_existing_item` (existing key) or
        :meth:`_insert_new_item` (new key).

        Time complexity: O(height · (log l + k)) amortised, dominated by
        KList ``insert_entry`` and ``split_inplace`` at each level.
        """
        x_item = x_entry.item
        x_key = x_item.key
        cur = self
        parent = None
        p_next_entry = None

        while True:
            node = cur.node
            node_rank = node.rank

            # Case 1: Found node with matching rank
            if node_rank == rank:
                res = node.set.retrieve(x_key)
                existing_x_entry = res[0]
                next_entry = res[1]
                # Fast path: update existing item
                if existing_x_entry:
                    if x_entry.left_subtree is not None:
                        raise ValueError(
                            f"Entry with key {x_key} already exists in the tree. "
                            "Cannot be inserted with a subtree again."
                        )
                    if rank == 1:
                        existing_x_entry.item.value = x_item.value
                        return InsertResult(self, False, next_entry)
                    return self._update_existing_item(cur, x_item, next_entry)

                return self._insert_new_item(cur, x_entry, next_entry)

            # Case 2: Current rank too small
            if node_rank < rank:
                cur = self._handle_rank_mismatch(cur, parent, p_next_entry, rank)
                continue

            # Case 3: Descend to next level
            res = node.set.retrieve(x_key)
            parent = cur
            next_entry = res[1]
            cur = next_entry.left_subtree if next_entry else node.right_subtree
            p_next_entry = next_entry

    def _handle_rank_mismatch(
        self,
        cur: GPlusTreeBase,
        parent: GPlusTreeBase,
        p_next: Entry,
        rank: int
    ) -> GPlusTreeBase:
        """Create or unfold a node to match *rank* when the current node's
        rank is too small.
        """
        dummy = self._get_dummy()

        if parent is None:
            old_node = self.node
            root_set, _, _ = self.SetClass().insert_entry(Entry(dummy, None))
            child_tree = self._make_empty_tree()
            child_tree.node = old_node
            self.node = self.NodeClass(rank, root_set, child_tree)
            return self

        # Unfold intermediate node between parent and current
        min_entry = cur.node.set.get_min()[0]
        min_replica = _get_replica(min_entry.item)
        new_set, _, _ = self.SetClass().insert_entry(Entry(min_replica, None))
        new_tree = self._make_empty_tree()
        new_tree.node = self.NodeClass(rank, new_set, cur)

        if p_next:
            p_next.left_subtree = new_tree
        else:
            parent.node.right_subtree = new_tree

        return new_tree

    def _update_existing_item(
        self,
        cur: GPlusTreeBase,
        new_item: Union[InternalItem, LeafItem],
        next_entry: Optional[Entry],
    ) -> InsertResult:
        """Descend from *cur* to the leaf and update the item value in-place.

        Only the leaf entry carries mutable *value* data, so an update
        must always reach rank 1.  The *next_entry* captured at the
        matching-rank level is forwarded unchanged.

        Time complexity: O(height * (log l + log k)) — one ``retrieve``
        per level, dominated by the tree height.
        """
        key = new_item.key
        while True:
            node = cur.node
            if node.rank == 1:
                # At leaf: only need the found entry, not the successor.
                entry = node.set.retrieve(key, with_next=False)[0]
                if entry:
                    entry.item.value = new_item.value
                return InsertResult(self, False, next_entry)
            successor = node.set.retrieve(key)[1]
            cur = successor.left_subtree if successor else node.right_subtree

    def _insert_new_item(
        self,
        cur: GPlusTreeBase,
        x_entry: Entry,
        next_entry: Entry,
    ) -> InsertResult:
        """Insert a brand-new item key into the tree.

        For internal nodes only a key replica is stored; for leaf nodes
        the full item is stored.  The first matching-rank node gets a
        simple insert.  Every subsequent level down to the leaves is
        split in place, with left and right halves relinked via a
        :class:`_SplitContext`.

        Args:
            cur: The current G+-tree node where insertion starts.
            x_entry: The entry to be inserted.
            next_entry: The next entry in the tree relative to *x_entry*.

        Returns:
            InsertResult with the updated tree, insertion flag, and next entry.
        """
        x_item = x_entry.item
        x_key = x_item.key
        replica = _get_replica(x_item)
        node = cur.node

        # Leaf: direct insert, no splitting needed
        if node.rank == 1:
            node.set, _, _ = node.set.insert_entry(x_entry)
            return InsertResult(self, True, next_entry)

        # Internal node: insert a replica and set up split context
        subtree = next_entry.left_subtree if next_entry else node.right_subtree
        insert_entry = Entry(replica, subtree)
        node.set, _, _ = node.set.insert_entry(insert_entry)

        # insert_entry is stored by reference in the KList, so we can
        # use it directly — no need for a second O(log l + log k) retrieve.
        ctx = _SplitContext(
            right_parent=cur,
            right_entry=next_entry if next_entry else None,
            left_parent=cur,
            left_x_entry=insert_entry,
        )
        cur = subtree

        # Split-and-relink at each level until we reach a leaf
        while True:
            node = cur.node
            is_leaf = node.rank == 1

            left_split, _, right_split, next_entry = node.set.split_inplace(x_key)

            right_insert = x_entry if is_leaf else Entry(replica, None)
            new_tree = self._handle_right_split(
                right_split, right_insert, is_leaf, node, next_entry, ctx
            )
            next_cur = self._handle_left_split(
                cur, left_split, next_entry, is_leaf, ctx
            )

            if is_leaf:
                self._link_split_leaves(cur, new_tree)
                return InsertResult(self, True, next_entry)

            cur = next_cur

    # -- helpers extracted from _insert_new_item -------------------------

    def _handle_right_split(
        self,
        right_split: AbstractSetDataStructure,
        insert_entry: Entry,
        is_leaf: bool,
        node: GPlusNodeBase,
        next_entry: Optional[Entry],
        ctx: _SplitContext,
    ) -> Optional[GPlusTreeBase]:
        """Insert the new item into the right half of a split and relink
        the parent.  Updates *ctx* in-place.

        Returns the newly created tree, or ``None`` when the right split
        was empty and we are not at a leaf.
        """
        if right_split.item_count() > 0 or is_leaf:
            right_split, _, _ = right_split.insert_entry(insert_entry)
            new_tree = self._make_empty_tree()
            new_tree.node = self.NodeClass(node.rank, right_split, node.right_subtree)

            if ctx.right_entry is not None:
                ctx.right_entry.left_subtree = new_tree
            else:
                ctx.right_parent.node.right_subtree = new_tree

            ctx.right_parent = new_tree
            ctx.right_entry = next_entry if next_entry else None
            return new_tree

        # right_split is empty (and not a leaf): keep existing references
        return None

    def _handle_left_split(
        self,
        cur: GPlusTreeBase,
        left_split: AbstractSetDataStructure,
        next_entry: Optional[Entry],
        is_leaf: bool,
        ctx: _SplitContext,
    ) -> Optional[GPlusTreeBase]:
        """Update or collapse the left half of a split and relink the
        parent.  Updates *ctx* in-place.

        Returns the next cursor to descend into.
        """
        if left_split.item_count() > 1 or is_leaf:
            cur.node.set = left_split
            if next_entry:
                cur.node.right_subtree = next_entry.left_subtree
            if ctx.left_x_entry is not None:
                ctx.left_x_entry.left_subtree = cur
            ctx.left_parent = cur
            ctx.left_x_entry = None
            return cur.node.right_subtree

        # Collapse single-item node (non-leaf)
        new_subtree = (
            next_entry.left_subtree if next_entry else cur.node.right_subtree
        )
        if ctx.left_x_entry is not None:
            ctx.left_x_entry.left_subtree = new_subtree
        else:
            ctx.left_parent.node.right_subtree = new_subtree
        # ctx.left_parent / ctx.left_x_entry unchanged for next iteration
        return new_subtree

    @staticmethod
    def _link_split_leaves(
        left_leaf: GPlusTreeBase, right_leaf: GPlusTreeBase
    ) -> None:
        """Wire up the leaf-level *next* pointers after a split."""
        right_leaf.node.next = left_leaf.node.next
        left_leaf.node.next = right_leaf

    def iter_leaf_nodes(self):
        """Iterate over all leaf-level GPlusNodes in left-to-right order.

        The initial descent to the leftmost leaf is *O(height)*.  The
        subsequent traversal via ``next`` pointers visits every leaf
        exactly once, making the full iteration *O(n)*.

        PERFORMANCE NOTE (Issue #5):
        When leaf sets contain recursively instantiated G-trees (e.g.
        KList expanded to GKPlusTree), the initial descent becomes
        expensive.  Adding a ``first_leaf`` pointer would give *O(1)*
        access and eliminate the descent.

        Yields:
            GPlusNode: Each leaf-level node in left-to-right order.
        """
        # TODO(#5): Replace this descent with: current = self.first_leaf if self.first_leaf else <fallback>
        # This would provide O(1) access instead of O(height) traversal
        # Descend to the leftmost leaf
        current = self
        
        # Exit early if the tree is empty
        if current.is_empty():
            return
        
        # Find the leftmost leaf node in current tree dimension
        while current and not current.is_empty() and current.node.right_subtree is not None:
            for entry in current.node.set:
                if entry.left_subtree is not None:
                    # If we have a left subtree, we need to go deeper
                    current = entry.left_subtree
                    break
            else:
                current = current.node.right_subtree
        
        while current is not None:
            yield current.node
            current = current.node.next
    

    def physical_height(self) -> int:
        """The physical pointer-follow height of the G⁺-tree.

        Defined recursively as the number of KList-node segments in this
        node’s k-list, plus the maximum ``physical_height()`` of any
        child subtree.

        Time complexity: O(n) — visits every node in the tree.
        """
        node = self.node
        base = node.set.physical_height()

        # If this is a leaf node, return the base height
        if node.rank == 1:
            return base

        # Find the tallest child among all left_subtrees and the right_subtree
        max_child = 0
        for entry in node.set:
            left = entry.left_subtree
            if left is not None:
                max_child = max(max_child, left.physical_height())
        max_child = max(max_child, node.right_subtree.physical_height())

        # total physical height = this node’s chain length + deepest child
        return base + max_child

    def print_structure(self, indent: int = 0, depth: int = 0, max_depth: int = 2) -> str:
        """Delegate to the standalone :func:`display.print_structure`."""
        from gplus_trees.display import print_structure as _print_structure
        return _print_structure(self, indent, depth, max_depth)


# Everything below this line has been extracted into dedicated modules:
#   Stats, gtree_stats_       → gplus_trees.tree_stats
#   print_pretty, collect_leaf_keys, print_structure → gplus_trees.display
# They are re-exported from the top of this file for backward compatibility.
