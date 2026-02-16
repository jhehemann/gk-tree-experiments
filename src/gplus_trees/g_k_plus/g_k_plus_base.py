"""GKPlusTree base implementation.

Defines :class:`GKPlusNodeBase` (a single tree node) and
:class:`GKPlusTreeBase` (the recursively-defined tree) for GK+-trees —
an extension of G+-trees with dimensional support.

Most algorithmic methods live in dedicated mixin modules:

- **insert.py** — insertion logic
- **navigation.py** — ``find_pivot``, ``get_min``, ``get_max``
- **conversion.py** — KList ↔ GKPlusTree threshold conversions
- **zip.py** — merge (zip) of two trees
- **unzip.py** — split (unzip) at a key
- **bulk_create.py** — bottom-up bulk creation

This file contains the class definitions, ``__init__``, counting / stats
methods, the public ``insert_entry`` API, and iteration.

Multi-dimensional complexity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GK+-trees support recursive nesting: when a leaf's KList exceeds
``k · l_factor`` items, it is promoted to a GK+-tree of the next higher
dimension. Hence ``node.set`` can itself be a ``GKPlusTreeBase``.

All complexities below are stated **per dimension** (single level of
nesting). When inner sets are GK+-trees of dimension *d+1*, every
set-level operation (insert, split, iterate) recurses into that inner
tree, so the true cost compounds across all *D* active dimensions.

Let h_d denote the height at dimension *d*. The worst-case total cost
of a per-dimension O(h_d · C_set) operation is
O(h_1 · h_2 · … · h_D · (log l + k)), where the innermost set is a
KList. In practice, inner trees are bounded by the conversion threshold
(~k · l_factor items), so h_{d+1} ≈ log(k · l_factor) is small.

Single-dimension complexity (n = items, k = KList capacity,
h = tree height ≈ log n, l = KList nodes per level):

+-----------------------+-------------------------------------------+
| Operation             | Time (per dimension)                      |
+=======================+===========================================+
| ``insert_entry``      | O(h · (log l + k))  amortised             |
| ``item_count``        | O(1) cached / O(n) first call             |
| ``real_item_count``   | O(1) cached / O(n) first call             |
| ``expanded_count``    | O(1) cached / O(n) first call (recurses)  |
| ``get_max_dim``       | O(n) (recursive across all dimensions)    |
| ``item_slot_count``   | O(n) (recursive across all dimensions)    |
| ``iter_real_entries`` | O(n), plus inner-tree iteration if nested |
| ``__iter__``          | O(n) total, O(h) setup; recurses into     |
|                       | inner-tree ``__iter__`` at each leaf       |
+-----------------------+-------------------------------------------+
"""

from __future__ import annotations
from typing import Optional, Type, Tuple


from gplus_trees.base import (
    AbstractSetDataStructure,
    Entry,
)
import logging

from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import (
    GPlusTreeBase, GPlusNodeBase, print_pretty, get_dummy
)

from gplus_trees.g_k_plus.base import GKTreeSetDataStructure
from gplus_trees.g_k_plus.zip import GKPlusZipMixin
from gplus_trees.g_k_plus.unzip import GKPlusUnzipMixin
from gplus_trees.g_k_plus.navigation import GKPlusNavigationMixin
from gplus_trees.g_k_plus.conversion import GKPlusConversionMixin
from gplus_trees.g_k_plus.insert import GKPlusInsertMixin
from gplus_trees.logging_config import get_logger

# Re-export bulk-creation and conversion symbols for backward compatibility.
# External code that does ``from gplus_trees.g_k_plus.g_k_plus_base import bulk_create_gkplus_tree``
# (or _tree_to_klist, _klist_to_tree, RankData) will continue to work.
from gplus_trees.g_k_plus.bulk_create import (  # noqa: F401
    RankData,
    bulk_create_gkplus_tree,
    _tree_to_klist,
    _klist_to_tree,
)

DEFAULT_DIMENSION = 1  # Default dimension for GKPlusTree
DEFAULT_L_FACTOR = 1.0  # Default threshold factor for KList to GKPlusTree conversion

logger = get_logger(__name__)


def IS_DEBUG():
    """Check if debug logging is enabled. Evaluated at call time so it
    reflects the current logging configuration (unlike a module-level constant)."""
    return logger.isEnabledFor(logging.DEBUG)


class GKPlusNodeBase(GPlusNodeBase):
    """Base class for GK+-tree nodes.
    Extends GPlusNodeBase with size support.
    """

    # These will be injected by the factory
    SetClass: Type[AbstractSetDataStructure]
    TreeClass: Type[GKPlusTreeBase]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class GKPlusTreeBase(GKPlusInsertMixin, GKPlusConversionMixin, GKPlusNavigationMixin, GKPlusZipMixin, GKPlusUnzipMixin, GPlusTreeBase, GKTreeSetDataStructure):
    """
    A GK+-tree is an extension of G+-tree with dimension support.
    It can automatically transform between KList and GKPlusTree based on item count.

    Attributes:
        node (Optional[GKPlusNodeBase]): The GK+-node that the tree contains.
        DIM (int): The dimension of the GK+-tree (class attribute).
        l_factor (float): The threshold factor for conversion between KList and GKPlusTree.
    """
    __slots__ = GPlusTreeBase.__slots__ + ("l_factor", "item_cnt", "size", "expanded_cnt")  # Add new slots

    # Will be set by the factory
    DIM: int = 1  # Default dimension value
    NodeClass: Type[GKPlusNodeBase]
    SetClass: Type[AbstractSetDataStructure]
    KListClass: Type[KListBase]

    def __init__(self, node: Optional[GKPlusNodeBase] = None,
                 l_factor: float = DEFAULT_L_FACTOR) -> None:
        """
        Initialize a new GKPlusTree.

        Args:
            node: The root node of the tree (if not empty)
            l_factor: Threshold factor for KList-to-GKPlusTree conversion
        """
        # Get node and dimension from parent class
        super().__init__(node)
        self.l_factor = l_factor
        self.item_cnt = None  # Initialize item count
        self.size = None  # Initialize tree size (excluding dummy items)
        self.expanded_cnt = None  # Initialize count for expanded leaves

    def __str__(self):
        if self.is_empty():
            return "Empty GKPlusTree"
        return f"GKPlusTree(dim={self.__class__.DIM}, node={self.node})"

    __repr__ = __str__

    # ── Hook overrides for GPlusTreeBase ────────────────────────────
    def _get_dummy(self) -> 'DummyItem':
        """Dimension-specific dummy item."""
        return get_dummy(dim=type(self).DIM)

    def _on_node_visited(self, cur: 'GKPlusTreeBase') -> None:
        """Invalidate cached size / count info when a node is mutated."""
        cur._invalidate_tree_size()

    def _make_empty_tree(self) -> 'GKPlusTreeBase':
        """Create an empty tree of the same type, forwarding *l_factor*."""
        return type(self)(l_factor=self.l_factor)

    def item_count(self, with_dim_dummy: bool = False) -> int:
        if with_dim_dummy:
            if self.item_cnt_ge_dummy is None:
                return self.get_item_count_ge_dummy()
            return self.item_cnt_ge_dummy

        if self.item_cnt is None:
            return self.get_tree_item_count()
        return self.item_cnt
    
    def real_item_count(self) -> int:
        if self.size is None:
            return self.get_size()
        return self.size

    def get_tree_item_count(self) -> int:
        """Get the number of items in the tree (only leafs), including all dummy items in all expanded dimensions.

        Complexity:
            O(n) — visits every subtree.  Result is cached in ``self.item_cnt``;
            subsequent calls return in O(1) until ``_invalidate_tree_size()``.
            When leaf sets are inner GK+-trees, their ``item_count`` may also
            recurse into the next dimension.
        """
        if self.is_empty():
            self.item_cnt = 0
            return self.item_cnt
        
        if self.node.rank == 1:  # indicates a leaf in current dim
            self.item_cnt = self.node.set.item_count()
            return self.item_cnt
        else:
            count = 0
            for entry in self.node.set:
                count += (entry.left_subtree.item_count()
                         if entry.left_subtree is not None else 0)

            count += (self.node.right_subtree.item_count()
                     if self.node.right_subtree is not None else 0)
            self.item_cnt = count
            return self.item_cnt

    def get_size(self) -> int:
        """Get the number of real items in the tree, excluding dummy items.

        Complexity:
            O(n) — visits every subtree.  Result is cached in ``self.size``;
            subsequent calls return in O(1) until ``_invalidate_tree_size()``.
            When leaf sets are inner GK+-trees, their ``real_item_count`` may
            also recurse into the next dimension.
        """
        if self.is_empty():
            self.size = 0
            return self.size

        node = self.node
        if node.rank == 1:  # indicates a leaf in current dim
            self.size = node.set.real_item_count()
            return self.size
        else:
            count = 0
            for entry in node.set:
                count += (entry.left_subtree.real_item_count()
                         if entry.left_subtree is not None else 0)

            count += (node.right_subtree.real_item_count()
                     if node.right_subtree is not None else 0)
            self.size = count
            return self.size

    def insert_entry(self, x_entry: Entry, rank: int) -> Tuple['GKPlusTreeBase', bool]:
        """
        Insert an entry into the GK+-tree. The rank is calculated automatically
        based on the entry's item key. The entire entry object is preserved to
        maintain external references.
        
        Args:
            x_entry (Entry): The entry to be inserted, containing an item and left_subtree.
            rank (int): The rank of the entry's item key.

        Returns:
            Tuple[GKPlusTreeBase, bool]: The updated tree and whether insertion was successful.
            
        Raises:
            TypeError: If entry is not an Entry object.

        Complexity:
            O(h · (log l + k)) amortised per dimension. When leaf sets
            are inner GK+-trees (expanded via ``check_and_convert_set``),
            set-level operations (insert, split) recurse into dimension
            d+1.  Total worst-case across D dimensions:
            O(h_1 · h_2 · … · h_D · (log l + k)).
        """
        if not isinstance(x_entry, Entry):
            raise TypeError(f"insert_entry(): expected Entry, got {type(x_entry).__name__}")
        if self.is_empty():
            return self._insert_empty(x_entry, rank)
        return self._insert_non_empty(x_entry, rank)

    def _invalidate_tree_size(self) -> None:
        """
        Invalidate the tree size. This is a placeholder for future implementation.
        """
        self.item_cnt = None
        self.size = None
        self.expanded_cnt = None

    def get_max_dim(self) -> int:
        """
        Get the maximum dimension of the GK+-tree.

        Returns:
            int: The maximum dimension of the tree.

        Complexity:
            O(N_total) — visits every node across all dimensions, including
            recursion into inner GK+-tree sets.
        """
        if self.is_empty():
            return self.DIM

        max_dim = self.DIM
        if isinstance(self.node.set, GKPlusTreeBase):
            # If the set is a GKPlusTreeBase, get its maximum dimension
            max_dim = max(max_dim, self.node.set.get_max_dim())

        # Get the max of the left subtrees of the entries in the KList
        for entry in self.node.set:
            if entry.left_subtree is not None:
                max_dim = max(max_dim, entry.left_subtree.get_max_dim())
        
        # Get the max dimension of the right subtree
        max_dim = max(max_dim, self.node.right_subtree.get_max_dim() if self.node.right_subtree else max_dim)

        return max_dim

    def expanded_count(self) -> int:
        if self.expanded_cnt is None:
            return self.get_expanded_leaf_count()
        return self.expanded_cnt

    def get_expanded_leaf_count(self) -> int:
        """
        Count all leaf nodes whose set is recursively instantiated with another
        GKPlusTree, having an extra dummy item for each expansion.

        Returns:
            int: The number of leaf nodes in the tree.

        Complexity:
            time & memory: best O(1), average O(log(n)), worst O(log(n)^2)
        """
        if self.is_empty():
            self.expanded_cnt = 0
            return 0

        count = 0
        node = self.node
        
        # Base case: The node is a leaf node (rank 1)
        if node.rank == 1:
            if isinstance(node.set, GKPlusTreeBase):
                count += 1
                count += node.set.expanded_count()  # recurse
            self.expanded_cnt = count
            return count
        
        # Recursive case: The node is not a leaf (rank > 1)
        for entry in node.set:
            if entry.left_subtree is not None:
                count += entry.left_subtree.expanded_count()

        if node.right_subtree is not None:
            count += node.right_subtree.expanded_count()
        self.expanded_cnt = count
        return count

    def item_slot_count(self):
        """Count the number of item slots in the tree.

        Complexity:
            O(N_total) — visits every subtree recursively across all dimensions.
            Calls ``item_slot_count`` on each ``node.set``, which recurses into
            inner GK+-trees when sets have been expanded.
        """
        if self.is_empty():
            if IS_DEBUG():
                logger.debug(f"[DIM {self.DIM}] item_slot_count: Tree is empty, returning 0")
            return 0

        node = self.node
        if node.rank == 1:
            return node.set.item_slot_count()

        count = 0
        for entry in node.set:
            if entry.left_subtree is not None:
                count += entry.left_subtree.item_slot_count()

        count += node.right_subtree.item_slot_count()
        count += node.set.item_slot_count()

        return count

    def iter_real_entries(self):
        """Iterate over all real entries (excluding dummies) in the gk-plus-tree.

        Complexity:
            O(n) in the current dimension, filtering by key >= 0.
            When leaf sets are inner GK+-trees, ``__iter__`` on those sets
            recurses into the next dimension (see ``__iter__`` docs).

        Yields:
            Entry: Each entry in the tree, excluding dummy entries.
        """
        if self.is_empty():
            return

        for entry in self:
            if entry.item.key >= 0:
                yield entry

    def __iter__(self):
        """Yields each entry of the gk-plus-tree in order, including all dummy entries.

        Complexity:
            Setup: O(h) for ``get_max_leaf`` + ``iter_leaf_nodes`` in the
            current dimension.  Total iteration: O(n) entries.

            When leaf node sets are inner GK+-trees (dimension d+1), the
            inner ``for entry in node.set`` triggers ``__iter__`` on that
            tree, which itself pays O(h_{d+1}) setup and O(n_{d+1}) per
            leaf.  This recurses through all active dimensions.
        """
        if self.is_empty():
            return

        # stop iteration after processing the last leaf node
        last_leaf = self.get_max_leaf()

        for node in self.iter_leaf_nodes():
            for entry in node.set:
                yield entry
            if node is last_leaf:
                # Stop iteration after processing the last leaf node
                break

