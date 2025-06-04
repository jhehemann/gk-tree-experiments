"""G+-tree base implementation"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple, Any, Type
from dataclasses import dataclass, asdict
from pprint import pprint
import collections
import math

from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
    RetrievalResult,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.profiling import (
    track_performance,
    PerformanceTracker
)

# Configure logging
logger = logging.getLogger(__name__)
# Clear all handlers to ensure we don't add duplicates
if logger.hasHandlers():
    logger.handlers.clear()
# Add a single handler with formatting
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Prevent propagation to the root logger to avoid duplicate logs
logger.propagate = False

t = Type["GPlusTreeBase"]

# Constants
DUMMY_KEY = int("-1", 16)
DUMMY_VALUE = None
DUMMY_ITEM = Item(DUMMY_KEY, DUMMY_VALUE)

DEBUG = False

class GPlusNodeBase:
    """
    Base class for G+-tree nodes. Factory will set:
      - SetClass  : which AbstractSetDataStructure to use for entries
      - TreeClass : which GPlusTree to build for child/subtree pointers
    """
    __slots__ = ("rank", "set", "right_subtree", "next")

    # these two get injected by your factory.py
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
        # self.right_subtree = right if right is not None else self.TreeClass()
        self.right_subtree = right
        self.next = None    # leaf‐chain pointer
    
class GPlusTreeBase(AbstractSetDataStructure):
    """
    A G+-tree is a recursively defined structure that is either empty or contains a single G+-node.
    Attributes:
        node (Optional[GPlusNode]): The G+-node that the tree contains. If None, the tree is empty.
    """
    __slots__ = ("node",)
    
    # Will be set by the factory
    NodeClass: Type[GPlusNodeBase]
    SetClass: Type[AbstractSetDataStructure]
    
    def __init__(self, node: Optional[GPlusNodeBase] = None):
        self.node: Optional[GPlusNodeBase] = node

    # @classmethod
    # def from_root(cls: Type[t], root_node: GPlusNodeBase) -> t:
    #     """
    #     Create a new tree instance wrapping an existing node.
    #     """
    #     tree = cls.__new__(cls)
    #     tree.node = root_node
    #     return tree

    # @track_performance
    def is_empty(self) -> bool:
        return self.node is None
    
    def __str__(self):
        return "Empty GPlusTree" if self.is_empty() else f"GPlusTree(node={self.node})"

    __repr__ = __str__
    
    # Public API
    def insert(self, x: Item, rank: int, x_left: Optional[GPlusTreeBase] = None) -> GPlusTreeBase:
        """
        Public method (average-case O(log n)): Insert an item into the G+-tree. 
        If the item already exists, updates its value at the leaf node.
        
        Args:
            x_item (Item): The item (key, value) to be inserted.
            rank (int): The rank of the item. Must be a natural number > 0.
        Returns:
            GPlusTreeBase: The updated G+-tree.

        Raises:
            TypeError: If x_item is not an Item or rank is not a positive int.
        """
        if not isinstance(x, Item):
            raise TypeError(f"insert(): expected Item, got {type(x).__name__}")
        if not isinstance(rank, int) or rank <= 0:
            raise TypeError(f"insert(): rank must be a positive int, got {rank!r}")
        if self.is_empty():
            return self._insert_empty(x, rank)
        return self._insert_non_empty(x, rank)
    
    def retrieve(
        self, key: int
    ) -> RetrievalResult:
        """
        Searches for an item with a matching key in the G+-tree.

        Iteratively traverses the tree (O(log n) with high probability)
        with O(k) additional memory by descending into left or right subtrees
        based on key comparisons.

        Args:
            key (int): The key to search for.

        Returns:
            RetrievalResult: Contains:
                found_item (Optional[Item]): The value associated with the key, or None if not found.
                next_pair (Tuple[Optional[Item], Optional[GPlusTreeBase]]):
                    The next item in sorted order and its associated subtree, or (None, None).
        """
        # logger.debug(f"retrieve() called with key: {key} on tree:\n{self.print_structure()}")

        if not isinstance(key, int) or key < 0:
            raise TypeError(f"retrieve(): key must be a non-negative int, got {key!r}")
        
        if self.is_empty():
            return RetrievalResult(None, None)

        cur = self
        found_entry: Optional[Entry] = None
        next_entry: Optional[Entry] = None
        
        while True:
            node = cur.node
            # Attempt to retrieve from this node's set
            res = node.set.retrieve(key)
            logger.debug(f"result: {res}")
            
            found_entry = res.found_entry
            next_entry = res.next_entry


            if node.rank == 1:
                if next_entry is None and node.next is not None:
                    # If leaf has a linked next node, update next_entry
                    next_entry = node.next.node.set.get_min().found_entry
                    return RetrievalResult(found_entry, next_entry)
                else:
                    # Leaf node: return found_entry and next_entry
                    return RetrievalResult(found_entry, next_entry)
            
            # Descend based on presence of next_entry
            cur = next_entry.left_subtree if next_entry else node.right_subtree
    
    def delete(self, item):
        raise NotImplementedError("delete not implemented yet")

    # Private Methods
    def _make_leaf_klist(self, x_item: Item, x_left: Optional[GPlusTreeBase] = None) -> AbstractSetDataStructure:
        """Builds a KList for a single leaf node containing the dummy and x_item."""
        TreeClass = type(self)
        SetClass = self.SetClass
        
        # start with a fresh empty set of entries
        leaf_set = SetClass()
        
        # insert the dummy entry, pointing at an empty subtree
        leaf_set = leaf_set.insert(DUMMY_ITEM, None)
        
        # now insert the real item, also pointing at an empty subtree
        leaf_set = leaf_set.insert(x_item, x_left)
        
        return leaf_set

    def _make_leaf_trees(self, x_item, x_left: Optional[GPlusTreeBase] = None) -> Tuple[GPlusTreeBase, GPlusTreeBase]:
        """
        Builds two linked leaf-level GPlusTreeBase nodes for x_item insertion.
        and returns the corresponding G+-trees.
        """
        TreeK = type(self)
        NodeK = self.NodeClass
        SetK = self.SetClass

        # Build right leaf
        right_set = SetK()
        right_set = right_set.insert(x_item, x_left)
        right_node = NodeK(1, right_set, None)
        right_leaf = TreeK(right_node)

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set = left_set.insert(DUMMY_ITEM, None)
        left_node = NodeK(1, left_set, None)
        left_leaf = TreeK(left_node)

        # Link leaves
        left_leaf.node.next = right_leaf
        return left_leaf, right_leaf

    def _insert_empty(self, x_item: Item, rank: int, x_left: Optional[GPlusTreeBase] = None) -> GPlusTreeBase:
        """Build the initial tree structure depending on rank."""
        inserted = True
        # Single-level leaf
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_item, x_left)
            self.node = self.NodeClass(rank, leaf_set, None)
            # logger.info(f"Tree after empty insert rank == 1:\n{self.print_structure()}")
            return self, inserted

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_item, x_left)
        root_set = self.SetClass().insert(DUMMY_ITEM, None)
        root_set = root_set.insert(_create_replica(x_item.key), l_leaf_t)
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        # logger.info(f"Tree after empty insert rank > 1:\n{self.print_structure()}")
        return self, inserted

    def _insert_non_empty(self, x_item: Item, rank: int, x_left: Optional[GPlusTreeBase] = None) -> GPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
        inserted = True
        cur = self
        parent = None
        p_next_entry = None

        # Loop until we find where to insert
        while True:
            node = cur.node
            node_rank = node.rank  # Cache attribute access
            
            # Case 1: Found node with matching rank - ready to insert
            if node_rank == rank:
                # Only retrieve once
                res = node.set.retrieve(x_item.key)
                
                # Fast path: update existing item
                if res.found_entry:
                    if x_left is not None:
                        raise ValueError(f"Item with key {x_item.key} already exists in the tree. Cannot be inserted with a subtree again.")
                    inserted = False
                    # Direct update for leaf nodes (common case)
                    if rank == 1:
                        res.found_entry.item.value = x_item.value
                        return self, inserted
                    return self._update_existing_item(cur, x_item)
                
                # Insert new item
                return self._insert_new_item(cur, x_item, res.next_entry, x_left)

            # Case 2: Current rank too small - handle rank mismatch
            if node_rank < rank:
                cur = self._handle_rank_mismatch(cur, parent, p_next_entry, rank)
                continue

            # Case 3: Descend to next level (current rank > rank)
            res = node.set.retrieve(x_item.key)
            parent = cur
            
            # Cache the next_entry to avoid repeated access
            next_entry = res.next_entry
            if next_entry:
                p_next_entry = next_entry
                cur = next_entry.left_subtree
            else:
                p_next_entry = None
                cur = node.right_subtree

    def _handle_rank_mismatch(
        self,
        cur: GPlusTreeBase,
        parent: GPlusTreeBase,
        p_next: Entry,
        rank: int
    ) -> GPlusTreeBase:
        """
        If the current node's rank < rank, we need to create or unfold a 
        node to match the new rank.
        This is done by creating a new G+-node and linking it to the parent.
        Attributes:
            cur (GPlusTreeBase): The current G+-tree.
            parent (GPlusTreeBase): The parent G+-tree.
            p_next (tuple): The next entry in the parent tree.
            rank (int): The rank to match.
        Returns:
            GPlusTreeBase: The updated G+-tree.
        """
        TreeClass = type(self)
        
        if parent is None:
            # create a new root node
            old_node = self.node
            root_set = self.SetClass().insert(DUMMY_ITEM, None)
            self.node = self.NodeClass(rank, root_set, TreeClass(old_node))
            return self

        # Unfold intermediate node between parent and current
        # Set replica of the current node's min as first entry.
        min_entry = cur.node.set.get_min().found_entry
        min_replica = _create_replica(min_entry.item.key)
        new_set = self.SetClass().insert(min_replica, None)
        new_tree = TreeClass()
        new_tree.node = self.NodeClass(rank, new_set, cur)
        
        if p_next:
            p_next.left_subtree = new_tree
        else:
            parent.node.right_subtree = new_tree

        return new_tree

    def _update_existing_item(
        self, cur: GPlusTreeBase, new_item: Item
    ) -> GPlusTreeBase:
        """Traverse to leaf (rank==1) and update the entry in-place."""
        inserted = False
        key = new_item.key
        while True:
            node = cur.node
            if node.rank == 1:
                entry = node.set.retrieve(key).found_entry
                if entry:
                    entry.item.value = new_item.value
                return self, inserted
            next = node.set.retrieve(key).next_entry
            cur = next.left_subtree if next else node.right_subtree
        
    # @track_performance
    def _insert_new_item(
        self,
        cur: 'GPlusTreeBase',
        x_item: Item,
        next_entry: Entry,
        x_left: Optional[GPlusTreeBase] = None,
    ) -> 'GPlusTreeBase':
        """
        Insert a new item key. For internal nodes, we only store the key. 
        For leaf nodes, we store the full item.
        
        Args:
            cur: The current G+-tree node where insertion starts
            x_item: The item to be inserted
            next_entry: The next entry in the tree relative to x_item
            
        Returns:
            The updated G+-tree
        """
        inserted = True
        x_key = x_item.key
        replica = _create_replica(x_key)
        TreeClass = type(self)

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_entry = None     # Entry in right parent points to current subtree
        left_parent = None     # Parent node for left-side updates
        left_x_entry = None    # x_item stored in left parent
        
        while True:
            node = cur.node
            is_leaf = node.rank == 1
            # Use correct item type based on node rank
            insert_obj = x_item if is_leaf else replica

            # First iteration - simple insert without splitting
            if right_parent is None:
                # Determine subtree for potential next iteration
                subtree = (
                    next_entry.left_subtree
                    if next_entry else node.right_subtree
                )
                insert_subtree = x_left if is_leaf else subtree

                # Insert the item
                node.set = node.set.insert(insert_obj, insert_subtree)

                # Early return if we're already at a leaf node
                if is_leaf:
                    return self, inserted
                
                # Assign parent tracking for next iteration
                right_parent = left_parent = cur
                right_entry = next_entry if next_entry else None
                left_x_entry = node.set.retrieve(x_key).found_entry
                cur = subtree
            else:
                # Node splitting required - get updated next_entry
                res = node.set.retrieve(x_key)
                next_entry = res.next_entry

                # Split node at x_key
                left_split, _, right_split = node.set.split_inplace(x_key)

                # --- Handle right side of the split ---
                # Determine if we need a new tree for the right split
                if right_split.item_count() > 0 or is_leaf:
                    # Insert item into right split and create new tree
                    insert_subtree = x_left if is_leaf else None
                    right_split = right_split.insert(insert_obj, insert_subtree)
                    new_tree = TreeClass()
                    new_tree.node = self.NodeClass(node.rank, right_split, node.right_subtree)

                    # Update parent reference to the new tree
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
                    else:
                        right_parent.node.right_subtree = new_tree

                    # Update right parent tracking
                    next_right_parent = new_tree
                    next_right_entry = next_entry if next_entry else None
                else:
                    # Keep existing parent references
                    next_right_parent = right_parent
                    next_right_entry = right_entry

                # Update right parent variables for next iteration
                right_parent = next_right_parent
                right_entry = next_right_entry
                
                # --- Handle left side of the split ---
                # Determine if we need to create/update using left split
                if left_split.item_count() > 1 or is_leaf:
                    # Update current node to use left split
                    cur.node.set = left_split
                    if next_entry:
                        cur.node.right_subtree = next_entry.left_subtree

                    # Update parent reference if needed
                    if left_x_entry is not None:
                        left_x_entry.left_subtree = cur
                    
                    # Make current node the new left parent
                    next_left_parent = cur
                    next_left_x_entry = None  # Left split never contains x_item
                    next_cur = cur.node.right_subtree
                else:
                    # Collapse single-item nodes for non-leaves
                    new_subtree = (
                        next_entry.left_subtree if next_entry else cur.node.right_subtree
                    )
                    
                    # Update parent reference
                    if left_x_entry is not None:
                        left_x_entry.left_subtree = new_subtree
                    else:
                        left_parent.node.right_subtree = new_subtree

                    # Prepare for next iteration
                    next_left_parent = left_parent
                    next_left_x_entry = left_x_entry
                    next_cur = new_subtree
                
                # Update left parent variables for next iteration
                left_parent = next_left_parent
                left_x_entry = next_left_x_entry

                # Update leaf node 'next' pointers if at leaf level
                if is_leaf:
                    new_tree.node.next = cur.node.next
                    cur.node.next = new_tree
                    return self, inserted  # Early return when leaf is processed
                    
                # Continue to next iteration with updated current node
                cur = next_cur

    def iter_leaf_nodes(self):
        """
        Iterates over all leaf-level GPlusNodes in the tree,
        starting from the leftmost leaf node and following `next` pointers.

        Yields:
            GPlusNode: Each leaf-level node in left-to-right order.
        """
        # Descend to the leftmost leaf
        current = self
        while current.node.rank > 1:
            result = current.node.set.get_min()
            if result.next_entry is not None:
                current = result.next_entry.left_subtree
            else:
                current = current.node.right_subtree

        # At this point, current is the leftmost leaf-level GPlusTreeBase
        while current is not None:
            yield current.node
            current = current.node.next
    
    # @track_performance
    def physical_height(self) -> int:
        """
        The “real” pointer-follow height of the G⁺-tree:
        –  the number of KListNode segments in this node’s k-list, plus
        –  the maximum physical_height() of any of its subtrees.
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
        if node.right_subtree is not None:
            max_child = max(max_child, node.right_subtree.physical_height())

        # total physical height = this node’s chain length + deepest child
        return base + max_child

    def print_structure(self, indent: int = 0, depth: int = 0, max_depth: int = 2):
        prefix = ' ' * indent
        if self.is_empty() or self is None:
            return f"{prefix}Empty {self.__class__.__name__}"
        
        if depth > max_depth:
            return f"{prefix}... (max depth reached)"
            
        result = []
        node = self.node

        kwargs_print = []
        if hasattr(node, 'size'):
            kwargs_print.append(f", size={node.size}")
        joined_kwargs = ", ".join(kwargs_print)

        result.append(f"{prefix}{node.__class__.__name__}(rank={node.rank}, set={type(node.set).__name__}{joined_kwargs})")
        
        result.append(node.set.print_structure(indent + 4))

        # Print right subtree
        if node.right_subtree is not None:
            right_node = node.right_subtree.node

            kwargs_print = []
            if hasattr(right_node, 'size'):
                kwargs_print.append(f", size={right_node.size}")
            joined_kwargs = ", ".join(kwargs_print)

            result.append(f"{prefix}    Right: {right_node.__class__.__name__}(rank={right_node.rank}, set={type(right_node.set).__name__}{joined_kwargs})")

            # result.append(f"{prefix}    GPlusNode(rank={right_node.rank}, set={type(right_node.set).__name__})")
            # result.append(f"{prefix}      Entries:")
            # print_klist_entries(right_node.set, indent + 8)
            result.append(right_node.set.print_structure(indent + 8))
        else:
            result.append(f"{prefix}    Right: Empty")

        # Print next node if rank == 1
        if node.rank == 1 and hasattr(node, 'next') and node.next:
            if not node.next.is_empty():
                
                next_node = node.next.node
                
                kwargs_print = []
                if hasattr(next_node, 'size'):
                    kwargs_print.append(f", size={next_node.size}")
                joined_kwargs = ", ".join(kwargs_print)

                result.append(f"{prefix}    Next: {next_node.__class__.__name__}(rank={next_node.rank}, set={(type(next_node.set).__name__)}{joined_kwargs})")
                # if next node has an attribute 'size', print it
               
                # result.append(f"{prefix}    GPlusNode(rank={next_node.rank}, set={(type(next_node.set).__name__)})")
                #result.append(f"{prefix}      Entries:")
                # print_klist_entries(next_node.set, indent + 8)
                result.append(next_node.set.print_structure(indent + 8))
            else:
                result.append(f"{prefix}    Next: Empty")
        elif node.rank == 1 and hasattr(node, 'next') and node.next is None:
                result.append(f"{prefix}    Next: Empty")
        return "\n".join(result)

@dataclass
class Stats:
    gnode_height: int
    gnode_count: int
    item_count: int
    real_item_count: int
    item_slot_count: int
    leaf_count: int
    rank: int
    is_heap: bool
    least_item: Optional[Any]
    greatest_item: Optional[Any]
    is_search_tree: bool
    internal_has_replicas: bool
    internal_packed: bool
    linked_leaf_nodes: bool
    all_leaf_values_present: bool
    leaf_keys_in_order: bool

def gtree_stats_(t: GPlusTreeBase,
                 rank_hist: Optional[Dict[int, int]] = None,
                 _is_root: bool = True,
                 ) -> Stats:
    """
    Returns aggregated statistics for a G⁺-tree in **O(n)** time.

    The caller can supply an existing Counter / dict for `rank_hist`;
    otherwise a fresh Counter is used.
    """
    if rank_hist is None:
        rank_hist = collections.Counter()

    # ---------- empty tree return ---------------------------------
    if t is None or t.is_empty():
        return Stats(gnode_height        = 0,
                     gnode_count         = 0,
                     item_count          = 0,
                     real_item_count     = 0,
                     item_slot_count     = 0,
                     leaf_count          = 0,
                     rank                = -1,
                     is_heap             = True,
                     least_item          = None,
                     greatest_item       = None,
                     is_search_tree      = True,
                     internal_has_replicas = True,
                     internal_packed     = True,
                     linked_leaf_nodes   = True,
                     all_leaf_values_present = True,
                     leaf_keys_in_order  = True,)

    node       = t.node
    node_set   = node.set
    node_right_subtree = node.right_subtree
    node_rank = node.rank
    node_item_count = node_set.item_count()
    rank_hist[node_rank] = rank_hist.get(node_rank, 0) + node_set.item_count()

    # ---------- recurse on children only if rank > 1 ------------------------------------
    right_stats = gtree_stats_(node_right_subtree, rank_hist, False)
    
    # Only recurse on child nodes if we are at a non-leaf node indicated by the
    # presence of a right subtree
    if node_right_subtree is not None:  
        child_stats = [gtree_stats_(e.left_subtree, rank_hist, False) for e in node_set]
    else:
        child_stats = []
    # ---------- aggregate ----------------------------------
    # Initialize with default values for the current node
    stats = Stats(
        gnode_height=0,
        gnode_count=0,
        item_count=0,
        real_item_count=0,
        item_slot_count=0,
        leaf_count=0,
        rank=node_rank,
        is_heap=True,
        least_item=None,
        greatest_item=None,
        is_search_tree=True,
        internal_has_replicas=True,
        internal_packed=(node_rank <= 1 or node_item_count > 1),
        linked_leaf_nodes=True,
        all_leaf_values_present=True,
        leaf_keys_in_order=True,
    )
    
    # Precompute common values using right subtree stats
    stats.gnode_count     = 1 + right_stats.gnode_count
    stats.item_count      = node_item_count + right_stats.item_count
    stats.real_item_count += right_stats.real_item_count
    stats.item_slot_count = node_set.item_slot_count() + right_stats.item_slot_count
    stats.leaf_count += right_stats.leaf_count
    # stats.gnode_height    = 1 + max(right_stats.gnode_height,
    #                                 max((cs.gnode_height for cs in child_stats), default=0))

    max_child_height = 0

    # Check search tree property for the node itself by comparing keys in order
    # regardless of child_stats
    prev_key = None
    for i, entry in enumerate(node_set):
        current_key = entry.item.key
        
        # Check search tree property within the node
        if prev_key is not None and prev_key >= current_key:
            stats.is_search_tree = False
            
        
        
        # Process child stats if they exist (will be empty for leaf nodes)
        if i < len(child_stats):
            cs = child_stats[i]
            
            max_child_height = max(max_child_height, cs.gnode_height)

            if node_rank >= 2 and entry.item.value is not None:
                stats.internal_has_replicas = False
            
            # Accumulate counts for common values
            stats.gnode_count += cs.gnode_count
            stats.item_count += cs.item_count
            stats.item_slot_count += cs.item_slot_count
            stats.leaf_count += cs.leaf_count
            stats.real_item_count += cs.real_item_count

            # Update boolean flags
            if stats.is_heap and not ((node_rank > cs.rank) and cs.is_heap):
                stats.is_heap = False
                
            stats.internal_has_replicas &= cs.internal_has_replicas
            stats.internal_packed &= cs.internal_packed
            stats.linked_leaf_nodes &= cs.linked_leaf_nodes
            
            # Additional search tree property checks with child stats
            if stats.is_search_tree:
                if not cs.is_search_tree:
                    stats.is_search_tree = False
                elif cs.least_item and cs.least_item.key < prev_key:
                    stats.is_search_tree = False
                elif cs.greatest_item and cs.greatest_item.key >= current_key:
                    stats.is_search_tree = False

        prev_key = current_key
    
    # Calculate final height
    stats.gnode_height = 1 + max(right_stats.gnode_height, max_child_height)

    # Fold in right subtree flags
    if stats.is_heap and not right_stats.is_heap:
        stats.is_heap = False
    
    if stats.is_search_tree:
        if not right_stats.is_search_tree:
            stats.is_search_tree = False
        elif right_stats.least_item and prev_key is not None and right_stats.least_item.key < prev_key:
            stats.is_search_tree = False


    stats.is_search_tree       &= right_stats.is_search_tree
    if right_stats.least_item and right_stats.least_item.key < prev_key:
        stats.is_search_tree = False

    stats.internal_has_replicas &= right_stats.internal_has_replicas
    stats.internal_packed       &= right_stats.internal_packed
    stats.linked_leaf_nodes     &= right_stats.linked_leaf_nodes

    # ----- LEAST / GREATEST -----
    if child_stats and child_stats[0].least_item is not None:
        stats.least_item = child_stats[0].least_item
    else:
        stats.least_item = node_set.get_min().found_entry.item

    if right_stats.greatest_item is not None:
        stats.greatest_item = right_stats.greatest_item
    else:
        # print_pretty(node_set)

        stats.greatest_item = node_set.get_max().found_entry.item

    # ---------- leaf walk ONCE at the root -----------------------------
    if node_rank == 1:          # leaf node: base values
        # Count non-dummy items
        true_count = 0
        all_values_present = True
        
        for entry in node_set:
            item = entry.item
            if item.key != DUMMY_KEY:
                true_count += 1
                if item.value is None:
                    all_values_present = False

        stats.all_leaf_values_present = all_values_present
        stats.real_item_count = true_count
        stats.leaf_count = 1

    # Root-level validation (only occurs once)
    if _is_root:         # root call (only true once)
        leaf_keys, leaf_values = [], []
        leaf_count, item_count = 0, 0
        last_leaf, prev_key = None, None
        keys_in_order = True
        for leaf in t.iter_leaf_nodes():
            last_leaf = leaf
            leaf_count += 1
            
            for entry in leaf.set:
                item = entry.item
                if item.key < 0:
                    continue

                item_count += 1
                key = item.key

                # Check orderin
                if prev_key is not None and key < prev_key:
                    keys_in_order = False
                prev_key = key

                leaf_keys.append(key)
                leaf_values.append(item.value)

        # Set values from leaf traversal
        stats.leaf_keys_in_order = keys_in_order

        # Check leaf_count and real_item_count consistency
        if leaf_count != stats.leaf_count or item_count != stats.real_item_count:
            # print(f"Leaf count mismatch: iter {leaf_count} != stats {stats.leaf_count}")
            # print(f"Or Item count mismatch: iter {item_count} != stats {stats.item_count}")
            # print(f"Or real_item_count mismatch: iter {item_count} != stats {stats.real_item_count}")
            stats.linked_leaf_nodes = False
            stats.leaf_count = max(leaf_count, stats.leaf_count)
            
            stats.real_item_count = max(item_count, stats.real_item_count)
        elif last_leaf is not None:
            # Check if greatest item matches last leaf's greatest item
            last_count = last_leaf.set.item_count()
            last_item = last_leaf.set.get_max().found_entry.item
            if stats.greatest_item is not last_item:
                stats.linked_leaf_nodes = False

    return stats

# @track_performance
def collect_leaf_keys(tree: 'GPlusTreeBase') -> list[str]:
    out = []
    for leaf in tree.iter_leaf_nodes():
        for e in leaf.set:
            if e.item.key != DUMMY_KEY:
                out.append(e.item.key)
    return out

def _find_capacity(set_cls):
    """
    Walks set_cls.SetClass until we find a subclass of KListBase,
    then returns its KListNodeClass.CAPACITY.
    """
    cls = set_cls
    # keep following the `.SetClass` link...
    while not issubclass(cls, KListBase):
        cls = cls.SetClass
    # now cls is a KListBase, so its node class has the capacity
    return cls.KListNodeClass.CAPACITY

# def print_prett(tree):
#     """
#     Print the B+-tree so that all nodes on the same layer
#     appear on the same line, correctly indented.
#     """
#     # 1) Determine indent unit from the tree’s capacity
#     capacity    = _find_capacity(tree.SetClass)
#     indent_unit = capacity
#     print(f"Indent unit: {indent_unit}")

#     layers = collections.defaultdict(list)  # rank -> list of (indent, segment)

#     def _collect(node):
#         rank       = node.rank
#         indent     = indent_unit ** rank 
#         keys_line  = " | ".join(str(entry.item.key) for entry in node.set)
#         layers[rank-1].append((indent, keys_line))

#         # Recurse left‐to‐right
#         for entry in node.set:
#             if entry.left_subtree is not None:
#                 _collect(entry.left_subtree.node)
#         if node.right_subtree is not None:
#             _collect(node.right_subtree.node)

#     _collect(tree.node)

def print_pretty(set):
    """
    Prints a B+-tree so:
      • Lines go from highest rank down to 1.
      • Within a line, nodes appear left→right in traversal order.
      • All columns have the same width, so initial indent and
        inter-node spacing are uniform.
    """
    if set is None:
        return f"{type(set).__name__}: None"

    if not (isinstance(set, GPlusTreeBase) or isinstance(set, KListBase)):
        raise TypeError(f"print_pretty() expects GPlusTreeBase or KListBase, got {type(set).__name__}")

    if set.is_empty():
        return f"{type(set).__name__}: Empty"
    
    SEP = " | "
    set_type = type(set).__name__

    if isinstance(set, KListBase):
        texts = []
        node = set.head
        node_idx = 0
        while node is not None:
            text = ("[" + SEP.join(str(e.item.key) for e in node.entries)+ "]")
            texts.append(text)
            node_idx += 1
            node = node.next
        # print(f"Klist Height: {height}")
        res_text = " ".join(texts)
        return f"({set_type}): {res_text}"

    tree = set

    # 1) First pass: collect each node's text and track max length
    layers_raw  = collections.defaultdict(list)  # rank -> list of node-strings
    max_len     = 0

    def collect(tree, parent=None):
        nonlocal max_len
        dim = tree.DIM
        p_dim = parent.DIM if parent else None
        other_dim = dim != p_dim and parent is not None
        node = tree.node
        rank = node.rank
        parent_rank = parent.node.rank if parent else None

        logger.debug(f"Other dim: {other_dim}")

        fill_rank = parent_rank - 1 if parent_rank is not None else rank
        while fill_rank > rank:
            layers_raw[fill_rank].append("")
            fill_rank -= 1
        
        text     = SEP.join(str(e.item.key) for e in node.set)
        if parent is None or not other_dim:
            layers_raw[rank].append(text)
            max_len = max(max_len, len(text))
        else:
            # handle leaf items' left subtrees in separate layer (only root)
            new_text = f"(D{dim}R{rank}) " + text
            layers_raw[0].append(new_text)
            max_len = max(max_len, len(new_text))

        # recurse left→right
        if not other_dim:
            for e in node.set:
                if e.left_subtree:
                    collect(e.left_subtree, tree)
            if node.right_subtree:
                collect(node.right_subtree, tree)

    collect(tree, None)

    # 2) Define a fixed column width: widest text + 1 space padding
    column_width = max_len + 1

    # 3) Build “slots” per layer, padding every entry to column_width
    #    and inserting blanks where no node lives.
    all_ranks = sorted(layers_raw.keys())
    # we’ll assume every line has the same number of “slots” = max number
    # of nodes in any single layer:
    max_slots = max(len(v) for v in layers_raw.values())
    
    layers = {}
    column_counts = [len(layers_raw[rank]) for rank in all_ranks]

    for rank in all_ranks:
        texts = layers_raw[rank]
        # pad or truncate texts list to max_slots
        padded = [
            ("" + txt.center(column_width) + "  " if i < len(texts) else "" + " " * column_width + "")
            for i, txt in enumerate(texts + [""] * max_slots)
        ][:max_slots]
        layers[rank] = padded
        
    # 4) Now accumulate, prefixing each line by an indent proportional to rank
    #    (so higher nodes are shifted right to reflect depth).
    out_lines = []
    cumm_indent = 0.0      # cumulative indent (number of columns)
    for i, rank in enumerate(all_ranks):
        # indent to reflect depth
        # prefix = " " * ((rank - 1) * column_width) + "  "
        if i == 0:
            # first line: constant indent
            prefix = "  "
        else:
            # print(layers[i+1])
            column_diff = column_counts[i-1] - column_counts[i]
            # print(f"all_ranks[{i-1}]: {all_ranks[i-1]}")
            # print(f"\nrank: {rank}")
            # print(f"column_diff: {column_diff}")
            cumm_indent += float(column_diff) / 2
            
            # print(f"cumm_indent: {cumm_indent}")
            spaces = int(math.floor(((2 + column_width) * cumm_indent) + 0.5))
            # print(f"spaces: {spaces}")
            prefix = "  " + spaces * " "
            # prefix = ""
        line   = "".join(layers[rank])
        layer_id = f"Rank {rank}" if rank > 0 else "LSub"

        out_lines.append(f"{layer_id}:{prefix}{line}")

    # join with newlines and return
    res_text = set_type + "\n" + "\n\n".join(reversed(out_lines)) + "\n"
    return res_text
    