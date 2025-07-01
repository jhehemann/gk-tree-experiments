"""G+-tree base implementation"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any, Type
from dataclasses import dataclass
import collections
import math

from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
)
from gplus_trees.klist_base import KListBase
import logging
from gplus_trees.base import logger

t = Type["GPlusTreeBase"]

# Constants
DUMMY_KEY = int("-1", 16)
DUMMY_VALUE = None
DUMMY_ITEM = Item(DUMMY_KEY, DUMMY_VALUE)

# Cache for dimension-specific dummy items
_DUMMY_ITEM_CACHE: Dict[int, Item] = {}

def get_dummy(dim: int) -> Item:
    """
    Get a dummy item for the specified dimension.
    This function caches created dummy items to avoid creating new instances
    for the same dimension repeatedly.
    
    Args:
        dim (int): The dimension for which to get a dummy item.
        
    Returns:
        Item: A dummy item with key=-(dim) and value=None
    """
    # return Item(-1, None)

    # Check if we already have a dummy item for this dimension in cache
    if dim in _DUMMY_ITEM_CACHE:
        return _DUMMY_ITEM_CACHE[dim]
    
    # Create a new dummy item for this dimension
    dummy_key = -(dim)  # Negative dimension as key 
    dummy_item = Item(dummy_key, None)
    
    # Cache it for future use
    _DUMMY_ITEM_CACHE[dim] = dummy_item
    
    return dummy_item

DEBUG = False

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
    """
    __slots__ = ("node",)
    
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
        insert_entry = Entry(x, x_left)
        if self.is_empty():
            return self._insert_empty(insert_entry, rank)
        return self._insert_non_empty(insert_entry, rank)

    def retrieve(
        self, key: int, with_next: bool = True
    ) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Searches for an item with a matching key in the G+-tree.

        Iteratively traverses the tree (O(log n) with high probability)
        with O(k) additional memory by descending into left or right subtrees
        based on key comparisons.

        Args:
            key (int): The key to search for.

        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (found_entry, next_entry) where:
                - found_entry: The entry with the matching key if found, otherwise None
                - next_entry: The subsequent entry in sorted order, or None if no next entry exists
        """
        if not isinstance(key, int):
            raise TypeError(f"retrieve(): key must be an int, got {type(key).__name__}")
        
        if self.is_empty():
            return None, None

        cur = self
        found_entry: Optional[Entry] = None
        next_entry: Optional[Entry] = None
        
        while True:
            node = cur.node
            # Attempt to retrieve from this node's set
            is_leaf = node.rank == 1
            flag = True if not is_leaf else with_next
            found_entry, next_entry = node.set.retrieve(key, with_next=flag)

            if is_leaf:
                # We've reached a leaf node
                if not with_next:
                    return found_entry, None

                # If no next entry found in current node, check linked leaf nodes
                if next_entry is None and node.next is not None:
                    # Find the next entry (no dummy) in the linked leaf list
                    next_node = node.next.node
                    for entry in next_node.set:
                        if entry.item.key > key:
                            next_entry = entry
                            break
                        
                return found_entry, next_entry

            cur = next_entry.left_subtree if next_entry else node.right_subtree
    
    def delete(self, item):
        raise NotImplementedError("delete not implemented yet")

    # Private Methods
    def _make_leaf_klist(self, x_entry: Entry) -> AbstractSetDataStructure:
        """Builds a KList for a single leaf node containing the dummy and x_item."""
        TreeClass = type(self)
        SetClass = self.SetClass
        
        # start with a fresh empty set of entries
        leaf_set = SetClass()
        
        # insert the dummy entry, pointing at an empty subtree
        leaf_set, _ = leaf_set.insert_entry(Entry(DUMMY_ITEM, None))

        # now insert the real item, also pointing at an empty subtree
        leaf_set, _ = leaf_set.insert_entry(x_entry)

        return leaf_set

    def _make_leaf_trees(self, x_entry: Entry) -> Tuple[GPlusTreeBase, GPlusTreeBase]:
        """
        Builds two linked leaf-level GPlusTreeBase nodes for x_item insertion.
        and returns the corresponding G+-trees.
        """
        TreeK = type(self)
        NodeK = self.NodeClass
        SetK = self.SetClass

        # Build right leaf
        right_set = SetK()
        right_set, _ = right_set.insert_entry(x_entry)
        right_node = NodeK(1, right_set, None)
        right_leaf = TreeK(right_node)

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set, _ = left_set.insert_entry(Entry(DUMMY_ITEM, None))
        left_node = NodeK(1, left_set, None)
        left_leaf = TreeK(left_node)

        # Link leaves
        left_leaf.node.next = right_leaf
        return left_leaf, right_leaf

    def _insert_empty(self, insert_entry: Entry, rank: int) -> GPlusTreeBase:
        """Build the initial tree structure depending on rank."""
        inserted = True
        # Single-level leaf
        if rank == 1:
            leaf_set = self._make_leaf_klist(insert_entry)
            self.node = self.NodeClass(rank, leaf_set, None)
            return self, inserted

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(insert_entry)
        root_set, _ = self.SetClass().insert_entry(Entry(DUMMY_ITEM, None))
        root_set, _ = root_set.insert_entry(Entry(_create_replica(insert_entry.item.key), l_leaf_t))
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        return self, inserted

    def _insert_non_empty(self, x_entry: Entry, rank: int) -> GPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
        x_item =  x_entry.item
        x_key = x_item.key
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
                res = node.set.retrieve(x_key)
                existing_x_entry = res[0]
                next_entry = res[1]
                # Fast path: update existing item
                if existing_x_entry:
                    if x_entry.left_subtree is not None:
                        raise ValueError(f"Entry with key {x_key} already exists in the tree. Cannot be inserted with a subtree again.")
                    inserted = False
                    # Direct update for leaf nodes (common case)
                    if rank == 1:
                        existing_x_entry.item.value = x_item.value
                        return self, inserted
                    return self._update_existing_item(cur, x_item)
                
                # Insert new item
                return self._insert_new_item(cur, x_entry, next_entry)

            # Case 2: Current rank too small - handle rank mismatch
            if node_rank < rank:
                cur = self._handle_rank_mismatch(cur, parent, p_next_entry, rank)
                continue

            # Case 3: Descend to next level (current rank > rank)
            res = node.set.retrieve(x_key)
            parent = cur
            
            # Cache the next_entry to avoid repeated access
            next_entry = res[1]
            if next_entry:
                cur = next_entry.left_subtree
            else:
                cur = node.right_subtree

            p_next_entry = next_entry

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
            root_set, _ = self.SetClass().insert_entry(Entry(DUMMY_ITEM, None))
            self.node = self.NodeClass(rank, root_set, TreeClass(old_node))
            return self

        # Unfold intermediate node between parent and current
        # Set replica of the current node's min as first entry.
        min_entry = cur.node.set.get_min()[0]
        min_replica = _create_replica(min_entry.item.key)
        new_set, _ = self.SetClass().insert_entry(Entry(min_replica, None))
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
                entry = node.set.retrieve(key)[0]
                if entry:
                    entry.item.value = new_item.value
                return self, inserted
            next = node.set.retrieve(key)[1]
            cur = next.left_subtree if next else node.right_subtree

    def _insert_new_item(
        self,
        cur: 'GPlusTreeBase',
        x_entry: Entry,
        next_entry: Entry,
    ) -> 'GPlusTreeBase':
        """
        Insert a new item key. For internal nodes, we only store the key. 
        For leaf nodes, we store the full item.
        
        Args:
            cur: The current G+-tree node where insertion starts
            x_entry: The entry to be inserted
            next_entry: The next entry in the tree relative to x_entry

        Returns:
            The updated G+-tree
        """
        inserted = True
        x_key = x_entry.item.key
        replica = _create_replica(x_key)
        TreeClass = type(self)

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_entry = None     # Entry in right parent points to current subtree
        left_parent = None     # Parent node for left-side updates
        left_x_entry = None    # x_entry stored in left parent

        while True:
            node = cur.node
            is_leaf = node.rank == 1
            # Use correct item type based on node rank
            # insert_obj = x_item if is_leaf else replica

            # First iteration - simple insert without splitting
            if right_parent is None:
                # Determine subtree for potential next iteration
                subtree = (
                    next_entry.left_subtree
                    if next_entry else node.right_subtree
                )
                # insert_subtree = x_left if is_leaf else subtree

                # Insert the item and return early if we're already at a leaf node
                if is_leaf:
                    node.set, _ = node.set.insert_entry(x_entry)
                    return self, inserted
                
                # Insert a replica and assign parent tracking for next iteration
                insert_entry = Entry(replica, subtree)
                node.set, _ = node.set.insert_entry(insert_entry)
                right_parent = left_parent = cur
                right_entry = next_entry if next_entry else None
                
                # TODO: Check if this can just be the insert_entry
                left_x_entry = node.set.retrieve(x_key)[0]
                cur = subtree
            else:
                # Node splitting required - get updated next_entry
                res = node.set.retrieve(x_key)
                next_entry = res[1]

                # Split node at x_key
                left_split, _, right_split = node.set.split_inplace(x_key)

                # --- Handle right side of the split ---
                # Determine if we need a new tree for the right split
                if right_split.item_count() > 0 or is_leaf:
                    # Insert item into right split and create new tree
                    if is_leaf:
                        right_split, _ = right_split.insert_entry(x_entry)
                    else:
                        right_split, _ = right_split.insert_entry(Entry(replica, None))

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
        starting from the leftmost leaf node and following `next` pointers or recursive .

        Yields:
            GPlusNode: Each leaf-level node in left-to-right order.
        """
        # Descend to the leftmost leaf
        current = self
        
        # Exit early if the tree is empty
        if current is None or current.is_empty():
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
        if node.rank != 1:
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
    set_thresholds_met: bool
    linked_leaf_nodes: bool
    all_leaf_values_present: bool
    leaf_keys_in_order: bool


# TODO: Add correct checks for GKPlusTreeBase nodes set thresholds met
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
                     set_thresholds_met = True,
                     internal_packed     = True,
                     linked_leaf_nodes   = True,
                     all_leaf_values_present = True,
                     leaf_keys_in_order  = True,)

    K = t.SetClass.KListNodeClass.CAPACITY
    if hasattr(t, 'l_factor'):
        threshold = t.l_factor * K
    else:
        threshold = K

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
        set_thresholds_met=True,
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
        # Only check if the current key is not a dummy key since dummy keys are not part of the search 
        if prev_key is not None and prev_key >= current_key:
            if hasattr(t, 'DIM'):
                # search tree property is violated if the keys equal to and greater than the current trees dummy item are not in order
                if current_key >= get_dummy(t.DIM).key:
                    stats.is_search_tree = False
            else:
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

            # Inherit child violations first
            if not cs.set_thresholds_met:
                stats.set_thresholds_met = False
                # if logger.isEnabledFor(logging.DEBUG):
                    # logger.debug(f"  Child violation inherited: cs.set_thresholds_met={cs.set_thresholds_met}")
            
            # Check current node violations (only if no child violations yet)
            if stats.set_thresholds_met:
                if isinstance(node_set, KListBase) and node_set.item_count() > threshold:
                    stats.set_thresholds_met = False
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.warning(f"  Current node KList violation: item_count()={node_set.item_count()}, threshold={threshold}")
                elif not isinstance(node_set, KListBase) and node_set.item_count() <= threshold:
                    stats.set_thresholds_met = False
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.warning(f"  Current node GPlusTree violation: item_count()={node_set.item_count()}, threshold={threshold}")
                
            stats.internal_has_replicas &= cs.internal_has_replicas
            stats.internal_packed &= cs.internal_packed
            stats.linked_leaf_nodes &= cs.linked_leaf_nodes
            
            # Additional search tree property checks with child stats
            if stats.is_search_tree:
                if not cs.is_search_tree:
                    stats.is_search_tree = False
                elif cs.least_item and cs.least_item.key < prev_key:
                    if hasattr(t, 'DIM'):
                        if cs.least_item.key >= get_dummy(t.DIM).key:
                            stats.is_search_tree = False
                    else:
                        stats.is_search_tree = False

                elif cs.greatest_item and cs.greatest_item.key >= current_key:
                    if hasattr(t, 'DIM'):
                        if cs.greatest_item.key >= get_dummy(t.DIM).key:
                            stats.is_search_tree = False
                    else:
                        stats.is_search_tree = False

        prev_key = current_key
    
    # Calculate final height
    stats.gnode_height = 1 + max(right_stats.gnode_height, max_child_height)

    # Fold in right subtree flags
    if stats.is_heap and not right_stats.is_heap:
        stats.is_heap = False

    # Check right subtree first
    if not right_stats.set_thresholds_met:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Right subtree <set_thresholds_met>: {right_stats.set_thresholds_met}")
        stats.set_thresholds_met = False
    
    # Check current node violations (always check, regardless of current state)
    if isinstance(node_set, KListBase) and node_set.item_count() > threshold:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Node set is KList and item_count > threshold: {node_set.item_count()} > {threshold}, KList: {print_pretty(node_set)}")
        stats.set_thresholds_met = False
    elif not isinstance(node_set, KListBase) and node_set.item_count() <= threshold:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Node set is {type(node_set)} and item_count <= threshold: {node_set.item_count()} <= {threshold}, GPlusTree: {print_pretty(node_set)}")
        stats.set_thresholds_met = False


    
    if stats.is_search_tree:
        if not right_stats.is_search_tree:
            stats.is_search_tree = False
        elif right_stats.least_item and prev_key is not None and right_stats.least_item.key < prev_key:
            if hasattr(t, 'DIM'):
                if right_stats.least_item.key >= get_dummy(t.DIM).key:
                    stats.is_search_tree = False
            else:
                stats.is_search_tree = False

    stats.is_search_tree       &= right_stats.is_search_tree
    if right_stats.least_item and right_stats.least_item.key < prev_key:
        if hasattr(t, 'DIM'):
            if right_stats.least_item.key >= get_dummy(t.DIM).key:
                stats.is_search_tree = False
        else:
            stats.is_search_tree = False

    stats.internal_has_replicas &= right_stats.internal_has_replicas
    stats.internal_packed       &= right_stats.internal_packed
    stats.linked_leaf_nodes     &= right_stats.linked_leaf_nodes

    # ----- LEAST / GREATEST -----
    if child_stats and child_stats[0].least_item is not None:
        stats.least_item = child_stats[0].least_item
    else:
        stats.least_item = node_set.find_pivot()[0].item

    if right_stats.greatest_item is not None:
        stats.greatest_item = right_stats.greatest_item
    else:
        stats.greatest_item = node_set.get_max()[0].item

    # ---------- leaf walk ONCE at the root -----------------------------
    if node_rank == 1:          # leaf node: base values
        # Count non-dummy items
        true_count = 0
        all_values_present = True
        
        for entry in node_set:
            item = entry.item
            if item.key >= 0:  # Skip dummy items
                true_count += 1
                if item.value is None:
                    all_values_present = False

        stats.all_leaf_values_present = all_values_present
        stats.real_item_count = true_count
        # logger.debug(f"  Leaf: {[e.item.key for e in node_set]} item_count={node_item_count}, real_item_count={stats.real_item_count}")
        # logger.debug(f"  Leaf: {print_pretty(node_set)} item_count={node_item_count}, real_item_count={stats.real_item_count}")
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
            stats.linked_leaf_nodes = False
            stats.leaf_count = max(leaf_count, stats.leaf_count)
            
            stats.real_item_count = max(item_count, stats.real_item_count)
        elif last_leaf is not None:
            # Check if greatest item matches last leaf's greatest item
            last_count = last_leaf.set.item_count()
            last_item = last_leaf.set.get_max()[0].item
            if stats.greatest_item is not last_item:
                stats.linked_leaf_nodes = False

    return stats

def collect_leaf_keys(tree: 'GPlusTreeBase') -> list[str]:
    out = []
    for leaf in tree.iter_leaf_nodes():
        for e in leaf.set:
            if e.item.key != DUMMY_KEY:
                out.append(e.item.key)
    return out

def _get_capacity(set_cls):
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

def print_pretty(set: AbstractSetDataStructure):
    """
    Prints a G+-tree so:
      • Lines go from highest rank down to 1.
      • Within a line, nodes appear left→right in traversal order.
      • All columns have the same width, so initial indent and
        inter-node spacing are uniform.
    """
    PRIMARY = '\033[32m'    # green
    SECONDARY = '\033[33m'  # yellow
    RESET = '\033[0m'
    
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
        res_text = " ".join(texts)
        return f"({set_type}): {res_text}"

    tree = set
    
    if hasattr(tree, 'DIM'):
        dim = tree.DIM if hasattr(tree, 'DIM') else None
        # logger.debug(f"print_pretty() called for {set_type} with DIM={dim}")
        dum_key = get_dummy(dim).key
    else:
        # logger.debug(f"print_pretty() called for {set_type} without DIM")
        dum_key = DUMMY_KEY

    # 1) First pass: collect each node's text and track max length
    layers_raw  = collections.defaultdict(list)  # rank -> list of node-strings
    max_len     = 0

    def collect(tree, parent=None):
        if tree.is_empty():
            return
        nonlocal max_len
        dim = tree.DIM if hasattr(tree, 'DIM') else None

        p_dim = parent.DIM if parent and hasattr(parent, 'DIM') else None
        other_dim = False
        other_dim_processed = False

        if parent is not None and dim != p_dim:
            other_dim = True
            other_dim_processed = True
            
        node = tree.node
        rank = node.rank
        parent_rank = parent.node.rank if parent else None

        fill_rank = parent_rank - 1 if parent_rank is not None else rank
        while fill_rank > rank:
            layers_raw[fill_rank].append("")
            fill_rank -= 1
        
        # text = SEP.join(str(e.item.key) for e in node.set)
        text = ""
        for e in node.set:
            if parent is not None and other_dim:
                if e.item.key < dum_key:
                    text += (SEP if text else "") + f"{SECONDARY}{e.item.key}{RESET}"
                else:
                    text += (SEP if text else "") + str(e.item.key)
            else:
                if e.item.key == dum_key:
                    text += (SEP if text else "") + f"{PRIMARY}{e.item.key}{RESET}"
                elif e.item.key < dum_key:
                    text += (SEP if text else "") + f"{SECONDARY}{e.item.key}{RESET}"
                else:
                    text += (SEP if text else "") + str(e.item.key)

        if parent is None or not other_dim:
            layers_raw[rank].append(text)
            max_len = max(max_len, len(text))
        else:
            # print leaf items' left subtrees in separate layer (only roots)
            dim_str = str(dim) if dim is not None else "?"
            new_text = f"(D{dim_str}R{rank}) " + text
            layers_raw[0].append(new_text)
            max_len = max(max_len, len(new_text))

        # recurse left→right
        if not other_dim:
            for e in node.set:
                if e.left_subtree:
                    collect(e.left_subtree, tree)
            if node.right_subtree:
                collect(node.right_subtree, tree)
        
        # Special case: if node.set is a tree of different dimension, traverse it
        if isinstance(node.set, GPlusTreeBase) and not node.set.is_empty() and not other_dim_processed:
            collect(node.set, tree)

    collect(tree, None)

    # 2) Define a fixed column width: widest text + 1 space padding
    column_width = (max_len // 2) + 1

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
            ("" + txt.center(column_width) + "  " if i < len(texts) else "" + " " * (column_width // 2)) + ""
            for i, txt in enumerate(texts + [""] * max_slots)
        ][:max_slots]
        layers[rank] = padded
        
    # 4) Now accumulate, prefixing each line by an indent proportional to rank
    #    (so higher nodes are shifted right to reflect depth).
    out_lines = []
    cumm_indent = 0.0      # cumulative indent (number of columns)
    for i, rank in enumerate(all_ranks):
        # indent to reflect depth
        if i == 0:
            # first line: constant indent
            prefix = "     "
        else:
            column_diff = column_counts[i-1] - column_counts[i]
            cumm_indent += float(column_diff) / 2
            spaces = int(math.floor(((2 + column_width) * cumm_indent) + 0.5))
            prefix = "     " + spaces * " "
        line   = "".join(layers[rank])
        layer_id = f"{PRIMARY}Rank {rank}{RESET}" if rank > 0 else f"{SECONDARY}Other Dims{RESET}"
        out_lines.append(f"{layer_id}:{prefix}{line}")

    # join with newlines and return
    res_text = set_type + "\n" + "\n\n".join(reversed(out_lines)) + "\n"
    return res_text
