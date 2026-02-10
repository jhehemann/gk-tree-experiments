"""G+-tree base implementation"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any, Type, Union
from dataclasses import dataclass
import collections
import math

from gplus_trees.base import (
    AbstractSetDataStructure,
    ItemData,
    LeafItem,
    InternalItem,
    DummyItem,
    Entry,
    _get_replica,
)
from gplus_trees.klist_base import KListBase
import logging
from gplus_trees.base import logger

# ── Backward-compatible re-exports ──────────────────────────────────
# These symbols used to live here; they now live in dedicated modules
# but are re-exported so existing ``from gplus_trees.gplus_tree_base
# import …`` statements keep working.
from gplus_trees.tree_stats import Stats, gtree_stats_, _get_capacity  # noqa: F401
from gplus_trees.display import print_pretty, collect_leaf_keys  # noqa: F401

t = Type["GPlusTreeBase"]

# Constants
DUMMY_KEY = int("-1", 16)
DUMMY_ITEM = DummyItem(ItemData(key=DUMMY_KEY))


# Cache for dimension-specific dummy items
_DUMMY_ITEM_CACHE: Dict[int, DummyItem] = {}

def get_dummy(dim: int) -> DummyItem:
    """
    Get a dummy item for the specified dimension.
    This function caches created dummy items to avoid creating new instances
    for the same dimension repeatedly.
    
    Args:
        dim (int): The dimension for which to get a dummy item.
        
    Returns:
        DummyItem: A dummy item with key=-(dim) and value=None
    """
    # Check if we already have a dummy item for this dimension in cache
    if dim in _DUMMY_ITEM_CACHE:
        return _DUMMY_ITEM_CACHE[dim]
    
    # Create a new dummy item for this dimension
    dummy_key = -(dim)  # Negative dimension as key 
    dummy_item = DummyItem(ItemData(key=dummy_key))
    
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
    
    # Public API
    def insert(self, x: Union[InternalItem, LeafItem], rank: int, x_left: Optional[GPlusTreeBase] = None) -> Tuple[t, bool, Optional[Entry]]:
        """
        Public method (average-case O(log n)): Insert an item into the G+-tree. 
        If the item already exists, updates its value at the leaf node.
        
        Args:
            x_item (Union[InternalItem, LeafItem]): The item (key, value) to be inserted.
            rank (int): The rank of the item. Must be a natural number > 0.
        Returns:
            Tuple[GPlusTreeBase, bool, Optional[Entry]]: The result containing the updated G+-tree, insertion status, and next entry.

        Raises:
            TypeError: If x_item is not an InternalItem or LeafItem or rank is not a positive int.
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
                    # Find the next entry > key in the next leaf node
                    next_node = node.next.node
                    for entry in next_node.set:
                        if entry.item.key > key:
                            next_entry = entry
                            break
                        
                return found_entry, next_entry

            cur = next_entry.left_subtree if next_entry else node.right_subtree
            # if next_entry and next_entry.left_subtree is not None:
            #     cur = next_entry.left_subtree
            # else:
            #     cur = cur.node.right_subtree
    
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
        leaf_set, _, _ = leaf_set.insert_entry(Entry(DUMMY_ITEM, None))

        # now insert the real item, also pointing at an empty subtree
        leaf_set, _, _ = leaf_set.insert_entry(x_entry)

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
        right_set, _, _ = right_set.insert_entry(x_entry)
        right_node = NodeK(1, right_set, None)
        right_leaf = TreeK(right_node)

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set, _, _ = left_set.insert_entry(Entry(DUMMY_ITEM, None))
        left_node = NodeK(1, left_set, None)
        left_leaf = TreeK(left_node)

        # Link leaves
        left_leaf.node.next = right_leaf
        # TODO(#5): Set first_leaf pointer here: left_leaf.first_leaf = left_leaf
        # TODO(#5): Set first_leaf pointer here: right_leaf.first_leaf = left_leaf
        return left_leaf, right_leaf

    def _insert_empty(self, insert_entry: Entry, rank: int) -> GPlusTreeBase:
        """Build the initial tree structure depending on rank."""
        inserted = True
        # Single-level leaf
        if rank == 1:
            leaf_set = self._make_leaf_klist(insert_entry)
            self.node = self.NodeClass(rank, leaf_set, None)
            return self, inserted, None

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(insert_entry)
        root_set, _, _ = self.SetClass().insert_entry(Entry(DUMMY_ITEM, None))
        root_set, _, _ = root_set.insert_entry(Entry(_get_replica(insert_entry.item), l_leaf_t))
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        return self, inserted, None

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
                        return self, inserted, next_entry
                    return self._update_existing_item(cur, x_item, next_entry)
                
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
            root_set, _, _ = self.SetClass().insert_entry(Entry(DUMMY_ITEM, None))
            self.node = self.NodeClass(rank, root_set, TreeClass(old_node))
            return self

        # Unfold intermediate node between parent and current
        # Set replica of the current node's min as first entry.
        min_entry = cur.node.set.get_min()[0]
        min_replica = _get_replica(min_entry.item)
        new_set, _, _ = self.SetClass().insert_entry(Entry(min_replica, None))
        new_tree = TreeClass()
        new_tree.node = self.NodeClass(rank, new_set, cur)
        
        if p_next:
            p_next.left_subtree = new_tree
        else:
            parent.node.right_subtree = new_tree

        return new_tree

    def _update_existing_item(
        self, cur: GPlusTreeBase, new_item: Union[InternalItem, LeafItem], next_entry: Entry
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
                return self, inserted, next_entry
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
        x_item = x_entry.item
        x_key = x_item.key
        replica = _get_replica(x_item)
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
                    node.set, _, _ = node.set.insert_entry(x_entry)
                    return self, inserted, next_entry

                # Insert a replica and assign parent tracking for next iteration
                insert_entry = Entry(replica, subtree)
                node.set, _, _ = node.set.insert_entry(insert_entry)
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
                left_split, _, right_split, next_entry = node.set.split_inplace(x_key)

                # --- Handle right side of the split ---
                # Determine if we need a new tree for the right split
                if right_split.item_count() > 0 or is_leaf:
                    # Insert item into right split and create new tree
                    if is_leaf:
                        right_split, _, _ = right_split.insert_entry(x_entry)
                    else:
                        right_split, _, _ = right_split.insert_entry(Entry(replica, None))

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
                    return self, inserted, next_entry

                # Continue to next iteration with updated current node
                cur = next_cur

    def iter_leaf_nodes(self):
        """
        Iterates over all leaf-level GPlusNodes in the tree,
        starting from the leftmost leaf node and following `next` pointers or recursive .
        
        PERFORMANCE NOTE (Issue #5): 
        Currently descends from root to find leftmost leaf in O(height) time.
        When leaf sets contain recursively instantiated G-trees (e.g., KList
        expanded to GKPlusTree), this traversal becomes expensive as it must
        traverse nested tree structures.
        
        OPTIMIZATION TODO:
        Add a first_leaf pointer to enable O(1) access to the first leaf.
        This would eliminate the descent phase and significantly speed up
        sequential access patterns.

        Yields:
            GPlusNode: Each leaf-level node in left-to-right order.
        """
        # TODO(#5): Replace this descent with: current = self.first_leaf if self.first_leaf else <fallback>
        # This would provide O(1) access instead of O(height) traversal
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


# Everything below this line has been extracted into dedicated modules:
#   Stats, gtree_stats_  → gplus_trees.tree_stats
#   print_pretty, collect_leaf_keys → gplus_trees.display
# They are re-exported from the top of this file for backward compatibility.
