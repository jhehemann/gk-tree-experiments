"""GKPlusTree base implementation"""

from __future__ import annotations
from typing import Optional, Type, TypeVar, Tuple, Union
from itertools import islice

from gplus_trees.utils import calc_rank_from_digest_k
from gplus_trees.base import (
    AbstractSetDataStructure,
    LeafItem,
    InternalItem,
    DummyItem,
    Entry,
    _get_replica,
)
import logging

from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import (
    GPlusTreeBase, GPlusNodeBase, print_pretty, get_dummy
)

from gplus_trees.g_k_plus.base import GKTreeSetDataStructure

class RankData:
    """Consolidated data structure for each rank level in bulk tree creation."""
    __slots__ = ('entries', 'child_indices', 'next_child_idx', 'boundaries', 'pivot_item')
    
    def __init__(self, 
                 entries: Optional[list[Entry]] = None,
                 child_indices: Optional[list[int]] = None,
                 next_child_idx: int = 0,
                 boundaries: Optional[list[int]] = None,
                 pivot_item: Optional[Union[LeafItem, InternalItem, DummyItem]] = None):
        self.entries = entries
        self.child_indices = child_indices
        self.next_child_idx = next_child_idx
        self.boundaries = boundaries
        self.pivot_item = pivot_item

t = TypeVar('t', bound='GKPlusTreeBase')

DEFAULT_DIMENSION = 1  # Default dimension for GKPlusTree
DEFAULT_L_FACTOR = 1.0  # Default threshold factor for KList to GKPlusTree conversion

from gplus_trees.g_k_plus.base import logger
IS_DEBUG = logger.isEnabledFor(logging.DEBUG)

# Cached import for performance - initialized on first use
_create_gkplus_tree = None


def _get_create_gkplus_tree():
    """Lazy import of create_gkplus_tree to avoid circular imports while caching for performance."""
    global _create_gkplus_tree
    if _create_gkplus_tree is None:
        from gplus_trees.g_k_plus.factory import create_gkplus_tree
        _create_gkplus_tree = create_gkplus_tree
    return _create_gkplus_tree


class GKPlusNodeBase(GPlusNodeBase):
    """Base class for GK+-tree nodes.
    Extends GPlusNodeBase with size support.
    """

    # These will be injected by the factory
    SetClass: Type[AbstractSetDataStructure]
    TreeClass: Type[GKPlusTreeBase]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class GKPlusTreeBase(GPlusTreeBase, GKTreeSetDataStructure):
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
        """Get the number of items in the tree (only leafs), including all dummy items in all expanded dimensions."""
        if self.is_empty():
            self.item_cnt = 0
            return self.item_cnt
        
        if self.node.rank == 1:  # indicates a leaf in current dim
            count = 0
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
        """Get the number of real items in the tree, excluding dummy items."""
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
            entry (Entry): The entry to be inserted, containing an item and left_subtree.
            rank (int): The rank of the entry's item key.

        Returns:
            Tuple[GKPlusTreeBase, bool]: The updated tree and whether insertion was successful.
            
        Raises:
            TypeError: If entry is not an Entry object.
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
        """Count the number of item slots in the tree."""
        if self.is_empty():
            return 0

        node = self.node
        if node.rank == 1:
            return node.set.item_slot_count()

        count = 0
        for entry in node.set:
            if entry.left_subtree is not None:
                count += entry.left_subtree.item_slot_count()

        if node.rank != 1:
            count += node.right_subtree.item_slot_count()
        count += node.set.item_slot_count()

        return count

    def find_pivot(self) -> Tuple[Optional[Entry], Optional[Entry]]:
        """
        Returns the pivot entry of a node in the next lower dimension.

        This method is always called from a node of the next lower dimension.
        The pivot entry is the first entry that is either the dummy entry of
        the lower dimension itself or the next larger entry in the current tree.

        Returns:
            Tuple[Optional[Entry], Optional[Entry]]: A tuple of (pivot_entry, next_entry)
        """

        if self.is_empty():
            return None, None

        dummy_pivot = get_dummy(self.DIM - 1).key

        pivot = None
        current = None
        for entry in self:
            current = entry
            if pivot is not None:
                break
            else:
                if entry.item.key == dummy_pivot or entry.item.key > dummy_pivot:
                    pivot = entry

        if pivot is None:
            raise ValueError(f"No pivot entry in tree {print_pretty(self)}")

        return pivot, current

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
    
    def get_min_leaf_tree(self) -> GKPlusNodeBase:
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

    def get_max_leaf(self) -> GKPlusNodeBase:
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

    def update(self, cur, x_entry) -> Tuple[GKPlusTreeBase, bool]:
        """
        Update the tree after an insertion or deletion.
        This method recalculates the item count and size of the tree.
        
        Returns:
            Tuple[GKPlusTreeBase, bool]: The updated tree and a boolean indicating success.
        """
        raise ValueError(
            f"Entry with key {x_entry.item.key} already exists in the "
            "tree. No updates allowed during insert implementation phase."
            )

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

    def _make_leaf_trees(self, x_entry: Entry) -> Tuple[GKPlusTreeBase, GKPlusTreeBase]:
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

    def _insert_empty(self, x_entry: Entry, rank: int) -> GKPlusTreeBase:
        """Build the initial tree structure depending on rank."""
        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Inserting {x_entry.item.key} into empty tree with rank {rank}")
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

    def _insert_non_empty(self, x_entry: Entry, rank: int) -> GKPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Inserting {x_entry.item.key} into non-empty tree with rank {rank}: {print_pretty(self)}")
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
        cur: GKPlusTreeBase,
        parent: GKPlusTreeBase,
        p_next: Entry,
        rank: int,
        left_parent: Optional[GKPlusTreeBase] = None,
    ) -> GKPlusTreeBase:
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
        # Locate the current node’s pivot and place its replica first in the intermediate node.
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

    def _insert_min(self, 
               x_entry: Entry, 
               rank: int, 
               add_dummy: bool = True
               ) -> Tuple['GKPlusTreeBase', bool, Optional[Entry], Optional[Entry]]:
        """
        Args:
            x_entry (Entry): The entry to be inserted, containing an item and left_subtree.
            rank (int): The rank of the entry's item key.
            add_dummy (bool): Whether to add a dummy entry for the dimension.

        Returns:
            Tuple[GKPlusTreeBase, bool, Optional[Entry], Optional[Entry]]: The updated tree,
            whether insertion was successful, the pivot entry, and the next entry.
            
        Raises:
            TypeError: If entry is not an Entry object.
        """
        if not isinstance(x_entry, Entry):
            raise TypeError(f"insert_min(): expected Entry, got {type(x_entry).__name__}")
        if self.is_empty():
            return self._insert_empty(x_entry, rank
    )
    
    def _insert_new_item(
        self,
        cur: GKPlusTreeBase,
        x_entry: Entry,
    ) -> GKPlusTreeBase:
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
        # Pre-cache all frequently used values to minimize overhead
        # Cache frequently accessed attributes and methods to reduce lookups
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
            # Cache node reference and minimize repeated attribute access
            cur._invalidate_tree_size()
            node = cur.node
            
            # Cache frequently used values in local variables for better performance
            is_leaf = node.rank == 1

            # Fast path: First iteration without splitting
            # if IS_DEBUG:
            #     logger.debug(f"[DIM {self.DIM}] Inserting {x_item.key} into node set at rank {node.rank}: {print_pretty(node.set)}")
            if right_parent is None:
                is_gkplus_type = isinstance(node.set, GKPlusTreeBase)
                insert_entry = x_entry if is_leaf else Entry(replica, None)
                if not is_gkplus_type:
                    res = node.set.insert_entry(insert_entry)
                    node.set, inserted, next_entry = res[0], res[1], res[2]
                    if not inserted:
                        return self.update(cur, x_entry)
                    if IS_DEBUG:
                        logger.debug(f"[DIM {self.DIM}] Inserted {x_item.key} into node set at rank {node.rank}: {print_pretty(node.set)}")
                    node.set = check_and_convert_set(node.set) # only KLists can be extended
                else:
                    digest = x_item.get_digest_for_dim(self.DIM + 1)
                    new_rank = calc_rank_from_digest_k(digest, capacity)
                    res = node.set.insert_entry(insert_entry, rank=new_rank)
                    node.set, inserted, next_entry = res[0], res[1], res[2]
                    if not inserted:
                        return self.update(cur, x_entry)
                    if IS_DEBUG:
                        logger.debug(f"[DIM {self.DIM}] Inserted {x_item.key} into node set at rank {node.rank}: {print_pretty(node.set)}")

                subtree = next_entry.left_subtree if next_entry else node.right_subtree
                insert_entry.left_subtree = subtree if not is_leaf else insert_entry.left_subtree


                # Fastest path for leaf nodes - direct return
                if is_leaf:                    
                    self._invalidate_tree_size()
                    # Check for next entry in the next leaf if it exists and no next entry was found
                    if next_entry is None and node.next is not None:
                        next_entry = node.next.find_pivot()[0]
                    return self, True, next_entry
                

                # Setup for next iteration with optimized assignments
                right_parent = left_parent = cur
                right_entry = next_entry
                left_x_entry = insert_entry
                # left_x_entry = x_entry
                cur = subtree
                continue

            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] Subsequent iteration: Splitting node set at rank {node.rank} for key {x_item.key} and inserting key to the right split; node set: {print_pretty(node.set)}")
            
            # Perform split operation and cache converted results
            if isinstance(node.set, KListBase):
                left_split, _, right_split, next_entry = node.set.split_inplace(x_key)
            else:
                # Create a new GKPlusTree with dummy and x_item to be inserted into the right part of the unzipped tree
                
                # logger.info(f"Creating new GKPlusTree for unzip in dimension {self.DIM + 1} with rank {new_rank} and capacity {capacity}")

                left_split, _, right_split, next_entry = node.set.unzip(x_key)
                # left_split, _, right_split, next_entry = node.set.split_inplace(x_key)

            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] Left split (item_count: {left_split.item_count()}, real_item_count: {left_split.real_item_count()}): {print_pretty(left_split)}")
                logger.debug(f"[DIM {self.DIM}] Right split (item_count: {right_split.item_count()}, real_item_count: {right_split.real_item_count()}): {print_pretty(right_split)}")
            left_split = check_and_convert_set(left_split)
            
            # Cache item counts early to avoid repeated method calls in conditionals
            right_item_count = right_split.item_count() # invalidate size after insertion
            left_item_count = left_split.item_count()
            
            # Handle right side creation - inline optimization for performance
            new_tree = None
            if right_item_count > 0 or is_leaf:
                insert_entry = x_entry if is_leaf else Entry(replica, None)
                if isinstance(right_split, GKPlusTreeBase):
                    digest = x_item.get_digest_for_dim(self.DIM + 1)
                    new_rank = calc_rank_from_digest_k(digest, capacity)
                    tree_insert = bulk_create_gkplus_tree([x_entry], self.DIM + 1, l_factor=self.l_factor, KListClass=self.SetClass)

                    
                    right_split = tree_insert.zip(right_split)
                    # right_split, _, _ = right_split.insert_entry(insert_entry, rank=new_rank)
                    if IS_DEBUG:
                        logger.debug(f"[DIM {self.DIM}] Inserted {x_item.key} into right split with rank {new_rank} item count {right_split.item_count()} and real_item_count {right_split.real_item_count()}: {print_pretty(right_split)}")
                    right_split._invalidate_tree_size()
                else:
                    right_split, _, _ = right_split.insert_entry(insert_entry)

                # Create new tree node
                right_split = check_and_convert_set(right_split)
                new_tree = TreeClass(l_factor=l_factor)
                if IS_DEBUG:
                    logger.debug(f"[DIM {self.DIM}] Creating new right tree with rank {node.rank} and item count {right_split.item_count()}: {print_pretty(right_split)} and right subtree {print_pretty(node.right_subtree)}")
                    logger.debug(f"[DIM {self.DIM}] Right split items: {[e.item.key for e in right_split]}")
                    
                new_tree.node = NodeClass(node.rank, right_split, node.right_subtree)
                if IS_DEBUG:
                    logger.debug(f"[DIM {self.DIM}] Created new right tree with rank {node.rank} and item count {right_split.item_count()}: {print_pretty(new_tree)}")

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

            # Update right parent variables for next iteration
            right_parent = next_right_parent
            right_entry = next_right_entry

            # Handle left side with optimized control flow
            if left_item_count > 1 or is_leaf:
                # Update current node efficiently
                cur.node.set = left_split
                if next_entry:
                    cur.node.right_subtree = next_entry.left_subtree

                if left_x_entry is not None:
                    # Only True once: When we have a left split and not yet updated the x_entry 
                    # inserted (without split) into the initial node
                    left_x_entry.left_subtree = cur

                # Setup for next iteration with minimal assignments
                next_left_parent = cur
                next_left_x_entry = None  # Left split never contains x_item
                next_cur = cur.node.right_subtree
            else:
                # Collapse single-item nodes for non-leaves
                new_subtree = next_entry.left_subtree if next_entry else None

                # Update parent reference efficiently
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

            # Handle leaf level with early return optimization
            if is_leaf:
                # Check for next entry in the next leaf if it exists and no next entry was found
                if next_entry is None and node.next is not None:
                        next_entry = node.next.find_pivot()[0]
                if new_tree and cur:
                    new_tree.node.next = cur.node.next
                    cur.node.next = new_tree

                    # Right subtrees of left splits are not reset in higher dimensional leaf nodes.
                    # Do it here to maintain search tree structure across dimensions.
                    cur.node.right_subtree = None  # No right subtree at leaf level
                self._invalidate_tree_size()
                return self, True, next_entry  # Early return when leaf is processed

            # Continue to next iteration with updated current node
            cur = next_cur

    def split_inplace(self, key: int
                      ) -> Tuple['GKPlusTreeBase',
                                 Optional['GKPlusTreeBase'],
                                 'GKPlusTreeBase']:
        """
        Split the tree into two parts around the given key.

        Args:
            key: The key value to split at

        Returns:
            A tuple of (left_return, key_subtree, right_return) where:
            - left_return: A tree containing all entries with keys < key
            - key_subtree: If key exists, its associated left subtree; else None
            - right_return: A tree with entries with keys ≥ key (except key itself)
        """

        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")

        # Cache frequently used attributes and classes for better performance
        TreeClass = type(self)
        NodeClass = TreeClass.NodeClass
        cached_l_factor = self.l_factor
        dummy = get_dummy(dim=TreeClass.DIM)
        check_and_convert_set = self.check_and_convert_set
        KListNodeCapacity = self.KListClass.KListNodeClass.CAPACITY
        tree_dim_plus_one = self.DIM + 1
        
        # Cache dummy properties for repeated use
        dummy_key = dummy.key

        # Case 1: Empty tree - return None left, right and key's subtree
        if self.is_empty():
            return self, None, TreeClass(l_factor=cached_l_factor), None

        # Initialize left and right return trees
        left_return = self
        right_return = TreeClass(l_factor=cached_l_factor)

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_entry = None     # Entry in right parent points to current subtree
        left_parent = None     # Parent node for left-side updates
        seen_key_subtree = None  # Cache last seen left_parents left subtree
        cur = left_return
        key_node_found = False

        # logger.debug(f"[DIM {self.DIM}] [SPLIT {key}] tree before split: {print_pretty(self)}")

        while True:
            # Cache node reference and minimize repeated attribute access
            node = cur.node
            cur._invalidate_tree_size()
            node_rank = node.rank
            is_leaf = node_rank == 1

            # Split node at key - cache results immediately
            left_split, key_subtree, right_split, next_entry = node.set.split_inplace(key)
            left_split = check_and_convert_set(left_split)

            # Cache item counts and next entry
            l_count = left_split.item_count()
            r_count = right_split.item_count()

            # --- Handle right side of the split ---
            # Determine if we need a new tree for the right split
            if r_count > 0:  # incl. dummy items
                # Cache type check to avoid repeated isinstance calls
                is_gkplus_type = isinstance(right_split, GKPlusTreeBase)
                
                if is_gkplus_type:
                    # Calculate the new rank for the item in the next dimension - use cached values
                    digest = dummy.get_digest_for_dim(tree_dim_plus_one)
                    new_rank = calc_rank_from_digest_k(digest, KListNodeCapacity)
                    right_split, _, _ = right_split.insert_entry(Entry(dummy, None), rank=new_rank)
                    right_split._invalidate_tree_size()
                    # TODO: Check why we need to invalidate size and why it is not done in insert_entry. Check it also for insert_entry()
                else:
                    right_split, _, _ = right_split.insert_entry(Entry(dummy, None))
                right_split = check_and_convert_set(right_split)

                # Cache node references for performance
                right_node = NodeClass(node_rank, right_split, node.right_subtree)

                if right_parent is None:
                    # Create a root node for right return tree
                    right_return.node = right_node
                    new_tree = right_return
                else:
                    new_tree = TreeClass(l_factor=cached_l_factor)
                    new_tree.node = right_node

                    # Update parent reference
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
                    else:
                        right_parent.node.right_subtree = new_tree

                if is_leaf:
                    # Prepare for updating 'next' pointers
                    new_tree.node.next = node.next

                # Prepare references for next iteration
                next_right_entry = next_entry
                next_cur = (
                    next_entry.left_subtree
                    if next_entry else new_tree.node.right_subtree
                )
                next_right_parent = new_tree
            else:
                if is_leaf and right_parent:
                    # Cache type check result - reuse from above if possible
                    is_gkplus_type = isinstance(right_split, GKPlusTreeBase)

                    # Create a leaf with a single dummy item
                    if is_gkplus_type:
                        # Calculate new rank for item in the next dimension - use cached values
                        digest = dummy.get_digest_for_dim(tree_dim_plus_one)
                        new_rank = calc_rank_from_digest_k(digest, KListNodeCapacity)
                        right_split, _, _ = right_split.insert_entry(Entry(dummy, None), rank=new_rank)
                        right_split._invalidate_tree_size()
                    else:
                        right_split, _, _ = right_split.insert_entry(Entry(dummy, None))

                    right_split = check_and_convert_set(right_split)
                    right_node = NodeClass(1, right_split, None)
                    new_tree = TreeClass(l_factor=cached_l_factor)
                    new_tree.node = right_node

                    # Update parent reference
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
                    else:
                        right_parent.node.right_subtree = new_tree

                    # Link leaf nodes
                    new_tree.node.next = node.next
                    next_right_parent = new_tree
                else:
                    next_right_parent = right_parent
                    next_cur = (
                        next_entry.left_subtree
                        if next_entry else node.right_subtree
                    )
            
                next_right_entry = right_entry
            
            # Update right parent variables for next iteration
            right_parent = next_right_parent
            right_entry = next_right_entry

            # --- Handle left side of the split ---
            # Determine if we need to create/update using left split
            if l_count > 1:  # incl. dummy items
                # Update current node to use left split
                node.set = left_split
                cur._invalidate_tree_size()

                if left_parent is None:
                    # Reuse left split as the root node for the left return tree
                    left_return.node = cur.node

                if is_leaf:
                    # Prepare for updating 'next' pointers
                    # do not rearrange subtrees at leaf level
                    # TODO: check if the right subtree can be set to None here, so we don't need to 
                    # do this later
                    l_last_leaf = node
                elif key_subtree:
                    # Highest node containing split key found
                    # All entries in its left subtree are less than key and
                    # are part of the left return tree
                    seen_key_subtree = key_subtree
                    node.right_subtree = key_subtree
                elif next_entry:
                    node.right_subtree = next_entry.left_subtree

                # Check if we need to update the left parent reference - optimized
                next_left_parent = left_parent if key_node_found else cur
            else:
                if is_leaf:
                    if left_parent or seen_key_subtree:
                        # Prepare unlinking leaf nodes and determine left return if needed
                        if l_count == 0:
                            # find the preceeding leaf node
                            if left_parent:
                                l_last_leaf = left_parent.get_max_leaf()
                            else:
                                # no left parent, so seen_key_subtree is the left return tree
                                left_return.node = seen_key_subtree.node
                                if left_return.item_count() == 1:
                                    # Only the dummy item in left return
                                    left_return.node = None
                                    l_last_leaf = None
                                else:
                                    # At least one non-dummy item in left return
                                    l_last_leaf = seen_key_subtree.get_max_leaf()
                                left_return._invalidate_tree_size()
                        else:
                            node.set = left_split
                            l_last_leaf = node
                    else:
                        # No non-dummy entry in left tree
                        self.node = None
                        l_last_leaf = None
                        
                    next_left_parent = left_parent
                else:
                    # Determine new subtree efficiently
                    if key_subtree is not None:
                        # Highest node containing split key found
                        # All entries in its left subtree are less than key and
                        # are part of the left return tree
                        new_subtree = key_subtree
                        seen_key_subtree = key_subtree
                    else:
                        # There must be a next entry, since nodes have at least two entries. 
                        # l_count <= 1 and we have not found the split key (not key_subtree)
                        new_subtree = next_entry.left_subtree

                    if left_parent and not key_node_found:
                        left_parent.node.right_subtree = new_subtree
                        
                    next_left_parent = left_parent

            left_parent = next_left_parent

            # Main return logic
            if is_leaf:
                if next_entry is None and not right_return.is_empty():
                    pivot, pivot_next = right_return.find_pivot()
                    if pivot and pivot.item.key > dummy_key:
                        next_entry = pivot
                    else:
                        next_entry = pivot_next

                if l_last_leaf: # unlink leaf nodes
                    l_last_leaf.next = None
                return_subtree = key_subtree
                return self, return_subtree, right_return, next_entry

            if key_subtree:
                # Do not update left parent reference from this point on
                key_node_found = True

            # Continue to next iteration with updated current node
            cur = next_cur
        
    def _create_dummy_singleton_tree(self, rank: int, dim: int, l_factor: float, KListClass) -> 'GKPlusTreeBase':
        """
        Create a dummy singleton tree with a single entry containing the dummy item.

        Args:
            dim: The dimension of the tree
            l_factor: The load factor for the tree
            set_type: The type of the set to use

        Returns:
            A GKPlusTreeBase instance with a single dummy entry
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

    def _zip_handle_same_rank_sets(self, other: 'GKPlusTreeBase', left: 'GKPlusTreeBase') -> 'GKPlusTreeBase':
        """
        Helper method to handle the common logic for zipping sets when trees have the same rank.
        Consolidates repeated code for finding pivots, handling subtrees, and linking nodes.
        """
        r_pivot, r_pivot_next = other.node.set.find_pivot()
        logger.debug(f"[DIM {self.DIM}] Found r_pivot: {r_pivot.item.key if r_pivot else None}, r_pivot_next: {r_pivot_next.item.key if r_pivot_next else None}")

        if not r_pivot:
            # This can only happen in leaf nodes where dummy entries may be the only entries
            logger.debug(f"[DIM {self.DIM}] No valid leftmost entry found")
            self.node.next = other.node.next
            self._invalidate_tree_size()
            self.node.set = self.check_and_convert_set(self.node.set)
            return self

        r_leftmost_subtree = r_pivot.left_subtree
        logger.debug(f"[DIM {self.DIM}] r_leftmost_subtree: {print_pretty(r_leftmost_subtree)}")

        if self.node.rank > 1:
            if r_leftmost_subtree is None:
                logger.debug(f"[DIM {self.DIM}] No left subtree found for r_pivot {r_pivot.item.key}, using self.node.right_subtree as its left subtree")
                
                if self.node.right_subtree.node.rank == 1:
                    logger.debug(f"[DIM {self.DIM}] Right subtree is a leaf {print_pretty(self.node.right_subtree)}, finding other min leaf tree to link it")
                    other_min_leaf_tree = other.get_min_leaf_tree()
                    logger.debug(f"[DIM {self.DIM}] Found other min leaf tree set: {print_pretty(other_min_leaf_tree.node.set)}")
                    self.node.right_subtree.node.next = other_min_leaf_tree

                r_pivot.left_subtree = self.node.right_subtree
            else:
                logger.debug(f"[DIM {self.DIM}] Found left subtree for r_pivot {r_pivot.item.key}: {print_pretty(r_leftmost_subtree)}")
                new_left = self.node.right_subtree.zip(r_leftmost_subtree)
                r_pivot.left_subtree = new_left
        else:
            # Leaf node case
            logger.debug(f"[DIM {self.DIM}] Linking leaf to be zipped into with other's next. ")
            logger.debug(f"[DIM {self.DIM}] Leaf to be zipped into: {print_pretty(self)}")
            logger.debug(f"[DIM {self.DIM}] Linked leaf: {print_pretty(other.node.next)}")
            self.node.next = other.node.next

        return None  # Signal to continue processing

    def zip(self, other: 'GKPlusTreeBase', is_root: bool = True) -> 'GKPlusTreeBase':
        """
        Zip two GKPlusTreeBase instances together, merging their entries.

        Args:
            other: Another GKPlusTreeBase instance to zip with. The other must not have the dimensions dummy item. It will be included during the zipping process.

        Returns:
            A new GKPlusTreeBase instance containing the merged entries
        """
        if not isinstance(other, GKPlusTreeBase):
            raise TypeError(f"other must be an instance of GKPlusTreeBase, got {type(other).__name__}")
        
        if other.DIM != self.DIM:
            raise ValueError(f"Dimension mismatch: self.DIM={self.DIM}, other.DIM={other.DIM}")

        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Zipping trees: ")
            logger.debug(f"[DIM {self.DIM}] Self (item_count: {self.item_count()}, real_item_count: {self.real_item_count()}): {print_pretty(self)}")
            logger.debug(f"[DIM {self.DIM}] Other (item_count: {other.item_count()}, real_item_count: {other.real_item_count()}): {print_pretty(other)}")
            logger.debug(f"[DIM {self.DIM}] Self items: {[e.item.key for e in self]}")
            logger.debug(f"[DIM {self.DIM}] Other items: {[e.item.key for e in other]}")

        # Early returns for empty trees
        if self.is_empty():
            if other.is_empty():
                return self
            
            if is_root:
                is_root = False
                other_rank = other.node.rank
                dummy_tree = self._create_dummy_singleton_tree(other_rank, other.DIM, other.l_factor, self.KListClass)
                self.node = dummy_tree.node
                self._invalidate_tree_size()
                logger.debug(f"[DIM {self.DIM}] Created dummy tree: {print_pretty(self)}")
                return self.zip(other)
            return other
        
        if other.is_empty():
            return self
        
        left = self
        left_rank = left.node.rank
        
        # Case 1: Left rank < other rank - insert left into other's leftmost position
        if left_rank < other.node.rank:
            if is_root:
                is_root = False  # Mark that we're no longer at root level

            r_pivot, r_pivot_next = other.node.set.find_pivot()
            logger.debug(f"[DIM {self.DIM}] Found r_pivot: {r_pivot.item.key if r_pivot else None}, r_pivot_next: {r_pivot_next.item.key if r_pivot_next else None}")

            l_pivot, l_pivot_next = left.node.set.find_pivot()
            logger.debug(f"[DIM {self.DIM}] Found l_pivot: {l_pivot.item.key if l_pivot else None}, l_pivot_next: {l_pivot_next.item.key if l_pivot_next else None}")

            # Handle r_pivot's left subtree
            if r_pivot and r_pivot.left_subtree:
                logger.debug(f"[DIM {self.DIM}] Found r_pivot {r_pivot.item.key} with left subtree: {print_pretty(r_pivot.left_subtree)}")
                logger.debug(f"[DIM {self.DIM}] Left tree before zipping (item_count: {left.item_count()}, real_item_count: {left.real_item_count()}): {print_pretty(left)}")
                left = left.zip(r_pivot.left_subtree)
                logger.debug(f"[DIM {self.DIM}] Left tree after zipping (item_count: {left.item_count()}, real_item_count: {left.real_item_count()}): {print_pretty(left)}")
                logger.debug(f"[DIM {self.DIM}] Left Items: {[e.item.key for e in left]}")
                logger.debug(f"[DIM {self.DIM}] Left next: {print_pretty(left.node.next)}")
                r_pivot.left_subtree.node = left.node
                r_pivot.left_subtree._invalidate_tree_size()
                other._invalidate_tree_size()
            else:
                logger.debug(f"[DIM {self.DIM}] No left subtree found for r_pivot {r_pivot.item.key if r_pivot else None}, using self as its left subtree")
                
                logger.debug(f"[DIM {self.DIM}] Left tree is a leaf {print_pretty(left)}, finding other min leaf tree to link it")
                other_min_leaf_tree = other.get_min_leaf_tree()
                logger.debug(f"[DIM {self.DIM}] Found other min leaf tree set: {print_pretty(other_min_leaf_tree.node.set)}")
                if left.node.rank == 1:
                    left.node.next = other_min_leaf_tree

                else:
                    max_left_leaf = left.get_max_leaf()
                    max_left_leaf.next = other_min_leaf_tree

                r_pivot.left_subtree = left
                logger.debug(f"[DIM {self.DIM}] Set r_pivot {r_pivot.item.key}'s left_subtree to self: {print_pretty(left)}")
                other._invalidate_tree_size()

            # Insert replica of left pivot into other's set
            replica_entry = Entry(_get_replica(l_pivot.item), None)
            
            if isinstance(other.node.set, KListBase):
                other.node.set, _, _ = other.node.set.insert_entry(replica_entry, other.node.rank)
            else:
                singleton_tree = bulk_create_gkplus_tree([replica_entry], self.DIM + 1, l_factor=self.l_factor, KListClass=self.SetClass)
                other.node.set = singleton_tree.zip(other.node.set)
            
            other._invalidate_tree_size()

            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] new_other (item_count: {other.item_count()}, real_item_count: {other.real_item_count()}): {print_pretty(other)}")
                logger.debug(f"[DIM {self.DIM}] Other Items: {[e.item.key for e in other]}")
                for i, leaf in enumerate(other.iter_leaf_nodes()):
                    logger.debug(f"[DIM {self.DIM}] Other Leaf {i}: {print_pretty(leaf.set)}")
                    logger.debug(f"[DIM {self.DIM}] Leaf next: {print_pretty(leaf.next)}")
                logger.debug(f"[DIM {self.DIM}] Self node next: {print_pretty(self.node.next)}")
                logger.debug(f"[DIM {self.DIM}] new_self (item_count: {self.item_count()}, real_item_count: {self.real_item_count()}): {print_pretty(self)}")
                logger.debug(f"[DIM {self.DIM}] Self Items: {[e.item.key for e in self]}")
            
            return other
        
        # Case 2: Self rank > other rank - recursively zip into right subtree
        elif self.node.rank > other.node.rank:
            new_right = self.node.right_subtree.zip(other, is_root=False)
            logger.debug(f"[DIM {self.DIM}] Zipped right subtree (item_count: {new_right.item_count()}, real_item_count: {new_right.real_item_count()}): {print_pretty(new_right)}")
            logger.debug(f"[DIM {self.DIM}] Items right subtree: {[e.item.key for e in new_right]}")
            self.node.right_subtree = new_right
            self._invalidate_tree_size()
            logger.debug(f"[DIM {self.DIM}] Self items: {[e.item.key for e in self]}")
            return self
        
        # Case 3: Same rank - merge sets
        else:
            # Normalize sets to same type before merging
            self_is_klist = isinstance(self.node.set, KListBase)
            other_is_klist = isinstance(other.node.set, KListBase)
            
            # Convert to same type if needed
            if self_is_klist != other_is_klist:
                if self_is_klist:
                    # Convert self from KList to GKPlusTree
                    left.node.set = bulk_create_gkplus_tree(self.node.set, other.node.set.DIM, other.l_factor, type(self.node.set))
                    logger.debug(f"[DIM {self.DIM}] Converted left to GKPlusTreeBase: {print_pretty(left)}")
                else:
                    # Convert other from KList to GKPlusTree
                    other.node.set = bulk_create_gkplus_tree(other.node.set, self.node.set.DIM, self.l_factor, type(other.node.set))
                    logger.debug(f"[DIM {self.DIM}] Converted other to GKPlusTreeBase: {print_pretty(other)}")
                    _, _, other.node.set, _ = other.node.set.unzip(-1)
                    logger.debug(f"[DIM {self.DIM}] unzipped other at {-1}: {print_pretty(other)}")

            # Now both sets are the same type - handle common logic
            if IS_DEBUG and self_is_klist and other_is_klist:
                logger.debug(f"[DIM {self.DIM}] Merging KLists: {print_pretty(self.node.set)} and {print_pretty(other.node.set)}")
            
            # Common pivot finding and subtree handling
            result = self._zip_handle_same_rank_sets(other, left)
            if result is not None:
                return result  # Early return for special cases
            
            # Merge the sets based on their types
            if self_is_klist and other_is_klist:
                # Both KLists - merge entries directly
                dummy_lower_dim = get_dummy(self.DIM - 1)
                for entry in other.node.set:
                    if entry.item.key > dummy_lower_dim.key:
                        self.node.set, _, _ = self.node.set.insert_entry(entry)
            else:
                # Both are GKPlusTreeBase (after conversion) - recursively zip
                zipped = left.node.set.zip(other.node.set)
                zipped = self.check_and_convert_set(zipped)
                self.node.set = zipped

            # Finalize the merge
            self.node.right_subtree = other.node.right_subtree
            logger.debug(f"[DIM {self.DIM}] Merged Sets sitting at root of tree: {print_pretty(self)}")
            logger.debug(f"[DIM {self.DIM}] root node next: {print_pretty(self.node.next)}")

            left._invalidate_tree_size()
            self._invalidate_tree_size()
            
            if self_is_klist and other_is_klist:
                self.node.set = self.check_and_convert_set(self.node.set)
                logger.debug(f"[DIM {self.DIM}] Zipped KLists: {print_pretty(self)}")
            else:
                logger.debug(f"[DIM {self.DIM}] Zipped GKPlusTreeBase sets: {print_pretty(self)}")
            
            return self
            # else:
            #     # Both sets are GKPlusTreeBase, merge them
            #     if IS_DEBUG:
            #         logger.debug(f"[DIM {self.DIM}] Merging GKPlusTrees: {print_pretty(self.node.set)} and {print_pretty(other.node.set)}")
            #     zipped = self.node.set.zip(other.node.set)
            #     zipped = self.check_and_convert_set(zipped)
            #     self.node.set = zipped
            #     logger.debug(f"[DIM {self.DIM}] Zipped GKPlusTrees: {print_pretty(self)}")
            #     if left_rank == 1:
            #         self.node.next = other.node.next
            #     left._invalidate_tree_size()
            #     self._invalidate_tree_size()
            #     return self


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

        node = self.node
        rank = node.rank
        is_leaf = rank == 1
        right_subtree = node.right_subtree
        
        self._invalidate_tree_size()

        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Unzipping tree at key {key}: {print_pretty(self)}")

        # Unzip the node set
        if isinstance(node.set, KListBase):
            left_set, left_subtree_of_key, right_set, next_entry = node.set.split_inplace(key)
        else:
            left_set, left_subtree_of_key, right_set, next_entry = node.set.unzip(key)

        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Split results - Left: {print_pretty(left_set)}")
            logger.debug(f"[DIM {self.DIM}] Split results - Key subtree: {print_pretty(left_subtree_of_key)}")
            logger.debug(f"[DIM {self.DIM}] Split results - Right: {print_pretty(right_set)}")
            if next_entry:
                logger.debug(f"[DIM {self.DIM}] Split results - Next entry: key={next_entry.item.key}, value={next_entry.item.value}")
                if next_entry.left_subtree:
                    logger.debug(f"[DIM {self.DIM}] Split results - Next entry left subtree node set: {print_pretty(next_entry.left_subtree.node.set)}")
            else:
                logger.debug(f"[DIM {self.DIM}] Split results - Next entry: None")

        # Update the current node's set
        node.set = left_set
        left_next = node.next

        # Handle leaf nodes
        if is_leaf:
            return self._handle_leaf_unzip(left_set, left_subtree_of_key, right_set, 
                                         next_entry, left_next)

        # Handle non-leaf nodes
        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Handle unzipping result at non-leaf node with rank {rank}")

        return self._handle_nonleaf_unzip(key, left_set, left_subtree_of_key, right_set, 
                                        next_entry, right_subtree, rank)

    def _retrieve_next(self, key: int) -> Optional[Entry]:
        """Retrieve the next entry in the tree after the given key."""
        pivot, pivot_next = self.find_pivot()
        dummy_key = get_dummy(self.DIM).key
        logger.debug(f"[DIM {self.DIM}] Retrieving next entry after key {key} - Pivot: {pivot.item.key if pivot else None}, Next: {pivot_next.item.key if pivot_next else None}")
        
        if pivot and pivot.item.key > dummy_key:
            logger.debug(f"[DIM {self.DIM}] Pivot found and key > {dummy_key} (dummy key): {pivot.item.key}. Make it the next entry.")
            return pivot
        else:
            logger.debug(f"[DIM {self.DIM}] No valid pivot found or key <= {dummy_key}. Returning pivot_next.")
            return pivot_next

    def _handle_leaf_unzip(self, left_set, left_subtree_of_key, right_set, next_entry, left_next):
        """Handle unzipping at leaf level."""
        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Handle unzipping result at leaf level")
        if left_set.is_empty():
            # Return right subtree directly
            right_return = self.lift(right_set, None, 1)
            logger.debug(f"[DIM {self.DIM}] Left is empty, Right: {print_pretty(right_return)}")
            if not right_return.is_empty():
                logger.debug(f"[DIM {self.DIM}] Right return has items, setting next to left next: {print_pretty(left_next)}")
                right_return.node.next = left_next
                logger.debug(f"[DIM {self.DIM}] Finding next entry in right return.")
                # check adding condition: Only search for next entry if it is not klist?
                # check adding condition: Only search for next entry if it has value==None?
                next_entry = right_return._retrieve_next(get_dummy(self.DIM).key)
            else:
                logger.debug(f"[DIM {self.DIM}] Right is empty, finding next entry in left next: {print_pretty(left_next)}")
                next_entry = left_next._retrieve_next(get_dummy(self.DIM).key) if left_next else None
                logger.debug(f"[DIM {self.DIM}] Found next entry in left next: key={next_entry.item.key if next_entry else None}, value={next_entry.item.value if next_entry else None}")
                
            logger.debug(f"[DIM {self.DIM}] Set left node to None (empty)")
            self.node = None
            return self, left_subtree_of_key, right_return, next_entry
        
        logger.debug(f"[DIM {self.DIM}] Left set has items, processing further")
        right_return = self.lift(right_set, None, 1)
        if not right_return.is_empty():
            logger.debug(f"[DIM {self.DIM}] Right has items, finding next entry in it: {print_pretty(right_return)}")
            right_return.node.next = left_next
            next_entry = right_return._retrieve_next(get_dummy(self.DIM).key)
        else:
            logger.debug(f"[DIM {self.DIM}] Right is empty, finding next entry in left next: {print_pretty(left_next)}")
            next_entry = left_next._retrieve_next(get_dummy(self.DIM).key) if left_next else None
        
        
        self.node.set = left_set
        logger.debug(f"[DIM {self.DIM}] Set left (self) node with left set: {print_pretty(left_set)}")
        logger.debug(f"[DIM {self.DIM}] Unlinking leaf nodes: Set self.node.next to None")
        self.node.next = None
        
        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Leaf unzip complete - Right: {print_pretty(right_return)}")
        
        return self, left_subtree_of_key, right_return, next_entry

    def _handle_nonleaf_unzip(self, key, left_set, left_subtree_of_key, right_set, next_entry, right_subtree, rank):
        """Handle unzipping at non-leaf level."""
        # Case 1: Found the split key with its left subtree
        if left_subtree_of_key:
            return self._handle_key_subtree_case(key, left_set, left_subtree_of_key, right_set, 
                                               next_entry, right_subtree, rank)
        
        # Cases 2: Unified handling for left set with multiple items or single/empty
        left_count = left_set.item_count()
        return self._handle_no_key_subtree_case(key, left_set, left_count, right_set, 
                                              next_entry, right_subtree, rank)

    def _handle_key_subtree_case(self, key, left_set, left_subtree_of_key, right_set, next_entry, right_subtree, rank):
        """Handle the case where the split key has a left subtree."""
        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Found left subtree of key: {print_pretty(left_subtree_of_key)}")
        
        # Update current node based on left set size
        if left_set.item_count() > 1:
            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] Left set has items, using as right subtree")
            self.node.right_subtree = left_subtree_of_key
        else:
            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] Left set minimal, using left subtree as node")
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
            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] Unlinked leaf nodes")
        
        return self, key_subtree, right_return, next_entry

    def _handle_no_key_subtree_case(self, key, left_set, left_count, right_set, next_entry, right_subtree, rank):
        """Unified handler for cases where left set has multiple items or single/empty items."""
        logger.debug(f"[DIM {self.DIM}] handle no key subtree case")
        key_subtree = None
        right_return = None
        
        if next_entry:
            logger.debug(f"[DIM {self.DIM}] Next entry found: key={next_entry.item.key}, value={next_entry.item.value}")
            if next_entry.left_subtree:
                logger.debug(f"[DIM {self.DIM}] Next entry has left subtree, unzipping it: {print_pretty(next_entry.left_subtree)}")
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
                    logger.debug(f"[DIM {self.DIM}] Split results - Update next entry to: key={new_next.item.key}, value={new_next.item.value}")
                    next_entry = new_next
                else:
                    logger.debug(f"[DIM {self.DIM}] No new next entry found, keeping original: key={next_entry.item.key}, value={next_entry.item.value}")
            else:
                logger.debug(f"[DIM {self.DIM}] Next entry has no left subtree, unzipping right subtree")
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
            logger.debug(f"[DIM {self.DIM}] No next entry, unzipping right subtree: {print_pretty(right_subtree)}")
            if right_subtree:
                l_right_subtree, key_subtree, r_leftmost_subtree, new_next = right_subtree.unzip(key)
                next_entry = new_next
                if IS_DEBUG:
                    logger.debug(f"[DIM {self.DIM}] Unzipped right subtree: {print_pretty(right_subtree)}")
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


    def lift(self, set: Optional[AbstractSetDataStructure], right: GKPlusTreeBase, rank: int) -> GKPlusTreeBase:
        """
        Create a GKPlusTree from a set, a right subtree at the specified rank.

        Args:
            set: The tree node set. If empty, return the right subtree.
            right: The tree node's right subtree
            rank: The rank of the tree

        Returns:
            A new GKPlusTreeBase instance containing the merged entries
        """
        
        if IS_DEBUG:
            logger.debug(f"[DIM {self.DIM}] Lifting set {print_pretty(set)} with rank {rank} and right subtree: {print_pretty(right)}")
        TreeClass = type(self)
        if right is not None and right.is_empty():
            right = None

        if set.is_empty():
            if right is None:
                # If both set and right subtree are empty, return an empty tree
                return TreeClass(l_factor=self.l_factor)
            return right
        
        new_node = self.NodeClass(rank, set, right)
        logger.debug(f"[DIM {self.DIM}] Created new node: {print_pretty(new_node.set)} with rank {rank} and right subtree: {print_pretty(right)}")
        
        return TreeClass(new_node, self.l_factor)



    def iter_real_entries(self):
        """
        Iterate over all real entries (excluding dummies) in the gk-plus-tree.

        Yields:
            Entry: Each entry in the tree, excluding dummy entries.
        """
        if self.is_empty():
            return

        for entry in self:
            if entry.item.key >= 0:
                yield entry

    def __iter__(self):
        """Yields each entry of the gk-plus-tree in order. including all dummy entries."""
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

    def check_and_convert_set(self,
                              set: AbstractSetDataStructure
                              ) -> AbstractSetDataStructure:
        if isinstance(set, KListBase):
            return self.check_and_expand_klist(set)
        elif isinstance(set, GKPlusTreeBase):
            return self.check_and_collapse_tree(set)
        else:
            raise TypeError(f"Unsupported set type: {type(set).__name__}. "
                            "Expected KListBase or GKPlusTreeBase.")

    def check_and_expand_klist(self,
                               klist: KListBase
                               ) -> AbstractSetDataStructure:
        """
        Check if a KList exceeds the threshold and should be converted to a GKPlusTree.

        Args:
            klist: The KList to check

        Returns:
            Either the original KList or a new GKPlusTree based on the threshold
        """
        # Check if the item count exceeds l_factor * CAPACITY
        k = klist.KListNodeClass.CAPACITY
        threshold = int(k * self.l_factor)
        if klist.item_count() > threshold:
            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] Klist with item count {klist.item_count()} exceeds threshold {threshold}, converting to GKPlusTree; KList: {print_pretty(klist)}")
            # Convert to GKPlusTree with increased dimension
            new_dim = type(self).DIM + 1
            new_tree = _klist_to_tree(klist, k, new_dim, self.l_factor)
            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] Converted KList to GKPlusTree: {print_pretty(new_tree)}")
            return new_tree

        return klist

    def check_and_collapse_tree(self,
                                tree: 'GKPlusTreeBase'
                                ) -> AbstractSetDataStructure:
        """
        Check if a GKPlusTree has few enough items to be collapsed into a KList.

        Args:
            tree: The GKPlusTree to check

        Returns:
            Either the original tree or a new KList based on the threshold
        """
        # Get the threshold based on the KList capacity
        k = self.KListClass.KListNodeClass.CAPACITY
        threshold = int(k * self.l_factor)
        
        if tree.DIM == 1:
            # We want to keep the GKPlusTree structure for dimension 1
            return tree
        
        if tree.is_empty():
            return tree

        tree_item_count = tree.item_count()
        expected_klist_size = tree_item_count - 1  # Exclude the tree's own dummy item
        real_item_count = tree.real_item_count()
        
        # Fast path: no dummy items in the tree
        if expected_klist_size == real_item_count:
            if expected_klist_size <= threshold:
                new_klist = _tree_to_klist(tree)
                if IS_DEBUG:
                    logger.debug(f"[DIM {self.DIM}] GKPlusTree with tree item count {tree_item_count}, real item count {real_item_count} and expected klist item count {expected_klist_size} is below threshold {threshold}: {print_pretty(tree)}")
                logger.debug(f"[DIM {self.DIM}] Converted GKPlusTree to KList: {print_pretty(new_klist)}")
                return new_klist
            return tree
        
        # The dummy item from the tree and from all expanded leafs are removed when collapsed
        expanded_leafs_count = tree.expanded_count()
        expected_klist_size -= expanded_leafs_count

        if expected_klist_size <= threshold:
            new_klist = _tree_to_klist(tree)
            if IS_DEBUG:
                logger.debug(f"[DIM {self.DIM}] GKPlusTree with tree item count {tree_item_count}, real item count {real_item_count} and expected klist item count {expected_klist_size} is below threshold {threshold}: {print_pretty(tree)}")
                logger.debug(f"[DIM {self.DIM}] Converted GKPlusTree to KList: {print_pretty(new_klist)}")
            return new_klist
        return tree


def _tree_to_klist(tree: GKPlusTreeBase) -> KListBase:
    """
    Convert a GKPlusTree to a KList.
    
    Args:
        tree: The GKPlusTree to convert
        
    Returns:
        A new KList containing all items from this tree
    """
    if not isinstance(tree, GKPlusTreeBase):
        raise TypeError("tree must be an instance of GKPlusTreeBase")

    if tree.is_empty():
        return tree.KListClass()
    
    klist = tree.KListClass()

    # Insert items with keys larger than current trees dummy key into the new klist
    # Smaller keys are dummies from higher dimensions, caused by gnode expansions within the tree to collapse. These are dropped.
    # Larger dummy keys are from lower dimensions, that must be preserved.
    # Note: dummy key of DIM j is -j.
    for entry in tree:
        tree_dummy = get_dummy(tree.DIM)
        if entry.item.key > tree_dummy.key:
            klist, _, _ = klist.insert_entry(entry)
    return klist


def _klist_to_tree(klist: KListBase, K: int, DIM: int, l_factor: float = 1.0) -> GKPlusTreeBase:
    """
    Convert a KList to a GKPlusTree by extracting its entries and creating a new tree from them.
    
    Args:
        klist: The KList to convert
        K: The KList capacity (number of items per klist node)
        DIM: The dimension for the new tree
        l_factor: The threshold factor for conversion
        
    Returns:
        A new GKPlusTree containing all items from the KList
    """
    if not isinstance(klist, KListBase):
        raise TypeError("klist must be an instance of KListBase")
    
    if klist.is_empty():
        return _get_create_gkplus_tree()(K, DIM, l_factor)
    return bulk_create_gkplus_tree(klist, DIM, l_factor, type(klist))


def _bulk_create_klist(entries: list[Entry], KListClass: type[KListBase]) -> KListBase:
    """Create a KList from a list of entries."""
    klist = KListClass()
    insert_entry_fn = klist.insert_entry  # Cache method reference
    for entry in entries:
        klist, _, _ = insert_entry_fn(entry)
    return klist


def _build_leaf_level_trees(
    rank_data_map: dict[int, RankData],
    KListClass: type[KListBase],
    NodeClass: type[GKPlusNodeBase],
    TreeClass: type[GKPlusTreeBase],
    l_factor: float,
) -> list[GKPlusTreeBase]:
    """
    Build leaf level trees from rank data for rank 1.
    
    Args:
        rank_data_map: Consolidated rank data map containing entries and boundaries
        KListClass: The KList class to use for creating klists
        NodeClass: The node class to use for creating nodes
        TreeClass: The tree class to use for creating trees
        l_factor: The threshold factor for conversion
    Returns:
        A list of GKPlusTreeBase instances representing the leaf level trees
    """
    
    # Extract rank 1 data
    rank_1_data = rank_data_map[1]
    entries = rank_1_data.entries
    boundaries_map = rank_1_data.boundaries
    
    threshold = KListClass.KListNodeClass.CAPACITY * l_factor
    leaf_trees = []
    prev_node = None

    for i in range(len(boundaries_map)):
        start_idx = boundaries_map[i]
        end_idx = boundaries_map[i + 1] if i + 1 < len(boundaries_map) else len(entries)
        node_entries = entries[start_idx:end_idx]

        if len(node_entries) <= threshold:
            node_set = _bulk_create_klist(node_entries, KListClass)
        else:
            node_set = bulk_create_gkplus_tree(node_entries, TreeClass.DIM + 1, l_factor, KListClass)
        leaf_node = NodeClass(1, node_set, None)
        leaf_tree = TreeClass(l_factor=l_factor)
        leaf_tree.node = leaf_node
        if prev_node is not None:
            prev_node.next = leaf_tree
        prev_node = leaf_node
        leaf_trees.append(leaf_tree)

    return leaf_trees

def _build_internal_levels(
    rank_data_map: dict[int, RankData],
    leaf_trees: list[GKPlusTreeBase],
    KListClass: type[KListBase],
    TreeClass: type[GKPlusTreeBase],
    NodeClass: type[GKPlusNodeBase],
    threshold: int,
    l_factor: float,
    max_rank: int,
) -> GKPlusTreeBase:
    """
    Build internal levels of the GKPlusTree from rank data and leaf level trees.
    
    Args:
        rank_data_map: Consolidated rank data map containing entries, boundaries, and child indices
        leaf_trees: List of leaf trees created
        TreeClass: The tree class to use for creating trees
        NodeClass: The node class to use for creating nodes
        KListClass: The KList class to use for creating klists
        threshold: The threshold for node set type (KList vs GKPlusTree)
        l_factor: The threshold factor for conversion
        max_rank: The maximum rank to process
        
    Returns:
        A new GKPlusTreeBase instance representing the root of the tree
    """
    rank_trees_map: dict[int, GKPlusTreeBase] = {}
    rank_trees_map[1] = leaf_trees  # Rank 1 trees are the leaf trees

    # Cache frequently accessed functions and values for better performance
    bulk_create_klist_fn = _bulk_create_klist
    bulk_create_gkplus_tree_fn = bulk_create_gkplus_tree
    tree_dim_plus_one = TreeClass.DIM + 1

    rank = 2
    sub_trees = leaf_trees
    subtrees_lift = []  # Pre-allocate list for subtrees to lift to the next level
    
    while rank <= max_rank:
        # Fast path: skip ranks with no entries
        rank_data = rank_data_map.get(rank)
        if not rank_data or not rank_data.entries:
            rank += 1
            continue

        # Extract data for this rank
        entries = rank_data.entries
        child_idx_list = rank_data.child_indices or []
        boundaries = rank_data.boundaries or []
        boundaries_len = len(boundaries)
        entries_len = len(entries)

        subtrees_lift.clear()  # reuse subtrees list
        prev_child_idx = -1
        
        # Cache sub_trees indexing
        sub_trees_get = sub_trees.__getitem__
        subtrees_lift_append = subtrees_lift.append
        
        # Handle rank node sets and subtrees
        for i in range(boundaries_len):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1] if i < boundaries_len - 1 else entries_len
            node_entries = entries[start_idx:end_idx]
            node_child_indices = child_idx_list[start_idx:end_idx]

            # Create node entries and attach left subtrees 
            # Skip the pivot entry as it has no left subtree
            for entry, child_idx in islice(zip(node_entries, node_child_indices), 1, None):
                next_child_idx = prev_child_idx + 1
                
                if child_idx is not None:
                    entry.left_subtree = sub_trees_get(child_idx)
                    # Check for collapsed subtrees and lift them to the next level
                    if next_child_idx < child_idx:
                        for i in range(next_child_idx, child_idx):
                            subtrees_lift.append(sub_trees[i])
                else:
                    # We can use the next child index because we lifted the earlier missing subtrees from the lower ranks
                    entry.left_subtree = sub_trees_get(next_child_idx)
                    child_idx = next_child_idx
                
                prev_child_idx = child_idx

            # Create node set
            node_entries_len = len(node_entries)
            if node_entries_len <= threshold:
                node_set = bulk_create_klist_fn(node_entries, KListClass)
            else:
                node_set = bulk_create_gkplus_tree_fn(
                    node_entries,
                    tree_dim_plus_one,
                    l_factor,
                    KListClass
                )

            # Create tree node with right subtree
            prev_child_idx += 1
            right_subtree = sub_trees_get(prev_child_idx)
            tree_node = NodeClass(rank, node_set, right_subtree)
            tree = TreeClass(l_factor=l_factor)
            tree.node = tree_node

            # Lift the subtree to the next level to be assigned as a subtree
            subtrees_lift_append(tree)

        # Lift the remaining subtrees
        remaining_start_idx = prev_child_idx + 1
        len_sub_trees = len(sub_trees)
        if remaining_start_idx < len_sub_trees:
            for i in range(remaining_start_idx, len_sub_trees):
                subtrees_lift.append(sub_trees[i])

        sub_trees, subtrees_lift = subtrees_lift, sub_trees
        rank += 1

    # The first and only tree after processing the highest rank is the root tree
    return sub_trees[0] 


def bulk_create_gkplus_tree(
    entries: Union[list[Entry], KListBase[Entry]],
    DIM: int,
    l_factor: float,
    KListClass: type[KListBase],
) -> GKPlusTreeBase:
    """
    Bottom-up bulk creation of a GKPlusTree from a list of entries.

    Key insights:
    1. All entries exist in leaf nodes
    2. Rank determines node boundaries: higher rank = start of new node
    3. Higher rank entries are replicated upward to their rank level
    4. Threshold determines KList vs GKPlusTree for node implementation
    5. Entries are already sorted from KList iteration
    
    Args:
        entries: KList or Python List of Entry objects to be inserted into the tree
        k: The capacity parameter for the tree
        DIM: The dimension of the tree
        l_factor: The threshold factor for conversion
        KListClass: The KList class to use for creating KLists
    Returns:
        A new GKPlusTreeBase instance containing the entries
    """
    if not entries:
        raise ValueError("entries must be a non-empty list or KListBase")

    k = KListClass.KListNodeClass.CAPACITY
    sample_tree = _get_create_gkplus_tree()(k, DIM, l_factor)
    
    # Pre-cache all constants and frequently used objects
    NodeClass = sample_tree.NodeClass
    TreeClass = type(sample_tree)
    get_replica_fn = _get_replica
    threshold = int(k * l_factor)
    dummy = Entry(get_dummy(DIM), None)
    dummy_key = dummy.item.key
    
    # Consolidated rank data structure  
    rank_data_map: dict[int, RankData] = {}
    max_rank = 1
    
    for entry in entries:
        entry_item = entry.item
        digest = entry_item.get_digest_for_dim(DIM)
        insert_rank = calc_rank_from_digest_k(digest, k)

        # Update max_rank early to avoid repeated comparisons
        if insert_rank > max_rank:
            max_rank = insert_rank
        
        # Process ranks from insert_rank down to 1, with cached lookups
        for rank in range(insert_rank, 0, -1):
            # Get or create rank data
            if rank not in rank_data_map:
                rank_data_map[rank] = RankData()
            
            rank_data = rank_data_map[rank]
            
            # Cache all values for this rank
            rank_entries = rank_data.entries
            boundaries = rank_data.boundaries
            child_indices = rank_data.child_indices
            pivot_item = rank_data.pivot_item
            next_child_idx = rank_data.next_child_idx
            
            if rank == insert_rank or rank == 1:
                # Handle insert rank and leaf rank
                insert_entry = entry if rank == 1 else Entry(get_replica_fn(entry_item), None)

                # Entry insertion
                if rank_entries is not None:
                    if pivot_item is not None:
                        rank_entries.append(Entry(get_replica_fn(pivot_item), None))
                        if child_indices is not None:
                            child_indices.append(None)
                    
                    rank_entries.append(insert_entry)
                    if rank > 1:
                        if child_indices is not None:
                            child_indices.append(next_child_idx)
                        else:
                            rank_data.child_indices = [None, next_child_idx]
                        next_child_idx += 1
                        rank_data.next_child_idx = next_child_idx
                else:
                    # Initialize new entries list for this rank
                    if rank == 1 or rank == max_rank or pivot_item is None:
                        # For leaf and root level or when there is no pivot flag, start with dummy
                        new_entries = [dummy, insert_entry]
                        pivot_item = dummy
                    else:
                        # For internal nodes with pivot, start with pivot entry
                        pivot_entry = Entry(get_replica_fn(pivot_item), None)
                        new_entries = [pivot_entry, insert_entry]
                    
                    rank_data.entries = new_entries
                    
                    # Initialize child indices for non-leaf ranks
                    if rank > 1:
                        rank_data.child_indices = [None, next_child_idx]
                        next_child_idx += 1
                        rank_data.next_child_idx = next_child_idx

                # At least two entries have been added (dummy and insert_entry)
                rank_entries_len = len(rank_entries) if rank_entries else 2

                # Boundary management
                if rank == insert_rank:
                    if pivot_item is not None:
                        # A non-None pivot item indicates a node boundary
                        boundary_pos = rank_entries_len - (2 if pivot_item is not None else 1)
                        if boundaries is not None:
                            boundaries.append(boundary_pos)
                        else:
                            rank_data.boundaries = [boundary_pos]
                else:
                    # We are at rank 1 ≠ insert_rank: set boundary at the end of entries
                    boundary_pos = rank_entries_len - 1
                    if boundaries is not None:
                        boundaries.append(boundary_pos)
                    else:
                        rank_data.boundaries = [0, boundary_pos]

                # Always reset pivot item after insertion
                rank_data.pivot_item = None
            else:
                # Handle non-insert/non-leaf ranks
                # Set the variables indicating a node boundary for lower ranks
                rank_data.pivot_item = entry_item
                rank_data.next_child_idx = next_child_idx + 1

            rank -= 1
        max_rank = insert_rank if insert_rank > max_rank else max_rank

    # Build leaf level
    leaf_trees = _build_leaf_level_trees(
        rank_data_map,
        KListClass,
        NodeClass,
        TreeClass,
        l_factor,
    )

    root_tree = _build_internal_levels(
        rank_data_map,
        leaf_trees,
        KListClass,
        TreeClass,
        NodeClass,
        threshold,
        l_factor,
        max_rank
    )
    # logger.debug(f"Bulk created GKPlusTree: {print_pretty(root_tree)}")

    return root_tree
