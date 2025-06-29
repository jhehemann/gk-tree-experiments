"""GKPlusTree base implementation"""

from __future__ import annotations
from typing import Optional, Type, TypeVar, Tuple, List
from itertools import islice, chain

import copy


from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
    RetrievalResult,
)
import logging
import pprint

from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import (
    GPlusTreeBase, GPlusNodeBase, print_pretty, get_dummy
)

from gplus_trees.g_k_plus.base import GKTreeSetDataStructure
from gplus_trees.g_k_plus.utils import (
    calc_ranks,
    calc_rank_for_dim,
    calc_rank_from_group_size,
    calculate_group_size,
)

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

    def find_pivot(self) -> RetrievalResult:
        """
        Returns the pivot entry of a node in the next lower dimension.

        This method is always called from a node of the next lower dimension.
        The pivot entry is the first entry that is either the dummy entry of
        the lower dimension itself or the next larger entry in the current tree.

        Returns:
            RetrievalResult: The pivot entry and the next entry in the tree.
        """

        if self.is_empty():
            return RetrievalResult(None, None)

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

        return RetrievalResult(pivot, current)

    # TODO: Check indifference: This may return an entry with dummy key y, although a subsequent 
    # leaf may have been expanded to a higher dimension with a dummy key x < y.
    # However, y is the first entry yielded when iterating over the tree.
    def get_min(self) -> RetrievalResult:
        """
        Get the minimum entry in the tree. This corresponds to the entry with the dummy item of the maximum dimension in successive first leaf nodes.
        Returns:
            RetrievalResult: The minimum entry and the next entry (if any).
        """
        if self.is_empty():
            return RetrievalResult(None, None)

        first_leaf = next(self.iter_leaf_nodes(), None)
        if first_leaf is None:
            raise ValueError("Tree is empty, cannot retrieve minimum.")

        return first_leaf.set.get_min()

    def get_max(self) -> RetrievalResult:
        """
        Get the maximum entry in the tree.
        Returns:
            RetrievalResult: The maximum entry and the next entry (if any).
        """
        max_leaf = self.get_max_leaf()
        return max_leaf.set.get_max()

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
        leaf_set, _ = leaf_set.insert_entry(Entry(get_dummy(type(self).DIM), None))

        # now insert the real item, also pointing at an empty subtree
        leaf_set, _ = leaf_set.insert_entry(x_entry)

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
        right_set, _ = right_set.insert_entry(x_entry)
        right_node = NodeK(1, right_set, None)
        right_leaf = TreeK(right_node, self.l_factor)

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set, _ = left_set.insert_entry(Entry(get_dummy(type(self).DIM), None))
        left_node = NodeK(1, left_set, None)
        left_leaf = TreeK(left_node, self.l_factor)

        # Link leaves
        left_leaf.node.next = right_leaf
        return left_leaf, right_leaf

    def _insert_empty(self, x_entry: Entry, rank: int) -> GKPlusTreeBase:
        """Build the initial tree structure depending on rank."""
        # Single-level leaf
        inserted = True
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_entry)
            self.node = self.NodeClass(rank, leaf_set, None)
            return self, inserted

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_entry)
        root_set, _ = self.SetClass().insert_entry(Entry(get_dummy(dim=self.DIM), None))
        root_set, _ = root_set.insert_entry(Entry(_create_replica(x_entry.item.key), l_leaf_t))
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        return self, inserted

    def _insert_non_empty(self, x_entry: Entry, rank: int) -> GKPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
        x_item = x_entry.item
        x_key = x_item.key
        if IS_DEBUG:
            logger.debug(
            f"[DIM {self.DIM}] [INSERTING {x_key} with rank {rank}] into tree: {print_pretty(self)}"
        )
        
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
            next_entry = node.set.retrieve(x_key).next_entry
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
        rank: int
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

        if parent is None:

            # create a new root node
            old_node = self.node
            dummy = get_dummy(dim=TreeClass.DIM)
            root_set, _ = self.SetClass().insert_entry(Entry(dummy, None))
            self.node = self.NodeClass(
                rank,
                root_set,
                TreeClass(old_node, self.l_factor)
            )
            return self
        
        # Unfold intermediate node between parent and current
        # Locate the current node’s pivot and place its replica first in the intermediate node.
        pivot = cur.node.set.find_pivot().found_entry
        pivot_replica = _create_replica(pivot.item.key)
        new_set, _ = self.SetClass().insert_entry(Entry(pivot_replica, None))
        new_tree = TreeClass(l_factor=self.l_factor)
        new_tree.node = self.NodeClass(rank, new_set, cur)

        if p_next:
            p_next.left_subtree = new_tree
        else:
            parent.node.right_subtree = new_tree

        return new_tree

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
        x_key = x_entry.item.key
        replica = _create_replica(x_key)
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
            if right_parent is None:
                res = node.set.retrieve(x_key)
                next_entry = res.next_entry
                subtree = next_entry.left_subtree if next_entry else node.right_subtree
                is_gkplus_type = isinstance(node.set, GKPlusTreeBase)
                insert_entry = x_entry if is_leaf else Entry(replica, subtree)
                if IS_DEBUG:
                    logger.debug(
                        f"[DIM {self.DIM}] [INSERTING {x_key}] into node: {print_pretty(node.set)}"
                    )

                if not is_gkplus_type:
                    node.set, inserted = node.set.insert_entry(insert_entry)
                    if not inserted:
                        return self.update(cur, x_entry)
                    if IS_DEBUG:
                        logger.debug(
                            f"[DIM {self.DIM} INSERT {x_key}] Node before conversion check: {print_pretty(node.set)}"
                    )
                    node.set = check_and_convert_set(node.set) # only KLists can be extended
                    if IS_DEBUG:
                        logger.debug(
                            f"[DIM {self.DIM} INSERT {x_key}] Node after conversion check: {print_pretty(node.set)}"
                        )
                else:
                    new_rank = calc_rank_for_dim(x_key, capacity, dim=self.DIM + 1)
                    node.set, inserted = node.set.insert_entry(insert_entry, rank=new_rank)
                    if not inserted:
                        return self.update(cur, x_entry)
                    if IS_DEBUG:
                        logger.debug(
                            f"[DIM {self.DIM} INSERT {x_key}] Node after insertion (no conversion check for tree inserts): {print_pretty(node.set)}"
                    )

                # Fastest path for leaf nodes - direct return
                if is_leaf:                    
                    self._invalidate_tree_size()
                    if IS_DEBUG:
                        logger.debug(
                            f"[DIM {self.DIM} INSERTED {x_key}] into tree (now at leaf -> return): {print_pretty(self)}"
                        )
                    return self, True

                # Setup for next iteration with optimized assignments
                right_parent = left_parent = cur
                right_entry = next_entry
                left_x_entry = insert_entry
                # left_x_entry = x_entry
                cur = subtree
                continue
            
            # Complex path: Node splitting required
            # Cache retrieve result to avoid redundant method calls
            res = node.set.retrieve(x_key)
            next_entry = res.next_entry

            # Perform split operation and immediately cache converted results
            left_split, _, right_split = node.set.split_inplace(x_key)
            if IS_DEBUG:
                logger.debug(
                    f"[DIM {self.DIM}] [INSERT SUBS {x_key}] Left split before conversion: {print_pretty(left_split)}"
                )
            left_split = check_and_convert_set(left_split)
            if IS_DEBUG:
                logger.debug(
                    f"[DIM {self.DIM}] [INSERT SUBS {x_key}] Left split after conversion: {print_pretty(left_split)}"
                )
            if IS_DEBUG:
                logger.debug(
                    f"[DIM {self.DIM}] [INSERT SUBS {x_key}] Right split before insertion (if any): {print_pretty(right_split)}"
                )
            # right_split = check_and_convert_set(right_split)
            
            # Cache item counts early to avoid repeated method calls in conditionals
            right_item_count = right_split.item_count() # invalidate size after insertion
            left_item_count = left_split.item_count()
            
            # Handle right side creation - inline optimization for performance
            new_tree = None
            if right_item_count > 0 or is_leaf:
                insert_entry = x_entry if is_leaf else Entry(replica, None)
                if isinstance(right_split, GKPlusTreeBase):
                    new_rank = calc_rank_for_dim(x_key, capacity, dim=self.DIM + 1)
                    right_split, _ = right_split.insert_entry(insert_entry, rank=new_rank)
                    right_split._invalidate_tree_size()
                    
                else:
                    right_split, _ = right_split.insert_entry(insert_entry)

                if IS_DEBUG:
                    logger.debug(
                        f"[DIM {self.DIM}] [INSERT SUBS {x_key}] Right split after inserting before conversion: {print_pretty(right_split)}"
                    )

                # Create new tree node
                right_split = check_and_convert_set(right_split)
                if IS_DEBUG:
                    logger.debug(
                        f"[DIM {self.DIM}] [INSERT SUBS {x_key}] Right split after conversion: {print_pretty(right_split)}"
                    )
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
                if new_tree and cur:
                    new_tree.node.next = cur.node.next
                    cur.node.next = new_tree

                    # Right subtrees of left splits are not reset in higher dimensional leaf nodes.
                    # Do it here to maintain search tree structure across dimensions.
                    cur.node.right_subtree = None  # No right subtree at leaf level
                self._invalidate_tree_size()
                if IS_DEBUG:
                    logger.debug(
                        f"[DIM {self.DIM}] [INSERTED {x_key} into tree: {print_pretty(self)}]"
                    )
                # logger.debug(
                #         f"[DIM {self.DIM}] [INSERTED {x_key} into tree structure: {self.print_structure()}]"
                #     )
                return self, True  # Early return when leaf is processed

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
            return self, None, TreeClass(l_factor=cached_l_factor)

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
            is_leaf = node.rank == 1

            # Split node at key - cache results immediately
            # logger.debug(f"[DIM {self.DIM}] [SPLIT (INIT) {key}] Node set BEFORE split: {print_pretty(node.set)}")
            left_split, key_subtree, right_split = node.set.split_inplace(key)
            if IS_DEBUG:
                logger.debug(
                    f"[DIM {self.DIM}] [SPLIT {key}] Left before conversion: {print_pretty(left_split)}"
                )
            left_split = check_and_convert_set(left_split)
            if IS_DEBUG:
                logger.debug(
                    f"[DIM {self.DIM}] [SPLIT {key}] Left after conversion: {print_pretty(left_split)}"
                )

            # Cache item counts and next entry
            l_count = left_split.item_count()
            r_count = right_split.item_count()
            next_entry = right_split.retrieve(key).next_entry

            # --- Handle right side of the split ---
            # Determine if we need a new tree for the right split
            if r_count > 0:  # incl. dummy items
                # Cache type check to avoid repeated isinstance calls
                is_gkplus_type = isinstance(right_split, GKPlusTreeBase)
                
                if is_gkplus_type:
                    # Calculate the new rank for the item in the next dimension - use cached values
                    new_rank = calc_rank_for_dim(dummy_key, KListNodeCapacity, dim=tree_dim_plus_one)
                    right_split, _ = right_split.insert_entry(Entry(dummy, None), rank=new_rank)
                    right_split._invalidate_tree_size()
                    # TODO: Check why we need to invalidate size and why it is not done in insert_entry. Check it also for insert_entry()
                else:
                    right_split, _ = right_split.insert_entry(Entry(dummy, None))
                
                right_split = check_and_convert_set(right_split)

                # Cache node references for performance
                node_rank = node.rank
                node_right_subtree = node.right_subtree
                right_node = NodeClass(node_rank, right_split, node_right_subtree)

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
                        new_rank = calc_rank_for_dim(dummy_key, KListNodeCapacity, dim=tree_dim_plus_one)
                        right_split, _ = right_split.insert_entry(Entry(dummy, None), rank=new_rank)
                        right_split._invalidate_tree_size()
                    else:
                        right_split, _ = right_split.insert_entry(Entry(dummy, None))

                    right_split = check_and_convert_set(right_split)
                    right_node = NodeClass(1, right_split, None)
                    new_tree = TreeClass(l_factor=cached_l_factor)
                    new_tree.node = right_node

                    # Update parent reference
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
                    else:
                        right_parent.node.right_subtree = new_tree

                    # Link leaf nodes - use cached node reference
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
                    if key_subtree:
                        # Highest node containing split key found
                        # All entries in its left subtree are less than key and
                        # are part of the left return tree
                        new_subtree = key_subtree
                        seen_key_subtree = key_subtree
                    elif next_entry:
                        new_subtree = next_entry.left_subtree
                    else:
                        new_subtree = node.right_subtree # Should not happen

                    if left_parent and not key_node_found:
                        left_parent.node.right_subtree = new_subtree
                        
                    next_left_parent = left_parent

            left_parent = next_left_parent

            # Main return logic
            if is_leaf:
                if l_last_leaf: # unlink leaf nodes
                    l_last_leaf.next = None
                return_subtree = key_subtree
                return self, return_subtree, right_return

            if key_subtree:
                # Do not update left parent reference from this point on
                key_node_found = True

            # Continue to next iteration with updated current node
            cur = next_cur

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
                logger.debug(f"[EXPAND] KList {print_pretty(klist)} has {klist.item_count()} items, which is > {threshold}, converting to GKPlusTree")
            # Convert to GKPlusTree with increased dimension
            new_dim = type(self).DIM + 1

            target_dim = new_dim
            return _klist_to_tree(klist, k, target_dim, self.l_factor)

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
        expected_klist_size = tree_item_count - 1  # Exclude the dummy item
        real_item_count = tree.real_item_count()
        
        # Fast path: no dummy items in the tree
        if expected_klist_size == real_item_count:
            if IS_DEBUG:
                logger.debug(f"Item count {tree_item_count} - 1 is equal to real item count {real_item_count} in tree {print_pretty(tree)}")
            if expected_klist_size <= threshold:
                # Collapse into a KList
                if IS_DEBUG:
                    logger.debug(f"[COLLAPSE] KList will have {expected_klist_size} items, which is <= {threshold}, collapsing to KList")
                return _tree_to_klist(tree)
            if IS_DEBUG:
                logger.debug(f"[RETURN] KList would have {expected_klist_size} items, which is > {threshold}, keeping as GKPlusTree")
            return tree
        
        # The dummy item from the tree and from all expanded leafs are removed when collapsed
        expanded_leafs_count = tree.expanded_count()
        expected_klist_size -= expanded_leafs_count
        if IS_DEBUG:
            logger.debug("")
            logger.debug(f"[CHECK] Tree {print_pretty(tree)}")
            logger.debug(f"Tree has {tree_item_count} items, "
                        f"expanded leafs {expanded_leafs_count}, "
                        f"expected KList size {expected_klist_size} (item count - expanded leafs - 1), "
                        f"threshold {threshold}")
        if expected_klist_size <= threshold:
            # Collapse into a KList
            if IS_DEBUG:
                logger.debug(f"[COLLAPSE] KList will have {expected_klist_size} items, which is <= {threshold}, collapsing to KList")
            return _tree_to_klist(tree)

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
        # logger.debug(f"[TREE TO KLIST] Processing entry {entry.item.key} with dummy {tree_dummy.key}")
        if entry.item.key > tree_dummy.key:
            # logger.debug(f"[TREE TO KLIST] Inserting entry {entry.item.key} as it is larger than the tree's dummy key {tree_dummy.key}")
            klist, _ = klist.insert_entry(entry)
            # logger.debug(f"[TREE TO KLIST] KList after insertion: {print_pretty(klist)}")
    return klist


def _klist_to_tree(klist: KListBase, K: int, DIM: int, l_factor: float = 1.0) -> GKPlusTreeBase:
    """
    Convert a KList to a GKPlusTree by inserting each (item, rank) pair.
    Uses the factory pattern to create a tree with the specified capacity K.
    
    Args:
        klist: The KList to convert
        K: The capacity parameter
        DIM: The dimension for the new tree
        l_factor: The threshold factor for conversion
        
    Returns:
        A new GKPlusTree containing all items from the KList
    """
    # Use cached import for performance
    create_gkplus_tree = _get_create_gkplus_tree()
    
    # Raise an error if klist is not a KListBase instance
    if not isinstance(klist, KListBase):
        raise TypeError("klist must be an instance of KListBase")
    
    if klist.is_empty():
        return create_gkplus_tree(K, DIM, l_factor)

    entries = [entry for entry in klist]
    return bulk_create_gkplus_tree(entries, DIM, l_factor, type(klist))


def bulk_create_gkplus_tree(
    entries: list[Entry],
    DIM: int,
    l_factor: float,
    KListClass: type[KListBase],
) -> GKPlusTreeBase:
    """
    Create a new GKPlusTree with optimized bottom-up bulk creation.
    
    This implementation builds the tree from leaves to root, which is much more
    efficient than the recursive top-down approach because:
    1. Single pass over data to calculate all ranks
    2. Direct leaf creation without recursion
    3. Iterative bottom-up construction
    4. Better memory locality and cache efficiency
    
    Args:
        klist: The KList to convert
        DIM: The dimension of the tree
        l_factor: The threshold factor for conversion
        
    Returns:
        A new GKPlusTreeBase instance
    """
    k = KListClass.KListNodeClass.CAPACITY

    if not entries:
        tree = _get_create_gkplus_tree()(k, DIM, l_factor)
        return tree

    # Use optimized bottom-up construction - Not working yet
    return _bulk_create_bottom_up(entries, k, DIM, l_factor, KListClass)


def _bulk_create_klist(entries: list[Entry], KListClass: type[KListBase]) -> KListBase:
    """
    Create a KList from a list of entries.
    
    Args:
        entries: A list of Entry objects
        KListClass: The KList class to use
        
    Returns:
        A new KListBase instance containing the entries
    """
    klist = KListClass()
    if IS_DEBUG:
        logger.debug(f"[BULK CREATE] Creating KList with {[entry.item.key for entry, _ in entries]} entries and their left subtrees:")
        for entry, _ in entries:
            if entry.left_subtree:
                logger.debug(f"  - {entry.item.key} -> left subtree: {print_pretty(entry.left_subtree)}")
            else:
                logger.debug(f"  - {entry.item.key} has no left subtree")

    insert_entry_fn = klist.insert_entry  # Cache method reference
    for entry, _ in entries:
        klist, _ = insert_entry_fn(entry)
    return klist


def _build_leaf_level_trees(
    entries: list[tuple[Entry, None]],
    boundaries_map: list[int],
    dummy: Entry,
    KListClass: type[KListBase],
    NodeClass: type[GKPlusNodeBase],
    TreeClass: type[GKPlusTreeBase],
    l_factor: float,
    # has_dummy: bool,
) -> list[GKPlusTreeBase]:
    """Build leaf level trees from entries."""
    threshold = KListClass.KListNodeClass.CAPACITY * l_factor
    leaf_trees = []
    prev_node = None
    if IS_DEBUG:
        logger.debug(f"[BULK CREATE] Starting leaf level creation with {len(entries)} entries and boundaries: {boundaries_map}")
    for i in range(len(boundaries_map)):
        start_idx = boundaries_map[i]
        end_idx = boundaries_map[i + 1] if i + 1 < len(boundaries_map) else len(entries)
        node_entries = entries[start_idx:end_idx]
        if IS_DEBUG:
            logger.debug(f"[BULK CREATE] Creating leaf node {i} with entries: %s", 
                    [entry.item.key for entry, _ in node_entries]
                )
        # if prev_node is None:
        #     # has_dummy must be True
        #     node_entries = [(dummy, None)] + node_entries

        # Create node set - always use KList for simplicity in bulk creation
        # This ensures we don't get into recursive loops and maintains correctness
        if len(node_entries) <= threshold:
            node_set = _bulk_create_klist(node_entries, KListClass)
        else:
            raw_entries = [entry for entry, _ in node_entries]
            node_set = bulk_create_gkplus_tree(raw_entries, TreeClass.DIM + 1, l_factor, KListClass)
        leaf_node = NodeClass(1, node_set, None)
        leaf_tree = TreeClass(l_factor=l_factor)
        leaf_tree.node = leaf_node
        if prev_node is not None:
            prev_node.next = leaf_tree
        prev_node = leaf_tree.node 
        leaf_trees.append(leaf_tree)

    return leaf_trees

def _build_internal_levels(
    rank_entries_map: dict[int, list[Entry]],
    boundaries_map: dict[int, list[int]],
    leaf_trees: list[GKPlusTreeBase],
    TreeClass: type[GKPlusTreeBase],
    NodeClass: type[GKPlusNodeBase],
    KListClass: type[KListBase],
    threshold: int,
    l_factor: float,
    max_rank: int,
) -> GKPlusTreeBase:
    """
    Build internal levels of the GKPlusTree from leaf level trees.
    
    Args:
        rank_entries_map: Map of ranks to entries
        boundaries_map: Map of rank boundaries
        leaf_trees: List of leaf trees created from entries
        dummy: The dummy entry to use
        TreeClass: The tree class to use for creating nodes
        NodeClass: The node class to use for creating nodes
        KListClass: The KList class to use for creating nodes
        threshold: The threshold for KList vs GKPlusTree conversion
        l_factor: The threshold factor for conversion
        DIM: The dimension of the tree
        
    Returns:
        A new GKPlusTreeBase instance representing the root of the tree
    """
    rank_trees_map: dict[int, GKPlusTreeBase] = {}
    rank_trees_map[1] = leaf_trees  # Rank 1 trees are the leaf trees
    if IS_DEBUG:
        logger.debug(f"[BULK CREATE] Starting internal level creation from rank {2} to {max_rank}")

    rank = 2
    sub_trees = leaf_trees
    # collapsed_trees = []

    while rank <= max_rank:
        subtrees_next = []
        entries = rank_entries_map.get(rank, [])
        boundaries = boundaries_map.get(rank, [])
        if IS_DEBUG:
            logger.debug(f"[BULK CREATE] Processing rank {rank} with entries: %s",
                    [entry.item.key for entry, _ in entries] if entries else []
                )
        if not entries:
            subtrees_next = sub_trees
            rank += 1
            continue

        trees = []
        last_child_idx = -1
        for i in range(len(boundaries)):
            
            # if i == 0:
            #     last_child_idx -= 1
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(entries)
            node_entries = entries[start_idx:end_idx]
            
            if IS_DEBUG:
                logger.debug(f"[BULK CREATE] Creating rank {rank} node {i} with entries: %s", 
                        [entry.item.key for entry, _ in node_entries]
                    )
            
            for entry, child_idx in islice(node_entries, 1, None):
                entry.left_subtree = sub_trees[child_idx] if child_idx and child_idx < len(sub_trees) else sub_trees[last_child_idx + 1]
                child_idx = last_child_idx + 1 if child_idx is None else child_idx
                if child_idx is not None:
                    if IS_DEBUG:
                        logger.debug(f"[BULK CREATE] Entry {entry.item.key} changing last child index from {last_child_idx} to {child_idx}")
                    subtrees_next.extend(sub_trees[last_child_idx + 1:child_idx])
                    last_child_idx = child_idx
                else:
                    subtrees_next.append(sub_trees[last_child_idx + 1])
                    if IS_DEBUG:
                        logger.debug(f"[BULK CREATE] Entry {entry.item.key} has no child, incrementing last child index from {last_child_idx} to {last_child_idx + 1}")
                    last_child_idx += 1

            # Create node set - always use KList for implementation in bulk creation
            # node_set = _bulk_create_klist(node_entries, KListClass)
            if len(node_entries) <= threshold:
                node_set = _bulk_create_klist(node_entries, KListClass)
            else:
                raw_entries = [entry for entry, _ in node_entries]
                node_set = bulk_create_gkplus_tree(raw_entries, TreeClass.DIM + 1, l_factor, KListClass)

            if IS_DEBUG:
                logger.debug(f"[BULK CREATE] Created node set for rank {rank} node {i}: {print_pretty(node_set)}")
                logger.debug(f"[BULK CREATE] Right subtree: {print_pretty(sub_trees[last_child_idx + 1]) if last_child_idx + 1 < len(sub_trees) else 'None'}")
            tree_node = NodeClass(rank, node_set, sub_trees[last_child_idx + 1])
            last_child_idx += 1
            tree = TreeClass(l_factor=l_factor)
            tree.node = tree_node
            trees.append(tree)
            subtrees_next.append(tree)
        
        subtrees_next.extend(sub_trees[last_child_idx + 1:])
        if IS_DEBUG:
            logger.debug(f"[BULK CREATE] Subtrees for next iteration:")
            for tree in subtrees_next:
                logger.debug(print_pretty(tree))
        sub_trees = subtrees_next
        rank_trees_map[rank] = trees
        rank += 1

    # for rank in rank_trees_map:
    #     for tree in rank_trees_map[rank]:
    #         if IS_DEBUG:
    #             logger.debug(f"[BULK CREATE] Rank {rank} tree: {print_pretty(tree)}")

    tree = rank_trees_map[max_rank][0]  # Get the root tree from the last rank
    if IS_DEBUG:
        logger.debug(f"[BULK CREATE] Finished internal level creation, root tree: {tree}")
    return tree


def _bulk_create_bottom_up(
    entries: list[Entry],
    k: int,
    DIM: int,
    l_factor: float,
    KListClass: type[KListBase],
) -> GKPlusTreeBase:
    """
    Bottom-up bulk creation of GK+-tree.
    
    Key insights:
    1. All entries exist in leaf nodes
    2. Rank determines node boundaries: higher rank = start of new node
    3. Higher rank entries are replicated upward to their rank level
    4. Threshold determines KList vs GKPlusTree for node implementation
    5. Entries are already sorted from KList iteration
    
    Args:
        klist: The sorted KList to convert
        k: The capacity parameter for the tree
        DIM: The dimension of the tree
        l_factor: The threshold factor for conversion
    """
    create_gkplus_tree_fn = _get_create_gkplus_tree()
    sample_tree = create_gkplus_tree_fn(k, DIM, l_factor)
    
    # Get required classes and functions
    NodeClass = sample_tree.NodeClass
    TreeClass = type(sample_tree)
    create_replica_fn = _create_replica
    group_size = calculate_group_size(k)
    
    # Cache frequently used values
    threshold = int(k * l_factor)
    dummy = Entry(get_dummy(DIM), None)
    rank_entries_map: dict[int, list[Entry]] = {}
    rank_entries_map[1] = [(dummy, None)]  # at least one entry at rank 1
    boundaries_map: dict[int, list[int]] = {}
    add_boundary_map: dict[int, bool] = {}
    prev_pivot_map: dict[int, int] = {}
    rank_node_slots_map: dict[int, Optional[list[int]]] = {}
    rank_node_slots_map[1] = [[(dummy, None)]]  # at least one entry at rank 1

    max_rank = 1
    for entry in entries:
        insert_rank = calc_rank_from_group_size(entry.item.key, group_size, DIM)
        entry_key = entry.item.key
        rank = insert_rank
        
        while rank > 0:
            # Batch dictionary lookups for better cache efficiency
            ranks = rank_entries_map.get(rank)
            boundaries = boundaries_map.get(rank)
            add_boundary = add_boundary_map.get(rank, True)
            node_slots = rank_node_slots_map.get(rank, None)
            prev_pivot = prev_pivot_map.get(rank, None)
            
            # Only process ranks that need entry insertion
            if rank == insert_rank or rank == 1:
                # Create fresh replica for each rank (except leaf which uses original entry)
                insert_entry = entry if rank == 1 else Entry(create_replica_fn(entry_key), None)
                
                # Calculate child index efficiently
                child_idx = None
                if rank > 1:
                    child_slots = rank_node_slots_map.get(rank - 1)
                    if child_slots:
                        child_idx = len(child_slots) - 1
                    elif child_slots is None and rank == 2:
                        child_idx = 0
                
                # Entry insertion
                if ranks is not None:
                    if IS_DEBUG:
                        logger.debug(f"[BULK CREATE] Inserting entry {insert_entry.item.key} with child index {child_idx} at rank {rank} into {[e.item.key for e, _ in ranks]}, node slots: {[[e.item.key for e, _ in slot] for slot in node_slots]}. Previous pivot: {prev_pivot}")
                    if prev_pivot is not None:
                        pivot = Entry(create_replica_fn(prev_pivot), None)
                        ranks.append((pivot, False))
                        node_slots[-1].append((pivot, False))
                    ranks.append((insert_entry, child_idx))
                    
                    if rank != insert_rank:
                        node_slots.append([(insert_entry, child_idx)])
                    else:    
                        node_slots[-1].append((insert_entry, child_idx))
                    if IS_DEBUG:
                        logger.debug(f"[BULK CREATE] Updated rank {rank} entries: %s",
                                [e.item.key for e, _ in ranks]
                            )
                else:
                    if IS_DEBUG:
                        logger.debug(f"[BULK CREATE] Creating new rank {rank} with entry {insert_entry.item.key} and child index {child_idx}. Previous pivot: {prev_pivot}")
                    # Create new rank entry list
                    new_entries = [(insert_entry, child_idx)]
                    new_entries_slots = [(insert_entry, child_idx)]
                    if insert_rank >= max_rank:
                        new_entries.insert(0, (dummy, False))
                        new_entries_slots.insert(0, (dummy, False))
                        prev_pivot = dummy.item.key
                    else:
                        new_entries.insert(0, (Entry(create_replica_fn(prev_pivot), None), False))
                        new_entries_slots.insert(0, (Entry(create_replica_fn(prev_pivot), False), None))
                    rank_entries_map[rank] = new_entries
                    if node_slots is not None:
                        if IS_DEBUG:
                            logger.debug(f"Node slots for rank {rank} are not None, appending new entries: %s",
                                    [e.item.key for e, _ in new_entries_slots])
                        node_slots[-1].extend(new_entries_slots)
                        if IS_DEBUG:
                            logger.debug(f"[BULK CREATE] Updated node slots for rank {rank}: %s",
                                    [[e.item.key for e, _ in slot] for slot in node_slots]
                                )
                    else:
                        rank_node_slots_map[rank] = [new_entries_slots]

                if IS_DEBUG:
                    logger.debug(f"[BULK CREATE] Entries for rank {rank}: %s",
                            [e.item.key for e, _ in rank_entries_map[rank]]
                        )
                    logger.debug(f"[BULK CREATE] Rank node slots for rank {rank}: %s",
                            [[e.item.key for e, _ in slot] for slot in rank_node_slots_map[rank]]
                        )
                
                # Boundary management
                if rank == insert_rank:
                    # Handle special case for rank 1 initialization
                    if rank == 1 and not add_boundary_map:
                        boundaries_map[rank] = [0]
                        add_boundary_map[rank] = False
                    elif add_boundary:
                        # Calculate boundary position once
                        if prev_pivot is not None:
                            boundary_pos = (len(ranks) - 2) if boundaries is not None else 0
                        else:
                            boundary_pos = (len(ranks) - 1) if boundaries is not None else 0

                        if boundaries is not None:
                            boundaries.append(boundary_pos)
                        else:
                            boundaries_map[rank] = [boundary_pos]
                        add_boundary_map[rank] = False
                        
                elif boundaries is not None:
                    # Non-insert rank boundary update
                    boundary_pos = len(ranks) - 1
                    boundaries.append(boundary_pos)
                    add_boundary_map[rank] = False
                else:
                    # Initialize boundaries for new rank
                    rank_size = len(rank_entries_map[rank])
                    boundaries_map[rank] = [0, rank_size - 1]
                    add_boundary_map[rank] = False

                if IS_DEBUG:
                    logger.debug(f"[BULK CREATE] Setting pivot for rank {rank} to None at {entry_key}. Previous pivot: {prev_pivot}")
                prev_pivot_map[rank] = None
                rank -= 1
                continue

            # Handle non-insert/non-leaf ranks: set boundary flag for lower ranks
            if not add_boundary:
                add_boundary_map[rank] = True
            
            
            if node_slots is not None:
                node_slots.append([])
            else:
                rank_node_slots_map[rank] = [[], []]
            if IS_DEBUG:
                logger.debug(f"[BULK CREATE] Changing pivot from {prev_pivot} to {entry_key} for rank {rank} at {entry_key}.")
            prev_pivot_map[rank] = entry_key
            if IS_DEBUG:
                logger.debug(f"[BULK CREATE] Adding boundary at rank {rank} at {entry_key}, setting it as pivot for next entry.")
            rank -= 1
        max_rank = insert_rank if insert_rank > max_rank else max_rank

    # Comment when testing
    if IS_DEBUG:
        logger.debug(
            "[BULK CREATE] Created rank entries map:\n%s",
            pprint.pformat(rank_entries_map)
        )
        logger.debug(
            "[BULK CREATE] Created rank node slots map:\n%s",
            pprint.pformat(rank_node_slots_map)
        )
        logger.debug(
            "[BULK CREATE] Created boundaries map:\n%s",
            pprint.pformat(boundaries_map)
        )
    # for entry_sample in entries:
    #     rank_sample = calc_rank_from_group_size(entry_sample.item.key, group_size, DIM)
    #     sample_tree, _ = sample_tree.insert_entry(entry_sample, rank_sample)
    if IS_DEBUG:
        logger.debug(
        "[BULK CREATE] Sample tree after inserting entries:\n%s",
        print_pretty(sample_tree)
        )

    leaf_trees = _build_leaf_level_trees(
        rank_entries_map[1],
        boundaries_map[1],
        dummy,
        KListClass,
        NodeClass,
        TreeClass,
        l_factor,
    )

    root_tree = _build_internal_levels(
        rank_entries_map, boundaries_map, leaf_trees,
        TreeClass, NodeClass, KListClass, threshold, l_factor, max_rank
    )

    if IS_DEBUG:
        logger.debug(f"[BULK CREATE] Root tree: {print_pretty(root_tree)}")

    # exit()
    return root_tree
