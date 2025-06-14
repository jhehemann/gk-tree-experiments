"""GKPlusTree base implementation"""

from __future__ import annotations
from typing import Optional, Type, TypeVar, Dict, Tuple, Any
import logging
from dataclasses import dataclass
import collections

from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
    RetrievalResult,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import (
    GPlusTreeBase, GPlusNodeBase, Stats, print_pretty, get_dummy
)

from gplus_trees.g_k_plus.base import GKTreeSetDataStructure

from gplus_trees.base import logger

logger.name = "GKPlusTree"

# # Configure logging
# logger = logging.getLogger("GKPlusTree")
# # Clear all handlers to ensure we don't add duplicates
# if logger.hasHandlers():
#     logger.handlers.clear()
# # Add a single handler with formatting
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)  # Set to DEBUG for detailed output
# # Prevent propagation to the root logger to avoid duplicate logs
# logger.propagate = False



t = TypeVar('t', bound='GKPlusTreeBase')

DEFAULT_DIMENSION = 1  # Default dimension for GKPlusTree
DEFAULT_L_FACTOR = 1.0  # Default threshold factor for KList to GKPlusTree conversion

class GKPlusNodeBase(GPlusNodeBase):
    """
    Base class for GK+-tree nodes.
    Extends GPlusNodeBase with size support.
    """
    __slots__ = GPlusNodeBase.__slots__ + ("size",)  # Only add new slots beyond what parent has

    # These will be injected by the factory
    SetClass: Type[AbstractSetDataStructure]
    TreeClass: Type[GKPlusTreeBase]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size = None  # Will be computed on-demand

    def get_node_item_count(self) -> bytes:
        """
        Get the number of items in this node (incl. dummy items).
        
        Returns:
            bytes: The hash value for this node and its subtrees.
        """
        return self.set.item_count() if self.set else 0  
    
    def _update_left_subtree(self, key: int, new_left: GKPlusTreeBase) -> GKPlusNodeBase:
        """
        Update the left subtree of the current node.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Updating {key}'s left subtree "
                  f"Set: {print_pretty(self.set)} "
                  f"New left: {print_pretty(new_left)}")
        entry = self.set.retrieve(key).found_entry

        if entry is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Found key {key}, updating left subtree.")
            entry.left_subtree = new_left
        return self

class GKPlusTreeBase(GPlusTreeBase, GKTreeSetDataStructure):
    """
    A GK+-tree is an extension of G+-tree with dimension support.
    It can automatically transform between KList and GKPlusTree based on item count.
    
    Attributes:
        node (Optional[GKPlusNodeBase]): The GK+-node that the tree contains.
        DIM (int): The dimension of the GK+-tree (class attribute).
        l_factor (float): The threshold factor for conversion between KList and GKPlusTree.
    """
    __slots__ = GPlusTreeBase.__slots__ + ("l_factor", "item_cnt",)  # Only add new slots beyond what parent has
    
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
            l_factor: Threshold factor for KList-to-GKPlusTree conversion (default: 0.75)
        """
        # Get node and dimension from parent class
        super().__init__(node)
        self.l_factor = l_factor
        self.item_cnt = None  # Initialize item count
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Creating GKPlusTree "
                  f"(dim={self.DIM}, l_factor={self.l_factor})")

    def __str__(self):
        return "Empty GKPlusTree" if self.is_empty() else f"GKPlusTree(dim={self.__class__.DIM}, node={self.node})"

    __repr__ = __str__
    
    def item_count(self) -> int:
        if self.is_empty():
            self.item_cnt = 0
            return self.item_cnt
        return self.get_tree_item_count()
    
    def get_tree_item_count(self) -> int:
        """ Get the number of items in the tree, including dummy items."""
        if self.node.rank == 1:  # indicates a leaf in current dim
            count = 0
            self.item_cnt = self.node.set.item_count()
            return self.item_cnt
        else:
            count = 0
            for entry in self.node.set:
                count += entry.left_subtree.get_tree_item_count() if entry.left_subtree is not None else 0

            count += self.node.right_subtree.get_tree_item_count() if self.node.right_subtree is not None else 0
            self.item_cnt = count
            return self.item_cnt
    
    def _invalidate_tree_size(self) -> None:
        """
        Invalidate the tree size. This is a placeholder for future implementation.
        """
        self.item_cnt = None

    def get_max_dim(self) -> int:
        """
        Get the maximum dimension of the GK+-tree.

        Returns:
            int: The maximum dimension of the tree.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Getting max dim for GKPlusTree (dim={self.DIM}).")
        if self.is_empty():
            return self.DIM
        
        max_dim = self.DIM
        if isinstance(self.node.set, GKPlusTreeBase):
            # If the set is a GKPlusTreeBase, we can directly use its dimension
            max_dim = max(max_dim, self.node.set.get_max_dim())
        elif isinstance(self.node.set, KListBase):
            # Otherwise, we need to check the entries in the KList
            for entry in self.node.set:
                if entry.left_subtree is not None:
                    max_dim = max(max_dim, entry.left_subtree.get_max_dim())

        return max_dim

    def get_expanded_leaf_count(self) -> int:
        """
        Count the number of leaf nodes whose set is recursively instantiated by another GKPlusTree, having an extra dummy item for that dimension.
        
        Returns:
            int: The number of leaf nodes in the tree.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Counting expanded leaf nodes in GKPlusTree: {print_pretty(self)}")
        if self.is_empty():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Tree is empty, returning expanded count 0.")
            return 0
        
        count = 0
        for leaf in self.iter_leaf_nodes():
            if isinstance(leaf.set, GKPlusTreeBase):
                # For GKPlusTreeBase, directly count its expanded leaves
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Leaf set is GKPlusTreeBase, count expanded leaves.")
                count += 1  # Count the leaf itself
                count += leaf.set.get_expanded_leaf_count()
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
        Returns the pivot entry for a node of the next higher dimension to be unfolded.
        
        This method is always called from a node of the next lower dimension. The pivot entry is the first entry that is either the dummy entry of the lower dimension itself or the next larger entry in the current tree.
        
        Returns:
            RetrievalResult: The pivot entry and the next entry in the tree (if any).
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Getting pivot entry from GKPlusTree {print_pretty(self)}")

        if self.is_empty():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Tree is empty, returning None for pivot.")
            
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Found pivot entry: {entry.item.key}")
                    pivot = entry
        
        if pivot is None:
            raise ValueError(f"No pivot entry in tree {print_pretty(self)}")
        
        return RetrievalResult(pivot, current)

    def get_min(self) -> RetrievalResult:
        """
        Get the minimum entry in the tree.
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
        Get the maximum node in the tree.
        Returns:
            GKPlusNodeBase: The maximum node in the tree.
        """
        if self.is_empty():
            return None
        
        node = self.node
        while node.rank > 1 :
            node = node.right_subtree.node
        
        return node

    def _make_leaf_klist(self, x_item: Item, x_left: Optional[GPlusTreeBase] = None) -> AbstractSetDataStructure:
        """Builds a KList for a single leaf node containing the dummy and x_item."""
        TreeClass = type(self)
        SetClass = self.SetClass
        
        # start with a fresh empty set of entries
        leaf_set = SetClass()
        
        # insert the dummy entry, pointing at an empty subtree
        leaf_set = leaf_set.insert(get_dummy(type(self).DIM), None)

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
        right_leaf = TreeK(right_node, self.l_factor)

        # Build left leaf with dummy entry
        left_set = SetK()
        left_set = left_set.insert(get_dummy(type(self).DIM), None)
        left_node = NodeK(1, left_set, None)
        left_leaf = TreeK(left_node, self.l_factor)

        # Link leaves
        left_leaf.node.next = right_leaf
        return left_leaf, right_leaf

    def _insert_empty(self, x_item: Item, rank: int, x_left: Optional[GKPlusTreeBase] = None) -> GKPlusTreeBase:
        """Build the initial tree structure depending on rank."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DIM {self.DIM} insert into empty tree "
                  f"(key={x_item.key}, rank={rank})")

        # Single-level leaf
        inserted = True
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_item, x_left)
            self.node = self.NodeClass(rank, leaf_set, None)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tree after empty insert (rank=1):{print_pretty(self)}")
            return self, inserted

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_item, x_left)
        root_set = self.SetClass().insert(get_dummy(dim=self.DIM), None)
        root_set = root_set.insert(_create_replica(x_item.key), l_leaf_t)
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tree after empty insert (rank={rank}):{print_pretty(self)}")
        return self, inserted

    def _insert_non_empty(self, x_item: Item, rank: int, x_left: Optional[GPlusTreeBase] = None) -> GKPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DIM {self.DIM} insert (key={x_item.key}, rank={rank}) "
                  f"into tree:\n{print_pretty(self)}")

        cur = self
        parent = None
        p_next_entry = None

        # path cache
        path_cache = []

        # Loop until we find where to insert
        while True:
            node = cur.node
            path_cache.append(node)
            node_rank = node.rank  # Cache attribute access
            
            # Case 1: Found node with matching rank - ready to insert
            if node_rank == rank:
                # Only retrieve once
                res = node.set.retrieve(x_item.key)
                
                # Fast path: update existing item
                if res.found_entry:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"Item with key {x_item.key} "
                                       "already exists in the tree.")
                    if x_left is not None:
                        raise ValueError(f"Item with key {x_item.key} already "
                                         "exists in the tree. Cannot be "
                                         "inserted with a subtree again.")
                    item = res.found_entry.item
                    # Direct update for leaf nodes (common case)
                    if rank == 1:
                        item.value = x_item.value
                        return self, False
                    return self._update_existing_item(cur, x_item)
                
                # Item will be inserted, add 1 to each node's size so far
                for node in path_cache:
                    if node.size is not None:
                        node.size += 1
                return self._insert_new_item(
                    cur,
                    x_item,
                    res.next_entry,
                    x_left
                )

            # Case 2: Current rank too small - handle rank mismatch
            if node_rank < rank:
                cur = self._handle_rank_mismatch(
                    cur,
                    parent,
                    p_next_entry,
                    rank
                )
                continue

            # Case 3: Descend to next level (current rank > rank)
            res = node.set.retrieve(x_item.key)
            parent = cur
            next_entry = res.next_entry
            if next_entry:
                p_next_entry = next_entry
                cur = next_entry.left_subtree
            else:
                p_next_entry = None
                cur = node.right_subtree

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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Rank mismatch: current rank {cur.node.rank}, "
                      f"target rank {rank}, creating new root node.")
            
            # create a new root node
            old_node = self.node
            dummy = get_dummy(dim=TreeClass.DIM)
            root_set = self.SetClass().insert(dummy, None)
            self.node = self.NodeClass(
                rank, 
                root_set,
                TreeClass(old_node, self.l_factor)
            )
            return self

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Rank mismatch: current rank {cur.node.rank}, "
                  f"target rank {rank}, unfolding node.")
        # Unfold intermediate node between parent and current
        # Set replica of the current node set's pivot as first entry.
        pivot = cur.node.set.find_pivot().found_entry
        pivot_replica = _create_replica(pivot.item.key)
        new_set = self.SetClass().insert(pivot_replica, None)
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
        x_item: Item,
        next_entry: Entry,
        x_left: Optional[GPlusTreeBase] = None
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
        # Pre-import and cache frequently used values
        from gplus_trees.g_k_plus.utils import calc_rank
        
        inserted = True
        x_key = x_item.key
        replica = _create_replica(x_key)
        TreeClass = type(self)
        
        # Pre-calculate rank for GKPlusTreeBase insertions to avoid repeated calculations
        cached_new_rank = calc_rank(
            x_key,
            self.KListClass.KListNodeClass.CAPACITY,
            dim=self.DIM + 1
        )

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_entry = None     # Entry in right parent points to current subtree
        left_parent = None     # Parent node for left-side updates
        left_x_entry = None    # x_item stored in left parent
        
        while True:
            node = cur.node
            cur._invalidate_tree_size()
            is_leaf = node.rank == 1
            insert_obj = x_item if is_leaf else replica

            # First iteration - simple insert without splitting
            if right_parent is None:
                # Determine subtree for potential next iteration
                subtree = next_entry.left_subtree if next_entry else node.right_subtree
                insert_subtree = x_left if is_leaf else subtree
                
                # Check node set type once and insert accordingly
                is_gkplus_tree = isinstance(node.set, GKPlusTreeBase)
                node.set, _ = self._insert_into_node_set(
                    node.set, insert_obj, insert_subtree, cached_new_rank, x_key, is_gkplus_tree
                )

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Checking initial node set after inserting {x_key}: {print_pretty(node.set)}")
                node.set = self.check_and_convert_set(node.set)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Node set after inserting {x_key} (and conversion): {print_pretty(node.set)}")
                
                # Get next_entry once after insertion
                retrieval_result = node.set.retrieve(x_key)
                next_entry = retrieval_result.next_entry

                # Early return for leaf nodes
                if is_leaf:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Early return: inserted {x_key} into leaf node, returning tree.")
                    return self, inserted

                # Setup parent tracking for next iteration
                right_parent = left_parent = cur
                right_entry = next_entry
                left_x_entry = retrieval_result.found_entry
                cur = subtree
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Subsequent iteration: Node "
                          f"splitting required before inserting key {x_key} at "
                          f"dimension {self.DIM}, node set: "
                          f"{print_pretty(node.set)}")

                # Node splitting required - get updated next_entry
                res = node.set.retrieve(x_key)
                next_entry = res.next_entry

                # Split node at x_key
                left_split, _, right_split = node.set.split_inplace(x_key)
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Checking left split: "
                          f"{print_pretty(left_split)}")
                
                left_split = self.check_and_convert_set(left_split)
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Left split after conversion: "
                          f"{print_pretty(left_split)}")

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Checking right split: "
                          f"{print_pretty(right_split)}")

                right_split = self.check_and_convert_set(right_split)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Right split after conversion: "
                          f"{print_pretty(right_split)}")

                next_entry = right_split.retrieve(x_key).next_entry

                # Handle right side of the split
                new_tree, right_split = self._handle_right_split(
                    right_split, insert_obj, x_left, is_leaf, cached_new_rank, x_key, node, TreeClass
                )
                next_entry = right_split.retrieve(x_key).next_entry if new_tree else next_entry

                # Update parent reference to the new tree or keep existing
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
                        next_entry.left_subtree if next_entry else None
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
                    cur.node.right_subtree = None  # No right subtree at leaf level
                    return self, inserted  # Early return when leaf is processed
                    
                # Continue to next iteration with updated current node
                cur = next_cur
    
    def print_subtree_sizes(self):
        """
        Check the subtree sizes in the tree.
        
        Returns:
            bool: True if the node counts are consistent, False otherwise.
        """
        # Check if the node counts are consistent        
        print(f"Subtree at rank {self.node.rank} "
              f"has {self.node.set.item_count()} entries, "
                f"size: {self.node.get_size()}")
        
        for entry in self.node.set:
            if entry.left_subtree is not None:
                entry.left_subtree.print_subtree_sizes()
        
        if self.node.right_subtree is not None:
            self.node.right_subtree.print_subtree_sizes()
        return True
    
    # Extension methods to check threshold and perform conversions
    def check_and_convert_set(self, set: AbstractSetDataStructure) -> AbstractSetDataStructure:
        if isinstance(set, KListBase):
            return self.check_and_expand_klist(set)
        elif isinstance(set, GKPlusTreeBase):
            return self.check_and_collapse_tree(set)
        else:
            raise TypeError(f"Unsupported set type: {type(set).__name__}. "
                            "Expected KListBase or GKPlusTreeBase.")
    
    def check_and_expand_klist(self, klist: KListBase) -> AbstractSetDataStructure:
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
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Checking KList {print_pretty(klist)}, item count: "
                  f"{klist.item_count()}, threshold: {threshold}.")

        if klist.item_count() > threshold:
            # Import locally to avoid circular dependency
            from gplus_trees.g_k_plus.utils import klist_to_tree
            
            # Convert to GKPlusTree with increased dimension
            new_dim = type(self).DIM + 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Converting KList to GKPlusTree (dime={new_dim}).")

            target_dim = new_dim
            return klist_to_tree(klist, k, target_dim, self.l_factor)

        return klist
    
    def check_and_collapse_tree(self, tree: 'GKPlusTreeBase') -> AbstractSetDataStructure:
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
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Checking GKPlusTree {print_pretty(tree)} "
                  f"item_count={tree.item_count()}, threshold={threshold}.")
        
        if tree.is_empty():
            return tree
        
        if tree.DIM == 1:
            # We want to keep the GKPlusTree structure for dimension 1
            return tree
            
        # Count the tree items (including dummy items)
        item_count = tree.item_count()
        if item_count - 1 <= threshold: # - 1 because the dummy item will be removed during conversion
            from gplus_trees.g_k_plus.utils import tree_to_klist
            # Collapse into a KList
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Collapsing GKPlusTree to KList.")
            return tree_to_klist(tree)

        return tree
    
    def split_inplace(self, key: int) -> Tuple['GKPlusTreeBase', Optional['GKPlusTreeBase'], 'GKPlusTreeBase']:
        """
        Split the tree into two parts around the given key.
        
        Args:
            key: The key value to split at
            
        Returns:
            A tuple of (left_return, key_subtree, right_return) where:
            - left_return: A tree containing all entries with keys < key
            - key_subtree: If key exists in the tree, its associated left subtree; otherwise, None
            - right_return: A tree containing all entries with keys ≥ key (except the entry with key itself)
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"SPLITTING TREE AT KEY {key}.")

        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")
        
        TreeClass = type(self)
        NodeClass = TreeClass.NodeClass
        dummy = get_dummy(dim=TreeClass.DIM)

        # Case 1: Empty tree - return None left, right and key's subtree
        if self.is_empty():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("")
            return self, None, TreeClass(l_factor=self.l_factor)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"SELF TREE TO SPLIT: {print_pretty(self)}")

        # Initialize left and right return trees
        left_return = self
        right_return = TreeClass(l_factor=self.l_factor)

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_entry = None     # Entry in right parent points to current subtree
        left_parent = None     # Parent node for left-side updates
        seen_key_subtree = None # Cache last seen left_parents left subtree
        
        cur = left_return
        key_node_found = False

        while True:
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Split node at key {key}.")
            
            node = cur.node
            is_leaf = node.rank == 1

            # Split node at key
            left_split, key_subtree, right_split = node.set.split_inplace(key)

            left_split = self.check_and_convert_set(left_split)
            # right_split = self.check_and_convert_set(right_split)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[DIM {self.DIM}] LEFT SPLIT after split "
                      f"(and conversion): {print_pretty(left_split)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[DIM {self.DIM}] KEY SUBTREE: "
                      f"{print_pretty(key_subtree)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[DIM {self.DIM}] RIGHT SPLIT after split "
                      f"(and conversion): {print_pretty(right_split)}")

            l_count = left_split.item_count()
            r_count = right_split.item_count()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[DIM {self.DIM}] Is leaf: {is_leaf}")

            # --- Handle right side of the split ---
            # Determine if we need a new tree for the right split
            if r_count > 0:     # incl. dummy items
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Right split item count "
                          f" > 0: {r_count}")
                
                if isinstance(right_split, GKPlusTreeBase):
                    from gplus_trees.g_k_plus.utils import calc_rank
                    # Calculate the new rank for the item in the next dimension
                    new_rank = calc_rank(
                        dummy.key,
                        self.KListClass.KListNodeClass.CAPACITY,
                        dim=self.DIM + 1
                    )

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] Inserting dummy {dummy.key} "
                              f"into right split of dimension {self.DIM + 1} "
                              f"with new rank {new_rank}")

                    right_split, _ = right_split.insert(dummy, rank=new_rank, x_left=None)
                else:
                    right_split = right_split.insert(dummy, left_subtree=None)

                right_split = self.check_and_convert_set(right_split)
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Right split after inserting dummy "
                          f"(and conversion): {print_pretty(right_split)}")

                next_entry = right_split.retrieve(key).next_entry
                right_node = NodeClass(
                    node.rank, right_split, node.right_subtree
                )
                
                if right_parent is None:
                    # Create a root node for right return tree
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] Right parent is None, creating"
                              " new root node.")

                    right_return.node = right_node
                    new_tree = right_return
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Right return tree: {print_pretty(new_tree)}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Right return tree root set: "
                              f"{print_pretty(new_tree.node.set)}")
                else:  
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] Right parent is not None, "
                              f"creating new tree node and make it a subtree of"
                              f" right parent.")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] Right parent tree before "
                              f"update: {print_pretty(right_parent)}")
                    new_tree = TreeClass(l_factor=self.l_factor)
                    new_tree.node = right_node
                    
                    # Update parent reference
                    if right_entry is not None:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Right entry is not None: "
                                  f"key={right_entry.item.key}, "
                                  f"value={right_entry.item.value}")
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Right entry left subtree "
                                  f"before update: "
                                  f"{print_pretty(right_entry.left_subtree)}")
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Updating left subtree of "
                                  f"right entry with new tree "
                                  f"{print_pretty(new_tree)}.")
                        right_entry.left_subtree = new_tree
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Right entry is None, "
                                  f"updating right subtree of right parent with"
                                  f" new tree {print_pretty(new_tree)}.")
                        right_parent.node.right_subtree = new_tree

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] New right parent tree after "
                              f"subtree update: {print_pretty(right_parent)}")

                if is_leaf:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] Is leaf: {is_leaf} --> "
                              f"Set next pointers for new tree (node) "
                              f"{print_pretty(new_tree.node.set)} to "
                              f"cur.node.next (set): {print_pretty(cur.node.next.node.set if cur.node.next else None)}")
                    # Prepare for updating 'next' pointers
                    new_tree.node.next = cur.node.next

                # Prepare references for next iteration
                next_right_entry = next_entry
                next_cur = (
                    next_entry.left_subtree 
                    if next_entry else new_tree.node.right_subtree
                )
                next_right_parent = new_tree
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[DIM {self.DIM}] Next right parent set: {print_pretty(next_right_parent.node.set) if next_right_parent else None}")
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Right split item count is {r_count}")
                if is_leaf and right_parent:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("We are at a leaf node and have a right parent "
                              "--> create a new right tree node.")

                    # Create a leaf with a single dummy item
                    if isinstance(right_split, GKPlusTreeBase):
                        from gplus_trees.g_k_plus.utils import calc_rank
                        # Calculate the new rank for the item in the next dimension
                        new_rank = calc_rank(
                            dummy.key,
                            self.KListClass.KListNodeClass.CAPACITY,
                            dim=self.DIM + 1
                        )
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Inserting {dummy.key} into right split "
                                  f"dim={self.DIM + 1}, rank={new_rank}")
                            
                        right_split, _ = right_split.insert(dummy, rank=new_rank, x_left=None)
                    else:
                        right_split = right_split.insert(dummy, left_subtree=None)

                    right_split = self.check_and_convert_set(right_split)
                    next_entry = right_split.retrieve(key).next_entry
                    
                    right_node = NodeClass(1, right_split, None)
                    new_tree = TreeClass(l_factor=self.l_factor)
                    new_tree.node = right_node

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] Right parent tree before "
                              f"subtree update: {print_pretty(right_parent)}")
                    
                    # Update parent reference
                    if right_entry is not None:
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Right entry is not None: "
                                  f"key={right_entry.item.key}, "
                                  f"value={right_entry.item.value}")
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Right entry left subtree "
                                  f"before update: {print_pretty(right_entry.left_subtree)}")
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Updating left subtree of right entry with new tree {print_pretty(new_tree)}")
                        right_entry.left_subtree = new_tree
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[DIM {self.DIM}] Right entry is None, "
                                  f"updating right subtree of right parent with"
                                  f" new tree {print_pretty(new_tree)}.")
                        right_parent.node.right_subtree = new_tree

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] New right parent tree after "
                              f"subtree update: {print_pretty(right_parent)}")

                    # Link leaf nodes
                    new_tree.node.next = cur.node.next
                    next_right_parent = new_tree
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("No node creation, keep existing parent refs.")
                    next_entry = right_split.retrieve(key).next_entry
                    next_right_parent = right_parent
                    next_cur = (
                        next_entry.left_subtree 
                        if next_entry else cur.node.right_subtree
                    )

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Next entry: {next_entry}")
                    if next_entry:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Next cur is next entry's "
                                  f"({next_entry.item.key}) left subtree: ",
                                  print_pretty(next_entry.left_subtree))
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Next entry is None, using current node's "
                                  "right subtree.")
                next_right_entry = right_entry

            # Update right parent variables for next iteration
            right_parent = next_right_parent
            right_entry = next_right_entry

            # --- Handle left side of the split ---
            # Determine if we need to create/update using left split
            if l_count > 1:     # incl. dummy items
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Left split item count: {l_count} (incl. dummies)")

                # Update current node to use left split
                cur.node.set = left_split
                cur._invalidate_tree_size()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Current node set after left split: "
                          f"{print_pretty(cur.node.set)}")

                if left_parent is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Left parent is None, set left tree to cur and "
                              "update self reference to this node.")

                    # Reuse left split as the root node for the left return tree
                    left_return = self = cur
                
                if is_leaf:
                    # Prepare for updating 'next' pointers
                    # do not rearrange subtrees at leaf level
                    l_last_leaf = cur.node
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DIM {self.DIM}] Is leaf: {is_leaf} --> Set "
                              f"l_last_leaf to cur (node): "
                              f"{print_pretty(cur.node.set)}.")
                elif key_subtree:
                    # Highest node containing split key found
                    # All entries in its left subtree are less than key and
                    # are part of the left return tree
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Highest node containing split key found. "
                              "Updating current node's right subtree with "
                              "key subtree.")

                    seen_key_subtree = key_subtree
                    cur.node.right_subtree = key_subtree   
                elif next_entry:
                    cur.node.right_subtree = next_entry.left_subtree
                
                # Check if we need to update the left parent reference
                if key_node_found:
                    next_left_parent = left_parent
                else:
                    # Make current node the new left parent
                    next_left_parent = cur  
                
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Left split item count: {l_count} (incl. dummies)")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Left split item count <= 1, handling accordingly.")

                if is_leaf:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Is leaf: {is_leaf}")
                    
                    if left_parent or seen_key_subtree:
                        if l_count == 0:
                            # find the previous leaf node by traversing the left parent
                            if left_parent:
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug("Left parent exists, so leaf is not "
                                          "collapsed. Update next pointers.")
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug("Find the previous leaf node by "
                                          "traversing the left parent to unlink"
                                          " leaf nodes.")
                                l_last_leaf = left_parent.get_max_leaf()
                            else:
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug("Left parent exists, so leaf is not "
                                          "collapsed. Update next pointers.")
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug("Find the previous leaf node by "
                                          "traversing the left parent to unlink"
                                          " leaf nodes.")

                                left_return = self = seen_key_subtree
                                l_last_leaf = seen_key_subtree.get_max_leaf()
                        else:
                            # Only for debugging purposes
                            entry_list = list(left_parent.node.set)
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Left parent exists with set: "
                                      f"{[e.item.key for e in entry_list]}, "
                                      "so leaf is not collapsed. Set "
                                      "l_last_leaf to cur and update leaf "
                                      "next pointers.")

                            cur.node.set = left_split
                            l_last_leaf = cur.node
                            
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"set l_last_leaf to cur.node "
                                      f"{print_pretty(cur.node.set)} with "
                                      f" l_last_leaf.ext: {print_pretty(l_last_leaf.next.node.set) if l_last_leaf.next else None}")
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Left parent is None at leaf. Only dummy item"
                                  " in left tree --> return empty left.")

                        # No non-dummy entry in left tree - return empty left tree
                        left_return = self = TreeClass(l_factor = self.l_factor)
                        l_last_leaf = None

                    next_left_parent = left_parent
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("We are at an internal node --> Collapsing "
                              "single-item nodes (Dummies are counted)")
                    
                    if key_subtree:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Highest node containing split key found. "
                                  "Using its left subtree as new subtree.")
                        
                        # Highest node containing split key found
                        # All entries in its left subtree are less than key and
                        # are part of the left return tree
                        new_subtree = key_subtree
                        seen_key_subtree = key_subtree
                    elif next_entry:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Next entry exists, using its left subtree as"
                                  "new subtree.")
                        new_subtree = next_entry.left_subtree
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("SHOULD NOT HAPPEN: No next entry in current "
                                  "node --> using current node's right subtree "
                                  "to proceed with left tree.")
                        
                        new_subtree = cur.node.right_subtree # Should not happen

                    if left_parent:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Left parent exists. Update right subtree "
                                  "only if it is not fixed yet.")

                        if not key_node_found:
                            left_parent.node.right_subtree = new_subtree
                            next_left_parent = left_parent
                        else:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug("Fixed: Keep left parent reference.")
                            next_left_parent = left_parent
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Left parent is None. Keep its reference.")
                        next_left_parent = left_parent

            left_parent = next_left_parent

            # Update leaf node 'next' pointers if at leaf level
            if is_leaf:
                # Unlink leaf nodes
                if l_last_leaf:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Left leaf node exists with items: "
                              f"{[e.item.key for e in l_last_leaf.set]}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Setting its next pointer to None.")
                    l_last_leaf.next = None

                # prepare key entry subtree for return
                return_subtree = key_subtree

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"LEFT (SELF) RETURN: {print_pretty(self)}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"MIDDLE RETURN: {print_pretty(return_subtree)}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"RIGHT RETURN: {print_pretty(right_return)}")

                return self, return_subtree, right_return

            if key_subtree:
                # Do not update left parent reference from this point on
                key_node_found = True

            # Continue to next iteration with updated current node
            cur = next_cur
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("")
    
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
        """Yields each entry of the gk-plus-tree in order."""
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

    def _insert_into_node_set(self, node_set, insert_obj: Item, insert_subtree, cached_rank: int, x_key: int, is_gkplus_tree: bool):
        """Helper method to insert into a node set, handling both GKPlusTreeBase and other types."""
        if is_gkplus_tree:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Inserting {x_key} into GKPlusTree (dim={self.DIM + 1}, rank={cached_rank})")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Node set (tree) before inserting {x_key}: {print_pretty(node_set)}")
            return node_set.insert(insert_obj, rank=cached_rank, x_left=insert_subtree)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Inserting {insert_obj.key} into node set: {print_pretty(node_set)}")
            return node_set.insert(insert_obj, left_subtree=insert_subtree), None

    def _handle_right_split(self, right_split, insert_obj, x_left, is_leaf, cached_new_rank, x_key, node, TreeClass):
        """Helper method to handle right split creation and insertion."""
        new_tree = None
        if right_split.item_count() > 0 or is_leaf:
            insert_subtree = x_left if is_leaf else None
            is_right_gkplus = isinstance(right_split, GKPlusTreeBase)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Inserting {x_key} into right split"
                      f"{' of dim ' + str(self.DIM + 1) + ' new rank ' + str(cached_new_rank) if is_right_gkplus else ''}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Right split before insert: "
                      f"{print_pretty(right_split)}")

            right_split, _ = self._insert_into_node_set(
                right_split, insert_obj, insert_subtree, cached_new_rank, x_key, is_right_gkplus
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[DIM {self.DIM}] Right split after inserting "
                      f"{x_key} before conversion: "
                      f"{print_pretty(right_split)}")

            right_split = self.check_and_convert_set(right_split)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[DIM {self.DIM}] Right split after insert "
                      f"(and conversion: {print_pretty(right_split)}")

            new_tree = TreeClass(l_factor=self.l_factor)
            new_tree.node = self.NodeClass(
                node.rank,
                right_split,
                node.right_subtree
            )
        
        return new_tree, right_split
