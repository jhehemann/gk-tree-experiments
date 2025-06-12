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
logger.setLevel(logging.INFO)  # Set to DEBUG for detailed output
# Prevent propagation to the root logger to avoid duplicate logs
logger.propagate = False

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
        logger.debug(f"Updating left subtree of key {key} in set {print_pretty(self.set)} with new left: {print_pretty(new_left)}")
        entry = self.set.retrieve(key).found_entry
        if entry is not None:
            logger.debug(f"Found entry with key {key}, updating left subtree.")
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
        # Call parent's __init__ with node and dimension
        super().__init__(node)
        # Add our additional attribute
        self.l_factor = l_factor
        logger.debug(f"Creating GKPlusTree with dimension {self.DIM} and l_factor {self.l_factor}")
        self.item_cnt = None  # Initialize item count
    
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
        if self.node.right_subtree is None:  # indicates a leaf in current dim
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
        logger.debug(f"Getting max dimension for GKPlusTree with dimension {self.DIM}.")
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
        logger.debug(f"Counting expanded leaf nodes in GKPlusTree: {print_pretty(self)}")
        if self.is_empty():
            logger.debug("Tree is empty, returning expanded count 0.")
            return 0
        
        count = 0
        for leaf in self.iter_leaf_nodes():
            expanded = False
            if isinstance(leaf.set, GKPlusTreeBase):
                # If the set is a GKPlusTreeBase, we can directly count its expanded leaves
                logger.debug(f"Leaf set is a GKPlusTreeBase, counting expanded leaves.")
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
        
        if node.right_subtree is not None:
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
        logger.debug(f"Getting pivot entry from GKPlusTree {print_pretty(self)}")
        if self.is_empty():
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
                    logger.debug(f"Found pivot entry: {entry}")
                    pivot = entry
        if pivot is None:
            raise ValueError(f"No pivot entry found in tree {print_pretty(self)}")
        
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
            raise ValueError("Tree is empty, cannot retrieve minimum entry.")
        
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
        while node.right_subtree is not None:
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
        inserted = True
        # Single-level leaf
        logger.debug(f"Dim {self.DIM} insert called into empty tree with KEY: {x_item.key}, RANK: {rank}")
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_item, x_left)
            self.node = self.NodeClass(rank, leaf_set, None)
            logger.debug(f"Tree after empty insert rank == 1:\n{print_pretty(self)}")
            return self, inserted

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_item, x_left)
        root_set = self.SetClass().insert(get_dummy(dim=self.DIM), None)
        root_set = root_set.insert(_create_replica(x_item.key), l_leaf_t)
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        logger.debug(f"Tree after empty insert rank == {rank}:\n{print_pretty(self)}")
        return self, inserted

    def _insert_non_empty(self, x_item: Item, rank: int, x_left: Optional[GPlusTreeBase] = None) -> GKPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
        logger.debug("")
        logger.debug(f"DIM {self.DIM} insert called with KEY: {x_item.key}, RANK: {rank} into tree:\n{print_pretty(self)}")
        
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
                    if x_left is not None:
                        raise ValueError(f"Item with key {x_item.key} already exists in the tree. Cannot be inserted with a subtree again.")
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
            logger.debug(f"Handling rank mismatch: current rank {cur.node.rank}, target rank {rank}, creating new root node.")
            old_node = self.node
            dummy = get_dummy(dim=TreeClass.DIM)
            root_set = self.SetClass().insert(dummy, None)
            self.node = self.NodeClass(rank, root_set, TreeClass(old_node))
            return self

        logger.debug(f"Handling rank mismatch: current rank {cur.node.rank}, target rank {rank}, unfolding node.")
        # Unfold intermediate node between parent and current
        # Set replica of the current node's min as first entry.
        min_entry = cur.node.set.find_pivot().found_entry
        min_replica = _create_replica(min_entry.item.key)
        new_set = self.SetClass().insert(min_replica, None)
        new_tree = TreeClass(l_factor=self.l_factor)
        new_tree.node = self.NodeClass(rank, new_set, cur)
       
        if p_next:
            # p_next.left_subtree = new_tree
            parent.node = parent.node._update_left_subtree(
                p_next.item.key, new_tree
            )
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
            cur._invalidate_tree_size()
            is_leaf = node.right_subtree is None
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
                if isinstance(node.set, GKPlusTreeBase):                    
                    from gplus_trees.g_k_plus.utils import calc_rank
                    # Calculate the new rank for the item in the next dimension
                    new_rank = calc_rank(
                        x_item.key,
                        self.KListClass.KListNodeClass.CAPACITY,
                        dim=self.DIM + 1
                    )
                    logger.debug(f"Inserting {insert_obj.key} into initial node set of dimension {self.DIM + 1} with new rank {new_rank}")

                    logger.debug(f"Node set (tree) before inserting {insert_obj.key}: {print_pretty(node.set)}")

                    node.set, _ = node.set.insert(insert_obj, rank=new_rank, x_left=insert_subtree)
                    # logger.debug(f"Node set after inserting {insert_obj.key} before conversion check: {print_pretty(node.set)}")
                else:
                    logger.debug(f"Inserting {insert_obj.key} into initial node set (KList): {print_pretty(node.set)}")
                    node.set = node.set.insert(insert_obj, left_subtree=insert_subtree)
                    # logger.debug(f"Node set after inserting {insert_obj.key} before conversion check: {print_pretty(node.set)}")

                logger.debug(f"Now checking initial insertion node set after inserting {insert_obj.key}: {print_pretty(node.set)}")
                # logger.debug(node.set.print_structure())
                node.set = self.check_and_convert_set(node.set)
                logger.debug(f"Node set after inserting {insert_obj.key} and conversion: {print_pretty(node.set)}")

                # logger.debug(f"New set after inserting {insert_obj.key} and conversion: {print_pretty(node.set)}")

                # Early return if we're already at a leaf node
                if is_leaf:
                    logger.debug(f"Early return: inserted {insert_obj.key} into leaf node, returning tree.")
                    return self, inserted
                
                # Assign parent tracking for next iteration
                right_parent = left_parent = cur
                right_entry = next_entry if next_entry else None
                left_x_entry = node.set.retrieve(x_key).found_entry
                cur = subtree
            else:
                # Node splitting required - get updated next_entry
                logger.debug(f"Subsequent iteration: Node splitting required before inserting key {x_key} at dimension {self.DIM}, node set to split: {print_pretty(node.set)}")
                res = node.set.retrieve(x_key)
                next_entry = res.next_entry

                # Split node at x_key
                left_split, _, right_split = node.set.split_inplace(x_key)
                
                logger.debug(f"Now checking left split: {print_pretty(left_split)}")
                left_split = self.check_and_convert_set(left_split)
                logger.debug(f"Left split after conversion: {print_pretty(left_split)}")

                logger.debug(f"Now checking right split: {print_pretty(right_split)}")
                right_split = self.check_and_convert_set(right_split)

                logger.debug(f"Right split after conversion: {print_pretty(right_split)}")

                # --- Handle right side of the split ---
                # Determine if we need a new tree for the right split
                if right_split.item_count() > 0 or is_leaf:
                    # Insert item into right split and create new tree
                    insert_subtree = x_left if is_leaf else None
                    if isinstance(right_split, GKPlusTreeBase):
                        from gplus_trees.g_k_plus.utils import calc_rank
                        # Calculate the new rank for the item in the next dimension
                        new_rank = calc_rank(
                            x_item.key,
                            self.KListClass.KListNodeClass.CAPACITY,
                            dim=self.DIM + 1
                        )
                        logger.debug(f"Inserting {insert_obj.key} into right split of dimension {self.DIM + 1} with new rank {new_rank}")
                        logger.debug(f"Right split before insert: {print_pretty(right_split)}")

                        right_split, _ = right_split.insert(insert_obj, rank=new_rank, x_left=insert_subtree)
                        
                    else:
                        logger.debug(f"Inserting {insert_obj.key} into right split: {print_pretty(right_split)}")
                        right_split = right_split.insert(insert_obj, left_subtree=insert_subtree)
                    logger.debug(f"Right split after inserting {insert_obj.key} before conversion: {print_pretty(right_split)}")

                    # logger.debug(f"Now checking right split after inserting {insert_obj.key}, before conversion: {print_pretty(right_split)}")
                    right_split = self.check_and_convert_set(right_split)
                    
                    logger.debug(f"Right split after insert and conversion: {print_pretty(right_split)}")

                    new_tree = TreeClass(l_factor=self.l_factor)
                    new_tree.node = self.NodeClass(node.rank, right_split, node.right_subtree)

                    # Update parent reference to the new tree
                    if right_entry is not None:                        
                        # TODO: Use right_entry instance after klist_to_tree has been updated to use inplace entries
                        # right_entry.left_subtree = new_tree
                        right_parent.node = right_parent.node._update_left_subtree(
                            right_entry.item.key, new_tree
                        )
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
        # logger.debug(f"Checking KList {print_pretty(klist)} with item count {klist.item_count()} against threshold {threshold}.")  
        if klist.item_count() > threshold:
            # Import locally to avoid circular dependency
            from gplus_trees.g_k_plus.utils import klist_to_tree
            
            # Convert to GKPlusTree with increased dimension
            new_dim = type(self).DIM + 1
            logger.debug(f"Converting KList to GKPlusTree with new dimension {new_dim}.")
            target_dim = new_dim
            # logger.debug(f"Target dimension for new GKPlusTree: {target_dim}.")
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
        
        logger.debug(f"Checking GKPlusTree {print_pretty(tree)} "
                     f"with item count {tree.item_count()} against threshold {threshold}.")
        if tree.is_empty():
            return tree
        
        if tree.DIM == 1:
            # We want to keep the GKPlusTree structure for dimension 1
            return tree
            
        # Count the tree items (including dummy items)
        item_count = tree.item_count()
        # logger.debug(f"Tree item count: {item_count}, threshold: {threshold}.")

        if item_count - 1 <= threshold: # - 1 because the dummy item will be removed during conversion
            from gplus_trees.g_k_plus.utils import tree_to_klist
            # Collapse into a KList
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
        logger.debug(f"SPLITTING TREE AT KEY {key}.")
        
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__!r}")
        
        TreeClass = type(self)
        NodeClass = TreeClass.NodeClass
        dummy = get_dummy(dim=TreeClass.DIM)

        # Case 1: Empty tree - return None left, right and key's subtree
        if self.is_empty():
            logger.debug("")
            return self, None, TreeClass()
        
        logger.debug(f"SELF TREE TO SPLIT: {print_pretty(self)}")
        # logger.debug(print_pretty(self))
        # logger.debug(f"Root set structure before split:\n{print_pretty(self.node.set)}")
        # logger.debug(self.print_structure())

        # Initialize left and right return trees
        left_return = self
        right_return = TreeClass()

        # Parent tracking variables
        right_parent = None    # Parent node for right-side updates
        right_entry = None     # Entry in right parent points to current subtree
        left_parent = None     # Parent node for left-side updates
        seen_key_subtree = None # Cache last seen left_parents left subtree
        
        cur = left_return
        key_node_found = False

        while True:
            # logger.debug(f"New iteration with cur: {print_pretty(cur.node.set)}")
            logger.debug(f"Split node at key {key}.")
            node = cur.node
            is_leaf = node.right_subtree is None

            # Node splitting required - get updated next_entry
            res = node.set.retrieve(key)
            next_entry = res.next_entry
            logger.debug(f"Next entry in cur: {next_entry}")

            # Split node at key
            left_split, key_subtree, right_split = node.set.split_inplace(key)

            left_split = self.check_and_convert_set(left_split)
            right_split = self.check_and_convert_set(right_split)

            logger.debug(f"LEFT SPLIT after split (and conversion): {print_pretty(left_split)}")
            logger.debug(f"KEY SUBTREE: {print_pretty(key_subtree)}")
            logger.debug(f"RIGHT SPLIT after split (and conversion): {print_pretty(right_split)}")

            l_count = left_split.item_count()
            r_count = right_split.item_count()

            # Log split information, is empty, and item count
            logger.debug(f"Is leaf: {is_leaf}")  

            # --- Handle right side of the split ---
            # Determine if we need a new tree for the right split
            if r_count > 0:     # incl. dummy items
                logger.debug(f"Right split item count is > 0: {r_count}")
                if isinstance(right_split, GKPlusTreeBase):
                    from gplus_trees.g_k_plus.utils import calc_rank
                    # Calculate the new rank for the item in the next dimension
                    new_rank = calc_rank(
                        dummy.key,
                        self.KListClass.KListNodeClass.CAPACITY,
                        dim=self.DIM + 1
                    )
                    logger.debug(f"Inserting {dummy.key} into right split of dimension {self.DIM + 1} with new rank {new_rank}")

                    right_split, _ = right_split.insert(dummy, rank=new_rank, x_left=None)
                else:
                    right_split = right_split.insert(dummy, left_subtree=None)

                right_split = self.check_and_convert_set(right_split)
                logger.debug(f"Right split after inserting dummy (and conversion): {print_pretty(right_split)}")

                next_entry = right_split.retrieve(key).next_entry
                right_node = NodeClass(
                    node.rank, right_split, node.right_subtree
                )
                
                if right_parent is None:
                    # Create a root node for right return tree
                    logger.debug("Right parent is None, creating new root node.")
                    right_return.node = right_node
                    new_tree = right_return
                else:
                    logger.debug("Right parent is not None, creating new tree.")
                    new_tree = TreeClass(l_factor=self.l_factor)
                    new_tree.node = right_node
                    
                    # Update parent reference
                    if right_entry is not None:
                        # right_entry.left_subtree = new_tree
                        right_parent.node = right_parent.node._update_left_subtree(
                            right_entry.item.key, new_tree
                        )
                    else:
                        right_parent.node.right_subtree = new_tree

                if is_leaf:
                    # Prepare for updating 'next' pointers
                    new_tree.node.next = cur.node.next

                # Prepare references for next iteration
                
                next_right_entry = next_entry
                next_cur = (
                    next_entry.left_subtree 
                    if next_entry else new_tree.node.right_subtree
                )
                next_right_parent = new_tree

            else:
                logger.debug(f"Right split item count is {r_count}")
                if is_leaf and right_parent:
                    logger.debug("We are at a leaf node and have a right parent --> create a new right tree node.")
                    # Create a leaf with a single dummy item
                    if isinstance(right_split, GKPlusTreeBase):
                        from gplus_trees.g_k_plus.utils import calc_rank
                        # Calculate the new rank for the item in the next dimension
                        new_rank = calc_rank(
                            dummy.key,
                            self.KListClass.KListNodeClass.CAPACITY,
                            dim=self.DIM + 1
                        )
                        logger.debug(f"Inserting {dummy.key} into right split of dimension {self.DIM + 1} with new rank {new_rank}")
                            
                        right_split, _ = right_split.insert(dummy, rank=new_rank, x_left=None)
                    else:
                        right_split = right_split.insert(dummy, left_subtree=None)
                    
                    right_node = NodeClass(1, right_split, None)
                    new_tree = TreeClass(l_factor=self.l_factor)
                    new_tree.node = right_node

                    # Update parent reference
                    if right_entry is not None:
                        # right_entry.left_subtree = new_tree
                        right_parent.node = right_parent.node._update_left_subtree(
                            right_entry.item.key, new_tree
                        )
                    else:
                        right_parent.node.right_subtree = new_tree
                    
                    # Prepare for updating 'next' pointers
                    # r_first_leaf = new_tree

                    # Link leaf nodes
                    new_tree.node.next = cur.node.next

                    next_right_parent = new_tree

                else:
                    logger.debug("No node creation, keeping existing parent references.")
                    next_right_parent = right_parent
                    next_cur = (
                        next_entry.left_subtree 
                        if next_entry else cur.node.right_subtree
                    )

                    # if is_leaf:
                    # No right parent at this point
                    # Prepare for updating 'next' pointers
                        # r_first_leaf = None
                
                next_right_entry = right_entry

            # Update right parent variables for next iteration
            right_parent = next_right_parent
            right_entry = next_right_entry

            # --- Handle left side of the split ---
            # Determine if we need to create/update using left split
            if l_count > 1:     # incl. dummy items
                # Update current node to use left split
                logger.debug(f"Left split item count: {l_count} (incl. dummy items)")
                cur.node.set = left_split
                cur._invalidate_tree_size()

                if left_parent is None:
                    logger.debug("Left parent is None, set the left tree to cur and update self reference to this node.")
                    # Reuse left split as the root node for the left return tree
                    left_return = self = cur
                    # logger.debug(f"Left tree: {print_pretty(left_return)}")
                
                if is_leaf:
                    logger.debug(f"Is leaf: {is_leaf} --> Set l_last_leaf to cur.")
                    # Prepare for updating 'next' pointers
                    # do not rearrange subtres at leaf level
                    l_last_leaf = cur.node
                elif key_subtree:
                    logger.debug(f"Highest node containing split key found. Updating current node's right subtree with key subtree.")
                    # Highest node containing split key found
                    # All entries in its left subtree are less than key and
                    # are part of the left return tree
                    cur.node.right_subtree = key_subtree
                    seen_key_subtree = key_subtree
                elif next_entry:
                    cur.node.right_subtree = next_entry.left_subtree
                
                # Check if we need to update the left parent reference
                if key_node_found:
                    next_left_parent = left_parent
                else:
                    # Make current node the new left parent
                    next_left_parent = cur  
                
            else:
                logger.debug(f"Left split item count: {l_count} (incl. dummy items)")
                logger.debug("Left split item count is <= 1, handling accordingly.")

                if is_leaf:
                    logger.debug(f"Is leaf: {is_leaf}")
                    if left_parent or seen_key_subtree:
                        if l_count == 0:
                            
                            # find the previous leaf node by traversing the left parent
                            if left_parent:
                                logger.debug("Left parent exists, so leaf is not collapsed. Update leaf next pointers.")
                                logger.debug(f"Find the previous leaf node by traversing the left parent to unlink leaf nodes.")
                                l_last_leaf = left_parent.get_max_leaf()
                            else:
                                logger.debug("Left parent exists, so leaf is not collapsed. Update leaf next pointers.")
                                logger.debug(f"Find the previous leaf node by traversing the left parent to unlink leaf nodes.")
                                left_return = self = seen_key_subtree
                                l_last_leaf = seen_key_subtree.get_max_leaf()
                                
                        else:
                            logger.debug(f"Left parent exists: {[e.item.key for e in left_parent.node.set]}, so leaf is not collapsed. Set l_last_leaf to cur and update leaf next pointers.")
                            
                            cur.node.set = left_split
                            l_last_leaf = cur.node
                            logger.debug(f"l_last_leaf.next: {l_last_leaf.next}")

                    else:
                        logger.debug("Left parent is None at leaf. Only dummy item in left tree --> return empty left.")
                        # No non-dummy entry in left tree - return empty left tree

                        # # Link leaf nodes
                        # if r_first_leaf:
                        #     logger.debug("Linking leaf nodes.")
                        #     r_first_leaf.node.next = cur.node.next


                        left_return = self = TreeClass(l_factor = self.l_factor)
                        l_last_leaf = None

                    next_left_parent = left_parent

                else:
                    logger.debug("We are at an internal node --> Collapsing single-item nodes (Note: Dummy items are counted)")                    
                    if key_subtree:
                        logger.debug(f"Highest node containing split key found. Using split key's left subtree as new subtree.")
                        # Highest node containing split key found
                        # All entries in its left subtree are less than key and
                        # are part of the left return tree
                        new_subtree = key_subtree
                        seen_key_subtree = key_subtree
                    elif next_entry:
                        logger.debug("Next entry exists, using its left subtree as new subtree.")
                        new_subtree = next_entry.left_subtree
                        # logger.debug(f"Next entry's left subtree: {print_pretty(new_subtree)}")
                    else:
                        logger.debug("SHOULD NOT HAPPEN: No next entry in current node --> using current node's right subtree to proceed with left tree.")
                        new_subtree = cur.node.right_subtree # Should not happen

                    if left_parent:
                        logger.debug("Left parent exists. Update right subtree only if it is not fixed yet.")
                        if not key_node_found:
                            # logger.debug(f"Not fixed: Update left parent's right subtree with new subtree: {print_pretty(new_subtree)}")
                            
                            left_parent.node.right_subtree = new_subtree
                            # logger.debug(f"Left parent tree: {print_pretty(left_parent)}")
                            next_left_parent = left_parent
                        else:
                            logger.debug("Fixed: Keep left parent reference.")
                            next_left_parent = left_parent
                    else:
                        logger.debug("Left parent is None. Keep left parent reference.")
                        next_left_parent = left_parent

            left_parent = next_left_parent

            # Update leaf node 'next' pointers if at leaf level
            if is_leaf:
                # Unlink leaf nodes
                if l_last_leaf:
                    logger.debug(f"Left leaf node exists with items: {[e.item.key for e in l_last_leaf.set]}")
                    logger.debug("Setting its next pointer to None.")
                    l_last_leaf.next = None

                # prepare key entry subtree for return
                return_subtree = res.found_entry.left_subtree if res.found_entry else None
                logger.debug("")
                logger.debug("LEFT (SELF) RETURN:")
                logger.debug(print_pretty(self))
                logger.debug("MIDDLE RETURN:")
                logger.debug(print_pretty(return_subtree))
                logger.debug("RIGHT RETURN:")
                logger.debug(print_pretty(right_return))
                return self, return_subtree, right_return

            if key_subtree:
                # Do not update left parent reference from this point on
                key_node_found = True

            # Continue to next iteration with updated current node
            cur = next_cur
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
