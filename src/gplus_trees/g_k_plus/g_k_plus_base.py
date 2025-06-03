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
    GPlusTreeBase, GPlusNodeBase, Stats, print_pretty
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
logger.setLevel(logging.INFO)
# Prevent propagation to the root logger to avoid duplicate logs
logger.propagate = False

t = TypeVar('t', bound='GKPlusTreeBase')

DEFAULT_DIMENSION = 1  # Default dimension for GKPlusTree
DEFAULT_L_FACTOR = 2  # Default threshold factor for KList to GKPlusTree conversion

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
    return Item(-1, None)

    # Check if we already have a dummy item for this dimension in cache
    if dim in _DUMMY_ITEM_CACHE:
        return _DUMMY_ITEM_CACHE[dim]
    
    # Create a new dummy item for this dimension
    dummy_key = -(dim)  # Negative dimension as key 
    dummy_item = Item(dummy_key, None)
    
    # Cache it for future use
    _DUMMY_ITEM_CACHE[dim] = dummy_item
    
    return dummy_item


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

    def _invalidate_tree_size(self) -> None:
        """
        Invalidate the tree size. This is a placeholder for future implementation.
        """
        self.size = None

    def get_size(self) -> bytes:
        """
        Get the hash value, computing it if necessary.
        
        Returns:
            bytes: The hash value for this node and its subtrees.
        """
        if self.size is None:
            return self.calculate_tree_size()
        return self.size

    def calculate_tree_size(self) -> int:
        """
        Calculate the size of the tree based on the number of items in leaf nodes.
        
        Returns:
            int: The total size of the tree.
        """
        dummy_key = get_dummy(dim=self.TreeClass.DIM).key
        if self.rank == 1:
            size = 0
            for entry in self.set:
                if entry.item.key == dummy_key:
                    continue

                size += 1
            self.size = size
            return self.size
        else:
            count = 0
            for entry in self.set:
                count += entry.left_subtree.node.get_size() if entry.left_subtree is not None else 0

            count += self.right_subtree.node.get_size() if self.right_subtree is not None else 0
            self.size = count
            return self.size

class GKPlusTreeBase(GPlusTreeBase, GKTreeSetDataStructure):
    """
    A GK+-tree is an extension of G+-tree with dimension support.
    It can automatically transform between KList and GKPlusTree based on item count.
    
    Attributes:
        node (Optional[GKPlusNodeBase]): The GK+-node that the tree contains.
        DIM (int): The dimension of the GK+-tree (class attribute).
        l_factor (float): The threshold factor for conversion between KList and GKPlusTree.
    """
    __slots__ = GPlusTreeBase.__slots__ + ("l_factor",)  # Only add new slots beyond what parent has
    
    # Default dimension value that will be overridden by factory-created subclasses
    DIM: int = 1  # Default dimension value, will be set by the factory
    
    # Will be set by the factory
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
    
    def __str__(self):
        return "Empty GKPlusTree" if self.is_empty() else f"GKPlusTree(dim={self.__class__.DIM}, node={self.node})"

    __repr__ = __str__
    
    def item_count(self) -> int:
        return self.node.get_size()
    
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

    def get_min(self) -> RetrievalResult:
        """
        Get the minimum entry in the tree.
        Returns:
            RetrievalResult: The minimum entry and the next entry (if any).
        """        
        if self.is_empty():
            return RetrievalResult(None, None)
        
        first_leaf = next(self.iter_leaf_nodes(), None)
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
    
    @classmethod
    def with_dimension(cls: Type[t], dim: int) -> Type[t]:
        """
        Create a new GKPlusTreeBase subclass with a specific dimension.
        
        Args:
            dim: The dimension value for the new class
            
        Returns:
            A new class with the specified dimension
        """
        new_class = type(f"{cls.__name__}Dim{dim}", (cls,), {"DIM": dim})
        return new_class
    
    # def insert(self, x: Item, rank: int) -> GKPlusTreeBase:
    #     """Insert an item into the GK+-tree and update tree size."""
    #     tree, inserted = super().insert(x, rank)
    #     self.node.get_size()
        
    #     return tree, inserted
    
    
    def _insert_non_empty(self, x_item: Item, rank: int) -> GKPlusTreeBase:
        """Optimized version for inserting into a non-empty tree."""
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
                return self._insert_new_item(cur, x_item, res.next_entry)

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
            old_node = self.node
            dummy = get_dummy(dim=TreeClass.DIM)
            root_set = self.SetClass().insert(dummy, None)
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
    
    def _insert_new_item(
        self,
        cur: GKPlusTreeBase,
        x_item: Item,
        next_entry: Entry
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
            node._invalidate_tree_size()
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
                
                # Insert the item
                node.set = node.set.insert(insert_obj, subtree)

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
                    right_split = right_split.insert(insert_obj, None)
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
    
    # def _insert_non# Main conversion methods
    @classmethod
    def from_klist(cls: Type[t], klist: KListBase, dim: int = None) -> t:
        """
        Convert a KList to a GKPlusTree.
        
        Args:
            klist: The KList to convert
            dim: The dimension for the new tree (if specified, creates a new class)
            
        Returns:
            A new GKPlusTree containing all items from the KList
        """
        # Use the specified dimension or increment the current one
        target_class = cls
        if dim is not None:
            target_class = cls.with_dimension(dim)
        
        new_tree = target_class()
        
        # If the KList is empty, return an empty tree
        if klist.is_empty():
            return new_tree
        
        # Insert all items from the KList into the tree
        # For each entry in the KList
        for entry in klist:
            # Insert the item with rank 1 (leaf level)
            new_tree = new_tree.insert(entry.item, 1)
            
            # If the entry has a left subtree, integrate it
            if entry.left_subtree is not None:
                # Get the updated entry after insertion
                result = new_tree.retrieve(entry.item.key)
                if result.found_entry is not None:
                    # If the left subtree is a GKPlusTree, use it directly
                    if isinstance(entry.left_subtree, GKPlusTreeBase):
                        left_subtree = entry.left_subtree
                    # If it's another type of tree, recursively convert its content
                    else:
                        # This is a placeholder - in a real implementation, you'd need
                        # a more sophisticated conversion from other tree types
                        left_subtree = cls.from_tree(entry.left_subtree)
                    
                    # Update the left subtree for this entry
                    result.found_entry.left_subtree = left_subtree
        
        return new_tree
    
    def to_klist(self) -> KListBase:
        """
        Convert this GKPlusTree to a KList.
        
        Returns:
            A new KList containing all items from this tree
        """
        if self.is_empty():
            return self.KListClass()
        
        klist = self.KListClass()
        dummy_key = get_dummy(dim=self.DIM).key
        # Process leaf nodes to build the KList
        for leaf_node in self.iter_leaf_nodes():
            for entry in leaf_node.set:
                # Skip dummy items
                if entry.item.key == dummy_key:
                    continue
                
                # Insert each item into the KList
                klist = klist.insert(entry.item, entry.left_subtree)
        
        return klist
    
    @classmethod
    def from_tree(cls: Type[t], tree: GKPlusTreeBase, dim: int = None) -> t:
        """
        Convert any GKPlusTree-like object to a GKPlusTree.
        
        Args:
            tree: The tree to convert
            dim: The dimension for the new tree (if specified, creates a new class)
            
        Returns:
            A new GKPlusTree with the same content as the original tree
        """
        # Use the specified dimension or the current one
        target_class = cls
        if dim is not None:
            target_class = cls.with_dimension(dim)
        
        # If it's already the right type, just return it
        if isinstance(tree, target_class):
            return tree
            
        # If it's empty, return an empty GKPlusTree
        if tree is None or tree.is_empty():
            return target_class()
            
        # Create a new GKPlusTree with the same structure
        new_tree = target_class()
        
        # Process leaf nodes to build the new tree
        dummy_key = get_dummy(dim=target_class.DIM).key
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                # Skip dummy items
                if entry.item.key == dummy_key:
                    continue
                
                # Insert the item
                new_tree = new_tree.insert(entry.item, 1)
                
                # If there's a left subtree, recursively convert it
                if entry.left_subtree is not None:
                    left_subtree = cls.from_tree(entry.left_subtree)
                    # Update the reference in the new entry
                    result = new_tree.retrieve(entry.item.key)
                    if result.found_entry:
                        result.found_entry.left_subtree = left_subtree
                        
        return new_tree
    
    # Extension methods to check threshold and perform conversions
    def check_and_expand_klist(self, klist: KListBase) -> AbstractSetDataStructure:
        """
        Check if a KList exceeds the threshold and should be converted to a GKPlusTree.
        
        Args:
            klist: The KList to check
            
        Returns:
            Either the original KList or a new GKPlusTree based on the threshold
        """
        if klist.is_empty():
            return klist
            
        # Check if the item count exceeds l_factor * CAPACITY
        capacity = klist.KListNodeClass.CAPACITY
        threshold = int(capacity * self.l_factor)
        
        if klist.item_count() > threshold:
            # Convert to GKPlusTree with increased dimension
            new_dim = self.__class__.DIM + 1
            new_tree_class = type(self).with_dimension(new_dim)
            return new_tree_class.from_klist(klist)
        
        return klist
    
    def check_and_collapse_tree(self, tree: 'GKPlusTreeBase') -> AbstractSetDataStructure:
        """
        Check if a GKPlusTree has few enough items to be collapsed into a KList.
        
        Args:
            tree: The GKPlusTree to check
            
        Returns:
            Either the original tree or a new KList based on the threshold
        """
        if tree.is_empty():
            return tree
            
        # Get the threshold based on the KList capacity
        capacity = self.KListClass.KListNodeClass.CAPACITY
        threshold = int(capacity * self.l_factor)
        
        # Count the actual items (excluding dummy items)
        dummy_key = get_dummy(dim=self.__class__.DIM).key
        item_count = 0
        for leaf_node in tree.iter_leaf_nodes():
            for entry in leaf_node.set:
                if entry.item.key != dummy_key:
                    item_count += 1
        
        if item_count <= threshold:
            # Collapse into a KList
            return tree.to_klist()
            
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
        
        logger.debug("SELF TREE TO SPLIT:")
        logger.debug(print_pretty(self))

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
            logger.debug(f"New iteration with cur: {print_pretty(cur.node.set)}")
            logger.debug(f"Split node at key {key}.")
            node = cur.node
            is_leaf = node.rank == 1

            # Node splitting required - get updated next_entry
            res = node.set.retrieve(key)
            next_entry = res.next_entry
            logger.debug(f"Next entry in cur: {next_entry}")

            # Split node at key
            left_split, key_subtree, right_split = node.set.split_inplace(key)

            logger.debug(f"LEFT SPLIT: {print_pretty(left_split)}")
            logger.debug(f"KEY SUBTREE: {print_pretty(key_subtree)}")
            logger.debug(f"RIGHT SPLIT: {print_pretty(right_split)}")

            l_count = left_split.item_count()
            r_count = right_split.item_count()

            # Log split information, is empty, and item count
            logger.debug(f"Is leaf: {is_leaf}")  

            # --- Handle right side of the split ---
            # Determine if we need a new tree for the right split
            if r_count > 0:     # incl. dummy items
                logger.debug(f"Right split item count is > 0: {r_count}")
                right_split = right_split.insert(dummy, None)
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
                    new_tree = TreeClass()
                    new_tree.node = right_node
                    
                    # Update parent reference
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
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
                    right_split = right_split.insert(dummy, None)
                    right_node = NodeClass(1, right_split, None)
                    new_tree = TreeClass()
                    new_tree.node = right_node

                    # Update parent reference
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
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
                cur.node._invalidate_tree_size()

                if left_parent is None:
                    logger.debug("Left parent is None, set the left tree to cur and update self reference to this node.")
                    # Reuse left split as the root node for the left return tree
                    left_return = self = cur
                    logger.debug(f"Left tree: {print_pretty(left_return)}")
                
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
                            logger.debug("Left parent exists, so leaf is not collapsed. Set l_last_leaf to cur and update leaf next pointers.")
                            
                            l_last_leaf = cur.node
                            logger.debug(f"l_last_leaf.next: {l_last_leaf.next}")

                    else:
                        logger.debug("Left parent is None at leaf. Only dummy item in left tree --> return empty left.")
                        # No non-dummy entry in left tree - return empty left tree

                        # # Link leaf nodes
                        # if r_first_leaf:
                        #     logger.debug("Linking leaf nodes.")
                        #     r_first_leaf.node.next = cur.node.next


                        left_return = self = TreeClass()
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
                        logger.debug(f"Next entry's left subtree: {print_pretty(new_subtree)}")
                    else:
                        logger.debug("SHOULD NOT HAPPEN: No next entry in current node --> using current node's right subtree to proceed with left tree.")
                        new_subtree = cur.node.right_subtree # Should not happen

                    if left_parent:
                        logger.debug("Left parent exists. Update right subtree only if it is not fixed yet.")
                        if not key_node_found:
                            logger.debug(f"Not fixed: Update left parent's right subtree with new subtree: {print_pretty(new_subtree)}")
                            
                            left_parent.node.right_subtree = new_subtree
                            logger.debug(f"Left parent tree: {print_pretty(left_parent)}")
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
                    logger.debug("Left leaf node exists.")
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
    
    def __iter__(self):
        """Yields each entry of the gk-plus-tree in order."""
        # if self.is_empty():
        #     return
        for node in self.iter_leaf_nodes():
            for entry in node.set:
                yield entry
