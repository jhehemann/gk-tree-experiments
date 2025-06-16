"""GKPlusTree base implementation"""

from __future__ import annotations
from typing import Optional, Type, TypeVar, Tuple

from gplus_trees.base import (
    AbstractSetDataStructure,
    Item,
    Entry,
    _create_replica,
    RetrievalResult,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import (
    GPlusTreeBase, GPlusNodeBase, print_pretty, get_dummy
)

from gplus_trees.g_k_plus.base import GKTreeSetDataStructure
from gplus_trees.g_k_plus.utils import calc_rank

t = TypeVar('t', bound='GKPlusTreeBase')

DEFAULT_DIMENSION = 1  # Default dimension for GKPlusTree
DEFAULT_L_FACTOR = 1.0  # Default threshold factor for KList to GKPlusTree conversion

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
    __slots__ = GPlusNodeBase.__slots__ + ("size",)  # Add new slots beyond parent

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
        entry = self.set.retrieve(key).found_entry

        if entry is not None:
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
    __slots__ = GPlusTreeBase.__slots__ + ("l_factor", "item_cnt",)  # Add new slots

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

    def __str__(self):
        if self.is_empty():
            return "Empty GKPlusTree"
        return f"GKPlusTree(dim={self.__class__.DIM}, node={self.node})"

    __repr__ = __str__

    def item_count(self) -> int:
        if self.is_empty():
            self.item_cnt = 0
            return self.item_cnt
        return self.get_tree_item_count()

    def insert_entry(self, entry: Entry) -> Tuple['GKPlusTreeBase', bool]:
        """
        Insert an entry into the GK+-tree. The rank is calculated automatically
        based on the entry's item key. The entire entry object is preserved to
        maintain external references.
        
        Args:
            entry (Entry): The entry to be inserted, containing an item and left_subtree.
        
        Returns:
            Tuple[GKPlusTreeBase, bool]: The updated tree and whether insertion was successful.
            
        Raises:
            TypeError: If entry is not an Entry object.
        """
        if not isinstance(entry, Entry):
            raise TypeError(f"insert_entry(): expected Entry, got {type(entry).__name__}")
        
        # Calculate the rank using the item's key
        capacity = self.KListClass.KListNodeClass.CAPACITY
        rank = calc_rank(entry.item.key, capacity, self.DIM)
        
        # Use Entry-aware insertion logic to preserve the Entry object
        return self._insert_entry_with_rank(entry, rank)

    def get_tree_item_count(self) -> int:
        """Get the number of items in the tree, including dummy items."""
        if self.node.rank == 1:  # indicates a leaf in current dim
            count = 0
            self.item_cnt = self.node.set.item_count()
            return self.item_cnt
        else:
            count = 0
            for entry in self.node.set:
                count += (entry.left_subtree.get_tree_item_count()
                         if entry.left_subtree is not None else 0)

            count += (self.node.right_subtree.get_tree_item_count()
                     if self.node.right_subtree is not None else 0)
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
        Count leaf nodes whose set is recursively instantiated by another
        GKPlusTree, having an extra dummy item for that dimension.

        Returns:
            int: The number of leaf nodes in the tree.
        """
        if self.is_empty():
            return 0

        count = 0
        for leaf in self.iter_leaf_nodes():
            if isinstance(leaf.set, GKPlusTreeBase):
                # For GKPlusTreeBase, directly count its expanded leaves
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
        Returns the pivot entry for a node of the next higher dimension to be
        unfolded.

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
        while node.rank > 1:
            node = node.right_subtree.node

        return node

    def _make_leaf_klist(self, x_item: Item,
                         x_left: Optional[GPlusTreeBase] = None
                         ) -> AbstractSetDataStructure:
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

    def _make_leaf_trees(self, x_item,
                         x_left: Optional[GPlusTreeBase] = None
                         ) -> Tuple[GPlusTreeBase, GPlusTreeBase]:
        """
        Builds two linked leaf-level GPlusTreeBase nodes for x_item insertion
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

    def _insert_empty(self, x_item: Item, rank: int,
                      x_left: Optional[GKPlusTreeBase] = None
                      ) -> GKPlusTreeBase:
        """Build the initial tree structure depending on rank."""

        # Single-level leaf
        inserted = True
        if rank == 1:
            leaf_set = self._make_leaf_klist(x_item, x_left)
            self.node = self.NodeClass(rank, leaf_set, None)
            return self, inserted

        # Higher-level root with two linked leaf children
        l_leaf_t, r_leaf_t = self._make_leaf_trees(x_item, x_left)
        root_set = self.SetClass().insert(get_dummy(dim=self.DIM), None)
        root_set = root_set.insert(_create_replica(x_item.key), l_leaf_t)
        self.node = self.NodeClass(rank, root_set, r_leaf_t)
        return self, inserted

    def _insert_non_empty(self, x_item: Item, rank: int,
                          x_left: Optional[GKPlusTreeBase] = None
                          ) -> GKPlusTreeBase:
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
                    if x_left is not None:
                        raise ValueError(
                            f"Item with key {x_item.key} already exists in the "
                            "tree. Cannot be inserted with a subtree again."
                        )
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
        # Pre-cache all frequently used values to minimize overhead
        # Cache frequently accessed attributes and methods to reduce lookups
        x_key = x_item.key
        replica = _create_replica(x_key)
        TreeClass = type(self)
        NodeClass = self.NodeClass
        check_and_convert_set = self.check_and_convert_set
        l_factor = self.l_factor
        capacity = self.KListClass.KListNodeClass.CAPACITY
        
        # Pre-calculate rank once to avoid repeated calculations in loops
        cached_new_rank = calc_rank(x_key, capacity, dim=self.DIM + 1)

        # Parent tracking variables - optimized for memory locality
        right_parent = None
        right_entry = None
        left_parent = None
        left_x_entry = None

        while True:
            # Cache node reference and minimize repeated attribute access
            node = cur.node
            cur._invalidate_tree_size()
            
            # Cache frequently used values in local variables for better performance
            is_leaf = node.rank == 1
            node_set = node.set
            insert_obj = x_item if is_leaf else replica

            # Fast path: First iteration without splitting
            if right_parent is None:
                # Minimize conditional evaluations - use direct assignment when possible
                subtree = next_entry.left_subtree if next_entry else node.right_subtree
                insert_subtree = x_left if is_leaf else subtree

                # Optimized type check and insert - avoid function call overhead
                is_gkplus_type = isinstance(node_set, GKPlusTreeBase)
                if is_gkplus_type:
                    node_set, _ = node_set.insert(
                        insert_obj, rank=cached_new_rank, x_left=insert_subtree
                    )
                else:
                    node_set = node_set.insert(insert_obj, left_subtree=insert_subtree)
                    # Update node set and fetch new next_entry
                    node.set = check_and_convert_set(node_set)

                # Fastest return path for most common case (leaf nodes) - early exit
                if is_leaf:
                    return self, True
                
                retrieval_result = node.set.retrieve(x_key)
                next_entry = retrieval_result.next_entry

                # Setup for next iteration with optimized assignments
                right_parent = left_parent = cur
                right_entry = next_entry
                left_x_entry = retrieval_result.found_entry
                cur = subtree
                continue

            # Complex path: Node splitting required

            # Complex path: Node splitting required (less common but more expensive)
            # Cache retrieve result to avoid redundant method calls
            res = node_set.retrieve(x_key)
            next_entry = res.next_entry

            # Perform split operation and immediately cache converted results
            left_split, _, right_split = node_set.split_inplace(x_key)
            left_split = check_and_convert_set(left_split)
            right_split = check_and_convert_set(right_split)
            
            # CRITICAL: Re-fetch next_entry after check_and_convert_set (not in-place)
            next_entry = right_split.retrieve(x_key).next_entry
            
            # Cache item counts early to avoid repeated method calls in conditionals
            right_item_count = right_split.item_count()
            left_item_count = left_split.item_count()
            
            # Handle right side creation - inline optimization for performance
            new_tree = None
            if right_item_count > 0 or is_leaf:
                insert_subtree = x_left if is_leaf else None
                
                # Fast type check and optimized insert operation
                if isinstance(right_split, GKPlusTreeBase):
                    right_split, _ = right_split.insert(
                        insert_obj, rank=cached_new_rank, x_left=insert_subtree
                    )
                else:
                    right_split = right_split.insert(insert_obj, left_subtree=insert_subtree)
                
                # Create new tree node efficiently
                right_split = check_and_convert_set(right_split)
                new_tree = TreeClass(l_factor=l_factor)
                new_tree.node = NodeClass(node.rank, right_split, node.right_subtree)

            # Update next_entry after potential modification - essential for correctness
            next_entry = (right_split.retrieve(x_key).next_entry
                         if new_tree else next_entry)

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

                # Update parent reference
                if left_x_entry is not None:
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
                    cur.node.right_subtree = None  # No right subtree at leaf level
                return self, True  # Early return when leaf is processed

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

        if tree.is_empty():
            return tree

        if tree.DIM == 1:
            # We want to keep the GKPlusTree structure for dimension 1
            return tree

        # Count the tree items (including dummy items)
        item_count = tree.item_count()
        # -1 because the dummy item will be removed during conversion
        if item_count - 1 <= threshold:
            # Collapse into a KList
            return _tree_to_klist(tree)

        return tree

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

        while True:
            # Cache node reference and minimize repeated attribute access
            node = cur.node
            is_leaf = node.rank == 1

            # Split node at key - cache results immediately
            left_split, key_subtree, right_split = node.set.split_inplace(key)
            left_split = check_and_convert_set(left_split)
            
            # Cache item counts early to minimize repeated method calls
            l_count = left_split.item_count()
            r_count = right_split.item_count()

            # --- Handle right side of the split ---
            # Determine if we need a new tree for the right split
            if r_count > 0:  # incl. dummy items
                # Cache type check to avoid repeated isinstance calls
                is_gkplus_type = isinstance(right_split, GKPlusTreeBase)
                
                if is_gkplus_type:
                    # Calculate the new rank for the item in the next dimension - use cached values
                    new_rank = calc_rank(dummy_key, KListNodeCapacity, dim=tree_dim_plus_one)
                    right_split, _ = right_split.insert(dummy, rank=new_rank, x_left=None)
                else:
                    right_split = right_split.insert(dummy, left_subtree=None)

                # Convert and get next_entry in one step
                right_split = check_and_convert_set(right_split)
                retrieve_result = right_split.retrieve(key)
                next_entry = retrieve_result.next_entry
                
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

                    # Update parent reference - optimized conditional
                    if right_entry is not None:
                        right_entry.left_subtree = new_tree
                    else:
                        right_parent.node.right_subtree = new_tree

                if is_leaf:
                    # Prepare for updating 'next' pointers
                    new_tree.node.next = node.next

                # Prepare references for next iteration - minimize assignments
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
                        new_rank = calc_rank(dummy_key, KListNodeCapacity, dim=tree_dim_plus_one)
                        right_split, _ = right_split.insert(dummy, rank=new_rank, x_left=None)
                    else:
                        right_split = right_split.insert(dummy, left_subtree=None)

                    right_split = check_and_convert_set(right_split)
                    retrieve_result = right_split.retrieve(key)
                    next_entry = retrieve_result.next_entry

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
                    retrieve_result = right_split.retrieve(key)
                    next_entry = retrieve_result.next_entry
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
                        if l_count == 0:
                            # find the previous leaf node by traversing the left parent
                            if left_parent:
                                l_last_leaf = left_parent.get_max_leaf()
                            else:
                                left_return.node = seen_key_subtree.node
                                l_last_leaf = seen_key_subtree.get_max_leaf()
                        else:
                            node.set = left_split
                            l_last_leaf = node
                    else:
                        # No non-dummy entry in left tree
                        self.node = None
                        # left_return = self
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

            # Update leaf node 'next' pointers if at leaf level - early return
            if is_leaf:
                # Unlink leaf nodes
                if l_last_leaf:
                    l_last_leaf.next = None

                # prepare key entry subtree for return
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
            klist = klist.insert(entry.item, entry.left_subtree)
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

    entries = list(klist)
    tree = create_gkplus_tree(K, DIM, l_factor)
    ranks = [] 
    for entry in entries:
        ranks.append(calc_rank(entry.item.key, K, DIM))

    # Insert entry instances with their ranks
    tree_insert = tree.insert
    for i, entry in enumerate(entries):
        if i < len(ranks):
            rank = int(ranks[i])
            tree, _ = tree_insert(entry.item, rank, entry.left_subtree)
    return tree
