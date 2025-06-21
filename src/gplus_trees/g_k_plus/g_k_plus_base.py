"""GKPlusTree base implementation"""

from __future__ import annotations
from typing import Optional, Type, TypeVar, Tuple, List

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
from gplus_trees.g_k_plus.utils import calc_ranks, calc_rank_for_dim, calculate_group_size

t = TypeVar('t', bound='GKPlusTreeBase')

DEFAULT_DIMENSION = 1  # Default dimension for GKPlusTree
DEFAULT_L_FACTOR = 1.0  # Default threshold factor for KList to GKPlusTree conversion

from gplus_trees.g_k_plus.base import logger

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
    __slots__ = GPlusTreeBase.__slots__ + ("l_factor", "item_cnt", "size")  # Add new slots

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

    def __str__(self):
        if self.is_empty():
            return "Empty GKPlusTree"
        return f"GKPlusTree(dim={self.__class__.DIM}, node={self.node})"

    __repr__ = __str__

    def item_count(self) -> int:
        if self.item_cnt is None:
            return self.get_tree_item_count()
        return self.item_cnt
    
    def real_item_count(self) -> int:
        if self.size is None:
            return self.get_size()
        return self.size

    def item_count_geq_dummy_leq_n(self, set: GKPlusTreeBase, n: int) -> bool:
        """
        Check if the item count of items with keys >= dummy is equal to or smaller than n.

        Args:
            n (int): The threshold count to compare against.

        Returns:
            bool: True if item count <= n, False otherwise.
        """
        if set.is_empty():
            return True

        dummy = get_dummy(set.DIM).key
        count = 0
        current = None
        for entry in set:
            current = entry
            if current.item.key > dummy:
                count += 1
            if count > n:
                return False  # Early exit if count exceeds n
        return True

    def get_tree_item_count(self) -> int:
        """Get the number of items in the tree, including dummy items."""
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
                count += (entry.left_subtree.get_tree_item_count()
                         if entry.left_subtree is not None else 0)

            count += (self.node.right_subtree.get_tree_item_count()
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

    def update(self, cur, x_entry) -> Tuple[GKPlusTreeBase, bool]:
        """
        Update the tree after an insertion or deletion.
        This method recalculates the item count and size of the tree.
        
        Returns:
            Tuple[GKPlusTreeBase, bool]: The updated tree and a boolean indicating success.
        """
        return self, False
        # raise ValueError(
        #     f"Entry with key {x_entry.item.key} already exists in the "
        #     "tree. Cannot be updated at this point."
        #     )
        # if x_entry.left_subtree is not None:
        #     raise ValueError(
        #         f"Entry with key {x_entry.item.key} already exists in the "
        #         "tree. Cannot be inserted with a subtree again."
        #     )
        # # Direct update for leaf nodes (common case)
        # if rank == 1:
        #     existing_x_entry.item.value = x_entry.item.value
        #     return self, False
        # return self._update_existing_item(cur, x_entry.item)

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
        
        cur = self
        parent = None
        p_next_entry = None

        # path cache
        path_cache = []

        # Loop until we find where to insert
        while True:
            path_cache.append(cur)
            node = cur.node
            node_rank = node.rank  # Cache attribute access

            # Case 1: Found node with matching rank - ready to insert
            if node_rank == rank:
                return self._insert_new_item(cur, x_entry, path_cache)

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
        # Set replica of the current node set's pivot as first entry.
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
        path_cache: list[GPlusNodeBase],
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
        logger.debug(
            f"[DIM {self.DIM}] [INSERTING {x_key} into tree: {print_pretty(self)}]"
        )

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
                logger.debug(
                    f"[DIM {self.DIM}] [INSERTING {x_key} into node: {print_pretty(node.set)}]"
                )

                if not is_gkplus_type:
                    node.set, inserted = node.set.insert_entry(insert_entry)
                    if not inserted:
                        return self.update(cur, x_entry)
                    logger.debug(
                        f"[DIM {self.DIM}] [Node before conversion check: {print_pretty(node.set)}]"
                    )
                    node.set = check_and_convert_set(node.set) # only KLists can be extended
                    logger.debug(
                        f"[DIM {self.DIM}] [Node after conversion check: {print_pretty(node.set)}]"
                    )
                else:
                    new_rank = calc_rank_for_dim(x_key, capacity, dim=self.DIM + 1)
                    node.set, inserted = node.set.insert_entry(insert_entry, rank=new_rank)
                    if not inserted:
                        return self.update(cur, x_entry)
                    
                logger.debug(
                    f"[DIM {self.DIM}] [INSERTED {x_key} into node: {print_pretty(node.set)}]"
                )
                
                # Item will be inserted, add 1 to each node's size so far
                for tree in path_cache:
                    if tree.size is not None:
                        tree.size += 1

                # Fastest path for leaf nodes - direct return
                if is_leaf:                    
                    self._invalidate_tree_size()
                    logger.debug(
                        f"[DIM {self.DIM}] [INSERTED {x_key} into tree: {print_pretty(self)}]"
                    )
                    logger.debug(
                        f"[DIM {self.DIM}] [INSERTED {x_key} into tree structure: {self.print_structure()}]"
                    )
                    return self, True

                # retrieval_result = node.set.retrieve(x_key)

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
            left_split = check_and_convert_set(left_split)
            # right_split = check_and_convert_set(right_split)
            
            # Cache item counts early to avoid repeated method calls in conditionals
            right_item_count = right_split.item_count()
            left_item_count = left_split.item_count()
            
            # Handle right side creation - inline optimization for performance
            new_tree = None
            if right_item_count > 0 or is_leaf:
                insert_entry = x_entry if is_leaf else Entry(replica, None)
                if isinstance(right_split, GKPlusTreeBase):
                    new_rank = calc_rank_for_dim(x_key, capacity, dim=self.DIM + 1)
                    right_split, _ = right_split.insert_entry(insert_entry, rank=new_rank)
                else:
                    right_split, _ = right_split.insert_entry(insert_entry)

                # Create new tree node
                right_split = check_and_convert_set(right_split)
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
                self._invalidate_tree_size()
                logger.debug(
                    f"[DIM {self.DIM}] [INSERTED {x_key} into tree: {print_pretty(self)}]"
                )
                logger.debug(
                        f"[DIM {self.DIM}] [INSERTED {x_key} into tree structure: {self.print_structure()}]"
                    )
                return self, True  # Early return when leaf is processed

            # Continue to next iteration with updated current node
            cur = next_cur

    # def print_subtree_sizes(self):
    #     """
    #     Check the subtree sizes in the tree.

    #     Returns:
    #         bool: True if the node counts are consistent, False otherwise.
    #     """
    #     # Check if the node counts are consistent
    #     print(f"Subtree at rank {self.node.rank} "
    #           f"has {self.node.set.item_count()} entries, "
    #           f"size: {self.node.get_size()}")

    #     for entry in self.node.set:
    #         if entry.left_subtree is not None:
    #             entry.left_subtree.print_subtree_sizes()

    #     if self.node.right_subtree is not None:
    #         self.node.right_subtree.print_subtree_sizes()
    #     return True

    # Extension methods to check threshold and perform conversions
    

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
            logger.debug(
                f"[DIM {self.DIM}] [SPLIT {key}] Left before conversion: {print_pretty(left_split)}"
            )
            left_split = check_and_convert_set(left_split)
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

        if tree.real_item_count() <= threshold:
            # Collapse into a KList
            logger.debug(f"[COLLAPSE] Tree {print_pretty(tree)} has {tree.real_item_count()} real items, which is <= {threshold}, collapsing to KList")
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
        logger.debug(f"[TREE TO KLIST] Processing entry {entry.item.key} with dummy {tree_dummy.key}")
        if entry.item.key > tree_dummy.key:
            logger.debug(f"[TREE TO KLIST] Inserting entry {entry.item.key} as it is larger than the tree's dummy key {tree_dummy.key}")
            klist, _ = klist.insert_entry(entry)
            logger.debug(f"[TREE TO KLIST] KList after insertion: {print_pretty(klist)}")
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

    return bulk_create_gkplus_tree(klist, DIM, l_factor)
    
    # entries = list(klist)
    # tree = create_gkplus_tree(K, DIM, l_factor)
    # ranks = [] 
    # for entry in entries:
    #     ranks.append(calc_rank_for_dim(entry.item.key, K, DIM))

    # # Insert entry instances with their ranks
    # tree_insert_entry = tree.insert_entry
    # for i, entry in enumerate(entries):
    #     if i < len(ranks):
    #         rank = int(ranks[i])
    #         tree, _ = tree_insert_entry(entry, rank)
    # return tree


def _create_node_from_entries(
    entries: List[Entry],
    rank: int,
    KListClass: type[KListBase],
    NodeClass: type[GKPlusNodeBase],
    DIM: int,
    l_factor: float = 1.0,
) -> GKPlusNodeBase:
    """Create a GKPlusTreeBase node from a list of entries."""
    k = KListClass.KListNodeClass.CAPACITY
    threshold = int(k * l_factor)
    if len(entries) <= threshold:
        node_set = KListClass()
        for entry in entries:
            node_set, _ = node_set.insert_entry(entry)
    else:
        # Recursive case 'Dimension': create a GKPlusTree with the next higher dimension
        # Instantiate node set with the resulting tree
        group_size = calculate_group_size(k)
        node_set, _ = _create_gkplus_tree_from_entries(entries, group_size, KListClass, DIM + 1, l_factor)
        logger.debug(f"[CREATE NODE] Created node set from entries {[entry.item.key for entry in entries]} to {print_pretty(node_set)}")
    return NodeClass(rank, node_set, None)


def create_gkplus_tree_rec(
    pairs: List[Tuple[Entry, int]],
    KListClass: type[KListBase],
    DIM: int,
    l_factor: float = 1.0,
    prev_leaf: Optional[GKPlusTreeBase] = None,
) -> GKPlusTreeBase:
    """ Create a GKPlusTree recursively from a list of (entry, rank) pairs.
    Args:
        pairs: A list of tuples (Entry, rank) - sorted by rank descending (1) and key ascending (2)
        K: The capacity parameter for the tree
        DIM: The dimension of the tree
        l_factor: The threshold factor for conversion
        add_dummy: Whether to add a dummy entry to the current pairs list
    Returns:
        A GKPlusTreeBase instance containing the entries
    """
    if not pairs:
        return None

    # Get the maximum rank from the pairs
    # Ensure it is at least 1 for the case where the pivot entry is the only entry
    max_rank = max(1, max(rank for _, rank in pairs))
    k = KListClass.KListNodeClass.CAPACITY
    node_set: Optional[AbstractSetDataStructure] = None
    tree = _get_create_gkplus_tree()(k, DIM, l_factor) 
    NodeClass = tree.NodeClass

    # set the pivot entry's rank to the maximum rank
    entry0, _ = pairs[0]
    pairs[0] = (entry0, max_rank)
    logger.debug(f"[REC CREATE NEW] Creating tree with keys {[pair[0].item.key for pair in pairs]} and ranks {[pair[1] for pair in pairs]}")

    # Base case: if the maximum rank is 1, create a leaf node
    if max_rank == 1:
        # Create a leaf node with the entries
        entries = [entry for entry, _ in pairs]
        node = _create_node_from_entries(entries, 1, KListClass, NodeClass, DIM, l_factor)
        if prev_leaf is not None:
            # Link the previous leaf to the new leaf
            prev_leaf.node.next = tree
            
            logger.debug(f"[REC CREATE] Linking previous leaf {print_pretty(prev_leaf)} to new leaf {tree.__class__.__name__} (Instance to be filled with entries)") 
        prev_leaf = tree
        tree.node = node
        # logger.debug(f"[REC CREATE FINISHED] Created tree: {print_pretty(tree)}")
        return tree, prev_leaf

    # Recursive case 'Subtree': create root, split pairs based on keys and create subtrees
    max_rank_entries: List[Entry] = [] # will become the root entries
    subtrees_pairs: List[List[Tuple[Entry, int]]] = [[]] # lower tree level subtree entries

    # Prepare node and subtree entries
    logger.debug(f"[REC CREATE] Preparing subtree pairs")
    subtree_idx = 0
    for pair in pairs:
        entry, rank = pair
        if rank < max_rank: # items in the left subtree of a higher rank item are strictly smaller
            logger.debug(f"[REC CREATE] Adding entry {entry.item.key} with rank {rank} to subtree {subtree_idx}")
            subtrees_pairs[subtree_idx].append(pair)
        else:
            logger.debug(f"[REC CREATE] Adding replica of entry {entry.item.key} with rank {rank} to max rank entries")
            # The item has maximum rank and will become a root entry
            replica = _create_replica(entry.item.key) # Non-leaf nodes contain replicas
            max_rank_entries.append(Entry(replica, None)) 
            
            # The item also becomes the pivot for the next subtree
            pair = (entry, 0) # Reset rank to 0 to be determined by the next subtrees max rank
            subtrees_pairs.append([pair]) # start a new subtree list with the pivot entry
            logger.debug(f"[REC CREATE] Starting new subtree with pivot entry {pair[0].item.key} and default rank {pair[1]}")
            subtree_idx += 1
            
    
    logger.debug(f"[REC CREATE] Max rank entries initialized: {[entry.item.key for entry in max_rank_entries]}")

    for i, subtree_pairs in enumerate(subtrees_pairs[:-1]):
        logger.debug(f"[REC CREATE] Subtree pairs for entry {max_rank_entries[i].item.key}: {[pair[0].item.key for pair in subtree_pairs]} with ranks {[pair[1] for pair in subtree_pairs]}")
    logger.debug(f"[REC CREATE] Finished preparing subtree pairs")

    # Attach subtrees to the max rank entries
    logger.debug(f"[REC CREATE] Creating and attaching subtrees to max rank entries")
    for i, entry in enumerate(max_rank_entries):
        if i < len(subtrees_pairs):
            subtree_pairs = subtrees_pairs[i]
            if subtree_pairs:
                # Create a GKPlusTree for the subtree
                logger.debug(f"[REC CREATE] Creating subtree for entry {entry.item.key} with pairs: {[pair[0].item.key for pair in subtree_pairs]} and ranks {[pair[1] for pair in subtree_pairs]}")
                subtree_tree, prev_leaf = create_gkplus_tree_rec(
                    subtree_pairs,
                    KListClass,
                    DIM,
                    l_factor,
                    prev_leaf=prev_leaf
                )
                entry.left_subtree = subtree_tree
    
    # Create the root node with max rank entries
    logger.debug(f"[REC CREATE] Creating root node with max rank entries: {[entry.item.key for entry in max_rank_entries]}")
    root_node = _create_node_from_entries(
        max_rank_entries,
        max_rank,
        KListClass,
        NodeClass,
        DIM,
        l_factor,
    )

    # Create a new GKPlusTreeBase instance from the entries in the last subtrees_pairs list and assign it to the node's right subtree
    # Reset pivot rank in right subtree to 0 to be determined by the next subtrees max rank
    r_subtree_pairs = subtrees_pairs[-1]
    entry0, _ = r_subtree_pairs[0] 
    r_subtree_pairs[0] = (entry0, 0)
    
    logger.debug(f"[REC CREATE] Creating right subtree with pairs: {[pair[0].item.key for pair in r_subtree_pairs]} and ranks {[pair[1] for pair in r_subtree_pairs]}")
    right_subtree, prev_leaf = create_gkplus_tree_rec(
        r_subtree_pairs,
        KListClass,
        DIM,
        l_factor, 
        prev_leaf=prev_leaf
    )

    root_node.right_subtree = right_subtree
    tree.node = root_node
    logger.debug(f"[REC CREATE FINISHED] Created tree: {print_pretty(tree)}")
    return tree, prev_leaf

def _create_gkplus_tree_from_entries(
    entries: list[Entry],
    group_size: int,
    KListClass: type[KListBase],
    DIM: int,
    l_factor: float,
) -> GKPlusTreeBase:
    """
    Create a GKPlusTree from a list of (entry, rank) pairs.
    
    Args:
        pairs: A list of tuples (Entry, rank)
        K: The capacity parameter for the tree
        DIM: The dimension of the tree
        l_factor: The threshold factor for conversion
        
    Returns:
        A new GKPlusTreeBase instance containing the entries
    """
    ranks = calc_ranks(entries, group_size, DIM)
    key_2_dim_1_rank = calc_rank_for_dim(entries[0].item.key, KListClass.KListNodeClass.CAPACITY, DIM)
    logger.debug(f"[CREATE] Creating GKPlusTree with entries {[entry.item.key for entry in entries]} and {ranks} ranks for dim {DIM} with l_factor {l_factor}")
    pairs = list(zip(entries, ranks))
    
    # TODO: Find a way to avoid O(len(pairs)) for dummy insertion 
    # Prepend with pivot entry, which has always the lowest key and max rank and no left subtree
    pivot = Entry(get_dummy(DIM), None)  # Set dummy as the pivot for a new tree
    logger.debug(f"[PIVOT] Set dummy as pivot for dim {DIM}: {pivot.item.key} with rank 0")
    logger.debug(f"[PIVOT] Pairs before adding pivot: {pairs}")
    pairs.insert(0, (pivot, 0))  # Set the rank to default value 0 – will be determined later
    logger.debug(f"[PIVOT] Pairs after adding pivot: {pairs}")
    
    return create_gkplus_tree_rec(pairs, KListClass, DIM, l_factor)

def bulk_create_gkplus_tree(
    klist: KListBase,
    DIM: int,
    l_factor: float,
) -> GKPlusTreeBase:
    """
    Create a new GKPlusTree with the specified parameters.
    
    Args:
        K: The capacity parameter for the tree
        DIM: The dimension of the tree
        l_factor: The threshold factor for conversion
        
    Returns:
        A new GKPlusTreeBase instance
    """
    KListClass = type(klist)
    k = KListClass.KListNodeClass.CAPACITY
    if klist.is_empty():
        tree = _get_create_gkplus_tree()(k, DIM, l_factor)
        return tree

    group_size = calculate_group_size(k)
    entries = list(klist)
    tree, _ = _create_gkplus_tree_from_entries(entries, group_size, KListClass, DIM, l_factor)
    
    return tree