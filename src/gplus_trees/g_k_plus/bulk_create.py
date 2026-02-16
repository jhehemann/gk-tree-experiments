"""Bulk creation and conversion utilities for GK+-trees.

Provides:
- ``bulk_create_gkplus_tree`` – bottom-up bulk creation from an entry sequence.
- ``_tree_to_klist`` / ``_klist_to_tree`` – bidirectional conversions.
- ``_bulk_create_klist`` – helper to populate a KList from entries.

Internal helpers ``_build_leaf_level_trees`` and ``_build_internal_levels``
are used by ``bulk_create_gkplus_tree`` and are not part of the public API.

Complexity summary (n = entries, k = KList capacity, h = resulting height):

+-------------------------------+------------------------------------------+
| Operation                     | Time                                     |
+===============================+==========================================+
| ``bulk_create_gkplus_tree``   | O(n · h) amortised; leaf sets that       |
|                               | exceed the threshold are recursively     |
|                               | bulk-created at dimension d+1.           |
| ``_tree_to_klist``            | O(n_{d+1}) at dimension d+1; iterates    |
|                               | the inner tree via ``__iter__`` which    |
|                               | recurses through all nested dimensions.  |
| ``_klist_to_tree``            | O(n · h_{d+1}) (delegates to bulk)       |
| ``_bulk_create_klist``        | O(n) (sorted inserts → appends)          |
+-------------------------------+------------------------------------------+
"""

from __future__ import annotations
from typing import Optional, Union
from itertools import islice

from gplus_trees.utils import calc_rank_from_digest_k
from gplus_trees.base import (
    LeafItem,
    InternalItem,
    DummyItem,
    Entry,
    _get_replica,
)
from gplus_trees.klist_base import KListBase
from gplus_trees.gplus_tree_base import get_dummy


# ---------------------------------------------------------------------------
# RankData – bookkeeping for bulk tree construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lazy factory import (avoids circular dependency)
# ---------------------------------------------------------------------------

_create_gkplus_tree = None


def _get_create_gkplus_tree():
    """Lazy import of ``create_gkplus_tree`` to avoid circular imports while caching for performance."""
    global _create_gkplus_tree
    if _create_gkplus_tree is None:
        from gplus_trees.g_k_plus.factory import create_gkplus_tree
        _create_gkplus_tree = create_gkplus_tree
    return _create_gkplus_tree


# ---------------------------------------------------------------------------
# KList ↔ Tree conversions
# ---------------------------------------------------------------------------

def _tree_to_klist(tree) -> KListBase:
    """Convert a GKPlusTree to a KList.

    Args:
        tree: The GKPlusTree to convert.

    Returns:
        A new KList containing all items from the tree.
    """
    # Import here to avoid circular dependency at module level
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

    if not isinstance(tree, GKPlusTreeBase):
        raise TypeError("tree must be an instance of GKPlusTreeBase")

    if tree.is_empty():
        return tree.KListClass()

    klist = tree.KListClass()

    # Insert items with keys larger than current tree's dummy key into the new klist.
    # Smaller keys are dummies from higher dimensions, caused by gnode expansions
    # within the tree to collapse.  These are dropped.
    # Larger dummy keys are from lower dimensions and must be preserved.
    # Note: dummy key of DIM j is -j.
    tree_dummy_key = get_dummy(tree.DIM).key
    for entry in tree:
        if entry.item.key > tree_dummy_key:
            klist, _, _ = klist.insert_entry(entry)
    return klist


def _klist_to_tree(klist: KListBase, K: int, DIM: int, l_factor: float = 1.0):
    """Convert a KList to a GKPlusTree by extracting its entries and creating a new tree.

    Args:
        klist: The KList to convert.
        K: The KList capacity (number of items per klist node).
        DIM: The dimension for the new tree.
        l_factor: The threshold factor for conversion.

    Returns:
        A new GKPlusTree containing all items from the KList.
    """
    if not isinstance(klist, KListBase):
        raise TypeError("klist must be an instance of KListBase")

    if klist.is_empty():
        return _get_create_gkplus_tree()(K, DIM, l_factor)
    return bulk_create_gkplus_tree(klist, DIM, l_factor, type(klist))


# ---------------------------------------------------------------------------
# Bulk creation helpers
# ---------------------------------------------------------------------------

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
    NodeClass,
    TreeClass,
    l_factor: float,
) -> list:
    """Build leaf level trees from rank data for rank 1.

    Args:
        rank_data_map: Consolidated rank data map containing entries and boundaries.
        KListClass: The KList class to use for creating klists.
        NodeClass: The node class to use for creating nodes.
        TreeClass: The tree class to use for creating trees.
        l_factor: The threshold factor for conversion.

    Returns:
        A list of tree instances representing the leaf level trees.
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
    leaf_trees: list,
    KListClass: type[KListBase],
    TreeClass,
    NodeClass,
    threshold: int,
    l_factor: float,
    max_rank: int,
):
    """Build internal levels of the GKPlusTree from rank data and leaf level trees.

    Args:
        rank_data_map: Consolidated rank data map containing entries, boundaries, and child indices.
        leaf_trees: List of leaf trees created.
        TreeClass: The tree class to use for creating trees.
        NodeClass: The node class to use for creating nodes.
        KListClass: The KList class to use for creating klists.
        threshold: The threshold for node set type (KList vs GKPlusTree).
        l_factor: The threshold factor for conversion.
        max_rank: The maximum rank to process.

    Returns:
        A tree instance representing the root of the tree.
    """
    rank_trees_map: dict = {}
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
                    # We can use the next child index because we lifted the earlier
                    # missing subtrees from the lower ranks
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def bulk_create_gkplus_tree(
    entries: Union[list[Entry], KListBase],
    DIM: int,
    l_factor: float,
    KListClass: type[KListBase],
):
    """Bottom-up bulk creation of a GKPlusTree from a list of entries.

    Key insights:
        1. All entries exist in leaf nodes.
        2. Rank determines node boundaries: higher rank = start of new node.
        3. Higher rank entries are replicated upward to their rank level.
        4. Threshold determines KList vs GKPlusTree for node implementation.
        5. Entries are already sorted from KList iteration.

    Args:
        entries: KList or Python List of Entry objects to be inserted into the tree.
        DIM: The dimension of the tree.
        l_factor: The threshold factor for conversion.
        KListClass: The KList class to use for creating KLists.

    Returns:
        A new GKPlusTreeBase instance containing the entries.
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

    return root_tree
