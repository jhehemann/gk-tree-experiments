from typing import TYPE_CHECKING, Any
from gplus_trees.klist_base import KListBase
from gplus_trees.base import Item
import numpy as np

# Use TYPE_CHECKING to avoid circular imports at runtime

from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import print_pretty


def tree_to_klist(tree: 'GKPlusTreeBase') -> KListBase:
        """
        Convert this GKPlusTree to a KList.
        
        Returns:
            A new KList containing all items from this tree
        """
        # Import inside function to avoid circular imports
        from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
        
        if not isinstance(tree, GKPlusTreeBase):
            raise TypeError("tree must be an instance of GKPlusTreeBase")

        if tree.is_empty():
            return tree.KListClass()
        
        klist = tree.KListClass()
        # Process leaf nodes to build the KList
        for entry in tree:
            # Insert each item into the KList
            klist = klist.insert(entry.item, entry.left_subtree)
        return klist

def klist_to_tree(klist, K, DIM):
    """
    Mimics the Rust create_gtree: build a tree by inserting each (item, rank) pair.
    Uses the factory pattern to create a tree with the specified capacity K.
    """    
    if not isinstance(klist, KListBase):
        raise TypeError("klist must be an instance of KListBase")
    
    if klist.is_empty():
        return create_gkplus_tree(K, DIM)
    
    entries = list(klist)
    tree = create_gkplus_tree(K, DIM)
    
    item_count = len(entries)

    # dummy item count
    dummy_count = sum(1 for entry in entries if entry.item.key < 0)
    # print(f"Dummy count: {dummy_count}, Item count: {item_count}")
    # Use rank 1 for dummy items
    dum_ranks = np.ones(dummy_count, dtype=int)
    # print(f"Generated {dummy_count} dummy ranks of ones: {dum_ranks}")

    p = 1.0 - (1.0 / (K))
    ranks = np.random.geometric(p, size=item_count-dummy_count)
    ranks = np.concatenate((dum_ranks, ranks))

    # print(f"Ranks after appending ones: {len(ranks)}, {ranks}")
    # print(f"Entries: {list(entry.item.key for entry in entries)}")

    
    tree_insert = tree.insert
    for i, entry in enumerate(entries):
        if i < len(ranks):
            # Convert numpy.int64 to Python int to avoid TypeError
            rank = int(ranks[i])
            tree, _ = tree_insert(entry.item, rank)
    return tree