from typing import List
from gplus_trees.klist_base import KListBase
import numpy as np
import hashlib

from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.g_k_plus.rank_utils import calc_rank, get_rand_rank, calc_rank_from_digest, calc_ranks_for_multiple_dimensions

from tests.logconfig import logger

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

        # Insert items with keys larger than current trees dummy key into the new klist
        # Smaller keys are dummies from higher dimensions, caused by gnode expansions within the tree to collapse. These are dropped.
        # Larger dummy keys are from lower dimensions, that must be preserved.
        # Note: dummy key of DIM j is -j.
        for entry in tree:
            tree_dummy = get_dummy(tree.DIM)
            if entry.item.key > tree_dummy.key:
                klist = klist.insert(entry.item, entry.left_subtree)
        return klist

def klist_to_tree(klist, K, DIM, l_factor=1.0) -> 'GKPlusTreeBase':
    """
    Mimics the Rust create_gtree: build a tree by inserting each (item, rank) pair.
    Uses the factory pattern to create a tree with the specified capacity K.
    """
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

    # TODO: Adjust to insert entry instances with their ranks
    tree_insert = tree.insert
    for i, entry in enumerate(entries):
        if i < len(ranks):
            rank = int(ranks[i])
            tree, _ = tree_insert(entry.item, rank, entry.left_subtree)
    return tree