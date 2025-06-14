from typing import List
from gplus_trees.klist_base import KListBase
import numpy as np
import hashlib

from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import get_dummy
from gplus_trees.base import calculate_group_size, count_trailing_zero_bits

from tests.logconfig import logger

def get_rand_rank(K: int) -> int:
    """
    Generate a random rank for an item in a GKPlus tree.
    
    Args:
        K: The capacity of the KList, used to determine the probability distribution.
        
    Returns:
        A random rank as an integer.
    """
    p = 1.0 - (1.0 / K)
    return np.random.geometric(p)

def calc_rank_from_digest(digest: bytes, group_size: int) -> int:
    tz = count_trailing_zero_bits(digest)
    return (tz // group_size) + 1

def calc_rank(key: int, k: int, dim: int) -> int:
    group_size = calculate_group_size(k)
    # Use keys' absolute value to hash to account for dummy keys (in testing)
    digest = hashlib.sha256(abs(key).to_bytes(32, 'big')).digest()
    for _ in range(dim - 1):
        # Rehash the digest for each dimension
        digest = hashlib.sha256(digest).digest()
    return calc_rank_from_digest(digest, group_size)

def calc_ranks_for_multiple_dimensions(keys: List[int], k: int, dimensions: int = 1) -> List[List[int]]:
    """
    Calculate ranks for a list of keys based on repeated hashing.
    
    Parameters:
        keys (List[int]): List of integer keys to calculate ranks for.
        k (int): Must be a power of 2, used to derive group size.
        dimensions (int): Number of hashing levels to apply.

    Returns:
        List[List[int]]: Ranks for each dimension, where each inner list contains ranks for all keys at that dimension.
    """
    group_size = calculate_group_size(k)
    # Initialize a list for each dimension
    rank_lists = [[] for _ in range(dimensions)]

    for key in keys:
        current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
        for dim in range(dimensions):
            rank = calc_rank_from_digest(current_hash, group_size)
            rank_lists[dim].append(rank)
            current_hash = hashlib.sha256(current_hash).digest()

    return rank_lists

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