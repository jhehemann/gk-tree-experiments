"""Pure utility functions for rank calculation without circular dependencies."""

import hashlib
from typing import List
from gplus_trees.utils import calculate_group_size, calc_rank_from_digest


def calc_rank_for_dim(key: int, k: int, dim: int) -> int:
    """
    Calculate rank for a key at a given dimension. If a key is negative, its absolute value is used for hashing to make dummy keys hashable.
    
    Args:
        key: The key to calculate rank for
        k: The capacity parameter
        dim: The dimension level
        
    Returns:
        The calculated rank
    """
    group_size = calculate_group_size(k)
    # Use keys' absolute value to hash to account for dummy keys (in testing)
    digest = hashlib.sha256(abs(key).to_bytes(32, 'big')).digest()
    for _ in range(dim - 1):
        # Rehash the digest for each dimension
        digest = hashlib.sha256(digest).digest()
    return calc_rank_from_digest(digest, group_size)


def calc_rank_from_group_size(key: int, group_size: int, dim: int = 1) -> int:
    """
    Calculate the rank for a key using its group size.

    Args:
        key: The key to calculate rank for
        group_size: The size of the group (log2(k))
        dim: The dimension level

    Returns:
        The calculated rank
    """
    # Use keys' absolute value to hash to account for dummy keys (in testing)
    digest = hashlib.sha256(abs(key).to_bytes(32, 'big')).digest()
    for _ in range(dim - 1):
        # Rehash the digest for each dimension
        digest = hashlib.sha256(digest).digest()
    return calc_rank_from_digest(digest, group_size)


def calc_ranks(entries: List[int], group_size: int, DIM: int) -> List[int]:
    """
    Calculate ranks for a list of keys based on hashing.
    
    Parameters:
        keys (List[int]): List of integer keys to calculate ranks for.
        k (int): Must be a power of 2, used to derive group size.

    Returns:
        List[int]: Ranks for each key.
    """
    ranks = []
    for entry in entries:
        ranks.append(calc_rank_from_group_size(entry.item.key, group_size, DIM))
    return ranks


def calc_ranks_multi_dims(keys: List[int], k: int, dimensions: int = 1) -> List[List[int]]:
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
