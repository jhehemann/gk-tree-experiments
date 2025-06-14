"""Pure utility functions for rank calculation without circular dependencies."""

import hashlib
import numpy as np
from typing import List
from gplus_trees.base import calculate_group_size, count_trailing_zero_bits


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
    """Calculate rank from a hash digest."""
    tz = count_trailing_zero_bits(digest)
    return (tz // group_size) + 1


def calc_rank(key: int, k: int, dim: int) -> int:
    """
    Calculate rank for a key at a given dimension.
    
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
