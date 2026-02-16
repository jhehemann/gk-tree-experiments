"""Pure utility functions for rank calculation without circular dependencies."""

from typing import List

from gplus_trees.utils import get_group_size, calc_rank_from_digest, get_digest

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
    # Start with dimension 1: hash(abs(key) + 1)
    digest = get_digest(key, 1)

    # For subsequent dimensions, hash(prev_digest + dim)
    for d in range(2, dim + 1):
        digest = get_digest(digest, d)

    return calc_rank_from_digest(digest, group_size)


def calc_rank(key: int, k: int, dim: int) -> int:
    """
    Calculate the rank for a key for a specific dimension based on repeated hashing.

    Args:
        key (int): The key to calculate rank for.
        k (int): The K-list node capacity parameter (must be a power of 2 to derive group size).
        dim (int): The dimension level to calculate rank for.

    Returns:
        int: The calculated rank in the specified dimension.
    """
    group_size = get_group_size(k)
    return calc_rank_from_group_size(key, group_size, dim)

def calc_ranks(keys: List[int], k: int, DIM: int = 1) -> List[int]:
    """
    Calculate ranks for a list of keys for a specific dimension based on repeated hashing.

    Args:
        keys (List[int]): List of integer keys to calculate ranks for.
        k (int): The K-list node capacity parameter (must be a power of 2 to derive group size).
        DIM (int): The dimension to calculate ranks for.

    Returns:
        List[int]: Rank for each key in the specified dimension.
    """

    group_size = get_group_size(k)
    ranks = []
    for key in keys:
        ranks.append(calc_rank_from_group_size(key, group_size, DIM))
    return ranks


def calc_ranks_multi_dims(keys: List[int], k: int, dimensions: int = 1) -> List[List[int]]:
    """
    Calculate ranks for a list of keys based on repeated hashing.

    Args:
        keys (List[int]): List of integer keys to calculate ranks for.
        k (int): The K-list node capacity parameter (must be a power of 2 to derive group size).
        dimensions (int): Number of hashing levels to apply.

    Returns:
        List[List[int]]: Ranks for each dimension, where each inner list contains ranks for all keys at that dimension.
    """
    group_size = get_group_size(k)
    num_keys = len(keys)
    rank_lists = [[0] * num_keys for _ in range(dimensions)] # Initialize rank_lists[dim][key_idx]

    for key_idx, key in enumerate(keys):
        current_hash = get_digest(key, 1)
        rank_lists[0][key_idx] = calc_rank_from_digest(current_hash, group_size)
        for dim in range(1, dimensions):
            current_hash = get_digest(current_hash, dim + 1)
            rank_lists[dim][key_idx] = calc_rank_from_digest(current_hash, group_size)

    return rank_lists
