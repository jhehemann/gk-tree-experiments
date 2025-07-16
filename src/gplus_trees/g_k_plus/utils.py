"""Pure utility functions for rank calculation without circular dependencies."""

import hashlib
from typing import List, Union
from gplus_trees.utils import get_group_size, calc_rank_from_digest


def get_digest(hash_input: Union[int, bytes], dim: int) -> bytes:
    """
    Generates a SHA-256 digest from an input value and a dimension integer.
    
    Args:
        hash_input (Union[int, bytes]): The input value to hash
        dim (int): The dimension level to incorporate into the digest.
    
    Returns:
        bytes: The resulting SHA-256 digest of the input value and the dimension.
    
    Raises:
        TypeError: If hash_input is not of type int or bytes.
    """
    if isinstance(hash_input, bytes):
        digest = hashlib.sha256(
            hash_input + int(dim).to_bytes(32, 'big')
        ).digest()
    elif isinstance(hash_input, int):
        digest = hashlib.sha256(
            abs(hash_input).to_bytes(32, 'big') + int(dim).to_bytes(32, 'big')
        ).digest()
    else:
        raise TypeError("key_or_digest must be int or bytes")
    
    return digest


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
