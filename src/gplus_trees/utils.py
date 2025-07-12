"""
Utility functions for GPlusTree operations, including key discovery based on hash ranks.
"""
import hashlib
from typing import List


def count_trailing_zero_bits(digest: bytes) -> int:
    tz = 0
    # look at each byte, starting from least-significant (rightmost)
    for byte in reversed(digest):
        if byte == 0:
            tz += 8
        else:
            # (byte & -byte) isolates the lowest set bit, 
            # bit_length()-1 gives its zero-based position within the byte
            tz += (byte & -byte).bit_length() - 1
            break
    return tz


def calculate_group_size(k: int) -> int:
    """
    Calculate the group size of trailing zero-groupings of an item key's hash to count based on an expected gplus node size k (power of 2).
    
    Parameters:
        k (int): The expected gplus node size, must be a positive power of 2.
    
    Returns:
        int: The group size, which is log2(k).
    
    Raises:
        ValueError: If k is not a positive power of 2.
    """
    if k <= 0 or (k & (k - 1)) != 0:
        raise ValueError("k must be a positive power of 2")
    return k.bit_length() - 1


def calc_rank_from_digest(digest: bytes, group_size: int) -> int:
    """Calculate rank from a hash digest."""
    tz = count_trailing_zero_bits(digest)
    return (tz // group_size) + 1

def calc_rank_from_digest_k(digest: bytes, k: int) -> int:
    """Calculate rank from a hash digest based on the group size derived from k."""
    group_size = calculate_group_size(k)
    return calc_rank_from_digest(digest, group_size)


def find_keys_for_rank_lists(rank_lists: List[List[int]], k: int, spacing: bool = False) -> List[int]:
    """Find keys whose repeated hashes match the rank lists at their positions."""
    group_size = calculate_group_size(k)
    key_count = len(rank_lists[0])
    result_keys: List[int] = []
    next_candidate_key = 1  # Start from 1 to reserve 0 as non-existing split key option
    MAX_SEARCH_LIMIT = 10_000_000

    for key_idx in range(key_count):
        key = next_candidate_key
        search_limit = next_candidate_key + MAX_SEARCH_LIMIT
        found_between_key = False

        while key < search_limit:
            current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
            match = True

            for rank_list in rank_lists:
                desired_rank = rank_list[key_idx]
                calculated_rank = calc_rank_from_digest(current_hash, group_size)
                if calculated_rank != desired_rank:
                    match = False
                    break
                current_hash = hashlib.sha256(current_hash).digest()

            if match:
                if not spacing or found_between_key:
                    result_keys.append(key)
                    next_candidate_key = key + 1
                    break
                else:
                    found_between_key = True
            key += 1
        else:
            raise ValueError(f"No matching key found for rank_lists[{key_idx}]")

    return result_keys
