"""
Utility functions for GPlusTree operations, including key discovery based on hash ranks.
"""
import hashlib
from typing import List
from gplus_trees.g_k_plus.utils import calc_rank_from_digest, calculate_group_size


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
