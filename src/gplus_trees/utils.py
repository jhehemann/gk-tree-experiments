"""
Utility functions for GPlusTree operations, including key discovery based on hash ranks.
"""

import hashlib


def get_digest(key: int, dim: int) -> bytes:
    """
    Domain-separated per-dimension digest of a key: H(⟨dim⟩ ‖ ⟨key⟩).

    Realizes the settled rank construction rho_j(x) = rank(H(⟨j⟩ ‖ x))
    (paper ``model.md`` §5, ADR-002, accepted 2026-07-07): every dimension is an
    *independent* uniform oracle keyed by the pair (dim, key), rather than an
    iterate of the previous dimension's digest. ⟨dim⟩ and ⟨key⟩ are fixed-length
    32-byte big-endian encodings, so distinct (dim, key) pairs never alias and the
    per-dimension rank vector is i.i.d. geometric for a fresh key — no chain-merge
    or cycle pathology.

    ``abs(key)`` keeps a dummy item's negative key aliased to its positive
    counterpart, unchanged from the pre-domain-separation construction.

    Args:
        key (int): The item key to derive the digest for.
        dim (int): The 1-based dimension level.

    Returns:
        bytes: The SHA-256 digest H(⟨dim⟩ ‖ ⟨key⟩).
    """
    return hashlib.sha256(int(dim).to_bytes(32, "big") + abs(int(key)).to_bytes(32, "big")).digest()


def count_trailing_zero_bits(digest: bytes) -> int:
    tz = 0
    for byte in reversed(digest):
        if byte == 0:
            tz += 8
        else:
            # (byte & -byte) isolates the lowest set bit,
            # bit_length()-1 gives its zero-based position within the byte
            tz += (byte & -byte).bit_length() - 1
            break
    return tz


def get_group_size(k: int) -> int:
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
    if group_size == 0:
        raise ValueError("group_size must be greater than 0")
    tz = count_trailing_zero_bits(digest)
    return (tz // group_size) + 1


def calc_rank_from_digest_k(digest: bytes, k: int) -> int:
    """Calculate rank from a hash digest based on the group size derived from k."""
    group_size = get_group_size(k)
    return calc_rank_from_digest(digest, group_size)


def find_keys_for_rank_lists(rank_lists: list[list[int]], k: int, spacing: bool = False) -> list[int]:
    """Find keys whose domain-separated per-dimension digests match the rank lists.

    Inverts rho_j(x) = rank(H(⟨j⟩ ‖ x)) (model.md §5, ADR-002): dimension j's rank
    is derived directly from the key via :func:`get_digest`, independently of the
    other dimensions (no iterated chaining).
    """
    group_size = get_group_size(k)
    key_count = len(rank_lists[0])
    # Fixed-length ⟨dim⟩ prefixes, precomputed once (key varies in the hot loop).
    dim_prefixes = [d.to_bytes(32, "big") for d in range(1, len(rank_lists) + 1)]
    result_keys: list[int] = []
    next_candidate_key = 1  # Start from 1 to reserve 0 as non-existing split key option
    MAX_SEARCH_LIMIT = 10_000_000_000

    for key_idx in range(key_count):
        key = next_candidate_key
        search_limit = next_candidate_key + MAX_SEARCH_LIMIT
        found_between_key = False

        while key < search_limit:
            key_bytes = abs(key).to_bytes(32, "big")
            match = True

            for dim_index, rank_list in enumerate(rank_lists):
                desired_rank = rank_list[key_idx]
                digest = hashlib.sha256(dim_prefixes[dim_index] + key_bytes).digest()
                calculated_rank = calc_rank_from_digest(digest, group_size)
                if calculated_rank != desired_rank:
                    match = False
                    break

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
