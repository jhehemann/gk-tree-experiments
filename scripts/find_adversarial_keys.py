#!/usr/bin/env python3
"""
Script to measure execution time of find_keys_for_successive_rank1 for different parameters.
"""

import hashlib
import os
import pathlib
import pickle

from tqdm import tqdm, trange

from gplus_trees.utils import calc_rank_from_digest, get_group_size


def find_rank_keys(key_count: int, max_dim: int, k: int, rank: int) -> list[int]:
    """
    Efficiently find keys whose domain-separated per-dimension digests all have the
    specified rank across ``max_dim`` dimensions.

    Inverts the settled rank construction rho_j(x) = rank(H(⟨j⟩ ‖ x)) (model.md §5,
    ADR-002): each dimension j is an independent digest H(⟨j⟩ ‖ key), so a candidate
    is rejected as soon as one dimension misses the target rank (early abort, cheapest
    dimension first). Under either construction this grinds one hash per surviving
    dimension; the expected candidate count is unchanged up to negligible slack.
    """
    group_size = get_group_size(k)
    result_keys = []
    key = 1

    # Pre-compute fixed-length ⟨dim⟩ prefixes to avoid repeated conversions.
    dim_prefixes = [d.to_bytes(32, "big") for d in range(1, max_dim + 1)]

    with trange(key_count, desc=f"Finding rank {rank} keys", leave=False) as pbar:
        while len(result_keys) < key_count:
            key_bytes = abs(key).to_bytes(32, "big")

            match_all_dims = True
            for prefix in dim_prefixes:
                digest = hashlib.sha256(prefix + key_bytes).digest()
                if calc_rank_from_digest(digest, group_size) != rank:
                    match_all_dims = False
                    break

            if match_all_dims:
                result_keys.append(key)
                pbar.update(1)
            key += 1

    return result_keys


def generate_adversarial_keys(key_count, capacity, max_dim, rank=1):
    """Get adversarial keys, using only file storage."""
    keys_dir = str(pathlib.Path(__file__).parent.parent / "benchmarks" / "adversarial_keys_new")
    os.makedirs(keys_dir, exist_ok=True)
    file_name = f"keys_sz{key_count}_k{capacity}_d{max_dim}.pkl"
    file_path = os.path.join(keys_dir, file_name)

    if not os.path.exists(file_path):
        succ_keys = find_rank_keys(key_count=key_count, max_dim=max_dim, k=capacity, rank=rank)
        with open(file_path, "wb") as f:
            pickle.dump(succ_keys, f)


def main():
    param_pairs = [
        (4, [1, 10, 20, 30, 40]),
        (8, [1, 10, 20, 40, 80]),
        (16, [1, 10, 20, 40, 80, 160]),
        (32, [1, 10, 20, 40, 80, 160, 320]),
    ]
    counts = [10000]

    print(f"Generating adversarial keys for counts={counts}, param_pairs(k,max_dims)={param_pairs}\n")
    for count in tqdm(counts, desc="Counts"):
        for k, dims in tqdm(param_pairs, desc="K-Dim pairs", leave=False):
            for dim in tqdm(dims, desc=f"Dims for k={k}", leave=False):
                generate_adversarial_keys(key_count=count, capacity=k, max_dim=dim, rank=1)


if __name__ == "__main__":
    main()
