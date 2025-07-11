#!/usr/bin/env python3
"""
Script to measure execution time of find_keys_for_successive_rank1 for different parameters.
"""
import hashlib
import os
import pathlib
import pickle
from typing import List
from gplus_trees.g_k_plus.utils import calc_rank_from_digest, calculate_group_size
from tqdm import tqdm
from tqdm import trange

def find_rank_1_keys(key_count: int, max_dim: int, k: int) -> List[int]:
    """
    Efficiently find keys whose repeated hashes all have rank 1 for 'dim' dimensions.
    Uses a tight loop and avoids unnecessary allocations for hot loop usage.
    """
    group_size = calculate_group_size(k)
    result_keys = []
    key = 1

    with trange(key_count, desc="Finding rank 1 keys", leave=False) as pbar:
        while len(result_keys) < key_count:
            current_hash = hashlib.sha256(key.to_bytes(32, 'big')).digest()
            for _ in range(max_dim):
                if calc_rank_from_digest(current_hash, group_size) != 1:
                    break
                current_hash = hashlib.sha256(current_hash).digest()
            else:
                result_keys.append(key)
                pbar.update(1)
            key += 1
    return result_keys


def generate_adversarial_keys(key_count, capacity, max_dim):
    """Get adversarial keys, using only file storage."""
    keys_dir = str(pathlib.Path(__file__).parent.parent / 'benchmarks' / 'adversarial_keys')
    os.makedirs(keys_dir, exist_ok=True)
    file_name = f"keys_sz{key_count}_k{capacity}_d{max_dim}.pkl"
    file_path = os.path.join(keys_dir, file_name)

    if not os.path.exists(file_path):
        succ_keys = find_rank_1_keys(
            key_count=key_count, max_dim=max_dim, k=capacity
        )
        with open(file_path, 'wb') as f:
            pickle.dump(succ_keys, f)


def main():
    param_pairs = [
        (4, [1, 10, 20, 30, 40]),
        (8, [1, 10, 20, 40, 80]),
        (16, [1, 10, 20, 40, 80, 160]),
        (32, [1, 10, 20, 40, 80, 160, 320])
    ]
    counts = [1000, 10000]

    print("Generating adversarial keys for counts={}, param_pairs(k,max_dims)={}\n".format(counts, param_pairs))
    for count in tqdm(counts, desc="Counts"):
        for k, dims in tqdm(param_pairs, desc="K-Dim pairs", leave=False):
            for dim in tqdm(dims, desc=f"Dims for k={k}", leave=False):
                generate_adversarial_keys(key_count=count, capacity=k, max_dim=dim)

if __name__ == '__main__':
    main()
