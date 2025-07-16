#!/usr/bin/env python3
"""
Script to measure execution time of find_keys_for_successive_rank1 for different parameters.
"""
import hashlib
import os
import pathlib
import pickle
from typing import List
from gplus_trees.utils import calc_rank_from_digest, get_group_size
from tqdm import tqdm
from tqdm import trange

def find_rank_keys(key_count: int, max_dim: int, k: int, rank: int) -> List[int]:
    """
    Efficiently find keys whose repeated hashes all have the specified rank for 'dim' dimensions.
    Uses a tight loop and avoids unnecessary allocations for hot loop usage.
    Updated to match the authoritative get_digest_for_dim() pattern.
    Optimized for computational efficiency.
    """
    group_size = get_group_size(k)
    result_keys = []
    key = 1
    
    # Pre-compute byte representations for dimensions to avoid repeated conversions
    dim_bytes = [d.to_bytes(32, 'big') for d in range(1, max_dim + 1)]
    one_bytes = dim_bytes[0]  # bytes for dimension 1
    
    # Reuse hasher objects to avoid object creation overhead
    hasher = hashlib.sha256()
    
    with trange(key_count, desc=f"Finding rank {rank} keys", leave=False) as pbar:
        while len(result_keys) < key_count:
            # Start with dimension 1: hash(abs(key) + 1)
            abs_key = abs(key)
            key_bytes = abs_key.to_bytes(32, 'big')
            
            hasher.update(key_bytes + one_bytes)
            current_hash = hasher.digest()
            hasher = hashlib.sha256()  # Reset hasher for next use
            
            # Check if dimension 1 has the desired rank
            if calc_rank_from_digest(current_hash, group_size) != rank:
                key += 1
                continue
            
            # For dimensions 2 to max_dim, hash with the dimension number
            match_all_dims = True
            for dim_idx in range(1, max_dim):  # dim_idx 1 = dimension 2, etc.
                hasher.update(current_hash + dim_bytes[dim_idx])
                current_hash = hasher.digest()
                hasher = hashlib.sha256()  # Reset hasher for next use
                
                if calc_rank_from_digest(current_hash, group_size) != rank:
                    match_all_dims = False
                    break
            
            if match_all_dims:
                result_keys.append(key)
                pbar.update(1)
            key += 1
    
    return result_keys


def generate_adversarial_keys(key_count, capacity, max_dim, rank=1):
    """Get adversarial keys, using only file storage."""
    keys_dir = str(pathlib.Path(__file__).parent.parent / 'benchmarks' / 'adversarial_keys_new')
    os.makedirs(keys_dir, exist_ok=True)
    file_name = f"keys_sz{key_count}_k{capacity}_d{max_dim}.pkl"
    file_path = os.path.join(keys_dir, file_name)

    if not os.path.exists(file_path):
        succ_keys = find_rank_keys(
            key_count=key_count, max_dim=max_dim, k=capacity, rank=rank
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
    counts = [10000]

    print("Generating adversarial keys for counts={}, param_pairs(k,max_dims)={}\n".format(counts, param_pairs))
    for count in tqdm(counts, desc="Counts"):
        for k, dims in tqdm(param_pairs, desc="K-Dim pairs", leave=False):
            for dim in tqdm(dims, desc=f"Dims for k={k}", leave=False):
                generate_adversarial_keys(key_count=count, capacity=k, max_dim=dim, rank=1)

if __name__ == '__main__':
    main()
