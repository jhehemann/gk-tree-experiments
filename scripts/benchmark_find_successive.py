#!/usr/bin/env python3
"""
Script to measure execution time of find_keys_for_successive_rank1 for different parameters.
"""
import time
from gplus_trees.utils import find_keys_for_successive_rank1


def measure(k, dim_limit, count=1, spacing=False, runs=3):
    """Measure average time over a number of runs and return last generated keys."""
    total = 0.0
    last_keys = None
    for _ in range(runs):
        start = time.time()
        keys = find_keys_for_successive_rank1(k=k, dim_limit=dim_limit, count=count, spacing=spacing)
        total += time.time() - start
        last_keys = keys
    return total / runs, last_keys


def main():
    ks = [8, 16, 32]
    dims = [10, 20, 40]
    runs = 1
    count = 1000

    print(f"Measuring find_keys_for_successive_rank1 over {runs} runs with count={count}\n")
    for k in ks:
        for dim in dims:
            avg_time, keys = measure(k=k, dim_limit=dim, count=count, runs=runs)
            print(f"k={k:2d}, dim={dim:3d} -> {avg_time:.6f} sec (avg over {runs} runs)")
            # print(f"  keys: {keys}")

if __name__ == '__main__':
    main()
