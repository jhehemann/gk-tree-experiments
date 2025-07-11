#!/usr/bin/env python3
"""
Benchmarks for the gplus-tree data structure.

This script measures:
 1. Hex conversion performance
 2. calculate_item_rank performance
 3. Full GPlusTree build times (random_gtree_of_size)
 4. random_klist_tree statistics
 5. Per-insert cost into trees of various sizes

Usage:
    python benchmarks.py [--num N] [--space S] [--node-size K] [--sizes 100 1000 10000] [--trials T]
"""
import argparse
import random
import time
import timeit
import gc
from pprint import pprint
from dataclasses import asdict
from statistics import mean, variance

from gplus_trees.base import calculate_rank, calculate_group_size
from stats_gplus_tree import random_klist_tree, random_gtree_of_size
from gplus_trees.gplus_tree_base import gtree_stats_, GPlusTreeBase
from gplus_trees.base import Item

def bench_rank(num: int, space: int, node_size: int, runs: int = 3) -> None:
    """Benchmark calculate_item_rank on random keys within `space`."""
    keys = random.sample(range(1, space), k=num)
    group_size = calculate_group_size(node_size)
    def _inner():
        for key in keys:
            _ = calculate_rank(key, group_size)
    t = timeit.timeit(_inner, number=runs) / runs
    print(f"[bench] calculate_item_rank:      {t:.4f}s for {num} calls")


def bench_build_gtree(sizes: list[int], node_size: int) -> None:
    """Measure random_gtree_of_size for various sizes."""
    for n in sizes:
        t0 = time.perf_counter()
        _ = random_gtree_of_size(n, node_size)
        elapsed = time.perf_counter() - t0
        print(f"[bench] random_gtree_of_size({n}): {elapsed:.4f}s")


def bench_klist_stats(n: int, node_size: int) -> None:
    """Build a single random_klist_tree and print its stats."""
    tree = random_klist_tree(n, node_size)
    stats = gtree_stats_(tree, {})
    print(f"[bench] random_klist_tree({n}, {node_size}) stats:")
    pprint(asdict(stats))


def measure_single_insert(n: int, node_size: int, space: int, trials: int = 200) -> tuple[float, float, float, float]:
    """
    Measure per-insert cost into a tree of exactly `n` items,
    averaged over `trials` independent trees.
    Returns (mean_time_s, variance_time_s, retrieve_avg, retrieve_var).
    """
    # Pre-build independent trees
    trees = [random_klist_tree(n, node_size) for _ in range(trials)]

    # Generate random keys within space and their ranks
    keys = random.sample(range(1, space), k=trials)
    group_size = calculate_group_size(node_size)
    ranks = [calculate_rank(key, group_size) for key in keys]

    # First measure retrieve performance on the existing trees
    gc.collect()
    gc.disable()
    try:
        retrieve_times = []
        for tree, key in zip(trees, keys):
            t0 = time.perf_counter()
            tree.retrieve(key)
            retrieve_times.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    
    retrieve_avg = mean(retrieve_times)
    retrieve_var = variance(retrieve_times)

    # Then measure insert performance
    gc.collect()
    gc.disable()
    try:
        times = []
        for tree, key, rank in zip(trees, keys, ranks):
            t0 = time.perf_counter()
            tree.insert(Item(key, f"val{key}"), rank=rank)
            times.append(time.perf_counter() - t0)
    finally:
        gc.enable()

    return mean(times), variance(times), retrieve_avg, retrieve_var


def bench_single_insert(sizes: list[int], node_size: int, space: int, trials: int) -> None:
    """Run measure_single_insert for each size and print results."""    
    for n in sizes:
        avg, var, retrieve_avg, retrieve_var = measure_single_insert(n, node_size, space, trials)
        print(
            f"[bench] Insert into size {n:<7} → avg {avg*1e6:8.2f} µs   σ²={var*1e12:8.2f} µs²"
        )
        print(
            f"[bench] Retrieve from size {n:<7} → avg {retrieve_avg*1e6:8.2f} µs   σ²={retrieve_var*1e12:8.2f} µs²"
        )


def main():
    parser = argparse.ArgumentParser(description="GPlusTree benchmarks")
    parser.add_argument("--num", type=int, default=10_000,
                        help="Number of calls for rank benchmarks")
    parser.add_argument("--space", type=int, default=1 << 24,
                        help="Key space for conversion and rank benchmarks")
    parser.add_argument("--node-size", type=int, default=64,
                        help="Target node size (k) for rank, build, and insert benchmarks")
    parser.add_argument("--sizes", nargs='+', type=int, default=[100, 1000, 10_000],
                        help="Tree sizes for single-insert benchmarks")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of trials for single-insert benchmarks")
    args = parser.parse_args()

    print("\n=== calculate_item_rank Benchmark ===")
    bench_rank(args.num, args.space, args.node_size)

    print("\n=== Full GPlusTree Build ===")
    bench_build_gtree([10, 100, 1000, 10_000, 100_000], args.node_size)

    print("\n=== random_klist_tree Stats ===")
    bench_klist_stats(10_000, args.node_size)

    print("\n=== Single-Insert Benchmarks ===")
    bench_single_insert(args.sizes, args.node_size, args.space, args.trials)

if __name__ == "__main__":
    main()
