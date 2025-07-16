#!/usr/bin/env python3
"""
Benchmark for the retrieve() method of GKPlusTreeBase class.

This script measures performance differences between:
1. retrieve(key, with_next=True) - returns both found entry and next entry
2. retrieve(key, with_next=False) - returns only found entry

The GK+ tree implementation should show more significant performance differences
compared to the regular G+ tree due to its multi-dimensional structure and
automatic conversion between KList and GKPlusTree based on thresholds.

Usage:
    python benchmark_gk_retrieve.py [--sizes 100 1000 10000] [--trials 1000] [--node-size 64]
"""

import argparse
import random
import time
import gc
import sys
import os
from statistics import mean, stdev
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gplus_trees.base import Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
from stats.stats_gk_plus_tree import random_gkplus_tree_of_size


@dataclass
class GKRetrieveBenchmarkResult:
    """Results from a GK+ tree retrieve benchmark run."""
    tree_size: int
    node_capacity: int
    dimension: int
    l_factor: float
    trials: int
    key_pattern: str  # 'existing', 'missing', 'mixed'
    
    # with_next=True results
    with_next_mean_us: float
    with_next_std_us: float
    with_next_min_us: float
    with_next_max_us: float
    
    # with_next=False results  
    without_next_mean_us: float
    without_next_std_us: float
    without_next_min_us: float
    without_next_max_us: float
    
    # Performance comparison
    speedup_factor: float  # how much faster without_next is
    overhead_percent: float  # percentage overhead of with_next
    
    # Tree structure info
    tree_height: int
    total_nodes: int
    leaf_nodes: int


def build_gk_test_tree(size: int, node_capacity: int, dimension: int = 1, l_factor: float = 1.0, 
                       seed: Optional[int] = None) -> Tuple[GKPlusTreeBase, List[int]]:
    """
    Build a GK+ test tree with the specified parameters.
    
    Returns:
        Tuple of (tree, list_of_keys_in_tree)
    """
    if seed is not None:
        random.seed(seed)
    
    tree = random_gkplus_tree_of_size(size, node_capacity, l_factor)
    
    # Extract all keys from the tree for testing
    keys = []
    for leaf in tree.iter_leaf_nodes():
        for entry in leaf.set:
            if entry.item.key >= 0:  # Skip dummy entries
                keys.append(entry.item.key)
    
    return tree, keys


def generate_test_keys(existing_keys: List[int], pattern: str, count: int) -> List[int]:
    """
    Generate test keys based on the specified pattern.
    
    Args:
        existing_keys: Keys that exist in the tree
        pattern: 'existing', 'missing', or 'mixed'
        count: Number of keys to generate
        
    Returns:
        List of test keys
    """
    if pattern == 'existing':
        return random.choices(existing_keys, k=count)
    elif pattern == 'missing':
        # Generate keys that don't exist in the tree
        max_key = max(existing_keys) if existing_keys else 1000000
        existing_set = set(existing_keys)
        missing_keys = []
        attempts = 0
        while len(missing_keys) < count and attempts < count * 10:
            key = random.randint(1, max_key * 2)
            if key not in existing_set:
                missing_keys.append(key)
            attempts += 1
        return missing_keys
    elif pattern == 'mixed':
        # 50% existing, 50% missing
        existing_count = count // 2
        missing_count = count - existing_count
        existing_sample = generate_test_keys(existing_keys, 'existing', existing_count)
        missing_sample = generate_test_keys(existing_keys, 'missing', missing_count)
        mixed = existing_sample + missing_sample
        random.shuffle(mixed)
        return mixed
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def benchmark_gk_retrieve_performance(
    tree: GKPlusTreeBase,
    test_keys: List[int],
    trials: int = 1000
) -> Tuple[List[float], List[float]]:
    """
    Benchmark GK+ tree retrieve performance with and without next parameter.
    
    Returns:
        Tuple of (with_next_times, without_next_times) in seconds
    """
    # Prepare test keys (cycle through them if we have more trials than keys)
    if len(test_keys) < trials:
        test_keys = (test_keys * ((trials // len(test_keys)) + 1))[:trials]
    else:
        test_keys = random.choices(test_keys, k=trials)
    
    # Benchmark with_next=True
    gc.collect()
    gc.disable()
    try:
        with_next_times = []
        for key in test_keys:
            start = time.perf_counter()
            tree.retrieve(key, with_next=True)
            end = time.perf_counter()
            with_next_times.append(end - start)
    finally:
        gc.enable()
    
    # Benchmark with_next=False
    gc.collect()
    gc.disable()
    try:
        without_next_times = []
        for key in test_keys:
            start = time.perf_counter()
            tree.retrieve(key, with_next=False)
            end = time.perf_counter()
            without_next_times.append(end - start)
    finally:
        gc.enable()
    
    return with_next_times, without_next_times


def get_tree_stats(tree: GKPlusTreeBase) -> Tuple[int, int, int]:
    """Get basic tree statistics."""
    if tree.is_empty():
        return 0, 0, 0
    
    height = tree.physical_height()
    total_nodes = 0
    leaf_nodes = 0
    
    # Count nodes by traversing the tree structure
    for leaf in tree.iter_leaf_nodes():
        leaf_nodes += 1
        # This is a rough estimate; for exact count we'd need to traverse all internal nodes too
        total_nodes += 1
    
    # Rough estimate of total nodes (this is not exact but gives an idea)
    total_nodes = leaf_nodes * 2  # Rough approximation
    
    return height, total_nodes, leaf_nodes


def run_gk_retrieve_benchmark(
    size: int,
    node_capacity: int,
    dimension: int,
    l_factor: float,
    pattern: str,
    trials: int = 1000,
    seed: Optional[int] = None
) -> GKRetrieveBenchmarkResult:
    """
    Run a complete GK+ tree retrieve benchmark for given parameters.
    """
    print(f"Building GK+ tree: size={size}, K={node_capacity}, dim={dimension}, l_factor={l_factor}")
    tree, existing_keys = build_gk_test_tree(size, node_capacity, dimension, l_factor, seed)
    
    print(f"Generating {trials} test keys with pattern '{pattern}'...")
    test_keys = generate_test_keys(existing_keys, pattern, trials)
    
    print(f"Running benchmark with {trials} trials...")
    with_next_times, without_next_times = benchmark_gk_retrieve_performance(tree, test_keys, trials)
    
    # Get tree statistics
    height, total_nodes, leaf_nodes = get_tree_stats(tree)
    
    # Convert to microseconds for readability
    with_next_us = [t * 1e6 for t in with_next_times]
    without_next_us = [t * 1e6 for t in without_next_times]
    
    # Calculate statistics
    with_next_mean = mean(with_next_us)
    with_next_std = stdev(with_next_us) if len(with_next_us) > 1 else 0.0
    without_next_mean = mean(without_next_us)
    without_next_std = stdev(without_next_us) if len(without_next_us) > 1 else 0.0
    
    speedup = with_next_mean / without_next_mean if without_next_mean > 0 else 1.0
    overhead = ((with_next_mean - without_next_mean) / without_next_mean * 100) if without_next_mean > 0 else 0.0
    
    return GKRetrieveBenchmarkResult(
        tree_size=size,
        node_capacity=node_capacity,
        dimension=dimension,
        l_factor=l_factor,
        trials=trials,
        key_pattern=pattern,
        with_next_mean_us=with_next_mean,
        with_next_std_us=with_next_std,
        with_next_min_us=min(with_next_us),
        with_next_max_us=max(with_next_us),
        without_next_mean_us=without_next_mean,
        without_next_std_us=without_next_std,
        without_next_min_us=min(without_next_us),
        without_next_max_us=max(without_next_us),
        speedup_factor=speedup,
        overhead_percent=overhead,
        tree_height=height,
        total_nodes=total_nodes,
        leaf_nodes=leaf_nodes
    )


def print_gk_benchmark_result(result: GKRetrieveBenchmarkResult):
    """Print a formatted GK+ tree benchmark result."""
    print(f"\n{'='*90}")
    print(f"GK+ TREE RETRIEVE BENCHMARK RESULTS")
    print(f"{'='*90}")
    print(f"Tree size: {result.tree_size:,}")
    print(f"Node capacity (K): {result.node_capacity}")
    print(f"Dimension: {result.dimension}")
    print(f"L-factor: {result.l_factor}")
    print(f"Key pattern: {result.key_pattern}")
    print(f"Trials: {result.trials:,}")
    print(f"Tree height: {result.tree_height}")
    print(f"Leaf nodes: {result.leaf_nodes}")
    print(f"{'-'*90}")
    
    print(f"{'Metric':<25} {'with_next=True':<20} {'with_next=False':<20} {'Difference':<15}")
    print(f"{'-'*90}")
    print(f"{'Mean (μs)':<25} {result.with_next_mean_us:<20.2f} {result.without_next_mean_us:<20.2f} {result.with_next_mean_us - result.without_next_mean_us:<15.2f}")
    print(f"{'Std Dev (μs)':<25} {result.with_next_std_us:<20.2f} {result.without_next_std_us:<20.2f} {result.with_next_std_us - result.without_next_std_us:<15.2f}")
    print(f"{'Min (μs)':<25} {result.with_next_min_us:<20.2f} {result.without_next_min_us:<20.2f} {result.with_next_min_us - result.without_next_min_us:<15.2f}")
    print(f"{'Max (μs)':<25} {result.with_next_max_us:<20.2f} {result.without_next_max_us:<20.2f} {result.with_next_max_us - result.without_next_max_us:<15.2f}")
    
    print(f"{'-'*90}")
    print(f"Performance Analysis:")
    print(f"  Speedup factor: {result.speedup_factor:.2f}x (with_next=False is {result.speedup_factor:.2f}x faster)")
    print(f"  Overhead: {result.overhead_percent:.1f}% (with_next=True is {result.overhead_percent:.1f}% slower)")
    
    if result.overhead_percent > 50:
        print(f"  ⚠️  HIGH OVERHEAD: with_next=True adds significant cost!")
    elif result.overhead_percent > 20:
        print(f"  ⚠️  MODERATE OVERHEAD: with_next=True adds noticeable cost")
    elif result.overhead_percent > 5:
        print(f"  ✓  LOW OVERHEAD: with_next=True adds minimal cost")
    else:
        print(f"  ✓  NEGLIGIBLE OVERHEAD: with_next parameter has minimal impact")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GK+ tree retrieve() method performance")
    parser.add_argument("--sizes", nargs='+', type=int, default=[100, 1000, 10000],
                        help="Tree sizes to test")
    parser.add_argument("--trials", type=int, default=1000,
                        help="Number of trials per benchmark")
    parser.add_argument("--node-size", type=int, default=4,
                        help="Node capacity for tree construction")
    parser.add_argument("--dimensions", nargs='+', type=int, default=[1, 2],
                        help="Dimensions to test")
    parser.add_argument("--l-factors", nargs='+', type=float, default=[0.5, 1.0, 2.0],
                        help="L-factor values to test")
    parser.add_argument("--patterns", nargs='+', choices=['existing', 'missing', 'mixed'],
                        default=['existing', 'missing', 'mixed'],
                        help="Key patterns to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results")
    parser.add_argument("--output", type=str, 
                        help="Output file to save detailed results (CSV format)")
    
    args = parser.parse_args()
    
    print("GK+ Tree Retrieve Method Benchmark")
    print("=" * 60)
    print(f"Testing tree sizes: {args.sizes}")
    print(f"Dimensions: {args.dimensions}")
    print(f"L-factors: {args.l_factors}")
    print(f"Key patterns: {args.patterns}")
    print(f"Trials per test: {args.trials}")
    print(f"Node capacity: {args.node_size}")
    print(f"Random seed: {args.seed}")
    
    results = []
    
    for size in args.sizes:
        for dimension in args.dimensions:
            for l_factor in args.l_factors:
                for pattern in args.patterns:
                    print(f"\n🔄 Running benchmark: size={size}, dim={dimension}, l_factor={l_factor}, pattern={pattern}")
                    try:
                        result = run_gk_retrieve_benchmark(
                            size=size,
                            node_capacity=args.node_size,
                            dimension=dimension,
                            l_factor=l_factor,
                            pattern=pattern,
                            trials=args.trials,
                            seed=args.seed
                        )
                        results.append(result)
                        print_gk_benchmark_result(result)
                    except Exception as e:
                        print(f"❌ Error in benchmark: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
    
    # Summary
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    
    if results:
        print(f"{'Size':<8} {'Dim':<4} {'L-factor':<8} {'Pattern':<10} {'Overhead %':<12} {'Speedup':<8} {'Mean w/ next':<12} {'Mean w/o next':<12} {'Height':<8}")
        print(f"{'-'*120}")
        for result in results:
            print(f"{result.tree_size:<8} {result.dimension:<4} {result.l_factor:<8.1f} {result.key_pattern:<10} "
                  f"{result.overhead_percent:<12.1f} {result.speedup_factor:<8.2f} "
                  f"{result.with_next_mean_us:<12.2f} {result.without_next_mean_us:<12.2f} {result.tree_height:<8}")
        
        # Calculate overall statistics
        avg_overhead = mean([r.overhead_percent for r in results])
        avg_speedup = mean([r.speedup_factor for r in results])
        
        print(f"\nOverall averages:")
        print(f"  Average overhead: {avg_overhead:.1f}%")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        
        # Analysis by dimension
        print(f"\nAnalysis by dimension:")
        for dim in args.dimensions:
            dim_results = [r for r in results if r.dimension == dim]
            if dim_results:
                dim_avg_overhead = mean([r.overhead_percent for r in dim_results])
                print(f"  Dimension {dim}: {dim_avg_overhead:.1f}% average overhead")
        
        # Analysis by l_factor
        print(f"\nAnalysis by L-factor:")
        for l_factor in args.l_factors:
            l_results = [r for r in results if r.l_factor == l_factor]
            if l_results:
                l_avg_overhead = mean([r.overhead_percent for r in l_results])
                print(f"  L-factor {l_factor}: {l_avg_overhead:.1f}% average overhead")
    
    # Save detailed results if requested
    if args.output:
        import csv
        print(f"\n💾 Saving detailed results to {args.output}")
        with open(args.output, 'w', newline='') as csvfile:
            if results:
                writer = csv.DictWriter(csvfile, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))
        print(f"✅ Results saved successfully")


if __name__ == "__main__":
    main()
