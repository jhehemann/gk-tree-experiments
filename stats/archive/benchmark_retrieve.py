#!/usr/bin/env python3
"""
Benchmark for the retrieve() method of GPlusTreeBase class.

This script measures performance differences between:
1. retrieve(key, with_next=True) - returns both found entry and next entry
2. retrieve(key, with_next=False) - returns only found entry

The benchmark tests various scenarios:
- Different tree sizes
- Keys that exist vs keys that don't exist
- Random vs sequential key patterns
- Different node capacities

Usage:
    python benchmark_retrieve.py [--sizes 100 1000 10000] [--trials 1000] [--node-size 64]
"""

import argparse
import random
import time
import timeit
import gc
import sys
import os
from statistics import mean, stdev
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gplus_trees.base import Item
from gplus_trees.factory import create_gplustree
from gplus_trees.gplus_tree_base import GPlusTreeBase
from stats.stats_gplus_tree import random_gtree_of_size


@dataclass
class RetrieveBenchmarkResult:
    """Results from a retrieve benchmark run."""
    tree_size: int
    node_capacity: int
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


def build_test_tree(size: int, node_capacity: int, seed: Optional[int] = None) -> Tuple[GPlusTreeBase, List[int]]:
    """
    Build a test tree with the specified size and return it along with a list of keys.
    
    Returns:
        Tuple of (tree, list_of_keys_in_tree)
    """
    if seed is not None:
        random.seed(seed)
    
    tree = random_gtree_of_size(size, node_capacity)
    
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


def benchmark_retrieve_performance(
    tree: GPlusTreeBase,
    test_keys: List[int],
    trials: int = 1000
) -> Tuple[List[float], List[float]]:
    """
    Benchmark retrieve performance with and without next parameter.
    
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


def run_retrieve_benchmark(
    size: int,
    node_capacity: int,
    pattern: str,
    trials: int = 1000,
    seed: Optional[int] = None
) -> RetrieveBenchmarkResult:
    """
    Run a complete retrieve benchmark for given parameters.
    """
    print(f"Building tree of size {size} with node capacity {node_capacity}...")
    tree, existing_keys = build_test_tree(size, node_capacity, seed)
    
    print(f"Generating {trials} test keys with pattern '{pattern}'...")
    test_keys = generate_test_keys(existing_keys, pattern, trials)
    
    print(f"Running benchmark with {trials} trials...")
    with_next_times, without_next_times = benchmark_retrieve_performance(tree, test_keys, trials)
    
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
    
    return RetrieveBenchmarkResult(
        tree_size=size,
        node_capacity=node_capacity,
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
        overhead_percent=overhead
    )


def print_benchmark_result(result: RetrieveBenchmarkResult):
    """Print a formatted benchmark result."""
    print(f"\n{'='*80}")
    print(f"RETRIEVE BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Tree size: {result.tree_size:,}")
    print(f"Node capacity: {result.node_capacity}")
    print(f"Key pattern: {result.key_pattern}")
    print(f"Trials: {result.trials:,}")
    print(f"{'-'*80}")
    
    print(f"{'Metric':<25} {'with_next=True':<20} {'with_next=False':<20} {'Difference':<15}")
    print(f"{'-'*80}")
    print(f"{'Mean (μs)':<25} {result.with_next_mean_us:<20.2f} {result.without_next_mean_us:<20.2f} {result.with_next_mean_us - result.without_next_mean_us:<15.2f}")
    print(f"{'Std Dev (μs)':<25} {result.with_next_std_us:<20.2f} {result.without_next_std_us:<20.2f} {result.with_next_std_us - result.without_next_std_us:<15.2f}")
    print(f"{'Min (μs)':<25} {result.with_next_min_us:<20.2f} {result.without_next_min_us:<20.2f} {result.with_next_min_us - result.without_next_min_us:<15.2f}")
    print(f"{'Max (μs)':<25} {result.with_next_max_us:<20.2f} {result.without_next_max_us:<20.2f} {result.with_next_max_us - result.without_next_max_us:<15.2f}")
    
    print(f"{'-'*80}")
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
    parser = argparse.ArgumentParser(description="Benchmark retrieve() method performance")
    parser.add_argument("--sizes", nargs='+', type=int, default=[100, 1000, 10000],
                        help="Tree sizes to test")
    parser.add_argument("--trials", type=int, default=1000,
                        help="Number of trials per benchmark")
    parser.add_argument("--node-size", type=int, default=64,
                        help="Node capacity for tree construction")
    parser.add_argument("--patterns", nargs='+', choices=['existing', 'missing', 'mixed'],
                        default=['existing', 'missing', 'mixed'],
                        help="Key patterns to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results")
    parser.add_argument("--output", type=str, 
                        help="Output file to save detailed results (CSV format)")
    
    args = parser.parse_args()
    
    print("G+ Tree Retrieve Method Benchmark")
    print("=" * 50)
    print(f"Testing tree sizes: {args.sizes}")
    print(f"Key patterns: {args.patterns}")
    print(f"Trials per test: {args.trials}")
    print(f"Node capacity: {args.node_size}")
    print(f"Random seed: {args.seed}")
    
    results = []
    
    for size in args.sizes:
        for pattern in args.patterns:
            print(f"\n🔄 Running benchmark: size={size}, pattern={pattern}")
            try:
                result = run_retrieve_benchmark(
                    size=size,
                    node_capacity=args.node_size,
                    pattern=pattern,
                    trials=args.trials,
                    seed=args.seed
                )
                results.append(result)
                print_benchmark_result(result)
            except Exception as e:
                print(f"❌ Error in benchmark: {e}")
                continue
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if results:
        print(f"{'Size':<8} {'Pattern':<10} {'Overhead %':<12} {'Speedup':<10} {'Mean w/ next (μs)':<18} {'Mean w/o next (μs)':<18}")
        print(f"{'-'*80}")
        for result in results:
            print(f"{result.tree_size:<8} {result.key_pattern:<10} {result.overhead_percent:<12.1f} "
                  f"{result.speedup_factor:<10.2f} {result.with_next_mean_us:<18.2f} {result.without_next_mean_us:<18.2f}")
        
        # Calculate overall statistics
        avg_overhead = mean([r.overhead_percent for r in results])
        avg_speedup = mean([r.speedup_factor for r in results])
        
        print(f"\nOverall averages:")
        print(f"  Average overhead: {avg_overhead:.1f}%")
        print(f"  Average speedup: {avg_speedup:.2f}x")
    
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
