"""
ASV benchmarks for GKPlusTreeBase operations.

This module contains comprehensive benchmarks for the GKPlusTreeBase class,
focusing on batch construction (full data structure creation) and 
retrieve() operations with various data patterns and sizes.
"""

import gc
import random
from typing import List

from gplus_trees.g_k_plus.factory import make_gkplustree_classes, create_gkplus_tree
from gplus_trees.g_k_plus.utils import calc_rank_from_group_size, calculate_group_size
from gplus_trees.g_k_plus.g_k_plus_base import bulk_create_gkplus_tree
from gplus_trees.base import Item, Entry
from benchmarks.benchmark_utils import BaseBenchmark, BenchmarkUtils


class GKPlusTreeBatchInsertBenchmarks(BaseBenchmark):
    """Benchmarks for batch GKPlusTreeBase construction via sequential inserts."""
    
    # Test different capacities, data sizes, distributions, and l_factors
    params = [
        [4, 8, 16, 32],  # K values (capacities)
        [1000, 10000],  # data sizes
        ['uniform', 'sequential', 'clustered'],  # data distributions
        [1.0, 2.0, 4.0, 8.0]  # l_factor values
    ]
    param_names = ['capacity', 'size', 'distribution', 'l_factor']
    
    min_run_count = 5
    
    def setup(self, capacity, size, distribution, l_factor):
        """Setup GKPlusTree and test data for tree construction benchmarking."""
        super().setup(capacity, size, distribution, l_factor)
        
        # Create GKPlusTree with specified capacity and l_factor
        self.tree_class, _, _, _ = make_gkplustree_classes(capacity)
        
        # Generate deterministic test data
        self.keys = BenchmarkUtils.generate_deterministic_keys(
            size=size,
            seed=42 + hash((capacity, size, distribution, l_factor)) % 1000,
            distribution=distribution
        )
        self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
        # Calculate ranks for entries
        group_size = calculate_group_size(capacity)
        self.ranks = [calc_rank_from_group_size(key, group_size) for key in self.keys]
        
        # Store l_factor for tree creation
        self.l_factor = l_factor

        gc.collect()
        gc.disable() # Enabled in teardown
    
    
    def time_insert_entry_tree_construction(self, capacity, size, distribution, l_factor):
        """Benchmark full tree construction by inserting all entries."""
        tree = self.tree_class(l_factor=l_factor)
        
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)


class GKPlusTreeRetrieveBenchmarks(BaseBenchmark):
    """Benchmarks for GKPlusTreeBase.retrieve() method."""
    
    # Test different capacities, data sizes, hit ratios, and l_factors (distribution fixed to 'sequential')
    params = [
        [4, 8, 16, 32],  # K values (capacities)
        [1000, 10000],  # data sizes
        [0.0, 1.0],  # hit ratios
        [1.0, 2.0, 4.0, 8.0]  # l_factor values
    ]
    param_names = ['capacity', 'size', 'hit_ratio', 'l_factor']
    
    min_run_count = 5
    
    # Class-level cache for trees and data
    _tree_cache = {}
    _data_cache = {}
    
    def setup(self, capacity, size, hit_ratio, l_factor):
        """Setup populated GKPlusTree and lookup keys for benchmarking."""
        super().setup(capacity, size, hit_ratio, l_factor)
        
        # Use only 'sequential' distribution
        distribution = 'sequential'
        
        # Create cache key for tree (hit_ratio doesn't affect tree structure)
        tree_cache_key = (capacity, size, l_factor, distribution)
        
        # Check if we have a cached tree for these parameters
        if tree_cache_key not in self._tree_cache:
            # Create and populate GKPlusTree (only when not cached)
            tree_class, _, klist_class, _ = make_gkplustree_classes(capacity)
            
            # Generate and insert test data
            base_seed = 42 + hash((capacity, size, distribution, l_factor)) % 1000
            insert_keys = BenchmarkUtils.generate_deterministic_keys(
                size=size,
                seed=base_seed,
                distribution=distribution
            )
            
            entries = BenchmarkUtils.create_test_entries(insert_keys)
            
            # Use bulk_create_gkplus_tree for efficient initial tree construction
            tree = bulk_create_gkplus_tree(entries, DIM=1, l_factor=l_factor, KListClass=klist_class)
            
            # Cache the tree and associated data
            self._tree_cache[tree_cache_key] = tree
            self._data_cache[tree_cache_key] = insert_keys
            
        # Reuse cached tree and data
        self.tree = self._tree_cache[tree_cache_key]
        self.insert_keys = self._data_cache[tree_cache_key]
        
        # Generate lookup keys with specified hit ratio (this is fast and hit_ratio specific)
        base_seed = 42 + hash((capacity, size, distribution, l_factor)) % 1000
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.insert_keys,
            hit_ratio=hit_ratio,
            seed=base_seed + 1000
        )
        gc.collect()
        gc.disable() # Enabled in teardown

    @classmethod
    def clear_cache(cls):
        """Clear the tree cache to free memory when needed."""
        cls._tree_cache.clear()
        cls._data_cache.clear()
        gc.collect()

    def time_retrieve_sequential(self, capacity, size, hit_ratio, l_factor):
        """Benchmark sequential retrieve operations."""
        for key in self.lookup_keys:
            self.tree.retrieve(key)


# Memory usage benchmarks
class GKPlusTreeMemoryBenchmarks(BaseBenchmark):
    """Benchmarks for memory usage of GKPlusTree operations."""
    
    params = [
        [4, 8, 16, 32],  # K values
        [1000, 10000],  # sizes
        [1.0, 2.0, 4.0, 8.0]  # l_factor values
    ]
    param_names = ['capacity', 'size', 'l_factor']
    
    def setup(self, capacity, size, l_factor):
        """Setup for memory benchmarking."""
        super().setup(capacity, size, l_factor)
        
        self.tree_class, _, _, _ = make_gkplustree_classes(capacity)
        self.keys = BenchmarkUtils.generate_deterministic_keys(size, seed=42)
        self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
        # Calculate ranks for entries
        group_size = calculate_group_size(capacity)
        self.ranks = [calc_rank_from_group_size(key, group_size) for key in self.keys]
        
        # Store l_factor for tree creation
        self.l_factor = l_factor

        gc.collect()
        gc.disable() # Enabled in teardown
    
    def peakmem_tree_construction(self, capacity, size, l_factor):
        """Measure peak memory during tree construction."""
        tree = self.tree_class(l_factor=l_factor)
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)
        return tree
