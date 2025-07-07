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
        [4, 8, 32],  # K values (capacities)
        [1000, 10000],  # data sizes
        ['uniform', 'sequential', 'clustered'],  # data distributions
        [1.0, 2.0, 4.0]  # l_factor values
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
    
    def peakmem_insert_entry_tree_construction(self, capacity, size, distribution, l_factor):
        """Measure peak memory usage during full tree construction."""
        tree = self.tree_class(l_factor=l_factor)
        
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)


class GKPlusTreeRetrieveBenchmarks(BaseBenchmark):
    """Benchmarks for GKPlusTreeBase.retrieve() method."""
    
    # Test different capacities, data sizes, hit ratios, and l_factors (distribution fixed to 'sequential')
    params = [
        [4, 8, 32],  # K values (capacities)
        [1000, 10000],  # data sizes
        [0.0, 1.0],  # hit ratios
        [1.0, 2.0, 4.0]  # l_factor values
    ]
    param_names = ['capacity', 'size', 'hit_ratio', 'l_factor']
    
    repeat = 5
    min_run_count = 3
    
    def setup(self, capacity, size, hit_ratio, l_factor):
        """Setup populated GKPlusTree and lookup keys for benchmarking."""
        super().setup(capacity, size, hit_ratio, l_factor)
        
        # Use only 'sequential' distribution
        distribution = 'sequential'
        
        # Create and populate GKPlusTree
        self.tree_class, _, self.klist_class, _ = make_gkplustree_classes(capacity)
        self.tree = self.tree_class(l_factor=l_factor)
        
        # Generate and insert test data
        base_seed = 42 + hash((capacity, size, distribution, l_factor)) % 1000
        self.insert_keys = BenchmarkUtils.generate_deterministic_keys(
            size=size,
            seed=base_seed,
            distribution=distribution
        )
        
        entries = BenchmarkUtils.create_test_entries(self.insert_keys)
        
        # Use bulk_create_gkplus_tree for efficient initial tree construction
        self.tree = bulk_create_gkplus_tree(entries, DIM=1, l_factor=l_factor, KListClass=self.klist_class)

        # Generate lookup keys with specified hit ratio
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.insert_keys,
            hit_ratio=hit_ratio,
            seed=base_seed + 1000
        )
        gc.collect()
        gc.disable() # Enabled in teardown

    def time_retrieve_sequential(self, capacity, size, hit_ratio, distribution, l_factor):
        """Benchmark sequential retrieve operations."""
        for key in self.lookup_keys:
            self.tree.retrieve(key)


class GKPlusTreeMixedWorkloadBenchmarks(BaseBenchmark):
    """Benchmarks for mixed insert/retrieve workloads."""
    
    params = [
        [4, 8, 32],  # K values
        [1000, 10000],  # sizes
        [0.1, 0.5, 0.9],  # insert ratios
        [1.0, 2.0, 4.0]  # l_factor values
    ]
    param_names = ['capacity', 'size', 'insert_ratio', 'l_factor']
    
    repeat = 5
    min_run_count = 3
    
    def setup(self, capacity, size, insert_ratio, l_factor):
        """Setup for mixed workload testing."""
        super().setup(capacity, size, insert_ratio, l_factor)
        
        # Create GKPlusTree
        self.tree_class, _, _, _ = make_gkplustree_classes(capacity)
        
        # Generate initial data and operations
        self.initial_keys = BenchmarkUtils.generate_deterministic_keys(
            size=size // 2,
            seed=42,
            distribution='uniform'
        )
        
        # Generate mixed operations
        random.seed(42)
        num_operations = size
        num_inserts = int(num_operations * insert_ratio)
        num_retrieves = num_operations - num_inserts
        
        # Keys for new insertions
        self.insert_keys = BenchmarkUtils.generate_deterministic_keys(
            size=num_inserts,
            seed=43,
            key_range=(max(self.initial_keys) + 1, max(self.initial_keys) + num_inserts + 1000)
        )
        
        # Keys for retrievals
        self.retrieve_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.initial_keys,
            hit_ratio=0.7,
            seed=44
        )[:num_retrieves]
        
        # Create mixed operation sequence
        operations = (['insert'] * num_inserts) + (['retrieve'] * num_retrieves)
        random.shuffle(operations)
        self.operations = operations
        
        # Prepare entries and ranks for insertions
        self.insert_entries = BenchmarkUtils.create_test_entries(self.insert_keys)
        # Store l_factor for tree creation
        self.l_factor = l_factor
        
        self.tree = bulk_create_gkplus_tree(self.insert_entries, DIM=1, l_factor=self.l_factor, KListClass=self.klist_class)
        
        gc.collect()
        gc.disable() # Enabled in teardown
    
    def time_mixed_workload(self, capacity, size, insert_ratio, l_factor):
        """Benchmark mixed insert/retrieve workload."""
        insert_idx = 0
        retrieve_idx = 0
        tree = self.tree
        for operation in self.operations:
            if operation == 'insert' and insert_idx < len(self.insert_entries):
                tree, _ = tree.insert_entry(self.insert_entries[insert_idx], self.insert_ranks[insert_idx])
                insert_idx += 1
            elif operation == 'retrieve' and retrieve_idx < len(self.retrieve_keys):
                tree.retrieve(self.retrieve_keys[retrieve_idx])
                retrieve_idx += 1


# Memory usage benchmarks
class GKPlusTreeMemoryBenchmarks(BaseBenchmark):
    """Benchmarks for memory usage of GKPlusTree operations."""
    
    params = [
        [4, 8, 32],  # K values
        [1000, 10000],  # sizes
        [1.0, 2.0, 4.0]  # l_factor values
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


# Comparison benchmarks between different algorithms
class CapacityComparisonBenchmarks(BaseBenchmark):
    """Benchmarks comparing different capacity configurations."""
    
    params = [
        [4, 8, 16, 32],  # capacities
        [1000, 10000],  # sizes
        ['insert', 'retrieve'],  # operation types
        [1.0, 2.0, 4.0]  # l_factor values
    ]
    param_names = ['capacity', 'size', 'operation', 'l_factor']

    repeat = 5
    min_run_count = 3
    
    def setup(self, capacity, size, operation, l_factor):
        """Setup for comparison benchmarking."""
        super().setup(capacity, size, operation, l_factor)
        
        # Create tree class and test data
        self.tree_class, _, _, _ = make_gkplustree_classes(capacity)
        self.keys = BenchmarkUtils.generate_deterministic_keys(size, seed=42)
        self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
        # Calculate ranks for insertions
        self.group_size = calculate_group_size(capacity)
        self.ranks = [calc_rank_from_group_size(entry.item.key, self.group_size) for entry in self.entries]
        
        if operation == 'retrieve':
            # Pre-populate tree for retrieve benchmarks
            self.tree = self.tree_class(l_factor=l_factor)
            for entry, rank in zip(self.entries, self.ranks):
                self.tree, _ = self.tree.insert_entry(entry, rank)
            
            # Generate lookup keys for retrieve operations
            self.lookup_keys = BenchmarkUtils.create_lookup_keys(
                insert_keys=self.keys,
                hit_ratio=0.8,
                seed=43
            )
        
        # Store l_factor for tree creation
        self.l_factor = l_factor
        
        gc.collect()
        gc.disable() # Enabled in teardown
    
    def time_operation(self, capacity, size, operation, l_factor):
        """Benchmark the specified operation with the given capacity."""
        if operation == 'insert':
            tree = self.tree_class(l_factor=l_factor)  # Fresh tree
            for entry, rank in zip(self.entries, self.ranks):
                tree, _ = tree.insert_entry(entry, rank)
        else:  # retrieve
            for key in self.lookup_keys:
                self.tree.retrieve(key)
