"""
ASV benchmarks for GKPlusTreeBase operations.

This module contains comprehensive benchmarks for the GKPlusTreeBase class,
focusing on insert_entry() and retrieve() operations with various
data patterns and sizes.
"""

import gc
import random
from typing import List

from gplus_trees.g_k_plus.factory import make_gkplustree_classes, create_gkplus_tree
from gplus_trees.g_k_plus.utils import calc_rank_from_group_size, calculate_group_size
from gplus_trees.base import Item, Entry
from benchmarks.benchmark_utils import BaseBenchmark, BenchmarkUtils, RobustTimer


class GKPlusTreeInsertBenchmarks(BaseBenchmark):
    """Benchmarks for GKPlusTreeBase.insert_entry() method."""
    
    # Test different capacities and data sizes
    params = [
        [4, 8, 16],  # K values (capacities)
        [1000],  # data sizes
        ['uniform', 'sequential', 'clustered']  # data distributions
    ]
    param_names = ['capacity', 'size', 'distribution']
    
    number = 1
    repeat = 5
    min_run_count = 3
    
    def setup(self, capacity, size, distribution):
        """Setup GKPlusTree and test data for benchmarking."""
        super().setup(capacity, size, distribution)
        
        # Create GKPlusTree class with specified capacity
        self.GKPlusTreeClass, _, _, _ = make_gkplustree_classes(capacity)
        
        # Generate deterministic test data
        self.keys = BenchmarkUtils.generate_deterministic_keys(
            size=size,
            seed=42 + hash((capacity, size, distribution)) % 1000,
            distribution=distribution
        )
        self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
        # Calculate ranks for entries
        group_size = calculate_group_size(capacity)
        self.ranks = [calc_rank_from_group_size(key, group_size) for key in self.keys]
    
    def time_insert_entry_sequential(self, capacity, size, distribution):
        """Benchmark sequential insertions into an empty GKPlusTree."""
        tree = self.GKPlusTreeClass()
        
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)
    
    def time_insert_entry_batch_construction(self, capacity, size, distribution):
        """Benchmark batch construction by inserting all entries."""
        tree = self.GKPlusTreeClass()
        
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)
    
    def peakmem_insert_entry_sequential(self, capacity, size, distribution):
        """Measure peak memory usage during sequential insertions."""
        tree = self.GKPlusTreeClass()
        
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)
        
        return tree


class GKPlusTreeRetrieveBenchmarks(BaseBenchmark):
    """Benchmarks for GKPlusTreeBase.retrieve() method."""
    
    # Test different capacities, data sizes, and hit ratios
    params = [
        [4, 8, 16],  # K values (capacities)
        [1000],  # data sizes
        [0.0, 0.5, 0.8, 1.0],  # hit ratios
        ['uniform', 'sequential', 'clustered']  # data distributions
    ]
    param_names = ['capacity', 'size', 'hit_ratio', 'distribution']
    
    number = 1
    repeat = 5
    min_run_count = 3
    
    def setup(self, capacity, size, hit_ratio, distribution):
        """Setup populated GKPlusTree and lookup keys for benchmarking."""
        super().setup(capacity, size, hit_ratio, distribution)
        
        # Create and populate GKPlusTree
        self.GKPlusTreeClass, _, _, _ = make_gkplustree_classes(capacity)
        self.tree = self.GKPlusTreeClass()
        
        # Generate and insert test data
        base_seed = 42 + hash((capacity, size, distribution)) % 1000
        self.insert_keys = BenchmarkUtils.generate_deterministic_keys(
            size=size,
            seed=base_seed,
            distribution=distribution
        )
        
        entries = BenchmarkUtils.create_test_entries(self.insert_keys)
        group_size = calculate_group_size(capacity)
        
        for entry in entries:
            rank = calc_rank_from_group_size(entry.item.key, group_size)
            self.tree, _ = self.tree.insert_entry(entry, rank)
        
        # Generate lookup keys with specified hit ratio
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.insert_keys,
            hit_ratio=hit_ratio,
            seed=base_seed + 1000
        )
    
    def time_retrieve_sequential(self, capacity, size, hit_ratio, distribution):
        """Benchmark sequential retrieve operations."""
        for key in self.lookup_keys:
            self.tree.retrieve(key)
    
    def time_retrieve_with_next(self, capacity, size, hit_ratio, distribution):
        """Benchmark retrieve operations with next entry (full retrieve)."""
        for key in self.lookup_keys:
            self.tree.retrieve(key, with_next=True)
    
    def time_retrieve_without_next(self, capacity, size, hit_ratio, distribution):
        """Benchmark retrieve operations without next entry (faster path)."""
        for key in self.lookup_keys:
            self.tree.retrieve(key, with_next=False)


# class GKPlusTreeScalabilityBenchmarks(BaseBenchmark):
#     """Benchmarks to test scalability of GKPlusTree operations."""
    
#     # Focus on larger sizes for scalability testing
#     params = [
#         [8, 16, 32],  # K values
#         [1000, 10000],  # larger sizes
#     ]
#     param_names = ['capacity', 'size']
    
#     number = 1
#     repeat = 3  # Fewer repeats for large datasets
#     min_run_count = 2
    
#     def setup(self, capacity, size):
#         """Setup for scalability testing."""
#         super().setup(capacity, size)
        
#         # Create GKPlusTree and test data
#         self.GKPlusTreeClass, _, _, _ = make_gkplustree_classes(capacity)
        
#         # Use uniform distribution for consistency
#         self.keys = BenchmarkUtils.generate_deterministic_keys(
#             size=size,
#             seed=42,
#             distribution='uniform'
#         )
#         self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
#         # Calculate ranks for entries
#         group_size = calculate_group_size(capacity)
#         self.ranks = [calc_rank_from_group_size(key, group_size) for key in self.keys]
    
#     def time_insert_scalability(self, capacity, size):
#         """Test how insert performance scales with data size."""
#         tree = self.GKPlusTreeClass()
        
#         for entry, rank in zip(self.entries, self.ranks):
#             tree, _ = tree.insert_entry(entry, rank)
    
#     def time_retrieve_scalability(self, capacity, size):
#         """Test how retrieve performance scales with data size."""
#         # First build the structure
#         tree = self.GKPlusTreeClass()
#         for entry, rank in zip(self.entries, self.ranks):
#             tree, _ = tree.insert_entry(entry, rank)
        
#         # Then time retrieval of a subset
#         lookup_keys = self.keys[::10]  # Every 10th key
#         for key in lookup_keys:
#             tree.retrieve(key)


class GKPlusTreeMixedWorkloadBenchmarks(BaseBenchmark):
    """Benchmarks for mixed insert/retrieve workloads."""
    
    params = [
        [8, 16, 32],  # K values
        [1000],  # sizes
        [0.1, 0.5, 0.9],  # insert ratios
    ]
    param_names = ['capacity', 'size', 'insert_ratio']
    
    number = 1
    repeat = 3
    
    def setup(self, capacity, size, insert_ratio):
        """Setup for mixed workload testing."""
        super().setup(capacity, size, insert_ratio)
        
        # Create GKPlusTree
        self.GKPlusTreeClass, _, _, _ = make_gkplustree_classes(capacity)
        
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
        group_size = calculate_group_size(capacity)
        self.insert_ranks = [calc_rank_from_group_size(key, group_size) for key in self.insert_keys]
    
    def time_mixed_workload(self, capacity, size, insert_ratio):
        """Benchmark mixed insert/retrieve workload."""
        # Start with initial data
        tree = self.GKPlusTreeClass()
        initial_entries = BenchmarkUtils.create_test_entries(self.initial_keys)
        group_size = calculate_group_size(capacity)
        
        for entry in initial_entries:
            rank = calc_rank_from_group_size(entry.item.key, group_size)
            tree, _ = tree.insert_entry(entry, rank)
        
        # Execute mixed operations
        insert_idx = 0
        retrieve_idx = 0
        
        for operation in self.operations:
            if operation == 'insert' and insert_idx < len(self.insert_entries):
                tree, _ = tree.insert_entry(self.insert_entries[insert_idx], self.insert_ranks[insert_idx])
                insert_idx += 1
            elif operation == 'retrieve' and retrieve_idx < len(self.retrieve_keys):
                tree.retrieve(self.retrieve_keys[retrieve_idx])
                retrieve_idx += 1


# class GKPlusTreeStructuralBenchmarks(BaseBenchmark):
#     """Benchmarks for structural properties and operations of GKPlusTree."""
    
#     params = [
#         [8, 16, 32],  # K values
#         [1000, 10000],  # sizes
#     ]
#     param_names = ['capacity', 'size']
    
#     def setup(self, capacity, size):
#         """Setup for structural benchmarking."""
#         super().setup(capacity, size)
        
#         self.GKPlusTreeClass, _, _, _ = make_gkplustree_classes(capacity)
#         self.keys = BenchmarkUtils.generate_deterministic_keys(size, seed=42)
#         self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
#         # Calculate ranks for entries
#         group_size = calculate_group_size(capacity)
#         self.ranks = [calc_rank_from_group_size(key, group_size) for key in self.keys]
    
#     def time_tree_construction_and_height(self, capacity, size):
#         """Benchmark tree construction and measure final height."""
#         tree = self.GKPlusTreeClass()
#         for entry, rank in zip(self.entries, self.ranks):
#             tree, _ = tree.insert_entry(entry, rank)
        
#         # Access height to ensure it's computed
#         height = tree.height()
#         return height
    
#     def track_tree_height(self, capacity, size):
#         """Track the tree height as a structural indicator."""
#         tree = self.GKPlusTreeClass()
#         for entry, rank in zip(self.entries, self.ranks):
#             tree, _ = tree.insert_entry(entry, rank)
#         return tree.height()
    
#     def track_tree_item_count(self, capacity, size):
#         """Track the number of items as a performance indicator."""
#         tree = self.GKPlusTreeClass()
#         for entry, rank in zip(self.entries, self.ranks):
#             tree, _ = tree.insert_entry(entry, rank)
#         return tree.item_count()


# Memory usage benchmarks
class GKPlusTreeMemoryBenchmarks(BaseBenchmark):
    """Benchmarks for memory usage of GKPlusTree operations."""
    
    params = [
        [8, 16, 32],  # K values
        [1000],  # sizes
    ]
    param_names = ['capacity', 'size']
    
    def setup(self, capacity, size):
        """Setup for memory benchmarking."""
        super().setup(capacity, size)
        
        self.GKPlusTreeClass, _, _, _ = make_gkplustree_classes(capacity)
        self.keys = BenchmarkUtils.generate_deterministic_keys(size, seed=42)
        self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
        # Calculate ranks for entries
        group_size = calculate_group_size(capacity)
        self.ranks = [calc_rank_from_group_size(key, group_size) for key in self.keys]
    
    def peakmem_tree_construction(self, capacity, size):
        """Measure peak memory during tree construction."""
        tree = self.GKPlusTreeClass()
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)
        return tree


# Comparison benchmarks between different algorithms
class CapacityComparisonBenchmarks(BaseBenchmark):
    """Benchmarks comparing different capacity configurations."""
    
    params = [
        [1000],  # sizes
        ['insert', 'retrieve']  # operation types
    ]
    param_names = ['size', 'operation']
    
    def setup(self, size, operation):
        """Setup for comparison benchmarking."""
        super().setup(size, operation)
        
        # Create trees with different capacities
        self.trees = {}
        self.keys = BenchmarkUtils.generate_deterministic_keys(size, seed=42)
        self.entries = BenchmarkUtils.create_test_entries(self.keys)
        
        for capacity in [4, 8, 16, 32]:
            GKPlusTreeClass, _, _, _ = make_gkplustree_classes(capacity)
            self.trees[capacity] = GKPlusTreeClass()
            
            if operation == 'retrieve':
                # Pre-populate for retrieve benchmarks
                group_size = calculate_group_size(capacity)
                tree = self.trees[capacity]
                for entry in self.entries:
                    rank = calc_rank_from_group_size(entry.item.key, group_size)
                    tree, _ = tree.insert_entry(entry, rank)
                self.trees[capacity] = tree
        
        # Lookup keys for retrieve operations
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.keys,
            hit_ratio=0.8,
            seed=43
        )
    
    def time_capacity_4(self, size, operation):
        """Benchmark with capacity 4."""
        if operation == 'insert':
            tree = self.trees[4].__class__()  # Fresh tree
            group_size = calculate_group_size(4)
            for entry in self.entries:
                rank = calc_rank_from_group_size(entry.item.key, group_size)
                tree, _ = tree.insert_entry(entry, rank)
        else:  # retrieve
            for key in self.lookup_keys:
                self.trees[4].retrieve(key)
    
    def time_capacity_8(self, size, operation):
        """Benchmark with capacity 8."""
        if operation == 'insert':
            tree = self.trees[8].__class__()
            group_size = calculate_group_size(8)
            for entry in self.entries:
                rank = calc_rank_from_group_size(entry.item.key, group_size)
                tree, _ = tree.insert_entry(entry, rank)
        else:
            for key in self.lookup_keys:
                self.trees[8].retrieve(key)
    
    def time_capacity_16(self, size, operation):
        """Benchmark with capacity 16."""
        if operation == 'insert':
            tree = self.trees[16].__class__()
            group_size = calculate_group_size(16)
            for entry in self.entries:
                rank = calc_rank_from_group_size(entry.item.key, group_size)
                tree, _ = tree.insert_entry(entry, rank)
        else:
            for key in self.lookup_keys:
                self.trees[16].retrieve(key)
    
    def time_capacity_32(self, size, operation):
        """Benchmark with capacity 32."""
        if operation == 'insert':
            tree = self.trees[32].__class__()
            group_size = calculate_group_size(32)
            for entry in self.entries:
                rank = calc_rank_from_group_size(entry.item.key, group_size)
                tree, _ = tree.insert_entry(entry, rank)
        else:
            for key in self.lookup_keys:
                self.trees[32].retrieve(key)
