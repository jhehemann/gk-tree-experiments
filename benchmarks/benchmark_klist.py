"""
ASV benchmarks for KListBase operations.

This module contains comprehensive benchmarks for the KListBase class,
focusing on batch construction (full data structure creation) and 
retrieve() operations with various data patterns and sizes.
"""

import gc
import random
from typing import List

from gplus_trees.factory import make_gplustree_classes
from gplus_trees.base import Item, Entry
from benchmarks.benchmark_utils import BaseBenchmark, BenchmarkUtils



class KListBatchInsertBenchmarks(BaseBenchmark):
    """Benchmarks for K-List construction via sequential inserts."""

    # For each K in [4, 8, 16, 32, 64, 128], test sizes = [K, 2K, 4K, 8K] via l_factor
    k_values = [4, 8, 16, 32, 64, 128]
    l_factors = [1, 2, 4, 8]

    params = [
        k_values,           # capacity (K)
        l_factors,          # l_factor (size = K * l_factor)
        ['uniform', 'sequential', 'clustered']  # data distributions
    ]
    param_names = ['capacity', 'l_factor', 'distribution']

    # Timing configuration for robust measurements
    number = 1  
    repeat = 5
    min_run_count = 3

    def setup(self, capacity, l_factor, distribution):
        """Setup KList and test data for batch insert benchmarking."""
        size = capacity * l_factor
        super().setup(capacity, size, distribution)

        # Create KList class with specified capacity
        _, _, self.KListClass, _ = make_gplustree_classes(capacity)

        # Generate deterministic test data
        self.keys = BenchmarkUtils.generate_deterministic_keys(
            size=size,
            seed=42 + hash((capacity, size, distribution)) % 1000,
            distribution=distribution
        )
        self.entries = BenchmarkUtils.create_test_entries(self.keys)

    def time_insert_entry_batch_construction(self, capacity, l_factor, distribution):
        """Benchmark full klist construction by inserting all entries."""
        klist = self.KListClass()

        # Insert all entries (this tests bulk insertion performance)
        for entry in self.entries:
            klist, _ = klist.insert_entry(entry)

    def peakmem_insert_entry_batch_construction(self, capacity, l_factor, distribution):
        """Measure peak memory usage during full klist construction."""
        klist = self.KListClass()

        for entry in self.entries:
            klist, _ = klist.insert_entry(entry)

        return klist


class KListRetrieveBenchmarks(BaseBenchmark):
    """Benchmarks for KListBase.retrieve() method."""

    # For each K in [4, 8, 16, 32, 64, 128], test sizes = [K, 2K, 4K, 8K] via l_factor
    k_values = [4, 8, 16, 32, 64, 128]
    l_factors = [1, 2, 4, 8]

    params = [
        k_values,           # capacity (K)
        l_factors,          # l_factor (size = K * l_factor)
        [0.0, 1.0],    # hit ratios
        ['uniform', 'sequential', 'clustered']  # data distributions
    ]
    param_names = ['capacity', 'l_factor', 'hit_ratio', 'distribution']

    number = 1
    repeat = 5
    min_run_count = 3

    def setup(self, capacity, l_factor, hit_ratio, distribution):
        """Setup populated KList and lookup keys for benchmarking."""
        size = capacity * l_factor
        super().setup(capacity, size, hit_ratio, distribution)

        # Create and populate KList
        _, _, self.KListClass, _ = make_gplustree_classes(capacity)
        self.klist = self.KListClass()

        # Generate and insert test data
        base_seed = 42 + hash((capacity, size, distribution)) % 1000
        self.insert_keys = BenchmarkUtils.generate_deterministic_keys(
            size=size,
            seed=base_seed,
            distribution=distribution
        )

        entries = BenchmarkUtils.create_test_entries(self.insert_keys)
        for entry in entries:
            self.klist, _ = self.klist.insert_entry(entry)

        # Generate lookup keys with specified hit ratio
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.insert_keys,
            hit_ratio=hit_ratio,
            seed=base_seed + 1000
        )

    def time_retrieve_sequential(self, capacity, l_factor, hit_ratio, distribution):
        """Benchmark sequential retrieve operations."""
        for key in self.lookup_keys:
            self.klist.retrieve(key)


# class KListScalabilityBenchmarks(BaseBenchmark):
#     """Benchmarks to test scalability of KList operations."""
    
#     # Focus on larger sizes for scalability testing
#     params = [
#         [8, 16, 32],  # K values
#         [1000, 5000, 10000, 20000, 50000],  # larger sizes
#     ]
#     param_names = ['capacity', 'size']
    
#     number = 1
#     repeat = 3  # Fewer repeats for large datasets
#     min_run_count = 2
    
#     def setup(self, capacity, size):
#         """Setup for scalability testing."""
#         super().setup(capacity, size)
        
#         # Create KList and test data
#         _, _, self.KListClass, _ = make_gplustree_classes(capacity)
        
#         # Use uniform distribution for consistency
#         self.keys = BenchmarkUtils.generate_deterministic_keys(
#             size=size,
#             seed=42,
#             distribution='uniform'
#         )
#         self.entries = BenchmarkUtils.create_test_entries(self.keys)
    
#     def time_insert_scalability(self, capacity, size):
#         """Test how insert performance scales with data size."""
#         klist = self.KListClass()
        
#         for entry in self.entries:
#             klist.insert_entry(entry)
    
#     def time_retrieve_scalability(self, capacity, size):
#         """Test how retrieve performance scales with data size."""
#         # First build the structure
#         klist = self.KListClass()
#         for entry in self.entries:
#             klist.insert_entry(entry)
        
#         # Then time retrieval of a subset
#         lookup_keys = self.keys[::10]  # Every 10th key
#         for key in lookup_keys:
#             klist.retrieve(key)


class KListMixedWorkloadBenchmarks(BaseBenchmark):
    """Benchmarks for mixed insert/retrieve workloads."""
    
    params = [
        [4, 8, 32],  # K values
        [1000, 10000],  # sizes
        [0.1, 0.5, 0.9],  # insert ratios (fraction of operations that are inserts)
    ]
    param_names = ['capacity', 'size', 'insert_ratio']
    
    number = 1
    repeat = 3
    
    def setup(self, capacity, size, insert_ratio):
        """Setup for mixed workload testing."""
        super().setup(capacity, size, insert_ratio)
        
        # Create KList
        _, _, self.KListClass, _ = make_gplustree_classes(capacity)
        
        # Generate initial data and operations
        self.initial_keys = BenchmarkUtils.generate_deterministic_keys(
            size=size // 2,  # Start with half the data
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
        
        # Keys for retrievals (mix of existing and non-existing)
        self.retrieve_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.initial_keys,
            hit_ratio=0.7,
            seed=44
        )[:num_retrieves]
        
        # Create mixed operation sequence
        operations = (['insert'] * num_inserts) + (['retrieve'] * num_retrieves)
        random.shuffle(operations)
        self.operations = operations
        
        # Prepare entries for insertions
        self.insert_entries = BenchmarkUtils.create_test_entries(self.insert_keys)
    
    def time_mixed_workload(self, capacity, size, insert_ratio):
        """Benchmark mixed insert/retrieve workload."""
        # Start with initial data
        klist = self.KListClass()
        initial_entries = BenchmarkUtils.create_test_entries(self.initial_keys)
        for entry in initial_entries:
            klist.insert_entry(entry)
        
        # Execute mixed operations
        insert_idx = 0
        retrieve_idx = 0
        
        for operation in self.operations:
            if operation == 'insert' and insert_idx < len(self.insert_entries):
                klist.insert_entry(self.insert_entries[insert_idx])
                insert_idx += 1
            elif operation == 'retrieve' and retrieve_idx < len(self.retrieve_keys):
                klist.retrieve(self.retrieve_keys[retrieve_idx])
                retrieve_idx += 1


# Memory usage benchmarks
class KListMemoryBenchmarks(BaseBenchmark):
    """Benchmarks for memory usage of KList operations."""
    
    params = [
        [4, 8, 32],  # K values
        [1000, 10000],  # sizes
    ]
    param_names = ['capacity', 'size']
    
    def setup(self, capacity, size):
        """Setup for memory benchmarking."""
        super().setup(capacity, size)
        
        _, _, self.KListClass, _ = make_gplustree_classes(capacity)
        self.keys = BenchmarkUtils.generate_deterministic_keys(size, seed=42)
        self.entries = BenchmarkUtils.create_test_entries(self.keys)
    
    def peakmem_klist_construction(self, capacity, size):
        """Measure peak memory during KList construction."""
        klist = self.KListClass()
        for entry in self.entries:
            klist.insert_entry(entry)
        return klist
    
    def track_klist_item_count(self, capacity, size):
        """Track the number of items as a performance indicator."""
        klist = self.KListClass()
        for entry in self.entries:
            klist.insert_entry(entry)
        return klist.item_count()
    
    def track_klist_physical_height(self, capacity, size):
        """Track the physical height (number of nodes) as a structural indicator."""
        klist = self.KListClass()
        for entry in self.entries:
            klist.insert_entry(entry)
        return klist.physical_height()
