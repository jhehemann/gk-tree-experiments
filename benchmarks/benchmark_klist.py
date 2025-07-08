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
    min_run_count = 5

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

        for entry in self.entries:
            klist, _ = klist.insert_entry(entry)


class KListRetrieveBenchmarks(BaseBenchmark):
    """Benchmarks for KListBase.retrieve() method."""

    # For each K in [4, 8, 16, 32, 64, 128], test sizes = [K, 2K, 4K, 8K] via l_factor
    k_values = [4, 8, 16, 32, 64, 128]
    l_factors = [1, 2, 4, 8]

    params = [
        k_values,           # capacity (K)
        l_factors,          # l_factor (size = K * l_factor)
        [0.0, 1.0],    # hit ratios
    ]
    param_names = ['capacity', 'l_factor', 'hit_ratio']

    min_run_count = 5

    # Class-level cache for initialized KLlists and insert keys
    _klist_cache = {}
    _data_cache = {}

    def setup(self, capacity, l_factor, hit_ratio):
        """Setup populated KList and lookup keys for benchmarking."""
        size = capacity * l_factor
        super().setup(capacity, size, hit_ratio)

        # Use deterministic key generation parameters
        cache_key = (capacity, size)
        if cache_key not in self._klist_cache:
            # Build and cache KList and insert keys
            self.KListClass = make_gplustree_classes(capacity)[2]
            base_seed = 42 + hash((capacity, size)) % 1000
            insert_keys = BenchmarkUtils.generate_deterministic_keys(
                size=size,
                seed=base_seed,
                distribution='sequential'
            )
            entries = BenchmarkUtils.create_test_entries(insert_keys)
            klist = self.KListClass()
            for entry in entries:
                klist, _ = klist.insert_entry(entry)
            self._klist_cache[cache_key] = klist
            self._data_cache[cache_key] = insert_keys

        # Reuse cached KList and insert keys
        self.klist = self._klist_cache[cache_key]
        self.insert_keys = self._data_cache[cache_key]

        # Generate lookup keys with specified hit ratio
        base_seed = 42 + hash((capacity, size)) % 1000
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.insert_keys,
            hit_ratio=hit_ratio,
            seed=base_seed + 1000
        )
        gc.collect()
        gc.disable() # Enabled in teardown

    def time_retrieve_sequential(self, capacity, l_factor, hit_ratio, distribution):
        """Benchmark sequential retrieve operations."""
        for key in self.lookup_keys:
            self.klist.retrieve(key)
