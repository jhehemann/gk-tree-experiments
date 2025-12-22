"""
Benchmarking utilities for G+ trees and K-lists.

This module provides common utilities and base classes for ASV benchmarking
that work optimally with ASV's built-in timing and stabilization mechanisms.

Reproducibility:
    All random data generation uses deterministic seeds by default.
    The default seed can be overridden via the BENCHMARK_SEED environment variable.
    
Logging:
    Benchmarks should be run with logging at INFO level or higher to avoid
    performance contamination from verbose debug output.
"""

import random
import gc
import os
import logging
import warnings
from typing import List, Tuple
import numpy as np

from gplus_trees.base import Entry, LeafItem, ItemData

# Default seed for deterministic benchmarking - can be overridden via environment variable
DEFAULT_BENCHMARK_SEED = int(os.environ.get('BENCHMARK_SEED', '42'))

# Logger for benchmark utilities
_logger = logging.getLogger(__name__)



class BenchmarkUtils:
    """Utility class for ASV benchmarking operations.
    
    This class provides methods for generating deterministic test data and
    performing common benchmark setup operations while ensuring reproducibility.
    """
    
    @staticmethod
    def check_logging_level():
        """
        Check if logging level is appropriate for benchmarking.
        
        Issues a warning if DEBUG or lower (more verbose) logging is enabled,
        as this can significantly contaminate benchmark results with I/O overhead.
        """
        gplus_logger = logging.getLogger("gplus_trees")
        # Use getEffectiveLevel() to handle NOTSET correctly
        effective_level = gplus_logger.getEffectiveLevel()
        if effective_level <= logging.DEBUG:
            raise ValueError(
                f"Logging level is set to {logging.getLevelName(effective_level)}. "
                "Benchmarks require logging to be at INFO level or higher to avoid "
                "performance contamination from verbose debug output."
            )
    
    @staticmethod
    def generate_deterministic_keys(size: int, 
                                  seed: int = None,
                                  key_range: Tuple[int, int] = (1, 1000000),
                                  distribution: str = 'uniform') -> List[int]:
        """
        Generate deterministic keys for benchmarking.
        
        Args:
            size: Number of keys to generate
            seed: Random seed for reproducibility. If None, uses DEFAULT_BENCHMARK_SEED.
            key_range: Range of key values (min, max)
            distribution: Distribution type ('uniform', 'clustered', 'sequential')
            
        Returns:
            List of deterministic keys
            
        Note:
            This function always produces the same output for the same inputs,
            ensuring benchmark reproducibility.
        """
        if seed is None:
            seed = DEFAULT_BENCHMARK_SEED
            
        random.seed(seed)
        np.random.seed(seed)
        
        min_key, max_key = key_range
        
        if distribution == 'uniform':
            # Sample without replacement for uniform distribution (no duplicates)
            population = np.arange(min_key, max_key + 1)
            return np.random.choice(population, size=size, replace=False).tolist()
        elif distribution == 'clustered':
            # Create clustered data with some hot spots - more efficient numpy operations
            cluster_centers = np.linspace(min_key, max_key, 5, dtype=int)
            cluster_size = size // 5
            keys = np.empty(0, dtype=int)
            
            for center in cluster_centers:
                cluster_keys = np.random.normal(center, (max_key - min_key) // 20, cluster_size)
                cluster_keys = np.clip(cluster_keys, min_key, max_key).astype(int)
                keys = np.concatenate([keys, cluster_keys])
            
            # Fill remaining with uniform distribution
            remaining = size - len(keys)
            if remaining > 0:
                uniform_keys = np.random.randint(min_key, max_key + 1, size=remaining)
                keys = np.concatenate([keys, uniform_keys])
            
            # Remove duplicates efficiently and trim to size
            unique_keys = np.unique(keys)
            if len(unique_keys) >= size:
                return unique_keys[:size].tolist()
            else:
                # If not enough unique keys, pad with more random unique keys
                additional_needed = size - len(unique_keys)
                # Build population of keys not already used
                full_population = np.arange(min_key, max_key + 1)
                remaining = np.setdiff1d(full_population, unique_keys)
                if len(remaining) < additional_needed:
                    raise ValueError("Not enough unique keys available to generate desired size")
                pad = np.random.choice(remaining, size=additional_needed, replace=False)
                return np.concatenate([unique_keys, pad]).tolist()
        elif distribution == 'sequential':
            return list(range(min_key, min_key + size))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def create_test_entries(keys: List[int]) -> List[Entry]:
        """Create Entry objects from a list of keys."""
        entries = []
        for key in keys:
            item_data = ItemData(key, f"value_{key}")
            entries.append(Entry(LeafItem(item_data), None))
        return entries
    
    @staticmethod
    def create_test_items(keys: List[int]) -> List[LeafItem]:
        """Create LeafItem objects from a list of keys."""
        entries = []
        for key in keys:
            item_data = ItemData(key, f"value_{key}")
            entries.append(LeafItem(item_data))
        return entries

    @staticmethod
    def create_lookup_keys(insert_keys: List[int], 
                          hit_ratio: float = 0.8,
                          seed: int = None,
                          num_lookups = 1000) -> List[int]:
        """
        Create keys for lookup operations with specified hit ratio.
        
        Args:
            insert_keys: Keys that were inserted (for hits)
            hit_ratio: Ratio of lookups that should be hits (0.0 to 1.0)
            seed: Random seed for reproducibility. If None, uses DEFAULT_BENCHMARK_SEED.
            num_lookups: Number of lookup keys to generate
            
        Returns:
            List of lookup keys
        """
        if seed is None:
            seed = DEFAULT_BENCHMARK_SEED
            
        random.seed(seed)
        
        # Handle edge case of empty insert_keys
        if not insert_keys:
            return np.random.randint(1, 1000001, size=1000).tolist()
        
        num_hits = int(num_lookups * hit_ratio)
        num_misses = num_lookups - num_hits
        
        # Select hits from inserted keys (allowing duplicates to maintain hit ratio)
        if num_hits > 0:
            hit_keys = random.choices(insert_keys, k=num_hits)
        else:
            hit_keys = []
        
        # Generate miss keys more efficiently using set for O(1) lookups
        insert_keys_set = set(insert_keys)
        max_key = max(insert_keys)
        min_key = min(insert_keys)
        
        # Pre-generate more miss candidates to reduce iterations
        candidate_range_size = (max_key * 2 - min_key + 1)
        if candidate_range_size > 0 and num_misses > 0:
            # Generate more candidates than needed to account for collisions
            candidates_needed = min(num_misses * 3, candidate_range_size)
            miss_candidates = np.random.randint(min_key, max_key * 2 + 1, size=candidates_needed)
            
            # Filter out keys that are in insert_keys
            miss_keys = []
            for key in miss_candidates:
                if key not in insert_keys_set and len(miss_keys) < num_misses:
                    miss_keys.append(int(key))
            
            # If we still need more miss keys, generate them one by one (fallback)
            while len(miss_keys) < num_misses:
                key = random.randint(min_key, max_key * 2)
                if key not in insert_keys_set:
                    miss_keys.append(key)
        else:
            miss_keys = []
        
        # Combine and shuffle
        lookup_keys = hit_keys + miss_keys
        random.shuffle(lookup_keys)
        return lookup_keys


class BaseBenchmark:
    """Base class for ASV benchmarks optimized for ASV's built-in timing.
    
    This class provides a standard setup/teardown pattern that ensures:
    - Garbage collection is disabled during timed sections
    - Logging level is appropriate for benchmarking
    - Consistent parameter handling across benchmarks
    
    ASV Timing:
        ASV handles timing automatically. The warmup_time and sample_time
        settings control how ASV stabilizes and measures performance.
    """
    
    # Parameters for benchmarking
    params = []
    param_names = []
    
    # Let ASV handle timing optimization automatically
    warmup_time = 0.1
    sample_time = 0.4
    
    def setup(self, *params):
        """Setup method called before each benchmark.
        
        This method should prepare test data and disable garbage collection
        to ensure consistent timing measurements. ASV calls this before
        the timed section begins.
        
        Subclasses should:
        1. Call super().setup(*params) first
        2. Prepare test data
        3. Call gc.collect() to clean up setup overhead
        4. Call gc.disable() to prevent GC during measurement
        """
        # Check logging level and warn if too verbose
        BenchmarkUtils.check_logging_level()

    def teardown(self, *params):
        """Teardown method called after each benchmark.
        
        Re-enables garbage collection after measurement completes.
        ASV calls this after the timed section ends.
        """
        # Always re-enable garbage collection after benchmarking
        if not gc.isenabled():
            gc.enable()
