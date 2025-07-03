"""
Benchmarking utilities for G+ trees and K-lists.

This module provides common utilities and base classes for robust benchmarking
that minimizes the impact of CPU and memory load variations.
"""

import gc
import random
import time
import statistics
from typing import List, Tuple, Optional, Iterator
import numpy as np

from gplus_trees.base import Item, Entry
from gplus_trees.factory import make_gplustree_classes, create_gplustree


class BenchmarkUtils:
    """Utility class for robust benchmarking operations."""
    
    @staticmethod
    def ensure_stable_system(warmup_iterations: int = 5, 
                           stabilization_time: float = 0.1) -> None:
        """
        Perform operations to stabilize the system before benchmarking.
        
        Args:
            warmup_iterations: Number of warmup iterations to perform
            stabilization_time: Time to wait for system stabilization
        """
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Warm up the CPU with some computation
        for _ in range(warmup_iterations):
            _ = sum(i * i for i in range(1000))
        
        # Wait for system stabilization
        time.sleep(stabilization_time)
    
    @staticmethod
    def generate_deterministic_keys(size: int, 
                                  seed: int = 42,
                                  key_range: Tuple[int, int] = (1, 1000000),
                                  distribution: str = 'uniform') -> List[int]:
        """
        Generate deterministic keys for benchmarking.
        
        Args:
            size: Number of keys to generate
            seed: Random seed for reproducibility
            key_range: Range of key values (min, max)
            distribution: Distribution type ('uniform', 'clustered', 'sequential')
            
        Returns:
            List of deterministic keys
        """
        random.seed(seed)
        np.random.seed(seed)
        
        min_key, max_key = key_range
        
        if distribution == 'uniform':
            return random.sample(range(min_key, max_key + 1), size)
        elif distribution == 'clustered':
            # Create clustered data with some hot spots
            cluster_centers = [min_key + i * (max_key - min_key) // 5 for i in range(5)]
            keys = []
            cluster_size = size // 5
            for center in cluster_centers:
                cluster_keys = np.random.normal(center, (max_key - min_key) // 20, cluster_size)
                cluster_keys = np.clip(cluster_keys, min_key, max_key).astype(int)
                keys.extend(cluster_keys.tolist())
            
            # Fill remaining with uniform distribution
            remaining = size - len(keys)
            if remaining > 0:
                keys.extend(random.sample(range(min_key, max_key + 1), remaining))
            
            return list(set(keys))[:size]  # Remove duplicates and trim to size
        elif distribution == 'sequential':
            return list(range(min_key, min_key + size))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def create_test_entries(keys: List[int]) -> List[Entry]:
        """
        Create Entry objects from a list of keys.
        
        Args:
            keys: List of keys to create entries for
            
        Returns:
            List of Entry objects
        """
        return [Entry(Item(key, f"value_{key}"), None) for key in keys]
    
    @staticmethod
    def create_lookup_keys(insert_keys: List[int], 
                          hit_ratio: float = 0.8,
                          seed: int = 42) -> List[int]:
        """
        Create keys for lookup operations with specified hit ratio.
        
        Args:
            insert_keys: Keys that were inserted (for hits)
            hit_ratio: Ratio of lookups that should be hits (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            List of lookup keys
        """
        random.seed(seed)
        
        num_lookups = len(insert_keys)
        num_hits = int(num_lookups * hit_ratio)
        num_misses = num_lookups - num_hits
        
        # Select hits from inserted keys
        hit_keys = random.sample(insert_keys, num_hits)
        
        # Generate miss keys (keys not in insert_keys)
        max_key = max(insert_keys) if insert_keys else 1000000
        min_key = min(insert_keys) if insert_keys else 1
        
        miss_keys = []
        attempts = 0
        while len(miss_keys) < num_misses and attempts < num_misses * 10:
            key = random.randint(min_key, max_key * 2)
            if key not in insert_keys and key not in miss_keys:
                miss_keys.append(key)
            attempts += 1
        
        # Combine and shuffle
        lookup_keys = hit_keys + miss_keys
        random.shuffle(lookup_keys)
        return lookup_keys


class RobustTimer:
    """Timer class for robust timing measurements."""
    
    def __init__(self, num_iterations: int = 1):
        self.num_iterations = num_iterations
        self.times = []
    
    def time_operation(self, operation_func, *args, **kwargs) -> float:
        """
        Time an operation multiple times and return the median time.
        
        Args:
            operation_func: Function to time
            *args, **kwargs: Arguments for the function
            
        Returns:
            Median execution time in seconds
        """
        BenchmarkUtils.ensure_stable_system()
        
        times = []
        for _ in range(self.num_iterations):
            # Force garbage collection before each iteration
            gc.collect()
            
            start_time = time.perf_counter()
            result = operation_func(*args, **kwargs)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        self.times = times
        return statistics.median(times)
    
    def get_statistics(self) -> dict:
        """Get timing statistics."""
        if not self.times:
            return {}
        
        return {
            'median': statistics.median(self.times),
            'mean': statistics.mean(self.times),
            'min': min(self.times),
            'max': max(self.times),
            'std': statistics.stdev(self.times) if len(self.times) > 1 else 0,
            'q25': np.percentile(self.times, 25),
            'q75': np.percentile(self.times, 75)
        }


class BaseBenchmark:
    """Base class for ASV benchmarks with robust timing."""
    
    # Parameters for benchmarking
    params = []
    param_names = []
    
    # Number of iterations for timing (can be overridden)
    number = 1
    repeat = 3
    min_run_count = 5
    
    def setup(self, *params):
        """Setup method called before each benchmark."""
        BenchmarkUtils.ensure_stable_system()
    
    def teardown(self, *params):
        """Teardown method called after each benchmark."""
        # Force garbage collection
        gc.collect()
