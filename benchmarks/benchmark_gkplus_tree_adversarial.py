"""
Adversarial benchmarks for GKPlusTree using keys with successive rank=1 across dimensions.
"""
import gc
import hashlib
from typing import List
import pathlib

from gplus_trees.g_k_plus.factory import make_gkplustree_classes
from gplus_trees.g_k_plus.g_k_plus_base import bulk_create_gkplus_tree
from benchmarks.benchmark_utils import BaseBenchmark, BenchmarkUtils
import os
import pickle


# Utility to load previously generated adversarial keys
def load_adversarial_keys_from_file(key_count: int, capacity: int, dim_limit: int) -> list:
    """
    Load adversarial keys from a pickle file corresponding to the given parameters.
    """
    keys_dir = os.path.join(os.path.dirname(__file__), 'adversarial_keys')
    file_name = f"keys_sz{key_count}_k{capacity}_d{dim_limit}.pkl"
    file_path = os.path.join(keys_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Adversarial key file not found: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class GKPlusTreeAdversarialInsertBenchmarks(BaseBenchmark):
    """Adversarial insert benchmarks: sequential insert of keys with successive rank=1."""
    params = [
        [8, 16, 32],    # K values (capacities)
        [10, 20, 30, 40, 50, 60, 70, 80],  # successive dimensions to enforce rank=1
        [1.0, 2.0, 4.0, 8.0]  # l_factor values
    ]
    param_names = ['capacity', 'dim_limit', 'l_factor']
    min_run_count = 5

    def setup(self, capacity, dim_limit, l_factor):
        super().setup(capacity, dim_limit, l_factor)
        key_count = 1000
        
        self.keys = load_adversarial_keys_from_file(
                key_count, capacity, dim_limit)
        self.items = BenchmarkUtils.create_test_items(self.keys)
        self.ranks = [1] * len(self.keys)
        self.tree_class, _, _, _ = make_gkplustree_classes(capacity)
        self.l_factor = l_factor

        gc.collect()
        gc.disable()

    def time_adversarial_insert(self, capacity, dim_limit, l_factor):
        """Benchmark sequential insertion of adversarial keys."""
        tree = self.tree_class(l_factor=l_factor)
        for item, rank in zip(self.items, self.ranks):
            tree, _ = tree.insert(item, rank)


class GKPlusTreeAdversarialRetrieveBenchmarks(BaseBenchmark):
    """Adversarial retrieve benchmarks: sequential retrieve of adversarial keys."""
    params = [
        [8, 16, 32],    # K values (capacities)
        [10, 20, 40, 80],  # successive dimensions to enforce rank=1
        [1.0, 2.0, 4.0, 8.0],  # l_factor values
        [1.0] # hit ratio
    ]
    param_names = ['capacity', 'dim_limit', 'hit_ratio', 'l_factor']
    min_run_count = 5

    _tree_cache = {}

    def setup(self, capacity, dim_limit, hit_ratio, l_factor):
        super().setup(capacity, dim_limit, hit_ratio, l_factor)
        size = 1000

        tree_cache_key = (capacity, size, dim_limit, l_factor)

        if tree_cache_key not in self._tree_cache:
            insert_keys = load_adversarial_keys_from_file(
                key_count=size,
                capacity=capacity,
                dim_limit=dim_limit
            )
            entries = BenchmarkUtils.create_test_entries(insert_keys)
            _, _, klist_class, _ = make_gkplustree_classes(capacity)
            tree = bulk_create_gkplus_tree(entries, dim_limit, l_factor, klist_class)
            self._tree_cache[tree_cache_key] = (tree, insert_keys)
        self.tree, self.insert_keys = self._tree_cache[tree_cache_key]

        # Generate lookup keys with specified hit ratio, using class-level cache
        base_seed = 42 + hash((size, capacity, dim_limit, hit_ratio)) % 1000
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.insert_keys,
            hit_ratio=hit_ratio,
            seed=base_seed + 1000
        )
        gc.collect(); gc.disable()

    def time_adversarial_retrieve(self, capacity, dim_limit, hit_ratio, l_factor):
        """Benchmark sequential retrieval of adversarial keys."""
        for key in self.lookup_keys:
            self.tree.retrieve(key)
