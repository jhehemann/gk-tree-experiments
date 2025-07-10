"""
Adversarial benchmarks for GKPlusTree using keys with successive rank=1 across dimensions.
"""
import gc
import hashlib
from typing import List

from gplus_trees.g_k_plus.factory import make_gkplustree_classes
from gplus_trees.g_k_plus.utils import calc_rank_from_group_size, calculate_group_size
from gplus_trees.g_k_plus.g_k_plus_base import bulk_create_gkplus_tree
from gplus_trees.base import Item, Entry
from benchmarks.benchmark_utils import BaseBenchmark, BenchmarkUtils
from gplus_trees.utils import find_keys_for_successive_rank1
import os
import pickle


# Module-level cache for adversarial keys
_adversarial_keys_cache = {}

class GKPlusTreeAdversarialInsertBenchmarks(BaseBenchmark):
    """Adversarial insert benchmarks: sequential insert of keys with successive rank=1."""
    params = [
        [8, 16, 32],    # K values (capacities)
        [10, 20, 30, 40, 50, 60],  # successive dimensions to enforce rank=1
        [1.0, 2.0, 4.0, 8.0]  # l_factor values
    ]
    param_names = ['capacity', 'dim_limit', 'l_factor']
    min_run_count = 5

    def setup(self, capacity, dim_limit, l_factor):
        super().setup(capacity, dim_limit, l_factor)
        key_count = 1000
        cache_key = (key_count, capacity, dim_limit)
        keys_dir = './adversarial_keys'
        os.makedirs(keys_dir, exist_ok=True)
        file_name = f"keys_sz{key_count}_k{capacity}_d{dim_limit}.pkl"
        file_path = os.path.join(keys_dir, file_name)

        if cache_key in _adversarial_keys_cache:
            succ_keys = _adversarial_keys_cache[cache_key]
        elif os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                succ_keys = pickle.load(f)
            _adversarial_keys_cache[cache_key] = succ_keys
        else:
            succ_keys = find_keys_for_successive_rank1(
                k=capacity, dim_limit=dim_limit, count=key_count, spacing=False
            )
            _adversarial_keys_cache[cache_key] = succ_keys
            print(f"Generated {len(succ_keys)} keys for k={capacity}, dim_limit={dim_limit} and saved to {file_path}. Keys: {succ_keys[:10]}...{succ_keys[-10:] if len(succ_keys) > 20 else succ_keys}")
            with open(file_path, 'wb') as f:
                pickle.dump(succ_keys, f)

        self.succ_keys = succ_keys
        self.entries = BenchmarkUtils.create_test_entries(self.succ_keys)
        group_size = calculate_group_size(capacity)
        self.ranks = [calc_rank_from_group_size(key, group_size) for key in self.succ_keys]
        self.tree_class, _, _, _ = make_gkplustree_classes(capacity)
        self.l_factor = l_factor
        gc.collect(); gc.disable()

    def time_adversarial_insert(self, capacity, dim_limit, l_factor):  # noqa: N802
        """Benchmark sequential insertion of adversarial keys."""
        tree = self.tree_class(l_factor=l_factor)
        for entry, rank in zip(self.entries, self.ranks):
            tree, _ = tree.insert_entry(entry, rank)


class GKPlusTreeAdversarialRetrieveBenchmarks(BaseBenchmark):
    """Adversarial retrieve benchmarks: sequential retrieve of adversarial keys."""
    params = [
        [8, 16, 32],    # K values (capacities)
        [10, 20, 40],  # successive dimensions to enforce rank=1
        [1.0, 2.0, 4.0, 8.0]  # l_factor values
    ]
    param_names = ['capacity', 'dim_limit', 'l_factor']
    min_run_count = 5

    _tree_cache = {}

    def setup(self, capacity, dim_limit, l_factor):
        super().setup(capacity, dim_limit, l_factor)
        # Use module-level adversarial key cache
        key_count = 1000
        cache_key = (key_count, capacity, dim_limit)
        if cache_key not in _adversarial_keys_cache:
            succ_keys = find_keys_for_successive_rank1(
                k=capacity, dim_limit=dim_limit, count=key_count, spacing=False
            )
            _adversarial_keys_cache[cache_key] = succ_keys
        else:
            succ_keys = _adversarial_keys_cache[cache_key]
        if cache_key not in self._tree_cache:
            entries = BenchmarkUtils.create_test_entries(succ_keys)
            _, _, klist_class, _ = make_gkplustree_classes(capacity)
            tree = bulk_create_gkplus_tree(entries, DIM=dim_limit, l_factor=l_factor, KListClass=klist_class)
            self._tree_cache[cache_key] = (tree, succ_keys)
        self.tree, self.lookup_keys = self._tree_cache[cache_key]
        gc.collect(); gc.disable()

    def time_adversarial_retrieve(self, capacity, dim_limit, l_factor):  # noqa: N802
        """Benchmark sequential retrieval of adversarial keys."""
        for key in self.lookup_keys:
            self.tree.retrieve(key)
