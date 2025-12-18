"""
Adversarial benchmarks for GKPlusTree using keys with successive rank=1 across dimensions.
"""
import gc

from gplus_trees.g_k_plus.factory import make_gkplustree_classes
from gplus_trees.g_k_plus.g_k_plus_base import bulk_create_gkplus_tree
from gplus_trees.g_k_plus.utils import calc_ranks
from benchmarks.benchmark_utils import BaseBenchmark, BenchmarkUtils
import os
import pickle


# Utility to load previously generated adversarial keys
def load_adversarial_keys_from_file(key_count: int, capacity: int, dim_limit: int) -> list:
    """
    Load adversarial keys from a pickle file corresponding to the given parameters.
    """
    keys_dir = os.path.join(os.path.dirname(__file__), 'adversarial_keys_new')
    file_name = f"keys_sz{key_count}_k{capacity}_d{dim_limit}.pkl"
    file_path = os.path.join(keys_dir, file_name)
    if not os.path.exists(file_path):
        # Find the next higher key_count file with same k and d
        available_files = [f for f in os.listdir(keys_dir) if f.startswith(f"keys_sz") and f.endswith(f"_k{capacity}_d{dim_limit}.pkl")]
        sizes = []
        for fname in available_files:
            try:
                sz = int(fname.split("_")[1][2:])
                sizes.append(sz)
            except Exception:
                continue
        sizes = sorted([s for s in sizes if s >= key_count])
        if sizes:
            next_file = f"keys_sz{sizes[0]}_k{capacity}_d{dim_limit}.pkl"
            file_path = os.path.join(keys_dir, next_file)
        else:
            raise FileNotFoundError(f"No adversarial key file found for k={capacity}, d={dim_limit} with size >= {key_count}")
    with open(file_path, 'rb') as f:
        keys = pickle.load(f)
        return keys[:key_count]


class GKPlusTreeAdversarialInsertBenchmarks(BaseBenchmark):
    """Adversarial insert benchmarks: sequential insert of keys with successive rank=1."""
    # benchmark individual dimension limits for each K
    l_factors = [1.0, 2.0, 4.0]  # l_factor values
    capacities_and_dim_limits = [
        (4, [1, 10, 20, 30, 40]),
        (8, [1, 10, 20, 40, 80]),
        (16, [1, 10, 20, 40, 80, 160]),
    ]
    # (32, [1, 10, 20, 40, 80, 160, 320])  # IGNORE 32 for now due to long runtimes

    params = [
        [(cap, lim) for cap, limits in capacities_and_dim_limits for lim in limits],
        l_factors,
    ]
    param_names = ['capacity_dim_limit', 'l_factor']
    min_run_count = 3

    def setup(self, capacity_dim_limit, l_factor):
        super().setup(*capacity_dim_limit, l_factor)
        key_count = 1000
        capacity, dim_limit = capacity_dim_limit

        self.keys = load_adversarial_keys_from_file(
                key_count, capacity, dim_limit)
        self.items = BenchmarkUtils.create_test_items(self.keys)
        self.ranks = calc_ranks(self.keys, capacity)
        self.tree_class, _, _, _ = make_gkplustree_classes(capacity)
        self.l_factor = l_factor

        gc.collect()
        gc.disable()  # Re-enabled in teardown by BaseBenchmark

    def time_adversarial_insert(self, capacity_dim_limit, l_factor):
        """Benchmark sequential insertion of adversarial keys."""
        tree = self.tree_class(l_factor=l_factor)
        for item, rank in zip(self.items, self.ranks):
            tree, _, _ = tree.insert(item, rank)


class GKPlusTreeAdversarialRetrieveBenchmarks(BaseBenchmark):
    """Adversarial retrieve benchmarks: sequential retrieve of adversarial keys.
    
    Cache Behavior:
        This benchmark uses class-level caching to avoid rebuilding trees
        for each parameter set. Trees are cached by (capacity, size, dim_limit, l_factor).
        The cache persists across benchmark runs within the same process.
    """
    params = [
        [4, 8, 16],    # K values (capacities)
        [10, 20, 40],  # successive dimensions to enforce rank=1
        [1.0, 2.0, 4.0],  # l_factor values
        [1.0] # hit ratio
    ]
    param_names = ['capacity', 'dim_limit', 'hit_ratio', 'l_factor']
    min_run_count = 3
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

        # Generate lookup keys with specified hit ratio
        seed_offset = hash((size, capacity, dim_limit, hit_ratio)) % 1000
        base_seed = 42 + seed_offset
        self.lookup_keys = BenchmarkUtils.create_lookup_keys(
            insert_keys=self.insert_keys,
            hit_ratio=hit_ratio,
            seed=base_seed + 1000
        )
        gc.collect()
        gc.disable()  # Re-enabled in teardown by BaseBenchmark
    
    @classmethod
    def clear_cache(cls):
        """Clear the tree cache to free memory when needed.
        
        Note: This is not called automatically. It can be used for memory
        management in long benchmark sessions if needed.
        """
        cls._tree_cache.clear()
        gc.collect()

    def time_adversarial_retrieve(self, capacity, dim_limit, hit_ratio, l_factor):
        """Benchmark sequential retrieval of adversarial keys."""
        for key in self.lookup_keys:
            self.tree.retrieve(key)
