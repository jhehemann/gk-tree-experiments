import cProfile
import io
import math
import pstats
import random
import time

from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.utils import calc_ranks
from gplus_trees.utils import find_keys_for_rank_lists
from tests.test_base import BaseTestCase

# Test with different sizes - focus on SINGLE ADVERSARIAL KEY insertion
sizes = [50, 100, 200, 400, 800, 1600, 3200, 6400]
K = 4
DIMENSIONS = 20


def create_dim_1_first_2_rank_lists(dimensions=10, size=10):
    rank_lists = []
    first = 0
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0:
            rank_list[first] = 2
        rank_lists.append(rank_list)
    return rank_lists, [first]


print("SIZE | Single Insert Time | Function Calls | Calls/log²(n) | Expected O((log n)²) Ratio")
print("-" * 90)

prev_time = None
prev_calls = None

for SIZE in sizes:
    rank_lists, insert_key_idx_list = create_dim_1_first_2_rank_lists(dimensions=DIMENSIONS, size=SIZE)

    # Generate keys
    keys = find_keys_for_rank_lists(rank_lists, K)

    # Extract insert key
    if insert_key_idx_list:
        insert_keys = []
        for insert_key_idx in reversed(insert_key_idx_list):
            insert_keys.append(keys.pop(insert_key_idx))

    # Build tree with all other keys
    tree = create_gkplus_tree(K=K)
    random.seed(1)
    random.shuffle(keys)
    ranks = calc_ranks(keys, K)
    base_test = BaseTestCase()

    for idx, key in enumerate(keys):
        item = base_test.make_item(key)
        rank = ranks[idx]
        tree.insert(item, rank)

    # Profile SINGLE insertion of the adversarial key
    profiler = cProfile.Profile()
    profiler.enable()

    start_insert = time.time()
    item = base_test.make_item(insert_keys[0])
    rank = rank_lists[0][0]  # rank 2
    tree.insert(item, rank)
    insertion_time = time.time() - start_insert

    profiler.disable()

    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    total_calls = ps.total_calls

    log_n = math.log2(SIZE)
    log_n_sq = log_n**2
    calls_per_log_sq = total_calls / log_n_sq

    if prev_time is not None:
        time_ratio = insertion_time / prev_time
        calls_ratio = total_calls / prev_calls
        # Expected ratio for O((log n)^2)
        log_ratio = math.log2(SIZE) / math.log2(SIZE // 2)
        expected_ratio = log_ratio**2
        print(
            f"{SIZE:4d} | {insertion_time:18.6f}s | {total_calls:14d} | {calls_per_log_sq:13.1f} | {time_ratio:.3f} (exp: {expected_ratio:.3f})"
        )
    else:
        print(f"{SIZE:4d} | {insertion_time:18.6f}s | {total_calls:14d} | {calls_per_log_sq:13.1f} | N/A")

    prev_time = insertion_time
    prev_calls = total_calls

print()
print("Analysis:")
print("=" * 90)
print("If Calls/log²(n) stays roughly constant, the function call complexity is O((log n)²)")
print("If the time ratio matches expected ratio when doubling n, time complexity is O((log n)²)")
