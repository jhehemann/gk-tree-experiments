import random
from gplus_trees.g_k_plus.utils import calc_ranks
from gplus_trees.gplus_tree_base import print_pretty
from gplus_trees.utils import find_keys_for_rank_lists
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import (
    GKPlusTreeBase,
)
import sys
import os
# Add project root to sys.path so that tests package can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from tests.test_base import BaseTestCase
import cProfile
import pstats

def iter_nodes(tree):
    """
    Iterate over all nodes in a GKPlusTreeBase in depth-first order.

    Yields:
        GKPlusNodeBase: Each node in the tree.
    """
    if tree.is_empty():
        return
    stack = [tree]
    while stack:
        tree = stack.pop()
        yield tree
        if tree.node.rank > 1:
            for entry in tree.node.set:
                if entry.left_subtree is not None:
                    stack.append(entry.left_subtree)
            if tree.node.right_subtree is not None:
                stack.append(tree.node.right_subtree)

def print_tree_nodes(tree):
    """
    Print the structure of the tree using print_pretty for the first node,
    and recursively for any node whose set is a GKPlusTreeBase.
    """
    tree_nodes = list(iter_nodes(tree))
    if not tree_nodes:
        print("Empty tree")
        return

    print(print_pretty(tree_nodes[0]))
    for tree in tree_nodes:
        if isinstance(tree.node.set, GKPlusTreeBase):
            print()
            print_tree_nodes(tree.node.set)

def create_rank_lists(num_lists, size, key_range):
    """Create multiple rank lists of unique keys."""
    return [random.sample(range(key_range[0], key_range[1]), size) for _ in range(num_lists)]

def create_uniform_rank_lists(dimensions=10, size=10, rank=1):
    """Create a list of rank lists, each filled with the same value."""
    return [[rank] * size for _ in range(dimensions)]

def generate_and_populate_tree(keys, k):
    """Create a GKPlusTree, insert items based on rank_lists, and return the tree."""
    tree = create_gkplus_tree(K=k)
    random.seed(1)  # For reproducibility
    random.shuffle(keys)
    ranks = calc_ranks(keys, k)
    base_test = BaseTestCase()
    for idx, key in enumerate(keys):
        item = base_test.make_item(key)
        rank = ranks[idx]
        # print(f"Inserting item: {item.key}, with rank: {rank}")
        tree.insert(item, rank)
        # print(f"Inserted item: {item.key} with rank: {rank}")
    return tree


def create_adversarial_1_rank_lists(dimensions=10, size=10):
    """
    Create rank lists where all elements are 1.
    """
    rank_lists = []
    for i in range(dimensions):
        rank_list = [1] * size
        rank_lists.append(rank_list)
    return rank_lists, None

def create_adversarial_rank_lists_i_eq_2(dimensions=10, size=10):
    """
    Create rank lists where in each list with index i, the element at i is 2, others are 1.
    """
    rank_lists = []
    for i in range(dimensions):
        rank_list = [1] * size
        if i < size:
            rank_list[i] = 2
        rank_lists.append(rank_list)
    return rank_lists, None


def create_middle_out_pattern_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: middle position is 2, rest are 1.
    - Second list: middle-1 and middle+1 are 2, rest are 1.
    - Third list: middle-2 and middle+2 are 2, rest are 1.
    - Continue for 'dimensions' lists.
    """
    rank_lists = []
    middle = size // 2
    for i in range(dimensions):
        rank_list = [1] * size
        positions = []
        if i == 0:
            positions = [middle]
        else:
            if middle - i >= 0:
                positions.append(middle - i)
            if middle + i < size:
                positions.append(middle + i)
        for pos in positions:
            rank_list[pos] = 2
        rank_lists.append(rank_list)
    return rank_lists, None


def create_middle_in_pattern_rank_lists(dimensions=10, size=10, k=4):
    """
    Create rank lists where in the first list, the kth and kth last positions are 2, rest are 1.
    In each subsequent list, the positions of 2 are shifted towards the middle by k.
    This continues for 'dimensions' lists.
    """
    rank_lists = []
    spacing_factor = 1
    spacing = spacing_factor  # Spacing factor to shift positions
    left = spacing
    right = size - spacing - 1
    for i in range(dimensions):
        rank_list = [1] * size
        if left <= right:
            rank_list[left] = 2
            rank_list[right] = 2
        rank_lists.append(rank_list)
        left += spacing
        right -= spacing
    return rank_lists, None


def create_dim_1_middle_2_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: middle position is 2, rest are 1.
    - All other lists: all ranks are 1.
    """
    rank_lists = []
    middle = size // 2
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0:
            rank_list[middle] = 2
        rank_lists.append(rank_list)
    return rank_lists, [middle]


def create_dim_1_first_2_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: first position is 2, rest are 1.
    - All other lists: all ranks are 1.
    """
    rank_lists = []
    first = 0
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0:
            rank_list[first] = 2
        rank_lists.append(rank_list)
    return rank_lists, [first]


def create_dim_1_first_and_middle_2_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: first and middle positions are 2, rest are 1.
    - All other lists: all ranks are 1.
    """
    rank_lists = []
    first = 0
    second = 5
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0:
            rank_list[first] = 2
            rank_list[second] = 2
        rank_lists.append(rank_list)
    return rank_lists, [first, second]


if __name__ == "__main__":
    # Adversarial rank lists
    K = 4           # K-list node capacity
    DIMENSIONS = 20  # Simulate key ranks up to this many dimensions
    SIZE = 100        # Number of keys to generate

    rank_lists, insert_key_idx_list = create_dim_1_first_2_rank_lists(dimensions=DIMENSIONS, size=SIZE)
    print("Adversarial rank lists:")
    for rl in rank_lists:
        display_rl = rl
        if len(rl) > 20:
            display_rl = rl[:10] + ['...'] + rl[-10:]
        colored_rl = [
            f"\033[31m{num}\033[0m" if num != 1 and num != '...' else str(num)
            for num in display_rl
        ]
        print(' '.join(colored_rl))

    def profile_find_keys():
        keys = find_keys_for_rank_lists(rank_lists, K)
        if len(keys) <= 20:
            print("Keys matching rank lists:", keys)
        else:
            print(f"Keys matching rank lists (showing first 10 and last 10 of {len(keys)}):", keys[:10] + ['...'] + keys[-10:])
        insert_keys = []
        print("Insert key indices (inserted after all other keys):", insert_key_idx_list)
        if insert_key_idx_list:
            for insert_key_idx in reversed(insert_key_idx_list):
                insert_keys.append(keys.pop(insert_key_idx))
        return keys, list(reversed(insert_keys))

    def profile_generate_tree(keys):
        tree = generate_and_populate_tree(keys, K)
        return tree
    
    def profile_insert_specific(insert_key, insert_key_idx):
        if insert_key is None:
            raise ValueError("Insert specific should not be called when insert_key is None.")
        base_test = BaseTestCase()
        item = base_test.make_item(insert_key)
        rank = rank_lists[0][insert_key_idx]
        tree.insert(item, rank)
        return tree
    
    # Profile find_keys_for_rank_lists
    print("\nProfiling adversarial key generation (find_keys_for_rank_lists)...")
    profiler1 = cProfile.Profile()
    profiler1.enable()
    keys, insert_keys = profile_find_keys()
    profiler1.disable()
    if insert_keys:
        print("Insert keys:", insert_keys)

    # Profile generate_and_populate_tree 
    print("\nPROFILING TREE GENERATION WITH ADVERSARIAL KEYS (generate_and_populate_tree)...")
    profiler2 = cProfile.Profile()
    profiler2.enable()
    tree = profile_generate_tree(keys)
    profiler2.disable()
    print("\nTree generated and populated with keys.")

    print("\nPROFILING RESULTS ADVERSARIAL KEY GENERATION (find_keys_for_rank_lists):")
    stats1 = pstats.Stats(profiler1)
    stats1.sort_stats('cumtime').print_stats(20)

    print("\nPROFILING RESULTS TREE GENERATION WITH ADVERSARIAL KEYS (generate_and_populate_tree):")
    stats2 = pstats.Stats(profiler2)
    stats2.sort_stats('cumtime').print_stats(20)

    if insert_keys:
        # Profile insert_specific
        for insert_key, insert_key_idx in zip(insert_keys, insert_key_idx_list):
            print(f"\nProfiling insert_specific: {insert_key}...")
            profiler3 = cProfile.Profile()
            profiler3.enable()
            tree = profile_insert_specific(insert_key, insert_key_idx)
            profiler3.disable()
            print(f"\nSpecific key {insert_key} inserted into the tree.")
            
            # Print stats for this specific insertion
            print(f"\nPROFILING RESULTS SPECIFIC KEY INSERTION (insert_specific) ({insert_key}):")
            stats3 = pstats.Stats(profiler3)
            stats3.sort_stats('cumtime').print_stats(20)
    
    # print_tree_nodes(tree)

    

