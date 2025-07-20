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
        print(f"Inserting item: {item.key}, with rank: {rank}")
        tree.insert(item, rank)
        print(f"Inserted item: {item.key} with rank: {rank}")
    return tree


def create_adversarial_rank_lists(dimensions=10, size=10):
    """
    Create rank lists where in each list with index i, the element at i is 2, others are 1.
    """
    rank_lists = []
    for i in range(dimensions):
        rank_list = [1] * size
        if i < size:
            rank_list[i] = 2
        rank_lists.append(rank_list)
    return rank_lists


def create_middle_pattern_rank_lists(dimensions=10, size=10):
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
    return rank_lists


def create_middle_pattern_rank_lists_single_2(dimensions=10, size=10):
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
    return rank_lists


def create_shifted_rank_lists(dimensions=10, size=10, k=4):
    """
    Create rank lists where in the first list, the kth and kth last positions are 2, rest are 1.
    In each subsequent list, the positions of 2 are shifted towards the middle by k.
    This continues for 'dimensions' lists.
    """
    rank_lists = []
    spacing_factor = 2
    spacing = spacing_factor * k  # Spacing factor to shift positions
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
    return rank_lists


def create_two_middle_twos_rank_lists(dimensions=10, size=10):
    """
    Generate rank lists where:
    - First list: two middle positions are 1, positions left and right of those are 2, rest are 1.
    - All other lists: all ranks are 1.
    """
    rank_lists = []
    middle1 = (size - 1) // 2
    middle2 = middle1 + 1
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0 and size >= 4:
            if middle1 - 1 >= 0:
                rank_list[middle1 - 1] = 2
            if middle2 + 1 < size:
                rank_list[middle2 + 1] = 2
        rank_lists.append(rank_list)
    return rank_lists


if __name__ == "__main__":
    # Adversarial rank lists
    k = 4  # K-list node capacity
    rank_lists = create_middle_pattern_rank_lists_single_2(dimensions=10, size=10)
    print("Adversarial rank lists:")
    for rl in rank_lists:
        colored_rl = [
            f"\033[31m{num}\033[0m" if num != 1 else str(num)
            for num in rl
        ]
        print(' '.join(colored_rl))

    def profile_find_keys():
        keys = find_keys_for_rank_lists(rank_lists, k)
        print("Keys matching rank lists:", keys)
        return keys

    def profile_generate_tree(keys):
        tree = generate_and_populate_tree(keys, k)
        return tree
    

    # Profile find_keys_for_rank_lists
    print("\nProfiling find_keys_for_rank_lists...")
    profiler1 = cProfile.Profile()
    profiler1.enable()
    keys = profile_find_keys()
    profiler1.disable()

    # Profile generate_and_populate_tree 
    print("\nProfiling generate_and_populate_tree...")
    profiler2 = cProfile.Profile()
    profiler2.enable()
    tree = profile_generate_tree(keys)
    profiler2.disable()
    print("\nTree generated and populated with keys.")
    
    # exit()

    print_tree_nodes(tree)

    print("\nProfiling output for find_keys_for_rank_lists:")
    stats1 = pstats.Stats(profiler1)
    stats1.sort_stats('cumtime').print_stats(20)

    print("\nProfiling output for generate_and_populate_tree and print_tree_nodes:")
    stats2 = pstats.Stats(profiler2)
    stats2.sort_stats('cumtime').print_stats(20)
