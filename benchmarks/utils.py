"""Utilities for benchmark data generation and tree creation."""

import random
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gplus_trees.base import Item, calculate_item_rank, calculate_group_size
from gplus_trees.gplus_tree import GPlusTree


def create_gtree(items: List[Tuple[Item, int]]) -> GPlusTree:
    """
    Build a GPlusTree from a list of (item, rank) pairs.
    
    This is the setup phase - not timed in benchmarks.
    
    Args:
        items: List of (Item, rank) tuples to insert
        
    Returns:
        GPlusTree with all items inserted
    """
    tree = GPlusTree()
    tree_insert = tree.insert
    for (item, rank) in items:
        tree_insert(item, rank)
    return tree


def generate_random_items(n: int, target_node_size: int, seed: int) -> List[Tuple[Item, int]]:
    """
    Generate deterministic random items for tree construction.
    
    This is the setup phase - not timed in benchmarks.
    
    Args:
        n: Number of items to generate
        target_node_size: Target node size (K) for rank calculation
        seed: Random seed for reproducibility
        
    Returns:
        List of (Item, rank) tuples
        
    Raises:
        ValueError: If key-space is too small for requested n
    """
    # Seed the RNG for deterministic generation
    rng = random.Random(seed)
    
    # Cache frequently used functions
    calc_rank = calculate_item_rank
    make_item = Item
    group_size = calculate_group_size(target_node_size)
    
    # Key space: 2^24 = 16,777,216 unique values
    space = 1 << 24
    if space <= n:
        raise ValueError(f"Key-space too small! Required: {n + 1}, Available: {space}")
    
    # Generate unique random indices deterministically
    indices = rng.sample(range(space), k=n)
    
    items = []
    append = items.append
    
    for idx in indices:
        key = idx
        val = f"val_{idx}"
        
        item = make_item(key, val)
        rank = calc_rank(item.key, group_size)
        append((item, rank))
    
    return items


def random_gtree_of_size(n: int, target_node_size: int, seed: int) -> GPlusTree:
    """
    Create a random GPlusTree with deterministic seeding.
    
    This combines setup (generation) and construction - not timed in benchmarks.
    
    Args:
        n: Number of items in the tree
        target_node_size: Target node size (K)
        seed: Random seed for reproducibility
        
    Returns:
        GPlusTree with n items
    """
    items = generate_random_items(n, target_node_size, seed)
    return create_gtree(items)
