"""Correctness verification for benchmark data structures."""

import logging
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gplus_trees.gplus_tree import GPlusTree, gtree_stats_, Stats, DUMMY_ITEM


# Tree invariant flags to check
TREE_FLAGS = (
    "is_heap",
    "is_search_tree",
    "internal_has_replicas",
    "internal_packed",
    "linked_leaf_nodes",
    "all_leaf_values_present",
    "leaf_keys_in_order",
)


def verify_invariants(tree: GPlusTree, stats: Stats) -> bool:
    """
    Check all tree invariants.
    
    This is the verify phase - not timed in benchmarks.
    
    Args:
        tree: The GPlusTree to verify
        stats: Computed statistics for the tree
        
    Returns:
        True if all invariants pass, False otherwise
    """
    all_passed = True
    
    # Check boolean invariants
    for flag in TREE_FLAGS:
        if not getattr(stats, flag):
            logging.error("Invariant failed: %s is False", flag)
            all_passed = False
    
    # Check non-empty tree invariants
    if not tree.is_empty():
        if stats.item_count <= 0:
            logging.error(
                "Invariant failed: item_count=%d ≤ 0 for non-empty tree",
                stats.item_count
            )
            all_passed = False
        if stats.item_slot_count <= 0:
            logging.error(
                "Invariant failed: item_slot_count=%d ≤ 0 for non-empty tree",
                stats.item_slot_count
            )
            all_passed = False
        if stats.gnode_count <= 0:
            logging.error(
                "Invariant failed: gnode_count=%d ≤ 0 for non-empty tree",
                stats.gnode_count
            )
            all_passed = False
        if stats.gnode_height <= 0:
            logging.error(
                "Invariant failed: gnode_height=%d ≤ 0 for non-empty tree",
                stats.gnode_height
            )
            all_passed = False
        if stats.rank <= 0:
            logging.error(
                "Invariant failed: rank=%d ≤ 0 for non-empty tree",
                stats.rank
            )
            all_passed = False
        if stats.least_item is None:
            logging.error(
                "Invariant failed: least_item is None for non-empty tree"
            )
            all_passed = False
        if stats.greatest_item is None:
            logging.error(
                "Invariant failed: greatest_item is None for non-empty tree"
            )
            all_passed = False
    
    return all_passed


def check_leaf_keys_and_values(
    tree: GPlusTree,
    expected_keys: Optional[List[int]] = None
) -> Tuple[List[int], bool, bool, bool]:
    """
    Traverse leaf nodes and verify key ordering and value presence.
    
    This is the verify phase - not timed in benchmarks.
    
    Args:
        tree: The GPlusTree to examine
        expected_keys: Optional list of keys that must match exactly
        
    Returns:
        (keys, presence_ok, all_have_values, order_ok)
    """
    keys = []
    all_have_values = True
    order_ok = True
    
    # Traverse leaf nodes and collect keys
    prev_key = None
    for leaf in tree.iter_leaf_nodes():
        leaf_set = leaf.set
        for entry in leaf_set:
            item = entry.item
            key = item.key
            
            # Skip dummy items
            if item is DUMMY_ITEM:
                continue
            
            keys.append(key)
            
            # Check if value is non-None
            if item.value is None:
                all_have_values = False
            
            # Check if keys are in sorted order
            if prev_key is not None and key < prev_key:
                order_ok = False
            
            prev_key = key
    
    # Check presence only if expected_keys is provided
    presence_ok = True
    if expected_keys is not None:
        if len(keys) != len(expected_keys):
            presence_ok = False
        else:
            # Check set equivalence
            presence_ok = set(keys) == set(expected_keys)
    
    return keys, presence_ok, all_have_values, order_ok
