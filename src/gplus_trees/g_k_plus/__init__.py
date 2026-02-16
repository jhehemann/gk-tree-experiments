"""
GKPlusTree module - Extension of G+Trees with dimensional support.

This module provides GKPlusTrees that can automatically transform between
KLists and GKPlusTrees based on item count thresholds.
"""

from gplus_trees.g_k_plus.factory import create_gkplus_tree, make_gkplustree_classes
from gplus_trees.g_k_plus.g_k_plus_base import DEFAULT_DIMENSION, DEFAULT_L_FACTOR, GKPlusNodeBase, GKPlusTreeBase

__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_L_FACTOR",
    "GKPlusNodeBase",
    "GKPlusTreeBase",
    "create_gkplus_tree",
    "make_gkplustree_classes",
]
