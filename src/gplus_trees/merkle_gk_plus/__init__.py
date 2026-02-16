"""
Merkle tree extensions for GKPlus-trees.

This package provides Merkle tree functionality for GKPlus-trees,
allowing verification of data integrity through cryptographic hashing.
"""

from gplus_trees.merkle_gk_plus.factory import create_merkle_gk_plustree, make_merkle_gk_plustree_classes
from gplus_trees.merkle_gk_plus.gk_plus_mkl_base import MerkleGKPlusNodeBase, MerkleGKPlusTreeBase

__all__ = [
    "MerkleGKPlusNodeBase",
    "MerkleGKPlusTreeBase",
    "create_merkle_gk_plustree",
    "make_merkle_gk_plustree_classes",
]
