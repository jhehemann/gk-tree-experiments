"""
Merkle tree extensions for GPlus-trees.

This package provides Merkle tree functionality for GPlus-trees,
allowing verification of data integrity through cryptographic hashing.
"""

from gplus_trees.merkle.factory import create_merkle_gplustree, make_merkle_gplustree_classes
from gplus_trees.merkle.gplus_mkl_base import MerkleGPlusNodeBase, MerkleGPlusTreeBase

__all__ = ["MerkleGPlusNodeBase", "MerkleGPlusTreeBase", "create_merkle_gplustree", "make_merkle_gplustree_classes"]
