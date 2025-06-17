"""
Merkle tree extensions for GPlus-trees.

This package provides Merkle tree functionality for GPlus-trees,
allowing verification of data integrity through cryptographic hashing.
"""

from gplus_trees.merkle.gplus_mkl_base import (
    MerkleGPlusNodeBase,
    MerkleGPlusTreeBase
)

from gplus_trees.merkle.factory import (
    make_merkle_gplustree_classes,
    create_merkle_gplustree
)

__all__ = [
    'MerkleGPlusNodeBase',
    'MerkleGPlusTreeBase',
    'make_merkle_gplustree_classes',
    'create_merkle_gplustree'
]