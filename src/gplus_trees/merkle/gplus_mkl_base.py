"""Merkle extension for GPlus-trees.

This module provides Merkle tree functionality for GPlus-trees,
allowing each node to store and update cryptographic hashes of its subtrees.
Uses shared ``MerkleNodeMixin`` / ``MerkleTreeMixin`` to avoid code
duplication with the GKPlus-tree Merkle variant.
"""

from typing import Type

from gplus_trees.gplus_tree_base import GPlusTreeBase, GPlusNodeBase
from gplus_trees.merkle_mixin import MerkleNodeMixin, MerkleTreeMixin


class MerkleGPlusNodeBase(MerkleNodeMixin, GPlusNodeBase):
    """GPlusNodeBase extended with Merkle hash support."""
    __slots__ = GPlusNodeBase.__slots__ + ("merkle_hash",)

    def __init__(self, rank, set, right=None):
        super().__init__(rank, set, right)
        self.merkle_hash = None


class MerkleGPlusTreeBase(MerkleTreeMixin, GPlusTreeBase):
    """GPlusTreeBase extended with Merkle tree operations."""
    NodeClass: Type[MerkleGPlusNodeBase]
