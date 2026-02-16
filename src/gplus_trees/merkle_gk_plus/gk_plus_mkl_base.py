"""Merkle extension for GKPlus-trees.

This module provides Merkle tree functionality for GKPlus-trees,
allowing each node to store and update cryptographic hashes of its subtrees.
Uses shared ``MerkleNodeMixin`` / ``MerkleTreeMixin`` to avoid code
duplication with the GPlus-tree Merkle variant.
"""

from gplus_trees.g_k_plus.g_k_plus_base import GKPlusNodeBase, GKPlusTreeBase
from gplus_trees.merkle_mixin import MerkleNodeMixin, MerkleTreeMixin


class MerkleGKPlusNodeBase(MerkleNodeMixin, GKPlusNodeBase):
    """GKPlusNodeBase extended with Merkle hash support."""

    __slots__ = (*GKPlusNodeBase.__slots__, "merkle_hash")

    def __init__(self, rank, set, right=None):
        super().__init__(rank, set, right)
        self.merkle_hash = None


class MerkleGKPlusTreeBase(MerkleTreeMixin, GKPlusTreeBase):
    """GKPlusTreeBase extended with Merkle tree operations."""

    NodeClass: type[MerkleGKPlusNodeBase]
