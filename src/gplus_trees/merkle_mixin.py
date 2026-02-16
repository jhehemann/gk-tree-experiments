"""Merkle tree mixins for G⁺-tree and Gᵏ⁺-tree extensions.

Provides shared ``MerkleNodeMixin`` and ``MerkleTreeMixin`` so the
Merkle logic is defined once and reused by both ``merkle`` and
``merkle_gk_plus`` packages.
"""

from __future__ import annotations

import hashlib
import struct

from gplus_trees.base import InternalItem, LeafItem
from gplus_trees.logging_config import get_logger

logger = get_logger(__name__)


class MerkleNodeMixin:
    """Mixin that adds Merkle hash support to any G⁺-tree node.

    The host class **must** already define ``__slots__`` containing at
    least ``("rank", "set", "right_subtree")``.  This mixin adds a
    ``merkle_hash`` slot.  The mixed class must include
    ``"merkle_hash"`` in its own ``__slots__`` declaration (see
    factory usage).
    """

    # NOTE: __slots__ is declared on the concrete subclasses created by
    # the factories; this mixin only provides methods.

    def compute_hash(self) -> bytes:
        """Compute the Merkle hash for this node and its subtrees."""
        h = hashlib.sha256()

        has_children = any(
            (entry.left_subtree is not None and not entry.left_subtree.is_empty()) for entry in self.set
        ) or (self.right_subtree is not None and not self.right_subtree.is_empty())

        # 0x00 for leaf, 0x01 for internal
        h.update(b"\x00" if not has_children else b"\x01")

        # Pack the rank and item count
        h.update(struct.pack(">I", self.rank))
        num_entries = self.set.item_count()
        h.update(struct.pack(">I", num_entries))

        for entry in self.set:
            # Left-subtree hash (if any)
            left = entry.left_subtree
            if left is not None and not left.is_empty():
                lh = left.node.get_hash()
                h.update(b"\x02")  # left-child tag
                h.update(struct.pack(">I", len(lh)))
                h.update(lh)

            # Entry key
            key_b = str(entry.item.key).encode()
            h.update(struct.pack(">I", len(key_b)))
            h.update(key_b)

            # Entry value (if present)
            if entry.item.value is not None:
                val_b = str(entry.item.value).encode()
                h.update(struct.pack(">I", len(val_b)))
                h.update(val_b)

        # Right-subtree hash
        right = self.right_subtree
        if right is not None and not right.is_empty():
            rh = right.node.get_hash()
            h.update(b"\x03")  # right-child tag
            h.update(struct.pack(">I", len(rh)))
            h.update(rh)

        self.merkle_hash = h.digest()
        return self.merkle_hash

    def get_hash(self) -> bytes:
        """Get the hash value, computing it if necessary."""
        if self.merkle_hash is None:
            return self.compute_hash()
        return self.merkle_hash

    def invalidate_hash(self) -> None:
        """Invalidate the stored hash value so it will be recomputed on next access."""
        self.merkle_hash = None


class MerkleTreeMixin:
    """Mixin that adds Merkle tree operations to any G⁺-tree.

    The host class **must** inherit from ``GPlusTreeBase`` (or a
    subclass) so that ``self.node``, ``self.is_empty()``, etc. are
    available.
    """

    def get_root_hash(self) -> bytes | None:
        """Get the Merkle root hash of the tree."""
        if self.is_empty():
            return None
        if isinstance(self.node, MerkleNodeMixin):
            return self.node.get_hash()
        return None

    def verify_integrity(self) -> bool:
        """Verify the integrity of the entire tree by recomputing hashes."""
        if self.is_empty():
            return True
        old_hash = self.get_root_hash()
        self._invalidate_all_hashes()
        new_hash = self.get_root_hash()
        return old_hash == new_hash

    def _invalidate_all_hashes(self) -> None:
        """Recursively invalidate all hashes in the tree."""
        if self.is_empty():
            return
        if isinstance(self.node, MerkleNodeMixin):
            self.node.invalidate_hash()
        for entry in self.node.set:
            if (
                entry.left_subtree
                and not entry.left_subtree.is_empty()
                and isinstance(entry.left_subtree, MerkleTreeMixin)
            ):
                entry.left_subtree._invalidate_all_hashes()
        if (
            self.node.right_subtree
            and not self.node.right_subtree.is_empty()
            and isinstance(self.node.right_subtree, MerkleTreeMixin)
        ):
            self.node.right_subtree._invalidate_all_hashes()

    def _invalidate_path_hashes(self, key: int) -> None:
        """Invalidate the hashes along the path to a specific key."""
        if self.is_empty():
            return
        cur = self.node
        while True:
            cur.invalidate_hash()
            if cur.rank == 1:
                break
            next_entry = cur.set.retrieve(key)[1]
            if next_entry is not None:
                cur = next_entry.left_subtree.node
            else:
                cur = cur.right_subtree.node

    def insert(self, x: LeafItem | InternalItem, rank: int, *args, **kwargs):
        """Insert an item with the given rank, then invalidate Merkle hashes."""
        result = super().insert(x, rank, *args, **kwargs)
        self._invalidate_path_hashes(x.key)
        return result

    def get_inclusion_proof(self, key: int) -> list:
        """Generate a Merkle inclusion proof for a given key."""
        proof: list[bytes] = []
        if self.is_empty():
            return proof
        self._build_inclusion_proof(key, proof)
        return proof

    def _build_inclusion_proof(self, key: int, proof: list) -> bool:
        """Build an inclusion proof by traversing the tree to *key*."""
        if self.is_empty():
            return False
        node = self.node
        res = node.set.retrieve(key)
        if node.rank == 1:
            return res[0] is not None

        for entry in node.set:
            if res[0] and entry.item.key == res[0].item.key:
                continue
            if (
                entry.left_subtree
                and not entry.left_subtree.is_empty()
                and isinstance(entry.left_subtree.node, MerkleNodeMixin)
            ):
                proof.append(entry.left_subtree.node.get_hash())

        next_subtree = res[1].left_subtree if res[1] else node.right_subtree
        if (
            node.right_subtree != next_subtree
            and not node.right_subtree.is_empty()
            and isinstance(node.right_subtree.node, MerkleNodeMixin)
        ):
            proof.append(node.right_subtree.node.get_hash())

        return next_subtree._build_inclusion_proof(key, proof)
