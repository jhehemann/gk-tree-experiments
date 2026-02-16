"""Shared invariant-checking utilities.

This module provides tree invariant validation that can be used by both
the stats scripts and the test suite, eliminating the circular
dependency between ``stats`` and ``tests``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gplus_trees.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from gplus_trees.gplus_tree_base import GPlusTreeBase
    from gplus_trees.tree_stats import Stats

TREE_FLAGS = (
    "is_heap",
    "is_search_tree",
    "internal_has_replicas",
    "internal_packed",
    "set_thresholds_met",
    "linked_leaf_nodes",
    "all_leaf_values_present",
    "leaf_keys_in_order",
)


class InvariantError(Exception):
    """Raised when a G⁺-tree invariant is violated."""


def assert_tree_invariants_raise(
    t: GPlusTreeBase,
    stats: Stats,
) -> None:
    """Check all invariants, raising :class:`InvariantError` on the first failure."""
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase

    for flag in TREE_FLAGS:
        if flag == "set_thresholds_met" and not isinstance(t, GKPlusTreeBase):
            continue
        if not getattr(stats, flag):
            raise InvariantError(f"Invariant failed: {flag} is False")

    if not t.is_empty():
        if stats.item_count <= 0:
            raise InvariantError(f"Invariant failed: item_count={stats.item_count} ≤ 0 for non-empty tree")
        if stats.item_slot_count <= 0:
            raise InvariantError(f"Invariant failed: item_slot_count={stats.item_slot_count} ≤ 0 for non-empty tree")
        if stats.gnode_count <= 0:
            raise InvariantError(f"Invariant failed: gnode_count={stats.gnode_count} ≤ 0 for non-empty tree")
        if stats.gnode_height <= 0:
            raise InvariantError(f"Invariant failed: gnode_height={stats.gnode_height} ≤ 0 for non-empty tree")
        if stats.rank <= 0:
            raise InvariantError(f"Invariant failed: rank={stats.rank} ≤ 0 for non-empty tree")
        if stats.least_item is None:
            raise InvariantError("Invariant failed: least_item is None for non-empty tree")
        if stats.greatest_item is None:
            raise InvariantError("Invariant failed: greatest_item is None for non-empty tree")

        if hasattr(t, "get_size"):
            size = t.real_item_count()
            if size != stats.real_item_count:
                raise InvariantError(
                    f"Invariant failed: t.real_item_count()={size} ≠ stats.real_item_count={stats.real_item_count}"
                )


def _get_dummy_for_tree(tree: GPlusTreeBase):
    """Return the sentinel dummy item for *tree* (works for both G⁺ and Gᵏ⁺)."""
    from gplus_trees.gplus_tree_base import DUMMY_ITEM, get_dummy

    dim = getattr(tree.__class__, "DIM", None)
    if dim is not None:
        return get_dummy(dim=dim)
    return DUMMY_ITEM


def check_leaf_keys_and_values(
    tree: GPlusTreeBase,
    expected_keys: list[int] | None = None,
) -> tuple[list[int], bool, bool, bool]:
    """Traverse leaf nodes and validate keys / values.

    Works for both :class:`GPlusTreeBase` and :class:`GKPlusTreeBase`.

    Returns
    -------
    (keys, presence_ok, all_have_values, order_ok)
    """
    dummy = _get_dummy_for_tree(tree)

    keys: list[int] = []
    all_have_values = True
    order_ok = True

    prev_key = None
    for leaf in tree.iter_leaf_nodes():
        for entry in leaf.set:
            item = entry.item
            key = item.key
            if prev_key is None:
                if item is not dummy:
                    order_ok = False
            else:
                keys.append(key)
                if item.value is None:
                    all_have_values = False
                if key < prev_key:
                    order_ok = False
            prev_key = key

    presence_ok = True
    if expected_keys is not None:
        if len(keys) != len(expected_keys):
            presence_ok = False
        else:
            presence_ok = set(keys) == set(expected_keys)

    return keys, presence_ok, all_have_values, order_ok
