"""Shared invariant-checking utilities.

This module provides tree invariant validation that can be used by both
the stats scripts and the test suite, eliminating the circular
dependency between ``stats`` and ``tests``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gplus_trees.gplus_tree_base import get_dummy
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
    for flag in TREE_FLAGS:
        if flag == "set_thresholds_met" and t.set_conversion_threshold is None:
            # Only variants with set conversion enforce this flag.
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

        size = t.tracked_size()
        if size is not None and size != stats.real_item_count:
            raise InvariantError(
                f"Invariant failed: t.real_item_count()={size} ≠ stats.real_item_count={stats.real_item_count}"
            )


def check_leaf_keys_and_values(
    tree: GPlusTreeBase,
    expected_keys: list[int] | None = None,
) -> tuple[list[int], bool, bool, bool]:
    """Traverse leaf nodes and validate keys / values.

    Works for both :class:`GPlusTreeBase` and :class:`GKPlusTreeBase`.

    For Gᵏ⁺-trees, the leaf stream carries dimension sentinels (negative
    keys): the tree's own dummy, expansion dummies of deeper dimensions and
    carried dummies from other dimensions.  An expanded first leaf set may
    even open the stream with a deeper dimension's sentinel.  Sentinels are
    validated by identity against the cached sentinel of their dimension
    (:func:`get_dummy`) instead of being treated as real keys; a sentinel
    that is not the cached object flips ``order_ok``.  Sentinel *placement*
    is deliberately not validated — a flat leaf-stream walk cannot know
    which set may carry which dummy; structural checks live in
    :func:`gplus_trees.tree_stats.gtree_stats_`.

    Returns
    -------
    (keys, presence_ok, all_have_values, order_ok)
    """
    dummy = tree.dummy_item
    # Only Gᵏ⁺-trees (``DIM`` class attribute) legitimately carry sentinels
    # in the leaf stream; for G⁺-trees a negative key stays an ordinary
    # (broken) key so their behaviour is unchanged.
    expansion_aware = hasattr(tree, "DIM")

    keys: list[int] = []
    all_have_values = True
    order_ok = True

    first = True
    prev_key = None
    for leaf in tree.iter_leaf_nodes():
        for entry in leaf.set:
            item = entry.item
            key = item.key
            if expansion_aware and key < 0:
                # prev_key stays untouched so real-key order is checked
                # across the sentinel.
                if item is not get_dummy(-key):
                    order_ok = False
                first = False
                continue
            if first:
                first = False
                if item is not dummy:
                    order_ok = False
                prev_key = key
                continue
            keys.append(key)
            if item.value is None:
                all_have_values = False
            if prev_key is not None and key < prev_key:
                order_ok = False
            prev_key = key

    presence_ok = True
    if expected_keys is not None:
        if len(keys) != len(expected_keys):
            presence_ok = False
        else:
            presence_ok = set(keys) == set(expected_keys)

    return keys, presence_ok, all_have_values, order_ok
