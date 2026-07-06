"""Utility functions for testing GPlusTree invariants."""

from gplus_trees.gplus_tree_base import (
    GPlusTreeBase,
    Stats,
)

# Re-export shared invariant utilities so existing imports keep working.
from gplus_trees.invariants import (  # noqa: F401
    TREE_FLAGS,
    InvariantError,
    assert_tree_invariants_raise,
    check_leaf_keys_and_values,
)


def assert_tree_invariants_tc(
    tc, t: GPlusTreeBase, stats: Stats, err_msg: str | None = "", exclude_checks: list | None = None
) -> None:
    """TestCase version: use inside unittest.TestCase methods."""
    if exclude_checks is None:
        exclude_checks = []
    for flag in TREE_FLAGS:
        if flag in exclude_checks:
            continue
        if flag == "set_thresholds_met" and t.set_conversion_threshold is None:
            # this flag is only relevant for variants with set conversion
            continue
        tc.assertTrue(getattr(stats, flag), f"Invariant failed: {flag} is False\n\n{err_msg}")

    if not t.is_empty():
        tc.assertGreater(
            stats.item_count, 0, f"Invariant failed: item_count={stats.item_count} ≤ 0 for non-empty tree\n\n{err_msg}"
        )
        tc.assertGreater(
            stats.item_slot_count,
            0,
            f"Invariant failed: item_slot_count={stats.item_slot_count} ≤ 0 for non-empty tree\n\n{err_msg}",
        )
        tc.assertGreater(
            stats.gnode_count,
            0,
            f"Invariant failed: gnode_count={stats.gnode_count} ≤ 0 for non-empty tree\n\n{err_msg}",
        )
        tc.assertGreater(
            stats.gnode_height,
            0,
            f"Invariant failed: gnode_height={stats.gnode_height} ≤ 0 for non-empty tree\n\n{err_msg}",
        )
        tc.assertGreater(stats.rank, 0, f"Invariant failed: rank={stats.rank} ≤ 0 for non-empty tree\n\n{err_msg}")
        tc.assertIsNotNone(stats.least_item, f"Invariant failed: least_item is None for non-empty tree\n\n{err_msg}")
        tc.assertIsNotNone(
            stats.greatest_item, f"Invariant failed: greatest_item is None for non-empty tree\n\n{err_msg}"
        )

        # if t tracks its own size, it must equal stats.real_item_count
        size = t.tracked_size()
        if size is not None:
            tc.assertEqual(
                size,
                stats.real_item_count,
                f"Invariant failed: tracked_size()={size} ≠ real_item_count={stats.real_item_count}, \n\n{err_msg}",
            )
