"""Statistics and invariant checking for G⁺-tree structures."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gplus_trees.logging_config import get_logger

if TYPE_CHECKING:
    from gplus_trees.gplus_tree_base import GPlusTreeBase

logger = get_logger(__name__)


@dataclass
class Stats:
    """Aggregated statistics for a G⁺-tree."""

    gnode_height: int
    gnode_count: int
    item_count: int
    real_item_count: int
    item_slot_count: int
    leaf_count: int
    rank: int
    is_heap: bool
    least_item: Any | None
    greatest_item: Any | None
    is_search_tree: bool
    internal_has_replicas: bool
    internal_packed: bool
    set_thresholds_met: bool
    linked_leaf_nodes: bool
    all_leaf_values_present: bool
    leaf_keys_in_order: bool
    inner_stats: list[Stats] | None = None


def gtree_stats_(
    t: GPlusTreeBase,
    rank_hist: dict[int, int] | None = None,
    _is_root: bool = True,
) -> Stats:
    """
    Returns aggregated statistics for a G⁺-tree in **O(n)** time.

    The caller can supply an existing Counter / dict for ``rank_hist``;
    otherwise a fresh Counter is used.
    """
    # Lazy imports to break circular dependency
    from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
    from gplus_trees.gplus_tree_base import get_dummy
    from gplus_trees.klist_base import KListBase

    if rank_hist is None:
        rank_hist = collections.Counter()

    # ---------- empty tree return ---------------------------------
    if t is None or t.is_empty():
        return Stats(
            gnode_height=0,
            gnode_count=0,
            item_count=0,
            real_item_count=0,
            item_slot_count=0,
            leaf_count=0,
            rank=-1,
            is_heap=True,
            least_item=None,
            greatest_item=None,
            is_search_tree=True,
            internal_has_replicas=True,
            set_thresholds_met=True,
            internal_packed=True,
            linked_leaf_nodes=True,
            all_leaf_values_present=True,
            leaf_keys_in_order=True,
            inner_stats=None,
        )

    K = t.SetClass.KListNodeClass.CAPACITY
    if hasattr(t, "l_factor"):
        threshold = t.l_factor * K
    else:
        threshold = K

    node = t.node
    node_set = node.set
    node_right_subtree = node.right_subtree
    node_rank = node.rank
    node_item_count = node_set.item_count()
    rank_hist[node_rank] = rank_hist.get(node_rank, 0) + node_set.item_count()

    # ---------- check inner (higher-dimension) tree if node_set is a GKPlusTreeBase ----
    # When a KList overflows, it is replaced by a recursively instantiated
    # GKPlusTreeBase at dimension D+1.  That inner tree has its own structural
    # invariants (heap, search tree, packed, replicas, thresholds, …) that
    # must also be validated.  We call gtree_stats_ on the inner tree and
    # merge its boolean flags into the outer stats.
    #
    # This is O(total gnodes across all dimensions) — each gnode is visited
    # exactly once because the inner call only processes the inner tree's
    # gnodes, not the outer tree's.
    inner_set_stats = None
    if isinstance(node_set, GKPlusTreeBase) and not node_set.is_empty():
        inner_set_stats = gtree_stats_(node_set, rank_hist=None, _is_root=True)

    # ---------- recurse on children only if rank > 1 ------------------------------------
    right_stats = gtree_stats_(node_right_subtree, rank_hist, False)

    # Only recurse on child nodes if we are at a non-leaf node indicated by the
    # presence of a right subtree
    if node_right_subtree is not None:
        child_stats = [gtree_stats_(e.left_subtree, rank_hist, False) for e in node_set]
    else:
        child_stats = []

    # ---------- aggregate ----------------------------------
    stats = Stats(
        gnode_height=0,
        gnode_count=0,
        item_count=0,
        real_item_count=0,
        item_slot_count=0,
        leaf_count=0,
        rank=node_rank,
        is_heap=True,
        least_item=None,
        greatest_item=None,
        is_search_tree=True,
        internal_has_replicas=True,
        internal_packed=(node_rank <= 1 or node_item_count > 1),
        set_thresholds_met=True,
        linked_leaf_nodes=True,
        all_leaf_values_present=True,
        leaf_keys_in_order=True,
        inner_stats=None,
    )

    # Precompute common values using right subtree stats
    stats.gnode_count = 1 + right_stats.gnode_count
    stats.item_count = node_item_count + right_stats.item_count
    stats.real_item_count += right_stats.real_item_count
    stats.item_slot_count = node_set.item_slot_count() + right_stats.item_slot_count
    stats.leaf_count += right_stats.leaf_count

    max_child_height = 0

    # Check search tree property for the node itself by comparing keys in order
    prev_key = None
    for i, entry in enumerate(node_set):
        current_key = entry.item.key

        # Check search tree property within the node
        if prev_key is not None and prev_key >= current_key:
            if hasattr(t, "DIM"):
                if current_key >= get_dummy(t.DIM).key:
                    stats.is_search_tree = False
            else:
                stats.is_search_tree = False

        # Process child stats if they exist (will be empty for leaf nodes)
        if i < len(child_stats):
            cs = child_stats[i]

            max_child_height = max(max_child_height, cs.gnode_height)

            if node_rank >= 2 and entry.item.value is not None:
                stats.internal_has_replicas = False

            # Accumulate counts
            stats.gnode_count += cs.gnode_count
            stats.item_count += cs.item_count
            stats.item_slot_count += cs.item_slot_count
            stats.leaf_count += cs.leaf_count
            stats.real_item_count += cs.real_item_count

            # Update boolean flags
            if stats.is_heap and not ((node_rank > cs.rank) and cs.is_heap):
                stats.is_heap = False

            # Inherit child violations first
            if not cs.set_thresholds_met:
                stats.set_thresholds_met = False

            # Check current node violations (only if no child violations yet)
            if stats.set_thresholds_met and (
                (isinstance(node_set, KListBase) and node_set.item_count() > threshold)
                or (not isinstance(node_set, KListBase) and node_set.item_count() <= threshold)
            ):
                stats.set_thresholds_met = False

            stats.internal_has_replicas &= cs.internal_has_replicas
            stats.internal_packed &= cs.internal_packed
            stats.linked_leaf_nodes &= cs.linked_leaf_nodes

            # Additional search tree property checks with child stats
            if stats.is_search_tree:
                if not cs.is_search_tree:
                    stats.is_search_tree = False
                elif cs.least_item and prev_key and cs.least_item.key < prev_key:
                    if hasattr(t, "DIM"):
                        if cs.least_item.key >= get_dummy(t.DIM).key:
                            stats.is_search_tree = False
                    else:
                        stats.is_search_tree = False
                elif cs.greatest_item and cs.greatest_item.key >= current_key:
                    if hasattr(t, "DIM"):
                        if cs.greatest_item.key >= get_dummy(t.DIM).key:
                            stats.is_search_tree = False
                    else:
                        stats.is_search_tree = False

        prev_key = current_key

    # Calculate final height
    stats.gnode_height = 1 + max(right_stats.gnode_height, max_child_height)

    # Fold in right subtree flags
    if stats.is_heap and not right_stats.is_heap:
        stats.is_heap = False

    # Check right subtree first
    if not right_stats.set_thresholds_met:
        stats.set_thresholds_met = False

    # Check current node violations (always check, regardless of current state)
    if (isinstance(node_set, KListBase) and node_set.item_count() > threshold) or (
        not isinstance(node_set, KListBase) and node_set.item_count() <= threshold
    ):
        stats.set_thresholds_met = False

    if stats.is_search_tree:
        if not right_stats.is_search_tree:
            stats.is_search_tree = False
        elif right_stats.least_item and prev_key is not None and right_stats.least_item.key < prev_key:
            if hasattr(t, "DIM"):
                if right_stats.least_item.key >= get_dummy(t.DIM).key:
                    stats.is_search_tree = False
            else:
                stats.is_search_tree = False

    stats.is_search_tree &= right_stats.is_search_tree
    if right_stats.least_item and right_stats.least_item.key < prev_key:
        if hasattr(t, "DIM"):
            if right_stats.least_item.key >= get_dummy(t.DIM).key:
                stats.is_search_tree = False
        else:
            stats.is_search_tree = False

    stats.internal_has_replicas &= right_stats.internal_has_replicas
    stats.internal_packed &= right_stats.internal_packed
    stats.linked_leaf_nodes &= right_stats.linked_leaf_nodes

    # ----- INNER (HIGHER-DIMENSION) TREE STATS -----
    # Merge boolean flags from the inner tree's stats into the outer stats.
    # This propagates any invariant violation in a higher-dimension tree
    # up to the outer tree's stats.  We also collect all inner_stats into
    # a flat list so callers can inspect per-dimension breakdowns.
    all_inner = []

    if inner_set_stats is not None:
        # Merge boolean flags from the inner tree
        stats.is_heap &= inner_set_stats.is_heap
        stats.is_search_tree &= inner_set_stats.is_search_tree
        stats.internal_has_replicas &= inner_set_stats.internal_has_replicas
        stats.internal_packed &= inner_set_stats.internal_packed
        stats.set_thresholds_met &= inner_set_stats.set_thresholds_met
        stats.linked_leaf_nodes &= inner_set_stats.linked_leaf_nodes
        stats.all_leaf_values_present &= inner_set_stats.all_leaf_values_present
        stats.leaf_keys_in_order &= inner_set_stats.leaf_keys_in_order
        all_inner.append(inner_set_stats)
        # Recursively collect any deeper inner stats
        if inner_set_stats.inner_stats:
            all_inner.extend(inner_set_stats.inner_stats)

    # Collect inner stats from child subtrees and right subtree
    for cs in child_stats:
        if cs.inner_stats:
            all_inner.extend(cs.inner_stats)
    if right_stats.inner_stats:
        all_inner.extend(right_stats.inner_stats)

    stats.inner_stats = all_inner if all_inner else None

    # ----- LEAST / GREATEST -----
    if child_stats and child_stats[0].least_item is not None:
        stats.least_item = child_stats[0].least_item
    else:
        stats.least_item = node_set.find_pivot()[0].item

    if right_stats.greatest_item is not None:
        stats.greatest_item = right_stats.greatest_item
    else:
        stats.greatest_item = node_set.get_max()[0].item

    # ---------- leaf walk ONCE at the root -----------------------------
    if node_rank == 1:  # leaf node: base values
        true_count = 0
        all_values_present = True

        for entry in node_set:
            item = entry.item
            if item.key >= 0:  # Skip dummy items
                true_count += 1
                if item.value is None:
                    all_values_present = False

        stats.all_leaf_values_present = all_values_present
        stats.real_item_count = true_count
        stats.leaf_count = 1

    # Root-level validation (only occurs once)
    if _is_root:
        leaf_keys, leaf_values = [], []
        leaf_count, item_count = 0, 0
        last_leaf, prev_key = None, None
        keys_in_order = True
        for leaf in t.iter_leaf_nodes():
            last_leaf = leaf
            leaf_count += 1

            for entry in leaf.set:
                item = entry.item
                if item.key < 0:
                    continue

                item_count += 1
                key = item.key

                # Check ordering
                if prev_key is not None and key < prev_key:
                    keys_in_order = False
                prev_key = key

                leaf_keys.append(key)
                leaf_values.append(item.value)

        # Set values from leaf traversal
        stats.leaf_keys_in_order = keys_in_order

        # Check leaf_count and real_item_count consistency
        if leaf_count != stats.leaf_count or item_count != stats.real_item_count:
            stats.linked_leaf_nodes = False
            stats.leaf_count = max(leaf_count, stats.leaf_count)
            stats.real_item_count = max(item_count, stats.real_item_count)
        elif last_leaf is not None:
            last_leaf.set.item_count()
            last_item = last_leaf.set.get_max()[0].item
            if stats.greatest_item is not last_item:
                stats.linked_leaf_nodes = False

    return stats


def _get_capacity(set_cls) -> int:
    """
    Walks ``set_cls.SetClass`` until we find a subclass of KListBase,
    then returns its ``KListNodeClass.CAPACITY``.
    """
    from gplus_trees.klist_base import KListBase

    cls = set_cls
    while not issubclass(cls, KListBase):
        cls = cls.SetClass
    return cls.KListNodeClass.CAPACITY
