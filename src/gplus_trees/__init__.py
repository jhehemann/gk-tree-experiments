"""
gplus_trees — History-independent ordered-set trees.

Quick-start imports::

    from gplus_trees import create_gplustree, create_gkplus_tree

See subpackage ``__init__`` files for the full public surface.
"""

# G⁺-tree
# Shared primitives
from gplus_trees.base import Entry, ItemData, LeafItem
from gplus_trees.factory import create_gplustree, make_gplustree_classes

# Gᵏ⁺-tree
from gplus_trees.g_k_plus import (
    GKPlusTreeBase,
    create_gkplus_tree,
    make_gkplustree_classes,
)
from gplus_trees.gplus_tree_base import GPlusTreeBase
from gplus_trees.invariants import (
    InvariantError,
    check_leaf_keys_and_values,
)

# Stats & invariants
from gplus_trees.tree_stats import Stats, gtree_stats_

__all__ = [
    # Primitives
    "Entry",
    # Gᵏ⁺-tree
    "GKPlusTreeBase",
    # G⁺-tree
    "GPlusTreeBase",
    "InvariantError",
    "ItemData",
    "LeafItem",
    # Stats & invariants
    "Stats",
    "check_leaf_keys_and_values",
    "create_gkplus_tree",
    "create_gplustree",
    "gtree_stats_",
    "make_gkplustree_classes",
    "make_gplustree_classes",
]
