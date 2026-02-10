"""
gplus_trees — History-independent ordered-set trees.

Quick-start imports::

    from gplus_trees import create_gplustree, create_gkplus_tree

See subpackage ``__init__`` files for the full public surface.
"""

# G⁺-tree
from gplus_trees.factory import create_gplustree, make_gplustree_classes  # noqa: F401
from gplus_trees.gplus_tree_base import GPlusTreeBase  # noqa: F401

# Gᵏ⁺-tree
from gplus_trees.g_k_plus import (  # noqa: F401
    GKPlusTreeBase,
    create_gkplus_tree,
    make_gkplustree_classes,
)

# Shared primitives
from gplus_trees.base import Entry, ItemData, LeafItem  # noqa: F401

__all__ = [
    # G⁺-tree
    "GPlusTreeBase",
    "create_gplustree",
    "make_gplustree_classes",
    # Gᵏ⁺-tree
    "GKPlusTreeBase",
    "create_gkplus_tree",
    "make_gkplustree_classes",
    # Primitives
    "Entry",
    "ItemData",
    "LeafItem",
]
