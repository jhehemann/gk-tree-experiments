"""Pretty-printing and display utilities for G⁺-tree structures."""

from __future__ import annotations

import collections
import math
from typing import TYPE_CHECKING

from gplus_trees.base import AbstractSetDataStructure

if TYPE_CHECKING:
    from gplus_trees.gplus_tree_base import GPlusTreeBase


# ANSI colour codes
PRIMARY = "\033[32m"  # green
SECONDARY = "\033[33m"  # yellow
RESET = "\033[0m"


def print_pretty(set: AbstractSetDataStructure | None) -> str:
    """
    Prints a G⁺-tree so:
      • Lines go from highest rank down to 1.
      • Within a line, nodes appear left→right in traversal order.
      • All columns have the same width, so initial indent and
        inter-node spacing are uniform.
    """
    from gplus_trees.gplus_tree_base import GPlusTreeBase
    from gplus_trees.klist_base import KListBase

    if set is None:
        return f"{type(set).__name__}: None"

    if not (isinstance(set, (GPlusTreeBase, KListBase))):
        raise TypeError(f"print_pretty() expects GPlusTreeBase or KListBase, got {type(set).__name__}")

    if set.is_empty():
        return f"{type(set).__name__}: Empty"

    SEP = " | "
    set_type = type(set).__name__

    if isinstance(set, KListBase):
        texts = []
        node = set.head
        while node is not None:
            text = "[" + SEP.join(str(e.item.key) for e in node.entries) + "]"
            texts.append(text)
            node = node.next
        res_text = " ".join(texts)
        return f"({set_type}): {res_text}"

    tree = set
    dum_key = tree.dummy_item.key

    # 1) First pass: collect each node's text and track max length
    layers_raw = collections.defaultdict(list)  # rank -> list of node-strings
    max_len = 0

    def collect(tree, parent=None):
        if tree.is_empty():
            return
        nonlocal max_len
        dim = tree.dimension
        p_dim = parent.dimension if parent else None
        other_dim = False
        other_dim_processed = False

        if parent is not None and dim != p_dim:
            other_dim = True
            other_dim_processed = True

        node = tree.node
        rank = node.rank
        parent_rank = parent.node.rank if parent else None

        fill_rank = parent_rank - 1 if parent_rank is not None else rank
        while fill_rank > rank:
            layers_raw[fill_rank].append("")
            fill_rank -= 1

        text = ""
        for e in node.set:
            if parent is not None and other_dim:
                if e.item.key < dum_key:
                    text += (SEP if text else "") + f"{SECONDARY}{e.item.key}{RESET}"
                else:
                    text += (SEP if text else "") + str(e.item.key)
            else:
                if e.item.key == dum_key:
                    text += (SEP if text else "") + f"{PRIMARY}{e.item.key}{RESET}"
                elif e.item.key < dum_key:
                    text += (SEP if text else "") + f"{SECONDARY}{e.item.key}{RESET}"
                else:
                    text += (SEP if text else "") + str(e.item.key)

        if parent is None or not other_dim:
            layers_raw[rank].append(text)
            max_len = max(max_len, len(text))
        else:
            new_text = f"(D{dim}R{rank}) " + text
            layers_raw[0].append(new_text)
            max_len = max(max_len, len(new_text))

        # recurse left→right
        if not other_dim:
            for e in node.set:
                if e.left_subtree:
                    collect(e.left_subtree, tree)
            if node.right_subtree:
                collect(node.right_subtree, tree)

        # Special case: if node.set is an inner tree of a deeper dimension, traverse it
        inner_set_tree = tree.inner_tree_of(node.set)
        if inner_set_tree is not None and not inner_set_tree.is_empty() and not other_dim_processed:
            collect(inner_set_tree, tree)

    collect(tree, None)

    # 2) Define a fixed column width: widest text + 1 space padding
    column_width = (max_len // 2) + 1

    # 3) Build "slots" per layer
    all_ranks = sorted(layers_raw.keys())
    max_slots = max(len(v) for v in layers_raw.values())

    layers = {}
    column_counts = [len(layers_raw[rank]) for rank in all_ranks]

    for rank in all_ranks:
        texts = layers_raw[rank]
        padded = [
            ("" + txt.center(column_width) + "  " if i < len(texts) else "" + " " * (column_width // 2)) + ""
            for i, txt in enumerate(texts + [""] * max_slots)
        ][:max_slots]
        layers[rank] = padded

    # 4) Accumulate with indent
    out_lines = []
    cumm_indent = 0.0
    for i, rank in enumerate(all_ranks):
        if i == 0:
            prefix = "     "
        else:
            column_diff = column_counts[i - 1] - column_counts[i]
            cumm_indent += float(column_diff) / 2
            spaces = math.floor(((2 + column_width) * cumm_indent) + 0.5)
            prefix = "     " + spaces * " "
        line = "".join(layers[rank])
        layer_id = f"{PRIMARY}Rank {rank}{RESET}" if rank > 0 else f"{SECONDARY}Other Dims{RESET}"
        out_lines.append(f"{layer_id}:{prefix}{line}")

    res_text = set_type + "\n" + "\n\n".join(reversed(out_lines)) + "\n"
    return res_text


def collect_leaf_keys(tree: GPlusTreeBase) -> list[str]:
    """Collect all non-dummy leaf keys from a G⁺-tree."""
    from gplus_trees.gplus_tree_base import DUMMY_KEY

    out = []
    for leaf in tree.iter_leaf_nodes():
        for e in leaf.set:
            if e.item.key != DUMMY_KEY:
                out.append(e.item.key)
    return out


def print_structure(
    tree: GPlusTreeBase,
    indent: int = 0,
    depth: int = 0,
    max_depth: int = 2,
) -> str:
    """Return a debugging-oriented structural dump of a G⁺-tree.

    Recursively prints each node's rank, set type, right subtree, and
    (for leaf nodes) linked-list next pointer.  The heavy lifting for
    KList nodes is delegated to :pymethod:`KListBase.print_structure`.
    """
    prefix = " " * indent
    if tree is None or tree.is_empty():
        return f"{prefix}Empty {tree.__class__.__name__}"

    if depth > max_depth:
        return f"{prefix}... (max depth reached)"

    result = []
    node = tree.node

    result.append(f"{prefix}{node.__class__.__name__}(rank={node.rank}, set={type(node.set).__name__})")
    result.append(node.set.print_structure(indent + 4))

    # Right subtree
    if node.right_subtree is not None:
        right_node = node.right_subtree.node
        result.append(
            f"{prefix}    Right: {right_node.__class__.__name__}(rank={right_node.rank}, "
            f"set={type(right_node.set).__name__})"
        )
        result.append(right_node.set.print_structure(indent + 8))
    else:
        result.append(f"{prefix}    Right: Empty")

    # Leaf-chain next pointer (leaf nodes only)
    if node.rank == 1 and node.next is not None:
        if not node.next.is_empty():
            next_node = node.next.node
            result.append(
                f"{prefix}    Next: {next_node.__class__.__name__}(rank={next_node.rank}, "
                f"set={type(next_node.set).__name__})"
            )
            result.append(next_node.set.print_structure(indent + 8))
        else:
            result.append(f"{prefix}    Next: Empty")
    elif node.rank == 1:
        result.append(f"{prefix}    Next: Empty")

    return "\n".join(result)
