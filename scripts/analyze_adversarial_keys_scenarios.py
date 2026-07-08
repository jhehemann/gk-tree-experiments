import argparse
import cProfile
import logging
import pstats
import random

from adversarial import analyze_chi, analyze_structure, bushy_pile_tree
from adversarial.mixed_rank import _bushy_leaf_profiles
from gplus_trees.base import ItemData, LeafItem
from gplus_trees.g_k_plus.factory import make_gkplustree_classes
from gplus_trees.g_k_plus.g_k_plus_base import (
    GKPlusTreeBase,
)
from gplus_trees.g_k_plus.utils import calc_ranks
from gplus_trees.gplus_tree_base import print_pretty
from gplus_trees.utils import find_keys_for_rank_lists


def iter_nodes(tree):
    """
    Iterate over all nodes in a GKPlusTreeBase in depth-first order.

    Yields:
        GKPlusNodeBase: Each node in the tree.
    """
    if tree.is_empty():
        return
    stack = [tree]
    while stack:
        tree = stack.pop()
        yield tree
        if tree.node.rank > 1:
            for entry in tree.node.set:
                if entry.left_subtree is not None:
                    stack.append(entry.left_subtree)
            if tree.node.right_subtree is not None:
                stack.append(tree.node.right_subtree)


def print_tree_nodes(tree):
    """
    Print the structure of the tree using print_pretty for the first node,
    and recursively for any node whose set is a GKPlusTreeBase.
    """
    tree_nodes = list(iter_nodes(tree))
    if not tree_nodes:
        print("Empty tree")
        return

    print(print_pretty(tree_nodes[0]))
    for tree in tree_nodes:
        if isinstance(tree.node.set, GKPlusTreeBase):
            print()
            print_tree_nodes(tree.node.set)


def create_rank_lists(num_lists, size, key_range):
    """Create multiple rank lists of unique keys."""
    return [random.sample(range(key_range[0], key_range[1]), size) for _ in range(num_lists)]


def create_uniform_rank_lists(dimensions=10, size=10, rank=1):
    """Create a list of rank lists, each filled with the same value."""
    return [[rank] * size for _ in range(dimensions)]


def create_adversarial_1_rank_lists(dimensions=10, size=10):
    """
    Create rank lists where all elements are 1.
    """
    rank_lists = []
    for _i in range(dimensions):
        rank_list = [1] * size
        rank_lists.append(rank_list)
    return rank_lists, None


def create_adversarial_rank_lists_i_eq_2(dimensions=10, size=10):
    """
    Create rank lists where in each list with index i, the element at i is 2, others are 1.
    """
    rank_lists = []
    for i in range(dimensions):
        rank_list = [1] * size
        if i < size:
            rank_list[i] = 2
        rank_lists.append(rank_list)
    return rank_lists, None


def create_middle_out_pattern_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: middle position is 2, rest are 1.
    - Second list: middle-1 and middle+1 are 2, rest are 1.
    - Third list: middle-2 and middle+2 are 2, rest are 1.
    - Continue for 'dimensions' lists.
    """
    rank_lists = []
    middle = size // 2
    for i in range(dimensions):
        rank_list = [1] * size
        positions = []
        if i == 0:
            positions = [middle]
        else:
            if middle - i >= 0:
                positions.append(middle - i)
            if middle + i < size:
                positions.append(middle + i)
        for pos in positions:
            rank_list[pos] = 2
        rank_lists.append(rank_list)
    return rank_lists, None


def create_middle_in_pattern_rank_lists(dimensions=10, size=10, k=4):
    """
    Create rank lists where in the first list, the kth and kth last positions are 2, rest are 1.
    In each subsequent list, the positions of 2 are shifted towards the middle by k.
    This continues for 'dimensions' lists.
    """
    rank_lists = []
    spacing_factor = 1
    spacing = spacing_factor  # Spacing factor to shift positions
    left = spacing
    right = size - spacing - 1
    for _i in range(dimensions):
        rank_list = [1] * size
        if left <= right:
            rank_list[left] = 2
            rank_list[right] = 2
        rank_lists.append(rank_list)
        left += spacing
        right -= spacing
    return rank_lists, None


def create_dim_1_middle_2_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: middle position is 2, rest are 1.
    - All other lists: all ranks are 1.
    """
    rank_lists = []
    middle = size // 2
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0:
            rank_list[middle] = 2
        rank_lists.append(rank_list)
    return rank_lists, [middle]


def create_dim_1_first_2_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: first position is 2, rest are 1.
    - All other lists: all ranks are 1.
    """
    rank_lists = []
    first = 0
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0:
            rank_list[first] = 2
        rank_lists.append(rank_list)
    return rank_lists, [first]


def create_dim_1_first_and_middle_2_rank_lists(dimensions=10, size=10):
    """
    Create rank lists with a pattern:
    - First list: first and middle positions are 2, rest are 1.
    - All other lists: all ranks are 1.
    """
    rank_lists = []
    first = 0
    second = 5
    for i in range(dimensions):
        rank_list = [1] * size
        if i == 0:
            rank_list[first] = 2
            rank_list[second] = 2
        rank_lists.append(rank_list)
    return rank_lists, [first, second]


def build_gkplus_tree(keys, k, l_factor=1.0):
    """Build a GK+ tree from ``keys`` — the measurement-compatible pipeline the
    bushy tests use: ``make_gkplustree_classes(k)`` → ``tree_cls(l_factor)``,
    dim-1 ranks via ``calc_ranks``, reassigning the tree on every insert.
    History independence makes the insert order irrelevant to the final tree.
    """
    tree_cls, _, _, _ = make_gkplustree_classes(k)
    tree = tree_cls(l_factor=l_factor)
    for item, rank in zip(
        (LeafItem(ItemData(key, f"v{key}")) for key in keys),
        calc_ranks(list(keys), k),
        strict=True,
    ):
        tree, _, _ = tree.insert(item, rank)
    return tree


def print_rank_matrix(rank_lists, title="Rank matrix (dimension x key):"):
    """Dimension x key matrix, ranks != 1 highlighted red — same display as the
    original scenario print, one row per dimension."""
    print(title)
    for j, rl in enumerate(rank_lists, start=1):
        display_rl = rl if len(rl) <= 20 else [*rl[:10], "...", *rl[-10:]]
        colored = [f"\033[31m{n}\033[0m" if n != 1 and n != "..." else str(n) for n in display_rl]
        print(f"dim {j}:  " + " ".join(colored))


def create_bushy_rank_lists(depth=3, branching=2, leaf_size=1):
    """Dimension-major rank lists for the bushy pile-tree attack.

    Reuses the authentic generator ``adversarial.mixed_rank._bushy_leaf_profiles``
    so the descending-lex ordering that *sustains* the nesting (rather than
    collapsing it to siblings) is exactly the verified one. Returns
    ``(rank_lists, None)`` to match the other ``create_*`` scenarios.
    """
    profiles = _bushy_leaf_profiles(depth, branching, leaf_size)
    return [[p[j] for p in profiles] for j in range(depth)], None


def bushy_predicted_height(branching, depth):
    """L16 closed form for the nesting-aware height: (B^(d+1) - B) / (B - 1)."""
    return (branching ** (depth + 1) - branching) // (branching - 1)


def measure_and_report(tree, *, expected_height=None):
    """Run the structural observers (analyze_structure / analyze_chi) and print."""
    h_nested, dim_max, dummy_count = analyze_structure(tree)
    max_chi, max_prod_chi = analyze_chi(tree)
    print("\nStructural measurements (analyze_structure / analyze_chi):")
    hline = f"  nesting-aware height : {h_nested}"
    if expected_height is not None:
        verdict = "OK" if h_nested == expected_height else "MISMATCH"
        hline += f"   (predicted {expected_height} -> {verdict})"
    print(hline)
    print(f"  max dimension        : {dim_max}")
    print(f"  max chi (conv. mult) : {max_chi}")
    print(f"  max prod chi         : {max_prod_chi}")
    print(f"  dummy count          : {dummy_count}")
    return {
        "height": h_nested,
        "dim": dim_max,
        "max_chi": max_chi,
        "max_prod_chi": max_prod_chi,
        "dummy_count": dummy_count,
    }


def _profiled(fn, enabled):
    """Run ``fn()``; return ``(result, profiler_or_None)``."""
    if not enabled:
        return fn(), None
    profiler = cProfile.Profile()
    profiler.enable()
    result = fn()
    profiler.disable()
    return result, profiler


def _print_profile(profiler, header, rows=12):
    if profiler is None:
        return
    print(f"\n{header}")
    pstats.Stats(profiler).sort_stats("cumtime").print_stats(rows)


RANK_LIST_SCENARIOS = {
    "adv1": create_adversarial_1_rank_lists,
    "i_eq_2": create_adversarial_rank_lists_i_eq_2,
    "middle_out": create_middle_out_pattern_rank_lists,
    "middle_in": create_middle_in_pattern_rank_lists,
    "dim1_middle": create_dim_1_middle_2_rank_lists,
    "dim1_first": create_dim_1_first_2_rank_lists,
    "dim1_first_middle": create_dim_1_first_and_middle_2_rank_lists,
}


def run_bushy(k=4, depth=3, branching=2, leaf_size=None, l_factor=1.0, *, measure=True, profile=False, matrix=True):
    """Bushy attack end to end: authentic keygen -> build -> structural measures.

    ``leaf_size=None`` auto-picks ``int(k*l_factor) - 1`` (>= threshold/branching
    so every dimension < depth converts, <= threshold so the depth-``d`` leaf
    blocks stay klists) — the regime the bushy tests use for k=4 (leaf=3).
    """
    if leaf_size is None:
        leaf_size = max(1, int(k * l_factor) - 1)
    print(
        f"\n=== bushy pile-tree: k={k}, depth={depth}, branching={branching}, leaf={leaf_size}, l_factor={l_factor} ==="
    )
    if matrix:
        # the clean pattern (one column per leaf-path); the built tree uses `leaf` copies per column
        pattern, _ = create_bushy_rank_lists(depth, branching, 1)
        print_rank_matrix(pattern, title="Rank pattern (dimension x leaf-path, one key each):")

    key_set, keygen_pr = _profiled(lambda: bushy_pile_tree(k, depth, branching, leaf_size), profile)
    keys = list(key_set.keys)
    n = len(keys)
    predicted = bushy_predicted_height(branching, depth)
    print(f"n = B^d*leaf = {n} keys; predicted height H_d = {predicted}")
    print("keys:", keys if n <= 20 else [*keys[:10], "...", *keys[-10:]])
    _print_profile(keygen_pr, "KEYGEN PROFILE (bushy_pile_tree):")

    tree, build_pr = _profiled(lambda: build_gkplus_tree(keys, k, l_factor), profile)
    print("Tree built.")
    _print_profile(build_pr, "TREE-BUILD PROFILE (build_gkplus_tree):")

    if measure:
        measure_and_report(tree, expected_height=predicted)
    return tree


def run_rank_list_scenario(name, k=4, dimensions=5, size=10, l_factor=1.0, *, measure=True, profile=False, matrix=True):
    """One of the rank-list scenarios: keygen (find_keys_for_rank_lists) -> build -> measures."""
    rank_lists, _ = RANK_LIST_SCENARIOS[name](dimensions=dimensions, size=size)
    print(f"\n=== scenario {name!r}: k={k}, dimensions={dimensions}, size={size} ===")
    if matrix:
        print_rank_matrix(rank_lists, title="Adversarial rank lists (dimension x key):")

    keys, keygen_pr = _profiled(lambda: find_keys_for_rank_lists(rank_lists, k), profile)
    n = len(keys)
    print("keys:", keys if n <= 20 else [*keys[:10], "...", *keys[-10:]])
    _print_profile(keygen_pr, "KEYGEN PROFILE (find_keys_for_rank_lists):")

    tree, build_pr = _profiled(lambda: build_gkplus_tree(keys, k, l_factor), profile)
    print("Tree built.")
    _print_profile(build_pr, "TREE-BUILD PROFILE (build_gkplus_tree):")

    if measure:
        measure_and_report(tree)
    return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate adversarial keys for a GK+ tree and measure the resulting tree "
        "(nesting-aware height, max dimension, conversion multiplicity chi). Default: bushy pile-tree."
    )
    parser.add_argument(
        "--scenario",
        default="bushy",
        choices=["bushy", *RANK_LIST_SCENARIOS],
        help="adversarial pattern to run (default: bushy)",
    )
    parser.add_argument("-k", "--k", type=int, default=4, help="k-list node capacity (default 4)")
    parser.add_argument("--l-factor", type=float, default=1.0, help="conversion threshold factor (default 1.0)")
    # bushy params
    parser.add_argument("-d", "--depth", type=int, default=3, help="[bushy] depth = number of dimensions (default 3)")
    parser.add_argument("-b", "--branching", type=int, default=2, help="[bushy] piles per dimension (default 2)")
    parser.add_argument(
        "-l",
        "--leaf",
        type=int,
        default=None,
        help="[bushy] keys per leaf-path (default: auto = int(k*l_factor)-1, which triggers conversion)",
    )
    # rank-list-scenario params
    parser.add_argument("--dimensions", type=int, default=5, help="[rank-list scenarios] number of dimensions")
    parser.add_argument("--size", type=int, default=10, help="[rank-list scenarios] keys per dimension")
    # what to run
    parser.add_argument("--no-measure", dest="measure", action="store_false", help="skip structural measurements")
    parser.add_argument("--profile", action="store_true", help="cProfile keygen and tree build (timing)")
    parser.add_argument("--no-matrix", dest="matrix", action="store_false", help="do not print the rank matrix")
    parser.add_argument(
        "--print-tree",
        action="store_true",
        help="print the built tree with print_pretty (use small params, e.g. -d 2, or output explodes)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True
    )
    logging.getLogger("gplus_trees").setLevel(level)

    if args.scenario == "bushy":
        tree = run_bushy(
            args.k,
            args.depth,
            args.branching,
            args.leaf,
            args.l_factor,
            measure=args.measure,
            profile=args.profile,
            matrix=args.matrix,
        )
    else:
        tree = run_rank_list_scenario(
            args.scenario,
            args.k,
            args.dimensions,
            args.size,
            args.l_factor,
            measure=args.measure,
            profile=args.profile,
            matrix=args.matrix,
        )

    if args.print_tree:
        print("\nTree structure (print_pretty; the first block is dimension 1, each")
        print("further block is the inner tree of a converted node one dimension deeper):")
        print_tree_nodes(tree)
