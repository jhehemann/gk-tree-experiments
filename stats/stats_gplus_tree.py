"""Statistics for G⁺-trees."""

import argparse
import logging
import math
import os
import random
import time
from datetime import datetime
from statistics import mean

import numpy as np

from gplus_trees.base import ItemData, LeafItem
from gplus_trees.factory import create_gplustree
from gplus_trees.gplus_tree_base import GPlusTreeBase
from gplus_trees.invariants import assert_tree_invariants_raise
from gplus_trees.tree_stats import gtree_stats_

logger = logging.getLogger(__name__)


# Assume create_gtree(items) builds a GPlusTree from a list of (LeafItem, rank) pairs.
def create_gtree(items, K=16):
    """
    Mimics the Rust create_gtree: build a tree by inserting each (item, rank) pair.
    Uses the factory pattern to create a tree with the specified capacity K.
    """
    tree = create_gplustree(K)
    tree_insert = tree.insert
    for item, rank in items:
        tree_insert(item, rank)
    return tree


# Create a random GPlusTree with n items and target node size (K) determining the rank distribution.
def random_gtree_of_size(n: int, target_node_size: int) -> GPlusTreeBase:
    # cache globals
    # calc_rank = calculate_item_rank
    # group_size = calculate_group_size(target_node_size)
    make_item_data = ItemData
    make_item = LeafItem
    p = 1.0 - (1.0 / (target_node_size))  # probability for geometric dist
    # logging.info(f"p = {p:.4f} for K = {target_node_size}")

    # we need at least n unique values; 2^24 = 16 777 216 > 1 000 000
    space = 1 << 24
    if space <= n:
        raise ValueError(f"Key-space too small! Required: {n + 1}, Available: {space}")

    indices = random.sample(range(1, space), k=n)  # Exclude dummy key 0

    # Pre-allocate items list
    items = [(None, None)] * n

    ranks = np.random.geometric(p, size=n)

    # Process all items in a single pass
    for i, idx in enumerate(indices):
        # Use the index directly as the key
        key = idx
        val = "val"
        items[i] = (make_item(make_item_data(key, val)), int(ranks[i]))

    return create_gtree(items, K=target_node_size)


# The function random_klist_tree just wraps random_gtree_of_size with a given K.
def random_klist_tree(n: int, K: int) -> GPlusTreeBase:
    return random_gtree_of_size(n, K)


def repeated_experiment(
    size: int,
    repetitions: int,
    K: int,
) -> None:
    """
    Repeatedly builds random GPlusTrees (with size items) using ranks drawn from a geometric distribution.
    Uses K as target node size to compute the geometric parameter. Aggregates statistics and timings over many trees.
    """
    t_all_0 = time.perf_counter()

    # Storage for stats and timings
    results = []  # List of tuples: (stats, phy_height)
    times_build = []
    times_stats = []
    times_phy = []

    # Generate results from repeated experiments.
    for _ in range(repetitions):
        # Time tree construction
        t0 = time.perf_counter()
        tree = random_klist_tree(size, K)
        times_build.append(time.perf_counter() - t0)

        # Time stats computation
        t0 = time.perf_counter()
        stats = gtree_stats_(tree, {})
        times_stats.append(time.perf_counter() - t0)
        # print("Tree stats:")
        # pprint(asdict(stats))

        # Time physical height computation
        t0 = time.perf_counter()
        phy_height = tree.physical_height()
        times_phy.append(time.perf_counter() - t0)

        results.append((stats, phy_height))

        assert_tree_invariants_raise(tree, stats)
        # print("Tree stats:")
        # pprint(asdict(stats))

    # Perfect height: ceil( log_{K+1}(size) )
    perfect_height = math.ceil(math.log(size, K)) if size > 0 else 0

    # Aggregate averages for stats
    avg_gnode_height = mean(s.gnode_height for s, _ in results)
    avg_gnode_count = mean(s.gnode_count for s, _ in results)
    avg_leaf_count = mean(s.leaf_count for s, _ in results)
    avg_real_item_count = mean(s.real_item_count for s, _ in results)
    avg_item_count = mean(s.item_count for s, _ in results)
    avg_item_slot_count = mean(s.item_slot_count for s, _ in results)
    avg_space_amp = mean((s.item_slot_count / s.item_count) for s, _ in results)
    avg_physical_height = mean(h for _, h in results)
    avg_height_amp = mean((h / perfect_height) for _, h in results) if perfect_height else 0
    avg_avg_gnode_size = mean((s.item_count / s.gnode_count) for s, _ in results)
    avg_max_rank = mean(s.rank for s, _ in results)

    # Aggregate averages for timings
    avg_build_time = mean(times_build)
    avg_stats_time = mean(times_stats)
    avg_phy_time = mean(times_phy)

    # Compute variances for stats
    var_gnode_height = mean((s.gnode_height - avg_gnode_height) ** 2 for s, _ in results)
    var_gnode_count = mean((s.gnode_count - avg_gnode_count) ** 2 for s, _ in results)
    var_leaf_count = mean((s.leaf_count - avg_leaf_count) ** 2 for s, _ in results)
    var_real_item_count = mean((s.real_item_count - avg_real_item_count) ** 2 for s, _ in results)
    var_item_count = mean((s.item_count - avg_item_count) ** 2 for s, _ in results)
    var_item_slot_count = mean((s.item_slot_count - avg_item_slot_count) ** 2 for s, _ in results)
    var_space_amp = mean(((s.item_slot_count / s.item_count) - avg_space_amp) ** 2 for s, _ in results)
    var_physical_height = mean((h - avg_physical_height) ** 2 for _, h in results)
    var_height_amp = mean(((h / perfect_height) - avg_height_amp) ** 2 for _, h in results) if perfect_height else 0
    var_avg_gnode_size = mean(((s.item_count / s.gnode_count) - avg_avg_gnode_size) ** 2 for s, _ in results)
    var_max_rank = mean((s.rank - avg_max_rank) ** 2 for s, _ in results)

    # Compute variances for timings
    var_build_time = mean((t - avg_build_time) ** 2 for t in times_build)
    var_stats_time = mean((t - avg_stats_time) ** 2 for t in times_stats)
    var_phy_time = mean((t - avg_phy_time) ** 2 for t in times_phy)

    # Prepare rows for stats and timings
    rows = [
        ("Real item count", avg_real_item_count, var_real_item_count),
        ("Item count", avg_item_count, var_item_count),
        ("Item slot count", avg_item_slot_count, var_item_slot_count),
        ("Space amplification", avg_space_amp, var_space_amp),
        ("G-node count", avg_gnode_count, var_gnode_count),
        ("Leaf count", avg_leaf_count, var_leaf_count),
        ("Avg G-node size", avg_avg_gnode_size, var_avg_gnode_size),
        ("Maximum rank", avg_max_rank, var_max_rank),
        ("G-node height", avg_gnode_height, var_gnode_height),
        ("Actual height", avg_physical_height, var_physical_height),
        ("Perfect height", perfect_height, None),
        ("Height amplification", avg_height_amp, var_height_amp),
    ]

    # Log table
    header = f"{'Metric':<20} {'Avg':>15} {'(Var)':>15}"
    sep_line = "-" * len(header)

    # logging.info(f"n = {size}; K = {K}; {repetitions} repetitions")
    logger.info(header)
    logger.info(sep_line)
    for name, avg, var in rows:
        if var is None:
            logger.info(f"{name:<20} {avg:>15}")
        else:
            var_str = f"({var:.2f})"
            avg_fmt = f"{avg:15.2f}"
            logger.info(f"{name:<20} {avg_fmt} {var_str:>15}")

    # Performance metrics
    sum_build = sum(times_build)
    sum_stats = sum(times_stats)
    sum_phy = sum(times_phy)
    total_sum = sum_build + sum_stats + sum_phy

    pct_build = (sum_build / total_sum * 100) if total_sum else 0
    pct_stats = (sum_stats / total_sum * 100) if total_sum else 0
    pct_phy = (sum_phy / total_sum * 100) if total_sum else 0

    perf_rows = [
        ("Build time (s)", avg_build_time, var_build_time, sum_build, pct_build),
        ("Stats time (s)", avg_stats_time, var_stats_time, sum_stats, pct_stats),
        ("Phy height time (s)", avg_phy_time, var_phy_time, sum_phy, pct_phy),
    ]

    # 2) Log a separate performance table
    header = f"{'Metric':<20}{'Avg(s)':>13}{'Var(s)':>13}{'Total(s)':>13}{'%Total':>10}"
    sep = "-" * len(header)

    logger.info("")  # blank line for separation
    logger.info("Performance summary:")
    logger.info(header)
    logger.info(sep)
    for name, avg, var, total, pct in perf_rows:
        logger.info(f"{name:<20}{avg:13.6f}{var:13.6f}{total:13.6f}{pct:10.2f}%")

    logger.info(sep)
    t_all_1 = time.perf_counter() - t_all_0
    logger.info("Execution time: %.3f seconds", t_all_1)


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run statistics experiments for G-Plus Trees.")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10, 100, 1000, 10_000, 100_000], help="List of tree sizes to test."
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[2, 4, 16, 64], help="List of K values (target node size) to test."
    )
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions for each experiment.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    log_dir = os.path.join(os.getcwd(), "stats/logs/gplus_tree_logs")
    os.makedirs(log_dir, exist_ok=True)

    # 2) Create a timestamped logfile name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{ts}.log")

    # 3) Configure logging to write to that file (and still print to console, if you like)
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),  # comment this out if you don't want console output
        ],
        force=True,  # Override any existing logging configuration
    )

    # Also apply the chosen level to the library logger so that
    # log records from gplus_trees.* are emitted at the requested level.
    logging.getLogger("gplus_trees").setLevel(log_level)

    # List of tree sizes to test.
    sizes = args.sizes
    # List of K values for which we want to run experiments.
    Ks = args.ks
    repetitions = args.repetitions

    for n in sizes:
        for K in Ks:
            logger.info("")
            logger.info("")
            logger.info(
                f"---------------- NOW RUNNING EXPERIMENT: n = {n}, K = {K}, repetitions = {repetitions} ----------------"
            )
            t0 = time.perf_counter()
            repeated_experiment(size=n, repetitions=repetitions, K=K)
            elapsed = time.perf_counter() - t0
            logger.info(f"Total experiment time: {elapsed:.3f} seconds")
