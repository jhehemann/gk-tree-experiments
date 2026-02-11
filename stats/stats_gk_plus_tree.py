"""Statistics for GK⁺-trees."""

import os
import logging
import math
import random
import time
from statistics import mean
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np
import argparse

from gplus_trees.base import ItemData, LeafItem
from gplus_trees.tree_stats import gtree_stats_
from gplus_trees.invariants import assert_tree_invariants_raise, check_leaf_keys_and_values

from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase
from benchmarks.benchmark_gkplus_tree_adversarial import load_adversarial_keys_from_file

logger = logging.getLogger(__name__)

# Assume create_gtree(items) builds a GKPlusTree from a list of (LeafItem, rank) pairs.
def create_gtree(items, K=16, l_factor=1.0) -> GKPlusTreeBase:
    """
    Mimics the Rust create_gtree: build a tree by inserting each (item, rank) pair.
    Uses the factory pattern to create a tree with the specified capacity K.
    """
    tree = create_gkplus_tree(K, l_factor=l_factor)
    tree_insert = tree.insert
    for (item, rank) in items:
        tree_insert(item, rank)
    return tree

# Create a random GPlusTree with n items and target node size (K) determining the rank distribution.
def random_gkplus_tree_of_size(n: int, target_node_size: int, l_factor: float) -> GKPlusTreeBase:
    # cache globals
    # calc_rank = calculate_item_rank
    # group_size = calculate_group_size(target_node_size)
    make_item_data = ItemData
    make_item = LeafItem
    p = 1.0 - (1.0 / (target_node_size))    # probability for geometric dist
    # logging.info(f"p = {p:.4f} for K = {target_node_size}")

    # we need at least n unique values; 2^24 = 16 777 216 > 1 000 000
    space = 1 << 24
    if space <= n:
        raise ValueError(f"Key-space too small! Required: {n + 1}, Available: {space}")

    indices = random.sample(range(1, space), k=n)

    # Pre-allocate items list
    items = [(None, None)] * n

    ranks = np.random.geometric(p, size=n)

    # all ranks are 1
    # ranks = np.ones(n, dtype=int)

    # logging.info(f"Ranks: {ranks}")

    # Process all items in a single pass
    for i, idx in enumerate(indices):
        # Use the index directly as the key
        key = idx
        val = "val"
        items[i] = (make_item(make_item_data(key, val)), int(ranks[i]))

    return create_gtree(items, K=target_node_size, l_factor=l_factor)

# The function random_klist_tree just wraps random_gtree_of_size with a given K.
def random_klist_tree(n: int, K: int, l_factor: float) -> GKPlusTreeBase:
    return random_gkplus_tree_of_size(n, K, l_factor=l_factor)

def repeated_experiment(
        size: int,
        repetitions: int,
        K: int,
        l_factor: float,
        adversarial_dim: int
    ) -> None:
    """
    Repeatedly builds random GPlusTrees (with size items) using ranks drawn from a geometric distribution.
    Uses K as target node size to compute the geometric parameter. Aggregates statistics and timings over many trees.
    """
    t_all_0 = time.perf_counter()

    # Storage for stats and timings
    results = []  # List of tuples: (stats, phy_height, tree_size)
    times_build = []
    times_stats = []
    times_phy = []
    times_size = []

    if adversarial_dim:
        # Generate keys for adversarial dimension
        keys = load_adversarial_keys_from_file(
            key_count=size,
            capacity=K,
            dim_limit=adversarial_dim
        )
        items = [LeafItem(ItemData(key, "val")) for key in keys]
        ranks = [1] * len(keys)

    # Generate results from repeated experiments.
    for _ in range(repetitions):
        # Time tree construction
        if adversarial_dim:
            t0 = time.perf_counter()
            tree = create_gkplus_tree(K=K, l_factor=l_factor)
            random.shuffle(items)
            for item, rank in zip(items, ranks):
                tree.insert(item, rank)
            times_build.append(time.perf_counter() - t0)
        else:    
            t0 = time.perf_counter()
            tree = random_klist_tree(size, K, l_factor=l_factor)
            times_build.append(time.perf_counter() - t0)

        # Time stats computation
        t0 = time.perf_counter()
        stats = gtree_stats_(tree, {})
        times_stats.append(time.perf_counter() - t0)

        # Time physical height computation
        t0 = time.perf_counter()
        phy_height = tree.physical_height()
        times_phy.append(time.perf_counter() - t0)

        # Time real item count computation
        t0 = time.perf_counter()
        tree_size = tree.real_item_count()
        times_size.append(time.perf_counter() - t0)

        results.append((stats, phy_height, tree_size))

        assert_tree_invariants_raise(tree, stats)
        # print("Tree stats:")
        # pprint(asdict(stats))

    # Perfect height: ceil( log_{K+1}(size) )
    perfect_height = math.ceil(math.log(size, K)) if size > 0 else 0

    # Aggregate averages for stats
    avg_gnode_height    = mean(s.gnode_height for s, _, _ in results)
    avg_gnode_count     = mean(s.gnode_count for s, _, _ in results)
    avg_leaf_count      = mean(s.leaf_count for s, _, _ in results)
    # avg_real_item_count = mean(s.real_item_count for s, _, _ in results)
    avg_tree_size       = mean(sz for _, _, sz in results)
    avg_item_count      = mean(s.item_count for s, _, _ in results)
    avg_item_slot_count = mean(s.item_slot_count for s, _, _ in results)
    avg_space_amp       = mean((s.item_slot_count / s.item_count) for s, _, _ in results)
    avg_physical_height = mean(h for _, h, _ in results)
    avg_height_amp      = mean((h / perfect_height) for _, h, _ in results) if perfect_height else 0
    avg_avg_gnode_size  = mean((s.item_count / s.gnode_count) for s, _, _ in results)
    avg_max_rank        = mean(s.rank for s, _, _ in results)

    # Aggregate averages for timings
    avg_build_time      = mean(times_build)
    avg_stats_time      = mean(times_stats)
    avg_phy_time        = mean(times_phy)
    avg_size_time       = mean(times_size)

    # Compute variances for stats
    var_gnode_height    = mean((s.gnode_height - avg_gnode_height)**2 for s, _, _ in results)
    var_gnode_count     = mean((s.gnode_count - avg_gnode_count)**2 for s, _, _ in results)
    var_leaf_count      = mean((s.leaf_count - avg_leaf_count)**2 for s, _, _ in results)
    # var_real_item_count = mean((s.real_item_count - avg_real_item_count)**2 for s, _, _ in results)
    var_item_count      = mean((s.item_count - avg_item_count)**2 for s, _, _ in results)
    var_tree_size       = mean((sz - avg_tree_size)**2 for _, _, sz in results)
    var_item_slot_count = mean((s.item_slot_count - avg_item_slot_count)**2 for s, _, _ in results)
    var_space_amp       = mean(((s.item_slot_count / s.item_count) - avg_space_amp)**2 for s, _, _ in results)
    var_physical_height = mean((h - avg_physical_height)**2 for _, h, _ in results)
    var_height_amp      = mean(((h / perfect_height) - avg_height_amp)**2 for _, h, _ in results) if perfect_height else 0
    var_avg_gnode_size  = mean(((s.item_count / s.gnode_count) - avg_avg_gnode_size)**2 for s, _, _ in results)
    var_max_rank        = mean((s.rank - avg_max_rank)**2 for s, _, _ in results)

    # Compute variances for timings
    var_build_time      = mean((t - avg_build_time)**2 for t in times_build)
    var_stats_time      = mean((t - avg_stats_time)**2 for t in times_stats)
    var_phy_time        = mean((t - avg_phy_time)**2 for t in times_phy)
    var_size_time       = mean((t - avg_size_time)**2 for t in times_size)

    # Prepare rows for stats and timings
    rows = [
        ("Real item count",       avg_tree_size,        var_tree_size),
        ("Item count",            avg_item_count,       var_item_count),
        ("Item slot count",       avg_item_slot_count,  var_item_slot_count),
        ("Space amplification",    avg_space_amp,        var_space_amp),
        ("G-node count",          avg_gnode_count,      var_gnode_count),
        ("Leaf count",            avg_leaf_count,       var_leaf_count),
        ("Avg G-node size",       avg_avg_gnode_size,   var_avg_gnode_size),
        ("Maximum rank",          avg_max_rank,         var_max_rank),
        ("G-node height",         avg_gnode_height,     var_gnode_height),
        ("Actual height",         avg_physical_height,  var_physical_height),
        ("Perfect height",        perfect_height,       None),
        ("Height amplification",  avg_height_amp,       var_height_amp),
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
    sum_phy   = sum(times_phy)
    sum_size  = sum(times_size)

    total_sum = sum_build + sum_stats + sum_phy + sum_size

    pct_build = (sum_build / total_sum * 100) if total_sum else 0
    pct_stats = (sum_stats / total_sum * 100) if total_sum else 0
    pct_phy   = (sum_phy   / total_sum * 100) if total_sum else 0
    pct_size  = (sum_size  / total_sum * 100) if total_sum else 0

    perf_rows = [
        ("Build time (s)", avg_build_time, var_build_time, sum_build, pct_build),
        ("Stats time (s)", avg_stats_time, var_stats_time, sum_stats, pct_stats),
        ("Phy height time (s)", avg_phy_time, var_phy_time, sum_phy, pct_phy),
        ("Size time (s)", avg_size_time, var_size_time, sum_size, pct_size),
    ]
    
    # 2) Log a separate performance table
    header = f"{'Metric':<20}{'Avg(s)':>13}{'Var(s)':>13}{'Total(s)':>13}{'%Total':>10}"
    sep    = "-" * len(header)

    logger.info("")  # blank line for separation
    logger.info("Performance summary:")
    logger.info(header)
    logger.info(sep)
    for name, avg, var, total, pct in perf_rows:
        logger.info(
            f"{name:<20}"
            f"{avg:13.6f}"
            f"{var:13.6f}"
            f"{total:13.6f}"
            f"{pct:10.2f}%"
        )

    logger.info(sep)
    t_all_1 = time.perf_counter() - t_all_0
    logger.info("Execution time: %.3f seconds", t_all_1)

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run statistics experiments for GK-Plus Trees.")
    parser.add_argument('--sizes', type=int, nargs='+', default=[10, 100, 1000, 10_000, 100_000],
                        help='List of tree sizes to test.')
    parser.add_argument('--ks', type=int, nargs='+', default=[2, 4, 16, 64],
                        help='List of K values (target node size) to test.')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions for each experiment.')
    parser.add_argument('--l_factor', type=float, default=1.0,
                        help='l_factor for the tree.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility.')
    parser.add_argument('--adversarial_dim', nargs='?', const=1, type=int, default=0,
                        help=(
                            "Only use keys with rank 1 in all dimensions up to dim. "
                            "Omitting the flag gives 0, "
                            "using --adversarial_dim       gives 1, "
                            "using --adversarial_dim N gives N."
                        ))
    parser.add_argument('--log-level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help='Set the logging level (default: INFO)')
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        
    log_dir = os.path.join(os.getcwd(), "stats/logs/gk_plus_tree_logs")
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
            logging.StreamHandler()         # comment this out if you don't want console output
        ],
        force=True  # Override any existing logging configuration
    )

    # Also apply the chosen level to the library logger so that
    # log records from gplus_trees.* are emitted at the requested level.
    logging.getLogger("gplus_trees").setLevel(log_level)

    # List of tree sizes to test.
    sizes = args.sizes
    # List of K values for which we want to run experiments.
    Ks = args.ks
    l_factor = args.l_factor
    repetitions = args.repetitions
    adversarial_dim = args.adversarial_dim

    for n in sizes:
        for K in Ks:
            logger.info("")
            logger.info("")
            logger.info(f"---------------- NOW RUNNING EXPERIMENT: n = {n}, K = {K}, l_factor: {l_factor}, repetitions = {repetitions}, adversarial_dim = {adversarial_dim} ----------------")
            t0 = time.perf_counter()
            # If you want to use adversarial or dim, you need to modify repeated_experiment and tree creation accordingly.
            repeated_experiment(size=n, repetitions=repetitions, K=K, l_factor=l_factor, adversarial_dim=adversarial_dim)
            elapsed = time.perf_counter() - t0
            logger.info(f"Total experiment time: {elapsed:.3f} seconds")