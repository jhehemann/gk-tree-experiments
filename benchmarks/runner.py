"""Core benchmark runner for G+ tree performance measurements."""

import logging
import math
import time
from dataclasses import dataclass
from statistics import mean
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gplus_trees.gplus_tree import GPlusTree, gtree_stats_, Stats

from .config import BenchmarkConfig, BenchmarkMetadata, get_git_commit_hash
from .utils import random_gtree_of_size
from .verify import verify_invariants


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    stats: Stats
    physical_height: int
    build_time: float
    stats_time: float
    phy_height_time: float


class BenchmarkRunner:
    """
    Manages the benchmark lifecycle with proper phase separation.
    
    Phases:
    1. Setup (not timed): Configuration and data generation
    2. Warmup (not timed): Optional warmup iterations
    3. Run (timed): Actual measurement
    4. Verify (not timed): Correctness checks
    5. Teardown (not timed): Cleanup
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._check_logging_level()
    
    def _check_logging_level(self) -> None:
        """Warn if verbose logging is enabled during measurement."""
        current_level = logging.getLogger().getEffectiveLevel()
        if current_level < logging.INFO:
            level_name = logging.getLevelName(current_level)
            logging.warning(
                "⚠️  Verbose logging (%s) is enabled. This may affect benchmark timing! "
                "Set log level to INFO or higher for accurate measurements.",
                level_name
            )
    
    def setup(self, size: int, k: int, repetitions: int) -> List[GPlusTree]:
        """
        Setup phase: Generate trees for benchmarking.
        
        NOT TIMED.
        
        Args:
            size: Number of items per tree
            k: Target node size
            repetitions: Number of trees to create
            
        Returns:
            List of pre-built trees ready for measurement
        """
        trees = []
        base_seed = self.config.seed
        
        for i in range(repetitions):
            # Use different seed for each repetition to get different trees
            # but deterministic based on base seed
            tree_seed = base_seed + i
            tree = random_gtree_of_size(size, k, tree_seed)
            trees.append(tree)
        
        return trees
    
    def warmup(self, trees: List[GPlusTree]) -> None:
        """
        Warmup phase: Run operations to warm caches.
        
        NOT TIMED.
        
        Args:
            trees: Trees to use for warmup
        """
        if self.config.skip_warmup:
            return
        
        # Perform a few operations to warm up caches
        warmup_count = min(5, len(trees))
        for i in range(warmup_count):
            tree = trees[i % len(trees)]
            _ = gtree_stats_(tree, {})
            _ = tree.physical_height()
    
    def run_single(self, tree: GPlusTree) -> BenchmarkResult:
        """
        Run measurement on a single tree.
        
        TIMED - only the actual operations are measured.
        
        Args:
            tree: Tree to measure (pre-built in setup)
            
        Returns:
            BenchmarkResult with timing and stats
        """
        # Time stats computation
        t0 = time.perf_counter()
        stats = gtree_stats_(tree, {})
        stats_time = time.perf_counter() - t0
        
        # Time physical height computation
        t0 = time.perf_counter()
        phy_height = tree.physical_height()
        phy_height_time = time.perf_counter() - t0
        
        # Build time is 0 here since tree was pre-built in setup
        build_time = 0.0
        
        return BenchmarkResult(
            stats=stats,
            physical_height=phy_height,
            build_time=build_time,
            stats_time=stats_time,
            phy_height_time=phy_height_time,
        )
    
    def verify(self, tree: GPlusTree, stats: Stats) -> bool:
        """
        Verify phase: Check correctness.
        
        NOT TIMED.
        
        Args:
            tree: Tree to verify
            stats: Stats computed during measurement
            
        Returns:
            True if all verifications pass
        """
        return verify_invariants(tree, stats)
    
    def run_benchmark(self, size: int, k: int, repetitions: int) -> Tuple[List[BenchmarkResult], BenchmarkMetadata]:
        """
        Run complete benchmark with proper phase separation.
        
        Args:
            size: Number of items per tree
            k: Target node size
            repetitions: Number of repetitions
            
        Returns:
            (results, metadata)
        """
        # Create metadata
        metadata = BenchmarkMetadata(
            commit_hash=get_git_commit_hash(),
            config=self.config,
            size=size,
            k=k,
            repetitions=repetitions,
        )
        
        # === SETUP PHASE (not timed) ===
        logging.debug("Setup: Generating %d trees...", repetitions)
        trees = self.setup(size, k, repetitions)
        
        # === WARMUP PHASE (not timed) ===
        if not self.config.skip_warmup:
            logging.debug("Warmup: Running warmup iterations...")
            self.warmup(trees)
        
        # === MEASUREMENT PHASE (timed) ===
        results = []
        all_verified = True
        
        for tree in trees:
            # Run timed measurement
            result = self.run_single(tree)
            results.append(result)
            
            # === VERIFY PHASE (not timed) ===
            if self.config.verify_only or logging.getLogger().isEnabledFor(logging.DEBUG):
                if not self.verify(tree, result.stats):
                    all_verified = False
        
        if self.config.verify_only:
            if all_verified:
                logging.info("✓ All verifications passed for n=%d, K=%d", size, k)
            else:
                logging.error("✗ Some verifications failed for n=%d, K=%d", size, k)
        
        # === TEARDOWN PHASE (not timed) ===
        # No explicit cleanup needed for Python
        
        return results, metadata
    
    def aggregate_and_report(
        self,
        results: List[BenchmarkResult],
        metadata: BenchmarkMetadata,
    ) -> None:
        """
        Aggregate results and report statistics.
        
        NOT TIMED.
        
        Args:
            results: List of benchmark results
            metadata: Benchmark metadata
        """
        size = metadata.size
        k = metadata.k
        
        # Perfect height: ceil(log_{K+1}(size))
        perfect_height = math.ceil(math.log(size, k + 1)) if size > 0 else 0
        
        # Aggregate statistics
        avg_gnode_height = mean(r.stats.gnode_height for r in results)
        avg_gnode_count = mean(r.stats.gnode_count for r in results)
        avg_item_count = mean(r.stats.item_count for r in results)
        avg_item_slot_count = mean(r.stats.item_slot_count for r in results)
        avg_space_amp = mean((r.stats.item_slot_count / r.stats.item_count) for r in results)
        avg_physical_height = mean(r.physical_height for r in results)
        avg_height_amp = mean((r.physical_height / perfect_height) for r in results) if perfect_height else 0
        avg_avg_gnode_size = mean((r.stats.item_count / r.stats.gnode_count) for r in results)
        avg_max_rank = mean(r.stats.rank for r in results)
        
        # Aggregate timing
        avg_build_time = mean(r.build_time for r in results)
        avg_stats_time = mean(r.stats_time for r in results)
        avg_phy_time = mean(r.phy_height_time for r in results)
        
        # Compute variances
        var_gnode_height = mean((r.stats.gnode_height - avg_gnode_height)**2 for r in results)
        var_gnode_count = mean((r.stats.gnode_count - avg_gnode_count)**2 for r in results)
        var_item_count = mean((r.stats.item_count - avg_item_count)**2 for r in results)
        var_item_slot_count = mean((r.stats.item_slot_count - avg_item_slot_count)**2 for r in results)
        var_space_amp = mean(((r.stats.item_slot_count / r.stats.item_count) - avg_space_amp)**2 for r in results)
        var_physical_height = mean((r.physical_height - avg_physical_height)**2 for r in results)
        var_height_amp = mean(((r.physical_height / perfect_height) - avg_height_amp)**2 for r in results) if perfect_height else 0
        var_avg_gnode_size = mean(((r.stats.item_count / r.stats.gnode_count) - avg_avg_gnode_size)**2 for r in results)
        var_max_rank = mean((r.stats.rank - avg_max_rank)**2 for r in results)
        
        var_build_time = mean((r.build_time - avg_build_time)**2 for r in results)
        var_stats_time = mean((r.stats_time - avg_stats_time)**2 for r in results)
        var_phy_time = mean((r.phy_height_time - avg_phy_time)**2 for r in results)
        
        # === OUTPUT: Metadata ===
        logging.info("")
        logging.info("=== METADATA ===")
        for line in str(metadata).split('\n'):
            logging.info(line)
        
        # === OUTPUT: Statistics Table ===
        rows = [
            ("Item count", avg_item_count, var_item_count),
            ("Item slot count", avg_item_slot_count, var_item_slot_count),
            ("Space amplification", avg_space_amp, var_space_amp),
            ("G-node count", avg_gnode_count, var_gnode_count),
            ("Avg G-node size", avg_avg_gnode_size, var_avg_gnode_size),
            ("Maximum rank", avg_max_rank, var_max_rank),
            ("G-node height", avg_gnode_height, var_gnode_height),
            ("Actual height", avg_physical_height, var_physical_height),
            ("Perfect height", perfect_height, None),
            ("Height amplification", avg_height_amp, var_height_amp),
        ]
        
        header = f"{'Metric':<20} {'Avg':>15} {'(Var)':>15}"
        sep_line = "-" * len(header)
        
        logging.info("")
        logging.info("=== STATISTICS ===")
        logging.info(header)
        logging.info(sep_line)
        for name, avg, var in rows:
            if var is None:
                logging.info(f"{name:<20} {avg:>15}")
            else:
                var_str = f"({var:.2f})"
                avg_fmt = f"{avg:15.2f}"
                logging.info(f"{name:<20} {avg_fmt} {var_str:>15}")
        
        # === OUTPUT: Performance Table ===
        sum_build = sum(r.build_time for r in results)
        sum_stats = sum(r.stats_time for r in results)
        sum_phy = sum(r.phy_height_time for r in results)
        total_sum = sum_build + sum_stats + sum_phy
        
        pct_build = (sum_build / total_sum * 100) if total_sum else 0
        pct_stats = (sum_stats / total_sum * 100) if total_sum else 0
        pct_phy = (sum_phy / total_sum * 100) if total_sum else 0
        
        perf_rows = [
            ("Build time (s)", avg_build_time, var_build_time, sum_build, pct_build),
            ("Stats time (s)", avg_stats_time, var_stats_time, sum_stats, pct_stats),
            ("Phy height time (s)", avg_phy_time, var_phy_time, sum_phy, pct_phy),
        ]
        
        header = f"{'Metric':<20}{'Avg(s)':>13}{'Var(s)':>13}{'Total(s)':>13}{'%Total':>10}"
        sep = "-" * len(header)
        
        logging.info("")
        logging.info("=== PERFORMANCE ===")
        logging.info(header)
        logging.info(sep)
        for name, avg, var, total, pct in perf_rows:
            logging.info(
                f"{name:<20}"
                f"{avg:13.6f}"
                f"{var:13.6f}"
                f"{total:13.6f}"
                f"{pct:10.2f}%"
            )
        logging.info(sep)
