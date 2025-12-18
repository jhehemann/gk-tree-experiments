#!/usr/bin/env python3
"""
Main entry point for G+ tree benchmarks.

This script runs performance benchmarks with proper phase separation:
- Setup (not timed): Data generation and configuration
- Warmup (not timed): Cache warming
- Run (timed): Actual measurement
- Verify (not timed): Correctness checks
- Teardown (not timed): Cleanup

Usage:
    # Run with default settings
    python -m benchmarks.run_benchmarks
    
    # Run with custom seed for reproducibility
    BENCHMARK_SEED=123 python -m benchmarks.run_benchmarks
    
    # Run in verify-only mode (no timing, only correctness)
    BENCHMARK_VERIFY_ONLY=true python -m benchmarks.run_benchmarks
    
    # Run with custom log level
    BENCHMARK_LOG_LEVEL=DEBUG python -m benchmarks.run_benchmarks
    
    # Combine options
    BENCHMARK_SEED=42 BENCHMARK_LOG_LEVEL=INFO python -m benchmarks.run_benchmarks
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from .config import BenchmarkConfig
from .runner import BenchmarkRunner


def setup_logging(config: BenchmarkConfig, log_dir: str = None) -> None:
    """
    Configure logging for benchmark output.
    
    Args:
        config: Benchmark configuration
        log_dir: Optional directory for log files
    """
    handlers = [logging.StreamHandler()]
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"benchmark_{ts}.log")
        handlers.append(logging.FileHandler(log_path, mode="w"))
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run G+ tree benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: from env or 42)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Tree sizes to benchmark (default: 10 100 1000)",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        help="Target node sizes (K values) to test (default: 2 16)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Number of repetitions per configuration (default: 200)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run in verify-only mode (no timing)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup phase",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory for log files (default: benchmarks/logs)",
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main benchmark execution.
    
    Returns:
        Exit code (0 for success)
    """
    # Parse arguments and create config
    args = parse_args()
    
    # Start with config from environment
    config = BenchmarkConfig.from_env()
    
    # Override with command-line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.sizes is not None:
        config.sizes = args.sizes
    if args.ks is not None:
        config.ks = args.ks
    if args.repetitions is not None:
        config.repetitions = args.repetitions
    if args.verify_only:
        config.verify_only = True
    if args.skip_warmup:
        config.skip_warmup = True
    if args.log_level is not None:
        config.log_level = args.log_level
    
    # Setup logging
    log_dir = args.log_dir or os.path.join(os.path.dirname(__file__), "logs")
    setup_logging(config, log_dir if not config.verify_only else None)
    
    # Log banner
    logging.info("=" * 70)
    logging.info("G+ TREE BENCHMARKS")
    logging.info("=" * 70)
    
    if config.verify_only:
        logging.info("Mode: VERIFY-ONLY (correctness checks, no timing)")
    else:
        logging.info("Mode: PERFORMANCE (timed measurements)")
    
    logging.info("")
    
    # Create runner
    runner = BenchmarkRunner(config)
    
    # Run benchmarks
    overall_start = time.perf_counter()
    
    for size in config.sizes:
        for k in config.ks:
            logging.info("")
            logging.info("=" * 70)
            logging.info(f"BENCHMARK: n={size}, K={k}, repetitions={config.repetitions}")
            logging.info("=" * 70)
            
            run_start = time.perf_counter()
            
            results, metadata = runner.run_benchmark(
                size=size,
                k=k,
                repetitions=config.repetitions
            )
            
            run_elapsed = time.perf_counter() - run_start
            
            if not config.verify_only:
                runner.aggregate_and_report(results, metadata)
            
            logging.info("")
            logging.info(f"Execution time: {run_elapsed:.3f} seconds")
    
    overall_elapsed = time.perf_counter() - overall_start
    
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"TOTAL EXECUTION TIME: {overall_elapsed:.3f} seconds")
    logging.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
