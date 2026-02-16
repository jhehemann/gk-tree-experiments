"""
Benchmarks package for GK+ trees and K-lists.

This package contains ASV benchmarks for performance testing of:
- KListBase operations (insert_entry, retrieve)
- GKPlusTreeBase operations (insert_entry, retrieve)
- Comparative performance analysis
- Memory usage benchmarks
- Scalability tests

The benchmarks are designed to be robust against CPU and memory load variations
by using multiple iterations, deterministic test data, and proper statistical analysis.
"""

# Import benchmark utilities for easier access
from .benchmark_utils import BaseBenchmark, BenchmarkUtils

__all__ = ["BaseBenchmark", "BenchmarkUtils"]
