# G<sup>+</sup>-trees 

A history-independent tree data structure for $O(\log n)$ tree operations with high probability and fast sequential access as introduced in [G-trees](https://g-trees.github.io/g_trees/).

## Benchmarks

Performance benchmarks are available in the `benchmarks/` directory. See [benchmarks/README.md](benchmarks/README.md) for detailed usage instructions.

Quick start:
```bash
# Run benchmarks
PYTHONPATH=src python -m benchmarks

# Or use the convenience script
./benchmarks/run.sh

# Verify correctness only (no timing)
./benchmarks/run.sh --verify-only
```
