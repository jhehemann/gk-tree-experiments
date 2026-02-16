# Isolated Benchmark System

This system implements benchmarking for the G+Trees project. All benchmarks run locally in complete isolation from the working directory.

## Features

- **Isolation**: Benchmarks run in separate directory (`../.isolated-benchmarks/`)
- **Local Development**: Workflow without external dependencies
- **Background Execution**: Does not block development work
- **Non-Interference**: Switch branches, commit, modify files during benchmark execution
- **Progress Tracking**: Monitor benchmark status without interruption
- **Reproducibility**: Deterministic by default with configurable seeds
- **Correctness**: Proper timing isolation and state management

## Quick Start

```bash
# 1. Initial setup (one time only)
./benchmark setup

# 2. Run benchmarks manually
./benchmark run HEAD                                # Benchmark current commit
./benchmark run 'HEAD^!'                            # Benchmark latest commit (best practice)
./benchmark run HEAD GKPlusTreeRetrieveBenchmarks   # Benchmark specific test
./benchmark run --quick 'HEAD^!'                    # Run a quick benchmark on the latest commit (useful for a sanity check)


# 3. Monitor progress
./benchmark status

# 4. Stop running benchmarks (if needed)
./benchmark stop

# 5. View results
./benchmark view
```

## Reproducibility & Correctness

### Deterministic Benchmarking

All benchmarks use deterministic random number generation to ensure reproducible results across machines and Python invocations:

- **Default Seed**: Benchmarks use seed `42` by default
- **Configurable**: Override via `BENCHMARK_SEED` environment variable
- **Consistent**: Same seed + same parameters = identical test data across runs
- **Process-Independent**: Uses stable hashing (SHA-256) for seed offsets, not Python's randomized `hash()`

```bash
# Run with custom seed for reproducibility testing
BENCHMARK_SEED=12345 ./benchmark run 'HEAD^!'

# Re-run with same seed to verify identical results
BENCHMARK_SEED=12345 ./benchmark run 'HEAD^!'
```

### Timing Methodology

Benchmarks use ASV's built-in timing mechanisms which:
- Exclude setup/teardown from measurements
- Disable garbage collection during timed sections
- Use high-resolution monotonic clocks
- Perform automatic warmup and statistical stabilization
- Re-enable garbage collection after measurements complete

### State Isolation

Benchmarks maintain proper state isolation:
- **Setup phase**: Not timed, prepares data structures
- **Measurement phase**: Timed, executes operation repeatedly
- **Teardown phase**: Not timed, cleanup and GC re-enablement
- **Caching**: Some benchmarks cache expensive setup (documented per-class)

### Verification Mode

To verify correctness without timing overhead:

```bash
# Run ASV in quick mode for faster verification
./benchmark run --quick 'HEAD^!'

# Or use ASV directly for more control
cd ../.isolated-benchmarks/gk-tree-experiments
poetry run asv run --quick --show-stderr HEAD^!
```

## Directory Structure

```
project-root/                          # Working directory (untouched)
├── benchmarks/
│   ├── isolated_benchmark_runner.py   # Main benchmark script
│   └── ISOLATED_BENCHMARKS.md         # This documentation
└── benchmark                          # Convenience wrapper

../.isolated-benchmarks/               # Isolated benchmark environment
├── gk-tree-experiments/               # Separate clone for benchmarking
├── results/                           # ASV benchmark results
├── html/                              # Generated HTML reports
├── logs/                              # Benchmark execution logs
└── status.json                        # Current benchmark status
```

## Manual Benchmarking

```bash
# Run all benchmarks for current commit
./benchmark run HEAD

# Run benchmarks for latest commit on current branch (best practice)
./benchmark run 'HEAD^!'

# Run specific benchmark
./benchmark run HEAD GKPlusTreeInsert

# Run benchmarks for specific branch
./benchmark run performance-refactor

# Run benchmarks for latest commit on specific branch (best practice)
./benchmark run 'performance-refactor^!'

# Run benchmarks for specific commit hash
./benchmark run abc1234
```

## Viewing Results

```bash
# Show current status
./benchmark status

# Stop running benchmarks
./benchmark stop

# Open HTML results in browser (Ctrl+C to stop the server)
./benchmark view

# View recent logs
./benchmark logs

# View specific log
./benchmark logs benchmark_performance-refactor_b387b538_20250704_120000.log
```

## Cleanup

```bash
# Remove all benchmark data
./benchmark clean
```


## Status Information

The status command shows:
- Current benchmark state (idle/running/completed/failed)
- Which commit is being benchmarked
- Progress messages
- Recent log files

## Logs

All benchmark executions are logged:
- Individual log file per benchmark run
- Includes full ASV output
- Timestamp and commit information
- Error details if benchmarks fail

## Troubleshooting

### Reproducibility Testing

To verify benchmarks are reproducible:

```bash
# Run benchmark twice with same seed
BENCHMARK_SEED=999 ./benchmark run 'HEAD^!'
# Wait for completion...
BENCHMARK_SEED=999 ./benchmark run 'HEAD^!'
# Compare results - should be identical (within statistical variance)
```

### Checking Benchmark Metadata

ASV automatically includes metadata in results:

```bash
# Results include:
# - Commit hash
# - Benchmark parameters (capacity, size, distribution, etc.)
# - Python version
# - System information
# View in: ../.isolated-benchmarks/results/
```

### Cache Behavior

Some benchmarks use class-level caching for expensive setup:
- `GKPlusTreeRetrieveBenchmarks`: Caches trees by (capacity, size, l_factor)
- `KListRetrieveBenchmarks`: Caches KLists by (capacity, size)
- `GKPlusTreeAdversarialRetrieveBenchmarks`: Caches trees by (capacity, size, dim_limit, l_factor)

Caches persist within a single ASV process but are cleared between different ASV runs.

### Check Isolation
```bash
# Verify separate directory exists
ls -la ../.isolated-benchmarks/
```

### View Detailed Logs
```bash
# List all log files
./benchmark logs

# View specific log
./benchmark logs benchmark_performance-refactor_b387b538_20250704_120000.log
```

### Clean Start
```bash
# Complete reset
./benchmark clean
./benchmark setup
```

## Best Practices


1. **Use deterministic seeds** for reproducible comparisons
2. **Run with `--quick` first** to catch errors before long benchmark runs
3. **Document your seed** when sharing results
4. **Verify reproducibility** by running with same seed multiple times
5. **Monitor for warnings** about logging levels during setup
