# G+ Tree Benchmarks

This directory contains benchmarks for measuring the performance and correctness of G+ tree implementations.

## Design Principles

The benchmarks are designed with **correctness first**:

1. **Reproducibility**: Deterministic seeding ensures identical results across runs
2. **Fair measurement**: Proper phase separation (setup, warmup, run, verify, teardown)
3. **Minimal overhead**: Only the intended operations are timed
4. **Correctness validation**: Built-in verification mode
5. **Clear metadata**: Every run includes commit hash, seed, and configuration

## Benchmark Lifecycle

Each benchmark follows a strict lifecycle:

```
┌──────────────────────────────────────────────────┐
│ 1. SETUP (not timed)                            │
│    - Load configuration                          │
│    - Generate deterministic test data            │
│    - Build data structures                       │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│ 2. WARMUP (not timed, optional)                 │
│    - Run operations to warm caches               │
│    - Ensure stable baseline                      │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│ 3. RUN (TIMED)                                   │
│    - Measure only the target operations          │
│    - No logging or I/O in hot path               │
│    - Use monotonic high-resolution clock         │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│ 4. VERIFY (not timed, optional)                 │
│    - Check data structure invariants             │
│    - Validate correctness                        │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│ 5. TEARDOWN (not timed)                         │
│    - Clean up resources                          │
└──────────────────────────────────────────────────┘
```

## Quick Start

### Running Benchmarks

```bash
# Run with default settings
python -m benchmarks

# Run from repository root with PYTHONPATH
PYTHONPATH=src python -m benchmarks
```

### Verify-Only Mode

To check correctness without performance measurement:

```bash
# Verify-only mode (fast, no timing)
BENCHMARK_VERIFY_ONLY=true python -m benchmarks

# Or with CLI flag
python -m benchmarks --verify-only
```

## Configuration

### Environment Variables

The benchmarks can be configured via environment variables:

- `BENCHMARK_SEED`: Random seed for reproducibility (default: 42)
- `BENCHMARK_VERIFY_ONLY`: Run in verify-only mode (default: false)
- `BENCHMARK_SKIP_WARMUP`: Skip warmup phase (default: false)
- `BENCHMARK_LOG_LEVEL`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)

### Command-Line Arguments

```bash
python -m benchmarks --help

# Examples:
python -m benchmarks --seed 123
python -m benchmarks --sizes 100 1000 10000
python -m benchmarks --ks 2 4 8 16
python -m benchmarks --repetitions 500
python -m benchmarks --verify-only
python -m benchmarks --log-level DEBUG
python -m benchmarks --log-dir ./my_logs
```

## Reproducibility

The benchmarks are designed to be fully reproducible:

### Same Machine, Same Seed → Same Results

```bash
# Run 1
BENCHMARK_SEED=42 python -m benchmarks

# Run 2 (will produce identical results)
BENCHMARK_SEED=42 python -m benchmarks
```

### Deterministic Data Generation

- All random data is generated using a seeded RNG
- Each repetition uses `seed + iteration` for variation while remaining deterministic
- Key-space: 2^24 (16,777,216 unique values)

## Output Format

Each benchmark run produces:

### 1. Metadata Section
```
=== METADATA ===
Commit: 415132ee48a597847d36bf694aedccea343ffb1b
Seed: 42
Size (n): 1000
Target node size (K): 16
Repetitions: 200
```

### 2. Statistics Section
```
=== STATISTICS ===
Metric                           Avg           (Var)
----------------------------------------------------
Item count                    109.10          (7.77)
Item slot count               123.06         (62.56)
Space amplification             1.13          (0.00)
...
```

### 3. Performance Section
```
=== PERFORMANCE ===
Metric                     Avg(s)       Var(s)     Total(s)    %Total
---------------------------------------------------------------------
Build time (s)           0.000896     0.000000     0.179251     76.63%
Stats time (s)           0.000257     0.000000     0.051380     21.96%
Phy height time (s)      0.000016     0.000000     0.003300      1.41%
```

## Timing Guarantees

### What IS timed:
- ✓ Tree statistics computation (`gtree_stats_`)
- ✓ Physical height calculation (`physical_height`)

### What is NOT timed:
- ✗ Data generation
- ✗ Tree construction (setup phase)
- ✗ Warmup iterations
- ✗ Correctness verification
- ✗ Logging and I/O operations
- ✗ Result aggregation

### Clock Used
- `time.perf_counter()`: Monotonic, high-resolution performance counter
- Immune to system clock adjustments
- Suitable for measuring short durations

## Logging Best Practices

### Default: INFO Level

```bash
# Recommended for benchmarks
python -m benchmarks --log-level INFO
```

### WARNING: Verbose Logging Affects Timing

```bash
# This will trigger a warning
python -m benchmarks --log-level DEBUG

# Output:
# ⚠️  Verbose logging (DEBUG) is enabled. This may affect benchmark timing!
#    Set log level to INFO or higher for accurate measurements.
```

### Log Files

Logs are saved to `benchmarks/logs/` by default:

```
benchmarks/logs/
├── benchmark_20231218_164932.log
├── benchmark_20231218_165401.log
└── ...
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| Item count | Average number of items stored |
| Item slot count | Average number of item slots (including duplicates) |
| Space amplification | Ratio of slots to items (overhead) |
| G-node count | Average number of nodes in the tree |
| Avg G-node size | Average items per node |
| Maximum rank | Highest rank value in the tree |
| G-node height | Height in terms of G-nodes |
| Actual height | Physical height of the tree |
| Perfect height | Theoretical optimal height (ceil(log_{K+1}(n))) |
| Height amplification | Ratio of actual to perfect height |

## Files

```
benchmarks/
├── __init__.py           # Package marker
├── __main__.py           # Module entry point
├── config.py             # Configuration and metadata
├── runner.py             # Core benchmark runner
├── utils.py              # Data generation utilities
├── verify.py             # Correctness verification
├── run_benchmarks.py     # Main entry point
├── README.md             # This file
└── logs/                 # Benchmark logs (created on first run)
```

## Extending Benchmarks

To add new benchmarks:

1. Create a new measurement function in `runner.py` or a new file
2. Follow the phase separation pattern (setup → warmup → run → verify → teardown)
3. Use `time.perf_counter()` for timing
4. Ensure deterministic data generation with seeded RNGs
5. Add verification in `verify.py` if needed
6. Update this README with new metrics or options

## Troubleshooting

### "Key-space too small" Error

Reduce the number of items or increase key space in `utils.py`.

### Inconsistent Results Across Runs

- Check if the same seed is used
- Verify no external state is affecting the benchmark
- Ensure logging level is INFO or higher

### High Variance in Timing

- Run more repetitions (`--repetitions`)
- Skip warmup if caches are pre-warmed
- Check for background processes affecting the system

## References

- G-trees paper: https://g-trees.github.io/g_trees/
- Python `time` module: https://docs.python.org/3/library/time.html#time.perf_counter
