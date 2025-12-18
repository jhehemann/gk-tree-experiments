# Isolated Benchmark System

This system implements benchmarking for the G+Trees project. All benchmarks run locally in complete isolation from the working directory.

## Features

- **Isolation**: Benchmarks run in separate directory (`../.isolated-benchmarks/`)
- **Local Development**: Workflow without external dependencies
- **Background Execution**: Does not block development work
- **Non-Interference**: Switch branches, commit, modify files during benchmark execution
- **Progress Tracking**: Monitor benchmark status without interruption

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
> **Note**: Set logging to INFO to avoid skewed benchmarks caused by expensive log output. 
`src/gplus_trees/logging_config.py`

## Directory Structure

```
project-root/                          # Working directory (untouched)
├── benchmarks/
│   ├── isolated_benchmark_runner.py   # Main benchmark script
│   └── ISOLATED_BENCHMARKS.md         # This documentation
└── benchmark                          # Convenience wrapper

../.isolated-benchmarks/                # Isolated benchmark environment
├── gplus-trees/                       # Separate clone for benchmarking
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
