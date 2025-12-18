#!/bin/bash
# Convenience script to run benchmarks with PYTHONPATH set
set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"
PYTHONPATH="${REPO_ROOT}/src" python -m benchmarks "$@"
