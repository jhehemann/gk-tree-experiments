# $\text{G}^k$-trees

History-independent ordered-set trees with high-probability balancing and fast sequential access, inspired by [G-trees](https://g-trees.github.io/g_trees/). Core operations target $O(\log^2 n)$ even against adaptive adversaries; full implementation is in progress—see repository issues for remaining work.

## Overview
- **$\text{G}^+$-tree**: single-dimension, history-independent tree (analogous to a $\text{B}^+$-tree, but for the G-tree family). Item ranks are derived from key hashes, so identical key sets yield identical layouts.
- **$\text{G}^{k+}$-tree**: multi-dimension extension of $\text{G}^+$-trees. Nodes have a size limit; when a node exceeds the limit, a new $\text{G}^{k+}$-tree is instantiated inside that node (a deeper dimension) and the overflowing items are inserted into it using ranks from hashing each item's previous hash.
- **$k$-list**: cache-friendly set data structure used by both tree types as the recursion anchor for the highest dimension (deepest level) within a node.

## Features
- History-independent layout; identical key sets yield identical trees.
- Target $O(\log^2 n)$ inserts and lookups even against adaptive adversaries (remaining engineering tracked in issues), with a leaf-level next-pointer for fast in-order scans.
- Pluggable capacity `K` via factories to tune branching and cache behavior.
- Multi-dimensional ranks ($\text{G}^{k+}$) to model higher-dimensional hashes by recursively hashing prior hashes when nodes overflow into a deeper tree.

## Installation
Requires Python 3.10–3.12.

Poetry (recommended):
```bash
poetry install && poetry shell
```

Pip (editable for local development):
```bash
pip install -e .
```

## Experiments
Statistics ($\text{G}^{k+}$):
```bash
python stats/stats_gk_plus_tree.py --sizes 1000 10000 --ks 4 16 --repetitions 3 --l_factor 1.0 --adversarial_dim 0 --seed 42
```
  - `--sizes`: tree sizes to test.
  - `--ks`: $k$-list node capacities.
  - `--repetitions`: repeats per parameter combo.
  - `--l_factor`: G-node size threshold (times $k$).
  - `--adversarial_dim`: use adversarial rank-1 keys up to that dimension (omit to disable).
  - `--seed`: make runs reproducible.

Statistics ($\text{G}^+$):
```bash
python stats/stats_gplus_tree.py --sizes 1000 10000 --ks 4 16 --repetitions 3 --seed 42
```
  - `--sizes`: tree sizes to test.
  - `--ks`: $k$-list node capacities.
  - `--repetitions`: repeats per parameter combo.
  - `--seed`: make runs reproducible.

Adversarial keys:
```bash
python scripts/find_adversarial_keys.py
python scripts/analyze_adversarial_keys_scenarios.py
python scripts/analyze_single_insertion.py
```
  - `find_adversarial_keys.py` writes rank-1 keys into benchmarks/adversarial_keys_new/ (counts/K/dim sets are defined inside the script).
  - `analyze_adversarial_keys_scenarios.py` profiles patterns; adjust `K`, `DIMENSIONS`, `SIZE` near the bottom of the script.
  - `analyze_single_insertion.py` inspects single insertion case for adversarial keys.

Benchmarks:
- Isolated performance runs via the `./benchmark` wrapper; see [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md) for workflow and scenarios.

## Tests
```bash
python -m unittest discover -s tests -v
python tests/test_runner.py --log-level INFO --verbosity 2
```
- `test_runner.py` flags:
  - `-f/--files`: target specific test files (supports `file.py::Class::method`).
  - `-c/--classes`: supply class lists per file (`file.py:ClassA,ClassB`).
  - `--verbosity`: 0–3 verbosity levels.
  - `--log-level`: logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL).

## Project layout
- [src/gplus_trees/](src/gplus_trees/): core implementations and factories for G+ and GK+ trees.
- [benchmarks/](benchmarks/): isolated ASV-based performance measurements (see [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)).
- [stats/](stats/): scripts for measuring structure statistics across parameters.
- [scripts/](scripts/): adversarial-key discovery and analysis tools.
- [tests/](tests/): unittest suites and helpers.
