# G+-trees

History-independent ordered-set trees with high-probability balancing and fast sequential access, inspired by [G-trees](https://g-trees.github.io/g_trees/). Core operations run in $O(\log n)$ with high probability.

## Overview
- **G+-tree**: single-dimension tree for ordered keys with deterministic structure given the key set.
- **GK+-tree**: multi-dimension variant that derives ranks from hashed keys and can switch between compact `KList` leaves and full trees based on load.
- **KList**: cache-friendly leaf representation used by both tree types.

## Features
- History-independent layout; identical key sets yield identical trees.
- $O(\log n)$ inserts and lookups with a leaf-level next-pointer for fast in-order scans.
- Pluggable capacity `K` via factories to tune branching and cache behavior.
- Optional multi-dimensional ranks (GK+) to model higher-dimensional hashes.

## Installation
Requires Python 3.10–3.12.

Poetry (recommended):
```bash
poetry install
```

Pip (editable for local development):
```bash
pip install -e .
```

## Quickstart
### G+-tree (single dimension)
```python
from gplus_trees.factory import create_gplustree
from gplus_trees.base import ItemData, LeafItem

tree = create_gplustree(K=4)

# Insert a key/value pair at rank 1 (leaf level)
item = LeafItem(ItemData(42, "meaning"))
tree, inserted, _ = tree.insert(item, rank=1)

# Look up the key; `next_entry` is the in-order successor
found_entry, next_entry = tree.retrieve(42)
print(found_entry.item.value)  # "meaning"
```

### GK+-tree (multi-dimension, rank from hash)
```python
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.base import ItemData, LeafItem
from gplus_trees.utils import get_digest, calc_rank_from_digest_k

tree = create_gkplus_tree(K=4, dimension=1, l_factor=1.0)

key = 99
digest = get_digest(key, dim=tree.DIM)
rank = calc_rank_from_digest_k(digest, k=4)

item = LeafItem(ItemData(key, "payload"))
tree, inserted, _ = tree.insert(item, rank=rank)
found_entry, _ = tree.retrieve(key)
print(found_entry.item.value)  # "payload"
```

### Working with ranks
- Ranks encode levels in the tree; larger ranks insert higher in the structure.
- For GK+, ranks are typically derived from hashes via `calc_rank_from_digest_k`.
- Leaf-level inserts always use `rank=1`.

## Benchmarks and statistics
- Statistics scripts
	```bash
	python stats/stats_gk_plus_tree.py --sizes 1000 --ks 4 --repetitions 10 --l_factor 1.0 --seed 42
	python stats/stats_gplus_tree.py   --sizes 1000 --ks 4 --repetitions 10 --seed 42
	```
- Benchmarks live in `benchmarks/` and can be run via the provided runner; see `benchmarks/BENCHMARKS.md` for scenarios.

## Tests
Run the full suite:
```bash
python -m pytest
```

## Project layout
- `src/gplus_trees/`: core implementations and factories for G+ and GK+ trees.
- `benchmarks/`: reproducible performance measurements.
- `stats/`: scripts for measuring structure statistics.
- `tests/`: unittest suites and helpers.

## License
MIT