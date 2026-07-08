# Provenance — adversarial key fixtures

23 frozen `.pkl` fixtures with adversarial rank-1 keys, deliberately tracked so benchmark
runs are reproducible. Each file is a pickled `list[int]` of keys whose domain-separated
per-dimension digests `H(⟨j⟩ ‖ key)` all have rank 1 across the file's dimension count
(see `find_rank_keys()` in `scripts/find_adversarial_keys.py:16`).

Regenerated 2026-07-07 under the domain-separation rank construction (ADR-0004 / paper
ADR-002); the previous set was produced under iterated hashing `H^j(key)`. The dimension
matrix and naming are unchanged — only the key values differ.

## Naming scheme

`keys_sz<count>_k<capacity>_d<max_dim>.pkl` (format string at
`scripts/find_adversarial_keys.py:72`), e.g. `keys_sz10000_k16_d160.pkl` = 10000 keys,
K=16, 160 dimensions.

## Generation parameters

Defined in `main()` at `scripts/find_adversarial_keys.py:82-88` — `counts = [10000]`,
`rank = 1`, and per-capacity dimension lists:

| K (capacity) | dims |
|---|---|
| 4 | 1, 10, 20, 30, 40 |
| 8 | 1, 10, 20, 40, 80 |
| 16 | 1, 10, 20, 40, 80, 160 |
| 32 | 1, 10, 20, 40, 80, 160, 320 |

The 23 tracked files cover exactly this matrix.

## Promotion process (new keys → frozen fixtures)

1. The generator writes to `benchmarks/adversarial_keys_new/` — the directory does not
   exist in the repo; it is created on demand (`os.makedirs(exist_ok=True)`,
   `scripts/find_adversarial_keys.py:70-71`) and is gitignored. It must never be committed.
2. After inspection/validation, selected files are **manually moved** into
   `benchmarks/adversarial_keys/` and committed — only then do they become frozen
   reference inputs. Record parameter changes here when promoting.

`README.md` remains the single source for how the generator is invoked; this file only
records the provenance and promotion chain.
