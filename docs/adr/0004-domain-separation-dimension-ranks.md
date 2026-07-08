# ADR-0004: Domain-separated dimension ranks

**Status:** Accepted · **Date:** 2026-07-07

## Context

The rank of a key at dimension `j` was derived by *iterated hashing*: dimension 1
hashed the key, and each deeper dimension hashed the previous dimension's digest
(`ρ_j(x) = rank(H^j(x))`). Iterating a random function loses entropy — chains
merge after a collision and cycles exist — so the dimensions are not independent.
In the paper's model that forces an explicit `O(len²·Q²/2^λ_H)` slack term into
every cross-dimension independence step (paper `model.md` §5, review Finding 8b),
which the paper carries as its proof bottleneck. The paper resolved this in
**ADR-002 (accepted 2026-07-07)** by adopting domain separation; this repository
is the implementation the paper's final campaigns measure, so it must match.

## Decision

Derive each dimension's rank directly from the key with a fixed-length dimension
tag, `ρ_j(x) := rank(H(⟨j⟩ ‖ x))`, `⟨j⟩` a 32-byte big-endian encoding
(`gplus_trees.utils.get_digest`). Dimensions become genuinely independent oracles:
distinct `(j, x)` pairs never alias, so for a fresh key the per-dimension rank
vector is i.i.d. geometric with no slack term, and oversized tie sets split almost
surely (no chain-merge/cycle pathology). All rank derivation flows through this one
primitive — the runtime path (`InternalItem.get_digest_for_dim`, consumed by insert
and bulk-create), the analysis helpers (`calc_rank*`, `calc_ranks_multi_dims`), and
the adversarial-key search (`find_keys_for_rank_lists`, `find_rank_keys`).

## Consequences

- **Structural fingerprints change value, not distribution.** Each dimension keeps
  the same marginal geometric rank distribution, so aggregate structure over random
  keys is statistically unchanged: an A/B sweep (same seed, iterated vs. domain
  separation) leaves G-node count, leaf count, maximum rank, and G-node height
  bit-identical, with only the nesting-aware physical height differing within
  variance. Merkle/fingerprint *values* differ; permutation-invariance (history
  independence) still holds.
- **Frozen adversarial fixtures were regenerated** under the new construction
  (`benchmarks/adversarial_keys/`, PROVENANCE promotion cycle). The generator grinds
  one hash per surviving dimension either way, so expected search cost is unchanged.
- **Empirical evidence produced under iterated hashing** (derisk campaign, HI probe,
  mixed-rank probe) retains its value as mechanism evidence — the two constructions
  are indistinguishable up to negligible slack via a standard hybrid argument — but
  paper-grade final campaigns run under domain separation.
- The `get_digest` primitive no longer accepts a chained `bytes` digest; it takes
  `(key: int, dim: int)` only.
