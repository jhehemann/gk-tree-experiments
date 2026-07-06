# ADR-0002: Hysteresis-free conversion threshold

**Status:** Accepted · **Date:** 2026-07-02

## Context

Conversion switches a node set's representation between k-list and nested
Gᵏ⁺-tree at the threshold k·l_factor. A workload oscillating around that count
re-converts on every step, each conversion costing O(n·h). The standard
engineering fix is hysteresis: expand at a higher count than you collapse.

## Decision

Expansion and collapse share one sharp threshold: expand when the item count
exceeds k·l_factor, collapse when it falls to or below it. No hysteresis.

History independence requires the representation to be a pure function of the
current set. With hysteresis, the representation would depend on whether the
threshold was crossed from above or below — two observably different
structures (and Merkle hashes) for the same set.

## Consequences

- Alternating insert/delete workloads at the threshold pay repeated
  conversions; this is the accepted price for history independence.
- Any future "optimization" that introduces hysteresis or lazy conversion
  breaks history independence — don't.
