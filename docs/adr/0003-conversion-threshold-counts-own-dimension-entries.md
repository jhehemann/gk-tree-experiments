# ADR-0003: Conversion threshold counts only own-dimension entries

**Status:** Accepted · **Date:** 2026-07-06

## Context

The KList↔tree conversion threshold (ADR-0002) compared `item_count()`, which
includes dummies carried in from other dimensions. Carried dummies accumulate
by one per expansion level, so once enough of them shared a node the threshold
became unsatisfiable: with K=2, l_factor=1.0, a node holding two carried
dummies plus one real key stays above the threshold in every dimension, and
bulk creation recursed through dimensions without bound (RecursionError).
Under that counting rule the canonical structure for such inputs does not
exist — it would be infinitely deep.

## Decision

Only entries of the set's own dimension weigh against the conversion
threshold: real items (key ≥ 0) and the owning tree's own dummy
(key = −DIM). Dummies carried in from other dimensions are exempt.
Expansion, collapse, and the `set_thresholds_met` invariant all share this
counting rule (`bulk_create._threshold_item_count`), keeping the conversion
hysteresis-free (ADR-0002).

## Consequences

- Dimensional recursion terminates: expansion beyond dimension d+1 requires
  more than `k·l_factor` counting entries in a node, and once the real keys'
  ranks diverge, leaf counts drop below the threshold regardless of how many
  carried dummies share the node.
- Structures without carried dummies are unchanged — the count equals the old
  `item_count()` there.
- A node set's KList form may exceed `k·l_factor` raw items (carried dummies
  on top of counting entries); that is expected, not an invariant violation.
