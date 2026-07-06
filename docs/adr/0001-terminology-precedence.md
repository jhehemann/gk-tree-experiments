# ADR-0001: Terminology precedence — paper over thesis over repo

**Status:** Accepted · **Date:** 2026-07-02

## Context

The project's vocabulary has three sources: the
[G-trees paper](https://g-trees.github.io/g_trees/) (first canonical
definitions: rank, G-node, k-list, G-tree, zip/unzip, dummy element, leafy
G⁺-variants), the
[master's thesis](https://monami.hs-mittweida.de/frontdoor/index/index/year/2025/docId/16118)
building on it (J. Hehemann, HS Mittweida, 2025: dimensions, Gᵏ⁺-trees, node
rank, boundary items), and this repo's code and docs. The conversion vocabulary — conversion, expansion, collapse,
conversion threshold, l_factor — is repo-coined, not thesis terminology: the
thesis only speaks of a node set exceeding "some threshold" and being
instantiated one dimension deeper, so these terms sit at precedence level 3.
The three sources had started to drift ("node" meaning three different
things, "rank" both an item and a node property).

## Decision

Terminology is adopted with this precedence:

1. **Formally defined in the G-trees paper** → adopt unchanged; do not rename
   unless strictly necessary.
2. **Introduced by the thesis** (not in the paper) → open for sharpening and
   renaming when it improves clarity.
3. **Repo-local wording** (informal names, notation, spellings that neither
   source formally defines) → freely adjustable to avoid overloading.

`CONTEXT.md` records the resulting canonical terms. Informal *word usage* in
the paper does not bind us: the paper's k-list definition casually says "node"
for its blocks, but "node" is not itself a defined term — we deliberately say
**segment** there to keep "node" unambiguous (G-node).

## Consequences

- Glossary disputes are resolved by source rank, not taste.
- Paper terms (rank, G-node, k-list, zip/unzip, dummy) are stable anchors;
  renames of thesis/repo terms stay cheap because they are marked as such.
- The repo's rank derivation (key hashes instead of true random draws) is an
  instantiation of the paper's geometric rank function, not a redefinition.
