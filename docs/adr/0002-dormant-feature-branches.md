# ADR-0002: Dormant feature branches — do not build on them

**Status:** Accepted · **Date:** 2026-07-02

## Context

The 2026-07-02 repository audit (`docs/repository-audit-2026-07-02.md`, F2) found two
feature branches whose status — pending or abandoned — is recorded nowhere. Anyone picking
them up risks building on discarded work; anyone ignoring them risks losing pending work.

## Decision

Both branches are declared **dormant: decision open**. They are kept, not deleted, and
nothing new is built on top of them without prior clarification.

- **`feat/inverse-node-keys`** — 43 commits ahead of `main`, 112 behind (as of 2026-07-02).
  Last commit `9094930` (2025-07-20, "refactor(gkplus): add debug logs"). Exists on origin.
  Topic per branch history: inverse node keys, incl. adversarial-key edge-case tests.
  Unclear whether pending or abandoned.
- **`feat/insert-min`** — 1 WIP commit `6f26237` (2025-07-25), local only. The commit
  message documents a known defect: insert_min does not work with the `add_dummy` flag
  (lower dims adding dummies later cannot insert min if a higher dim's dummy is already
  inserted).

**Why not deleted:** status unclear; deleting risks losing pending work. Kept until an
explicit decision. When resuming `feat/inverse-node-keys`, weigh the rebase cost
(112 commits of divergence) against salvaging the 43 commits.

## Removed `develop` branch (rollback note)

As part of the same cleanup, the repo switched to trunk-based development (`main` only;
work on short-lived `claude/*` branches via PR). `develop` was deleted locally and on
origin after verifying it carried 0 commits ahead of `main`. Its tip SHA is preserved:

- Tip: `df17bb21ac86a683672e4013fdfde25ed7865303`
- Restore if ever needed: `git push origin df17bb21ac86a683672e4013fdfde25ed7865303:refs/heads/develop`

## Consequences

- `git branch` shows only `main`, the two dormant `feat/*` branches, and short-lived
  work branches.
- `CLAUDE.md` points here; agents must not resume the dormant branches without the owner
  deciding their fate.
