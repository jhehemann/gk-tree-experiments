# ADR-0001: Record decisions in ADRs

**Status:** Accepted · **Date:** 2026-07-02

## Context

The 2026-07-02 repository audit (F6) found no place where decisions and their rationale
survive: conclusions lived at best in commit messages. After a multi-month pause, exactly
this reconstruction is the most expensive part of resuming work — for humans and agents.

## Decision

Decisions with lasting consequences are recorded as Architecture Decision Records in
`docs/adr/`, one file per decision:

- **Filename:** `NNNN-<slug>.md`, numbered consecutively from `0001`.
- **Structure:** Title, Status (Proposed/Accepted/Superseded), Date, Context, Decision,
  Consequences.
- **Append-only:** an accepted ADR is not edited into a different decision; it is
  superseded by a new ADR that references it.

## Consequences

- `docs/adr/README.md` keeps the index.
- `CLAUDE.md` points here; agents check ADRs touching the area they work in.
