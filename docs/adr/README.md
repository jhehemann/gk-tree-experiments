# Architecture Decision Records

Decisions with lasting consequences are recorded here, one file per decision:
`NNNN-<slug>.md`, numbered consecutively from `0001`. Why: the 2026-07-02
repository audit (F6) found decisions surviving only in commit messages — after
a multi-month pause, reconstructing them is the most expensive part of resuming
work.

| # | Title | Status | Date |
|---|-------|--------|------|
| [0001](0001-terminology-precedence.md) | Terminology precedence — paper over thesis over repo | Accepted | 2026-07-02 |

## Format

Minimal by default — an ADR can be a single paragraph (1–3 sentences: context,
decision, why). Add sections (Status, Context/Decision/Consequences, Considered
Options) only when they genuinely add value. Append-only: an accepted ADR is
not edited into a different decision; it is superseded by a new ADR that
references it.
