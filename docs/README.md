# docs/ — knowledge layer

- **`journal/`** — chronological experiment findings ("what did the runs show?").
- **`adr/`** — decisions + rationale, append-only (see `adr/README.md`).
- **`agents/`** — conventions consumed by the installed agent skills (issue tracker,
  triage labels, domain docs).

Agent invariants live in the repo-root `CLAUDE.md`; the human entrypoint is the
repo-root `README.md`.

## Conventions

- **Tree-variant module naming:** when adding a new tree variant, use `<name>_base.py`
  for the implementation base (consistent with `klist_base.py`, `gplus_tree_base.py`,
  `gk_plus_mkl_base.py`); plain `base.py` is reserved for abstract bases. The existing
  `g_k_plus/base.py` vs. `g_k_plus/g_k_plus_base.py` pair is aligned the next time that
  package is touched — no rename for its own sake (audit F5).
