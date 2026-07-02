# gk-tree-experiments

Agent entrypoint: invariants and pointers only. Human documentation lives in
`README.md` (project overview, install, experiment/test commands, layout) and
`benchmarks/BENCHMARKS.md` (benchmark details) — do not duplicate it here.

## Environment & commands

- Use the project-local `./.venv` (build backend: `poetry.core`). Setup: `poetry install`
  (installs dev deps), or `python3 -m venv .venv && ./.venv/bin/pip install -e . && ./.venv/bin/pip install pytest ruff`.
  NEVER run a bare `pip install -e .` outside `./.venv` — it lands in the pyenv base install.
- Imports: `tests/` and `stats/` are importable via the tracked root `conftest.py`
  (`sys.path.insert`), not via the editable install; `gplus_trees` resolves from `src/`
  via the editable install. On import errors, check `conftest.py` first.
- Tests: `pytest` (all) · `pytest -k <name>` · `pytest tests/gplus/`
- Lint/format: `ruff check .` / `ruff format .` — config single source of truth: `pyproject.toml`
- mypy runs locally only (`mypy src/`), deliberately not in CI — dated rationale in `pyproject.toml`.

## Branches & CI

- Trunk-based: `main` is the only long-lived branch. Work on short-lived `claude/*`
  branches, integrate via PR against `main`. There is no `develop`.
- CI gates every push to `main` and every PR against `main`.
- Dormant branches — do NOT build on them: `feat/inverse-node-keys`, `feat/insert-min`.
  Open decision, tracked as a local issue under `.scratch/repo-hygiene/issues/`.

## Knowledge layer

- Experiment findings: `docs/journal/` (chronological). Decisions + rationale: `docs/adr/`.
- Fixture provenance: `benchmarks/adversarial_keys/PROVENANCE.md`.

## Agent scaffolding

`.claude/`, `.agents/`, `skills-lock.json` are machine-local and gitignored (deliberate
decision, 2026-07-02). A fresh clone does not reproduce the skill setup from the repo;
skills come from `mattpocock/skills`, managed via the local `skills-lock.json`.

## Do not "clean up"

- Benchmark isolation outside the repo (`../.isolated-benchmarks/`, wrapper `./benchmark`) — by design.
- Tracked `.pkl` fixtures in `benchmarks/adversarial_keys/` — frozen reference inputs.
- Mixin decomposition in `src/gplus_trees/g_k_plus/` and the mirrored test layout
  `tests/{gk_plus,gplus,merkle}`.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature-slug>/` in this repo; there is no external tracker and PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles use their default strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
