# Cleanup backlog

Deliberately deferred — each item is its own test-driven PR, not part of a hygiene sweep.

- **Split oversized test files** (audit F8c): `tests/gplus/test_insert.py` (~1439 lines),
  `tests/gk_plus/test_gk_plus_zip.py` (~1418), `tests/test_klist.py` (~1128). Split along
  test-class seams; keep the mirrored `tests/{gk_plus,gplus,merkle}` layout.
- **Align `g_k_plus` base-module naming** (audit F5): `g_k_plus/base.py` (abstract) vs.
  `g_k_plus/g_k_plus_base.py` (implementation) — align to the `<name>_base.py` convention
  (see `docs/README.md`) the next time the package is touched. No rename for its own sake.
