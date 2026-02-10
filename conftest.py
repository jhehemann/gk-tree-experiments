"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``tests`` and ``stats``
# packages are importable when running via ``pytest`` from any directory.
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
