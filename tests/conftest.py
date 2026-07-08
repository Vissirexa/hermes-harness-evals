"""Make the repo root importable so tests can `import tools.config_drift_check`.

``tools/`` is a standalone-script directory (deliberately not registered in
pyproject's packages.find — it's meant to run via `python -m
tools.config_drift_check`, not be pip-installed), so it needs the repo root on
sys.path. Python 3's implicit namespace packages make `tools/` importable
without an `__init__.py` once the repo root is on sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
