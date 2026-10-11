"""Compatibility entrypoint; implementation lives in coexecution.libero.loading."""

import sys as _sys
from pathlib import Path as _Path

# Direct script calls and historical imports must work outside the repository cwd.
_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from coexecution.libero.loading import (
    contextmanager,
    task_checkpoint_backbone,
)

__all__ = [
    "contextmanager",
    "task_checkpoint_backbone",
]
