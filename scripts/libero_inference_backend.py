"""Compatibility entrypoint; implementation lives in coexecution.libero.backend."""

import sys as _sys
from pathlib import Path as _Path

# Direct script calls and historical imports must work outside the repository cwd.
_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from coexecution.libero.backend import (
    KEYS,
    Backend,
    endpoint_layout,
    extra_buffer_shift,
    hashlib,
    json,
    np,
    project_commands,
    re,
    time,
)

__all__ = [
    "KEYS",
    "Backend",
    "endpoint_layout",
    "extra_buffer_shift",
    "hashlib",
    "json",
    "np",
    "project_commands",
    "re",
    "time",
]
