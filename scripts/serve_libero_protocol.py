"""Compatibility entrypoint; implementation lives in coexecution.libero.service."""

import sys as _sys
from pathlib import Path as _Path

# Direct script calls and historical imports must work outside the repository cwd.
_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from coexecution.libero.service import (
    LOGGER,
    Backend,
    GrootOptimizations,
    Path,
    add_optimization_arguments,
    argparse,
    endpoint_layout,
    json,
    load_core,
    logging,
    main,
    optimization_config,
    os,
    serve,
    supervise_workers,
    threading,
)

__all__ = [
    "LOGGER",
    "Backend",
    "GrootOptimizations",
    "Path",
    "add_optimization_arguments",
    "argparse",
    "endpoint_layout",
    "json",
    "load_core",
    "logging",
    "main",
    "optimization_config",
    "os",
    "serve",
    "supervise_workers",
    "threading",
]


if __name__ == "__main__":
    main()
