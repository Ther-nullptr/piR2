"Compatibility exports for GR00T floating-point producers."

# Preserve historical attribute imports, including private kernel helpers.

from coexecution.optimization.groot.floating_connections import (
    FloatingConnections,
    prepare_norm,
    torch,
)

__all__ = [
    "FloatingConnections",
    "prepare_norm",
    "torch",
]
