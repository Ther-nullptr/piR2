"""Compatibility imports for the GR00T optimization configuration and scope.

Canonical implementations live in :mod:`coexecution.optimization`. Importing
this module does not import Torch or compile kernels.
"""

from coexecution.optimization.config import (
    OptimizationConfig,
    add_optimization_arguments,
    optimization_config,
)
from coexecution.optimization.session import GrootOptimizations
from coexecution.optimization.tactics import (
    apply_tactics,
    checked_tactic,
    read_json,
    read_shape_tactics,
)

__all__ = [
    "GrootOptimizations",
    "OptimizationConfig",
    "add_optimization_arguments",
    "apply_tactics",
    "checked_tactic",
    "optimization_config",
    "read_json",
    "read_shape_tactics",
]
