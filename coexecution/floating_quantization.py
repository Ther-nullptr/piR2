"Compatibility exports for Thor projections and GR00T installation."

# Preserve historical attribute imports, including private kernel helpers.

from coexecution.optimization.backends.thor.linear import (
    THOR_FP8_TACTICS,
    FloatingInputGroup,
    FloatingLinear,
    FloatingPacked,
    FloatingProjectionGroup,
    torch,
)
from coexecution.optimization.groot.floating import (
    FloatingProjections,
    inventory,
)

__all__ = [
    "THOR_FP8_TACTICS",
    "FloatingInputGroup",
    "FloatingLinear",
    "FloatingPacked",
    "FloatingProjectionGroup",
    "FloatingProjections",
    "inventory",
    "torch",
]
