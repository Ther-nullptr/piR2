"Compatibility exports for integer backends and GR00T installation."

# ruff: noqa: F401
# Preserve historical attribute imports, including private kernel helpers.

from coexecution.optimization.backends.integer.linear import (
    _LAUNCHERS,
    IntegerLinear,
    PackedActivation,
    PreparedLinear,
    SimpleNamespace,
    _prepare,
    install_shape_tactics,
    pack_swiglu,
    prepare,
    prepare_gelu,
    selected_tactic,
    torch,
    triton,
)
from coexecution.optimization.groot.integer import (
    F,
    IntegerProjectionGroup,
    Path,
    TransformerINT,
    inventory,
)

__all__ = [
    "F",
    "IntegerLinear",
    "IntegerProjectionGroup",
    "PackedActivation",
    "Path",
    "PreparedLinear",
    "SimpleNamespace",
    "TransformerINT",
    "install_shape_tactics",
    "inventory",
    "pack_swiglu",
    "prepare",
    "prepare_gelu",
    "selected_tactic",
    "torch",
    "triton",
]
