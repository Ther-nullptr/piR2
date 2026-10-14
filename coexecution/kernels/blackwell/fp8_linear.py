"""Lazy loader for the owned FP8 APIs consumed by the GR00T Thor adapter.

The CUDA implementation and original MIT notice are retained in this directory;
unrelated upstream model/sampling adapters are not part of the runtime surface.
"""

from .extension_loader import require_backend

OP_NAMESPACE = "pir2_cutlass_fp8"
REQUIRED_OPS = (
    "supports_device",
    "set_decode_policy",
    "set_swap_ab_enabled",
    "pack_linear_weight",
    "quantize_bf16",
    "linear_forward_packed_tactic",
)


def require_fp8_backend(device=None):
    return require_backend("fp8", OP_NAMESPACE, REQUIRED_OPS, device)
