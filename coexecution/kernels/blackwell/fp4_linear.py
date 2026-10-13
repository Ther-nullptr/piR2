"""Lazy loader for the owned FP4 APIs consumed by the GR00T Thor adapter."""

from .extension_loader import require_backend

OP_NAMESPACE = "pir2_cutlass_fp4"
REQUIRED_OPS = (
    "supports_device",
    "set_decode_policy",
    "pack_linear_weight_gemm_bf16",
    "pack_activation_gemm_bf16",
    "linear_forward_packed_gemm_bf16_tactic",
)


def require_fp4_backend(device=None):
    return require_backend("fp4", OP_NAMESPACE, REQUIRED_OPS, device)
