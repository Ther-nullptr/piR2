"""INT8/INT4 packing and linear projections using the pinned reference backend.

Quantization and packing follow robotics-the-speedup-paradox revision
239b4a3ef2268048571c9f508ad5700f98398c2f. See sources.lock.json and
THIRD_PARTY_NOTICES.md for upstream provenance; no third-party kernel is copied.
"""

from types import SimpleNamespace

import torch
import triton

from coexecution.kernels.ampere_ada.integer import (
    IntegerLinear,
    PackedActivation,
    _prepare,
)
from coexecution.kernels.common.fused import prepare_gelu

_LAUNCHERS = {}


def selected_tactic(q, rows):
    return getattr(q, "shape_tactics", {}).get(rows, q.tactic)


def install_shape_tactics(q, choices):
    """Select existing reference kernels before launch/capture, without repacking."""
    q.shape_tactics = {
        rows: tactic
        for (bits, rows, k, n, bias), tactic in choices.items()
        if (bits, k, n, bias)
        == (q.bits, q.in_features, q.out_features, q.bias is not None)
    }
    original = getattr(q, "_reference_ops", q._ops)
    q._reference_ops = original

    def bind(operation):
        def run(a, sa, b, sb, bias, bits, tactic):
            return operation(
                a, sa, b, sb, bias, bits, q.shape_tactics.get(a.shape[0], tactic)
            )

        return run

    q._ops = SimpleNamespace(
        gemm=bind(original.gemm), gemm_biasless=bind(original.gemm_biasless)
    )


def prepare(x, bits, lut=None, up=None):
    k = x.shape[-1]
    rows = x.numel() // k
    pad = triton.cdiv(k, 128) * 128
    data = torch.empty(
        (rows, pad if bits == 8 else pad // 2),
        device=x.device,
        dtype=torch.int8 if bits == 8 else torch.uint8,
    )
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    mode = 0 if lut is None else 2 if up is None else 1
    up = x if up is None else up
    tensors = (x, up, x if lut is None else lut, data, scales)
    constants = (
        k,
        pad,
        bits,
        mode,
        triton.next_power_of_2(pad),
        True,
        x.view(rows, k).stride(0),
        up.view(rows, k).stride(0),
    )
    key = (
        rows,
        constants,
        tuple((t.device, t.dtype, t.data_ptr() % 16) for t in tensors),
    )
    launch = _LAUNCHERS.get(key)
    if launch is None:
        kernel = _prepare[(rows,)](
            *tensors,
            *constants,
            num_warps=4 if pad <= 2048 else 8,
            enable_fp_fusion=False,
        )
        _LAUNCHERS[key] = kernel[(rows, 1, 1)]
    else:
        launch(*tensors, *constants)
    return PackedActivation(data, scales, tuple(x.shape[:-1]), k, bits)


def pack_swiglu(gate, up, lut, bits):
    return prepare(gate, bits, lut, up)


class PreparedLinear(IntegerLinear):
    def pack_input(self, x):
        if getattr(x, "_robotics_packed_input", False):
            return x.for_bits(self.bits)
        return prepare(x, self.bits)

    def forward_gelu(self, x, up=None, *, approximate="tanh"):
        return self.forward_packed(
            prepare(x, self.bits, prepare_gelu(x.device, x.dtype, approximate), up)
        )
