"""Reversible Thor FP8/FP4 projections using the pinned Speedup Paradox backend.

E4M3 or block-scaled E2M1 weights are packed once. Each call dynamically quantizes
the input and executes CUTLASS low-bit GEMM with FP32 accumulation and BF16 output.
Original parameters remain alive for restoration; this is not weight replacement
or a claim that total model memory falls in proportion to the quantization bits.
"""

import torch

from coexecution.optimization.backends.thor.packing import FloatingPacked

# Thor SM110, B1/H40: three randomized rounds on the executed M/K/N/bias
# signatures. Keep only choices with >10% median gain over prefill tactic 2.
# These are FP8 backend IDs, independent of the Ada integer tactic tables.
THOR_FP8_TACTICS = {
    (41, 256, 1536, True): 1,
    (41, 1536, 1536, True): 1,
    (41, 1536, 3072, True): 6,
    (41, 1536, 6144, True): 6,
    (41, 6144, 1536, True): 6,
    (128, 4096, 2048, True): 1,
    (128, 4096, 4096, True): 6,
}


class FloatingLinear(torch.nn.Module):
    """Explicit E4M3 or block-scaled E2M1 GEMM, with independent packed formats."""

    def __init__(self, linear, name, precision="fp8", *, fast_fp8=False):
        super().__init__()
        if precision not in ("fp8", "fp4"):
            raise ValueError("Floating precision must be fp8 or fp4")
        alignment = 16 if precision == "fp8" else 32
        if (
            not linear.weight.is_cuda
            or linear.weight.dtype != torch.bfloat16
            or linear.in_features % alignment
            or linear.out_features % alignment
        ):
            raise ValueError(f"{precision} requires aligned CUDA BF16 Linear weights")
        if torch.cuda.get_device_capability(linear.weight.device) != (11, 0):
            raise ValueError("The experimental FP adapter is validated for Thor SM110")
        self.precision = precision
        self.fast_fp8 = fast_fp8
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if precision == "fp8":
            from robotics_kernels.blackwell.fp8_linear import require_fp8_backend

            require_fp8_backend(linear.weight.device)
            self.ops = torch.ops.robotics_cutlass_fp8
            pack = self.ops.pack_linear_weight
        else:
            from robotics_kernels.blackwell.fp4_linear import require_fp4_backend

            require_fp4_backend(linear.weight.device)
            self.ops = torch.ops.robotics_cutlass_fp4
            pack = self.ops.pack_linear_weight_gemm_bf16
        weight, scale, metadata = pack(
            linear.weight.detach().contiguous(), "pir2", name
        )
        self.register_buffer("packed_weight", weight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("metadata", metadata)
        # The immutable BF16 bias shares its original storage.
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach()
        )

    def pack_input(self, value):
        if isinstance(value, FloatingPacked):
            if (
                value.precision != self.precision
                or value.shape[-1] != self.in_features
                or value.data.device != self.packed_weight.device
            ):
                raise ValueError("Floating packed input does not match its consumer")
            return value.data, value.scale
        if (
            value.dtype != torch.bfloat16
            or value.device != self.packed_weight.device
            or value.ndim < 1
            or value.shape[-1] != self.in_features
            or value.numel() == 0
            or value.requires_grad
        ):
            raise ValueError(
                "Floating quantization requires nonempty CUDA BF16 inference activations"
            )
        matrix = value.reshape(-1, self.in_features).contiguous()
        if self.precision == "fp8" and self.fast_fp8:
            from coexecution.optimization.backends.thor.packing import prepare_fp8

            return prepare_fp8(matrix)
        return (
            self.ops.quantize_bf16(matrix)
            if self.precision == "fp8"
            else self.ops.pack_activation_gemm_bf16(matrix)
        )

    def forward_packed(self, packed, scale, leading_shape):
        if self.precision == "fp8":
            signature = (
                packed.shape[0],
                self.in_features,
                self.out_features,
                self.bias is not None,
            )
            result = self.ops.linear_forward_packed_tactic(
                packed,
                scale,
                self.packed_weight,
                self.weight_scale,
                self.bias,
                THOR_FP8_TACTICS.get(signature, 2),
            )
        else:
            # The default fuses bias for M>1. Explicit nonzero tactics omit
            # that epilogue, so use them only for M=1, which adds bias separately.
            result = self.ops.linear_forward_packed_gemm_bf16_tactic(
                packed,
                scale,
                self.packed_weight,
                self.weight_scale,
                self.metadata,
                self.bias,
                "pir2",
                "",
                3 if packed.shape[0] == 1 else 0,
            )
        return result.reshape(*leading_shape, self.out_features)

    def forward(self, value):
        packed, scale = self.pack_input(value)
        return self.forward_packed(packed, scale, value.shape[:-1])


class FloatingInputGroup:
    """Share packing within one known, read-only QKV/KV or gate/up invocation.

    GEMMs, weight scales and tactics stay independent. Owner-call boundaries
    clear the cache even after exceptions; direct projection calls never reuse
    it. Tensor identity alone is deliberately not a cross-call cache key.
    Like the policy adapter, this group supports serial inference only.
    """

    def __init__(self, linears):
        self.linears = linears
        self.active = False
        self.cache = None

    def run(self, forward, *args, **kwargs):
        previous = self.active, self.cache
        self.active, self.cache = True, None
        try:
            return forward(*args, **kwargs)
        finally:
            self.active, self.cache = previous

    def project(self, value, index):
        linear = self.linears[index]
        if not self.active:
            return linear(value)
        if self.cache is None or self.cache[0] is not value or index in self.cache[3]:
            packed, scale = linear.pack_input(value)
            self.cache = (value, packed, scale, set())
        _, packed, scale, used = self.cache
        result = linear.forward_packed(packed, scale, value.shape[:-1])
        used.add(index)
        if len(used) == len(self.linears):
            self.cache = None
        return result


class FloatingProjectionGroup:
    """One concatenated GEMM per same-input owner invocation.

    FP8 deliberately requantizes concatenated weights with one shared scale.
    FP4 preserves block-16 row scales. Grouped and independent quantization
    therefore need separate numerical validation, especially for FP8.
    """

    def __init__(self, modules, name, precision, fast_fp8, contiguous=False):
        self.sizes = [m.out_features for m in modules]
        merged = torch.nn.Linear(
            modules[0].in_features,
            sum(self.sizes),
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        merged.weight = torch.nn.Parameter(
            torch.cat([m.weight.detach() for m in modules]), requires_grad=False
        )
        if any(m.bias is not None for m in modules):
            merged.bias = torch.nn.Parameter(
                torch.cat(
                    [
                        m.bias.detach()
                        if m.bias is not None
                        else m.weight.new_zeros(m.out_features)
                        for m in modules
                    ]
                ),
                requires_grad=False,
            )
        self.linear = FloatingLinear(merged, name, precision, fast_fp8=fast_fp8)
        self.contiguous = contiguous
        self.active, self.cache = False, None

    run = FloatingInputGroup.run

    def project(self, value, index):
        if (
            not self.active
            or self.cache is None
            or self.cache[0] is not value
            or index in self.cache[2]
        ):
            parts = self.linear(value).split(self.sizes, dim=-1)
            if self.contiguous:
                parts = tuple(p.contiguous() for p in parts)
            if not self.active:
                return parts[index]
            self.cache = (value, parts, set())
        _, parts, used = self.cache
        result = parts[index]
        used.add(index)
        if len(used) == len(self.sizes):
            self.cache = None
        return result
