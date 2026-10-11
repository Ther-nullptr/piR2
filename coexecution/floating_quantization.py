"""Reversible Thor FP8/FP4 projections using the pinned Speedup Paradox backend.

E4M3 or block-scaled E2M1 weights are packed once. Each call dynamically quantizes
the input and executes CUTLASS low-bit GEMM with FP32 accumulation and BF16 output.
Original parameters remain alive for restoration; this is not weight replacement
or a claim that total model memory falls in proportion to the quantization bits.
"""

import torch

from coexecution.floating_packing import FloatingPacked
from coexecution.groot_projections import inventory

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
            from coexecution.floating_packing import prepare_fp8

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


class FloatingProjections:
    """Pack selected Linear weights, then install instance-local forwards."""

    def __init__(
        self,
        model,
        executed=None,
        *,
        expanded=False,
        precision="fp8",
        fast_fp8=False,
        shared_inputs=False,
        grouped=False,
        swiglu=False,
    ):
        if precision not in ("fp8", "fp4"):
            raise ValueError("Floating precision must be fp8 or fp4")
        alignment = 16 if precision == "fp8" else 32
        all_sites, groups, _, text = inventory(model, expanded=True)
        selected = all_sites if expanded else inventory(model)[0]
        if executed is not None:
            selected = {
                name: site for name, site in selected.items() if name in executed
            }
        self.saved = []
        self.precision = precision
        self.linears = {}
        self.groups = []
        self.coverage = {
            "backend": (
                "robotics_cutlass_fp8.linear_forward_packed_tactic"
                if precision == "fp8"
                else "robotics_cutlass_fp4.linear_forward_packed_gemm_bf16_tactic"
            ),
            "format": (
                "E4M3 per tensor, FP32 scales and accumulation, BF16 output"
                if precision == "fp8"
                else "E2M1, UE4M3 block-16 scales in CUTLASS swizzled layout, FP32 accumulation, BF16 output"
            ),
            "tactic": (
                "Thor exact M/K/N/bias table, otherwise 2"
                if precision == "fp8"
                else "0 for M>1 (bias-aware), 3 for M=1"
            ),
            "activation_scale_refresh": "Every invocation",
            "activation_quantizer": "Triton two-pass"
            if fast_fp8
            else "Pinned reference",
            "linears": [],
            "fallbacks": [],
            "original_weights_retained": True,
            "quantized_weight_bytes": 0,
            "weight_scale_bytes": 0,
            "attention": "QK, softmax and PV remain floating-point attention",
            "shared_input_groups": [],
            "merged_projection_groups": [],
            "swiglu_producers": [],
        }
        for name, (module, role, scope) in all_sites.items():
            reason = None
            if name not in selected:
                reason = "Excluded by scope or explicit coverage"
            elif module.in_features % alignment or module.out_features % alignment:
                reason = f"{precision} GEMM requires K and N divisible by {alignment}"
            if reason:
                self.coverage["fallbacks"].append(
                    {"name": name, "role": role, "scope": scope, "reason": reason}
                )
        for name, module in model.named_modules():
            if type(module).__name__ == "CategorySpecificLinear":
                self.coverage["fallbacks"].append(
                    {
                        "name": name,
                        "scope": "category",
                        "reason": "CategorySpecificLinear remains BF16",
                    }
                )
        skipped = {row["name"] for row in self.coverage["fallbacks"]}
        selected = {
            name: site for name, site in selected.items() if name not in skipped
        }
        if not selected:
            raise ValueError("Floating quantization selected no supported projections")
        try:
            for name, (module, role, scope) in selected.items():
                quantized = FloatingLinear(module, name, precision, fast_fp8=fast_fp8)
                self.linears[name] = quantized
                self.coverage["linears"].append(
                    {
                        "name": name,
                        "role": role,
                        "scope": scope,
                        "shape_nk": list(module.weight.shape),
                        "has_bias": module.bias is not None,
                    }
                )
                self.coverage["quantized_weight_bytes"] += (
                    quantized.packed_weight.numel()
                    * quantized.packed_weight.element_size()
                )
                self.coverage["weight_scale_bytes"] += (
                    quantized.weight_scale.numel()
                    * quantized.weight_scale.element_size()
                )
            for name, (module, _, _) in selected.items():
                self.saved.append(
                    (module, module.forward, "forward" in module.__dict__)
                )
                module.forward = self.linears[name].forward
            self.sites = selected
            if swiglu:
                from coexecution.floating_packing import prepare_swiglu

                values = (
                    torch.arange(
                        65536, device=next(model.parameters()).device, dtype=torch.int32
                    )
                    .to(torch.int16)
                    .view(torch.bfloat16)
                )
                lut = torch.nn.functional.silu(values)
                for mlp, down in text:
                    if not all(
                        id(m) in {id(s[0]) for s in selected.values()}
                        for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
                    ):
                        continue
                    self.saved.append((mlp, mlp.forward, "forward" in mlp.__dict__))

                    def forward(x, mlp=mlp, lut=lut):
                        packed = prepare_swiglu(
                            mlp.gate_proj(x), mlp.up_proj(x), lut, precision
                        )
                        return mlp.down_proj(packed)

                    mlp.forward = forward
                    self.coverage["swiglu_producers"].append(down)
            if shared_inputs or grouped:
                modules = dict(model.named_modules())
                for names in groups:
                    if not all(name in self.linears for name in names):
                        continue
                    owner_name = names[0].rpartition(".")[0]
                    assert all(name.rpartition(".")[0] == owner_name for name in names)
                    owner = modules[owner_name]
                    if grouped:
                        group = FloatingProjectionGroup(
                            [selected[name][0] for name in names],
                            owner_name,
                            precision,
                            fast_fp8,
                            contiguous=selected[names[0]][1:] == ("AttnQ", "text"),
                        )
                        # Individual packed weights are no longer used by grouped sites.
                        for name in names:
                            self.linears.pop(name)
                        self.coverage["merged_projection_groups"].append(
                            {
                                "owner": owner_name,
                                "members": list(names),
                                "weight_scale": "one tensor per merged group"
                                if precision == "fp8"
                                else "unchanged block-16 row scales",
                            }
                        )
                    else:
                        group = FloatingInputGroup(
                            [self.linears[name] for name in names]
                        )
                    self.groups.append(group)
                    original = owner.forward
                    self.saved.append((owner, original, "forward" in owner.__dict__))
                    owner.forward = lambda *args, g=group, f=original, **kw: g.run(
                        f, *args, **kw
                    )
                    for index, name in enumerate(names):
                        selected[name][0].forward = lambda x, g=group, i=index: (
                            g.project(x, i)
                        )
                    if not grouped:
                        self.coverage["shared_input_groups"].append(
                            {
                                "owner": owner_name,
                                "members": list(names),
                                "reuse": "One owner invocation; identical unmodified input tensor",
                                "gemms": "Independent original weight scales and tactics",
                            }
                        )
            buffers = [
                *self.linears.values(),
                *(
                    g.linear
                    for g in self.groups
                    if isinstance(g, FloatingProjectionGroup)
                ),
            ]
            self.coverage["quantized_weight_bytes"] = sum(
                q.packed_weight.numel() * q.packed_weight.element_size()
                for q in buffers
            )
            self.coverage["weight_scale_bytes"] = sum(
                q.weight_scale.numel() * q.weight_scale.element_size() for q in buffers
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        for module, forward, had_forward in reversed(self.saved):
            if had_forward:
                module.forward = forward
            else:
                del module.forward
        self.saved.clear()
        for group in self.groups:
            group.active, group.cache = False, None
        self.groups.clear()
        self.linears.clear()
