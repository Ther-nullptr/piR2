"""Reversible GR00T installation of Thor FP8/FP4 projections."""

import torch

from coexecution.optimization.backends.thor.linear import (
    FloatingInputGroup,
    FloatingLinear,
    FloatingProjectionGroup,
)
from coexecution.optimization.groot.projections import inventory


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
                from coexecution.optimization.backends.thor.packing import (
                    prepare_swiglu,
                )

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
