"""Reversible GR00T model installation for pinned integer projections."""

from pathlib import Path

import torch
import torch.nn.functional as F
from robotics_kernels.ampere_ada.integer import IntegerProjectionGroup
from robotics_kernels.common.fused import prepare_gelu

from coexecution.optimization.backends.integer.linear import (
    PreparedLinear,
    prepare,
    selected_tactic,
)
from coexecution.optimization.groot.projections import inventory


class TransformerINT:
    def __init__(
        self, model, executed, native=False, expanded=False, condition_group=False
    ):
        sites, groups, activations, text = inventory(model, expanded)
        if condition_group:
            dit = model.action_head.model
            names = {id(module): name for name, module in model.named_modules()}
            # PointwiseFusion passes the same activated condition to every block
            # and the output modulation. Reuse the reference projection group.
            groups.append(
                [names[id(block.norm1.linear)] for block in dit.transformer_blocks]
                + [names[id(dit.proj_out_1)]]
            )
        self.sites = {n: s for n, s in sites.items() if n in executed}
        self.groups = [g for g in groups if all(n in self.sites for n in g)]
        self.activations = [a for a in activations if a[2] in self.sites]
        self.text = [(m, n) for m, n in text if n in self.sites]
        self.quant = {
            bits: {
                n: PreparedLinear.from_linear(
                    m, bits=bits, pack_reuse=True, biasless_epilogue=True
                )
                for n, (m, _, _) in self.sites.items()
            }
            for bits in (8, 4)
        }
        entries = [(m, "forward") for m, _, _ in self.sites.values()]
        entries += [(m, attr) for m, attr, _, _ in self.activations]
        entries += [(m, "forward") for m, _ in self.text]
        self.saved = [(m, a, a in m.__dict__, getattr(m, a)) for m, a in entries]
        device = next(model.parameters()).device
        values = (
            torch.arange(65536, device=device, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        self.silu_lut = F.silu(values)
        self.grouped = {bits: [] for bits in (8, 4)}
        for bits in (8, 4):
            for names in self.groups:
                # Text Q/K RMSNorm requires contiguous inputs; other consumers accept row strides.
                contiguous = (
                    self.sites[names[0]][1] == "AttnQ"
                    and self.sites[names[0]][2] == "text"
                )
                group = IntegerProjectionGroup(
                    [self.quant[bits][n] for n in names], contiguous_outputs=contiguous
                )
                group.linear.pack_input = lambda x, bits=bits: (
                    x.for_bits(bits)
                    if getattr(x, "_robotics_packed_input", False)
                    else prepare(x, bits)
                )
                self.grouped[bits].append((names, group))
        self.native = native
        if native:
            from torch.utils.cpp_extension import load

            assert all(
                q.in_features <= 8192 and q.out_features % 8 == 0
                for q in self.quant[8].values()
            )
            load(
                name="pir2_integer_dispatch",
                sources=[
                    str(
                        Path(__file__).resolve().parents[2] / "csrc/integer_dispatch.cu"
                    )
                ],
                extra_cuda_cflags=["-O3", "--fmad=false", "--ftz=false"],
                is_python_module=False,
                verbose=True,
            )
            for bits in (8, 4):
                models = list(self.quant[bits].values()) + [
                    g.linear for _, g in self.grouped[bits]
                ]
                for q in models:
                    q.forward = lambda x, q=q: self.dispatch(q, x)
                    q.forward_gelu = lambda x, up=None, approximate="tanh", q=q: (
                        self.dispatch(
                            q, x, prepare_gelu(x.device, x.dtype, approximate), up
                        )
                    )
        self.mode = "bf16"

    def dispatch(self, q, x, lut=None, up=None):
        if getattr(x, "_robotics_packed_input", False):
            return q.forward_packed(x.for_bits(q.bits))
        if self.native:
            y = torch.ops.pir2_integer_dispatch.linear(
                x,
                q.packed_weight,
                q.weight_scale,
                q.bias,
                lut,
                up,
                q.bits,
                selected_tactic(q, x.numel() // x.shape[-1]),
            )
            return y[:, : q.out_features].reshape(*x.shape[:-1], q.out_features)
        return q.forward_packed(prepare(x, q.bits, lut, up))

    def set(self, mode):
        for groups in self.grouped.values():
            for _, group in groups:
                group._input = group._output = None
                group._remaining.clear()
        for m, attr, had, original in self.saved:
            if had:
                setattr(m, attr, original)
            elif attr in m.__dict__:
                delattr(m, attr)
        self.mode = mode
        if mode == "bf16":
            return
        bits = {"w8a8": 8, "w4a4": 4}[mode]
        linears = self.quant[bits]
        for name, (m, _, _) in self.sites.items():
            m.forward = linears[name].forward
        for names, group in self.grouped[bits]:
            for i, name in enumerate(names):
                self.sites[name][0].forward = lambda x, g=group, i=i: g.project(x, i)
        for activation, attr, down, approximate in self.activations:
            setattr(activation, attr, lambda x: x)
            self.sites[down][0].forward = lambda x, q=linears[down], a=approximate: (
                q.forward_gelu(x, approximate=a)
            )
        for mlp, down in self.text:

            def forward(x, m=mlp, q=linears[down], bits=bits):
                gate, up = m.gate_proj(x), m.up_proj(x)
                return self.dispatch(q, gate, self.silu_lut, up)

            mlp.forward = forward

    def close(self):
        self.set("bf16")
