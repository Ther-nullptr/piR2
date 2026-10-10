"""Instance-local GR00T INT8/INT4 adapter using the pinned reference backend.

Quantization and packing follow robotics-the-speedup-paradox revision
239b4a3ef2268048571c9f508ad5700f98398c2f. See sources.lock.json and
THIRD_PARTY_NOTICES.md for upstream provenance; no third-party kernel is copied.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from robotics_kernels.ampere_ada.integer import (
    IntegerLinear,
    IntegerProjectionGroup,
    PackedActivation,
    _prepare,
)
from robotics_kernels.common.fused import prepare_gelu

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


def inventory(model, expanded=False):
    names = {id(m): n for n, m in model.named_modules()}
    sites, groups, activations, text = {}, [], [], []

    def add(m, role, scope):
        assert isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16
        name = names[id(m)]
        sites[name] = (m, role, scope)
        return name

    for name, m in model.named_modules():
        kind = type(m).__name__
        if kind == "Qwen3VLVisionAttention":
            add(m.qkv, "AttnQKV", "vision")
            add(m.proj, "AttnO", "vision")
        elif kind in ("Qwen3VLVisionMLP", "Qwen3VLVisionPatchMerger"):
            add(m.linear_fc1, "FFNup", "vision")
            down = add(m.linear_fc2, "FFNdown", "vision")
            act_kind = type(m.act_fn).__name__
            assert act_kind in ("GELUTanh", "GELU"), act_kind
            approximate = "tanh" if act_kind == "GELUTanh" else m.act_fn.approximate
            activations.append((m.act_fn, "forward", down, approximate))
        elif kind == "Qwen3VLTextAttention":
            groups.append(
                [
                    add(getattr(m, a), r, "text")
                    for a, r in [
                        ("q_proj", "AttnQ"),
                        ("k_proj", "AttnK"),
                        ("v_proj", "AttnV"),
                    ]
                ]
            )
            add(m.o_proj, "AttnO", "text")
        elif kind == "Qwen3VLTextMLP":
            assert m.config.hidden_act == "silu"
            groups.append(
                [add(m.gate_proj, "FFNgate", "text"), add(m.up_proj, "FFNup", "text")]
            )
            down = add(m.down_proj, "FFNdown", "text")
            text.append((m, down))
        elif kind == "BasicTransformerBlock":
            a = m.attn1
            scope = (
                "dit"
                if name.startswith("action_head.model.transformer_blocks.")
                else "vl_transformer"
            )
            qkv = [
                add(getattr(a, attr), role, scope)
                for attr, role in [
                    ("to_q", "AttnQ"),
                    ("to_k", "AttnK"),
                    ("to_v", "AttnV"),
                ]
            ]
            groups.append(qkv[1:] if a.is_cross_attention else qkv)
            add(a.to_out[0], "AttnO", scope)
            assert type(m.ff.net[0]).__name__ == "GELU"
            add(m.ff.net[0].proj, "FFNup", scope)
            down = add(m.ff.net[2], "FFNdown", scope)
            activations.append((m.ff.net[0], "gelu", down, m.ff.net[0].approximate))
    if expanded:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and name not in sites:
                add(m, "AdditionalLinear", "additional")
    return sites, groups, activations, text


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
                    str(Path(__file__).resolve().parent / "csrc/integer_dispatch.cu")
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
