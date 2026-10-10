"""Instance-local GR00T BF16 adapters; parameters and upstream sources stay intact.

Migrated from the GR00T non-GEMM experiment. The existing coexecution.fused_ops
kernels preserve upstream rounding; see THIRD_PARTY_NOTICES.md for provenance.
"""

import json
import types
from collections import Counter
from pathlib import Path

import torch
import triton
import triton.language as tl

from coexecution import fused_ops as ops


@triton.jit
def _modulation_kernel(
    X,
    S,
    H,
    Y,
    N: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    SS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    si = (i // (T * D)) * SS + i % D
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    s = tl.load(S + si, i < N, 0).to(tl.float32)
    h = tl.load(H + si, i < N, 0).to(tl.float32)
    scale = (1.0 + s).to(tl.bfloat16).to(tl.float32)
    product = (x * scale).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, product + h, i < N)


def modulation(x, s, h):
    if not (
        x.dtype == s.dtype == h.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.ndim == 3
        and s.ndim == h.ndim == 2
        and s.shape == h.shape == (x.shape[0], x.shape[2])
        and s.stride(-1) == h.stride(-1) == 1
        and s.stride(0) == h.stride(0)
        and x.is_cuda
        and x.device == s.device == h.device
        and not any(t.requires_grad for t in (x, s, h))
    ):
        raise ValueError("Modulation requires CUDA BF16 [B,T,D] and [B,D] scale/shift")
    y = torch.empty_like(x)
    ops._launch(
        _modulation_kernel,
        (triton.cdiv(x.numel(), 256),),
        (x, s, h, y),
        (x.numel(), x.shape[-1], x.shape[-2], s.stride(0), 256),
    )
    return y


def same(a, b):
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.flatten().contiguous().view(torch.uint8),
            b.flatten().contiguous().view(torch.uint8),
        )
    )


def clone_forward(fn, replacements):
    # Keep Transformers' deprecation wrapper, but also rebind its wrapped
    # function. Patching only wrapper globals leaves text RoPE untouched.
    wrapped = getattr(fn, "__wrapped__", None)

    def cell(value):
        return (lambda: value).__closure__[0]

    closure = fn.__closure__
    if wrapped is not None and closure:
        closure = tuple(
            cell(clone_forward(wrapped, replacements))
            if c.cell_contents is wrapped
            else c
            for c in closure
        )
    result = types.FunctionType(
        fn.__code__,
        {**fn.__globals__, **replacements},
        fn.__name__,
        fn.__defaults__,
        closure,
    )
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


class NonGemmAdapters:
    def __init__(self, policy, adaln_backend="triton"):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as q

        if policy.model.training or adaln_backend not in ("triton", "native"):
            raise ValueError(
                "Adapters require an evaluation model and triton/native backend"
            )
        self.q = q
        self.validation = False
        self.calls = Counter()
        self.fixtures = {}
        self.records = []
        self.modulation = modulation
        if adaln_backend == "native":
            from torch.utils.cpp_extension import load

            source = Path(__file__).with_name("csrc") / "fusion_torch.cu"
            load(
                name="gr00t_exact_fusion",
                sources=[str(source)],
                extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false", "--ftz=false"],
                is_python_module=False,
            )
            self.modulation = torch.ops.gr00t_exact_fusion.adaln_modulation
        for name, m in policy.model.named_modules():
            kind = type(m).__name__
            if kind in (
                "Qwen3VLVisionAttention",
                "Qwen3VLTextAttention",
                "Qwen3VLTextRMSNorm",
                "AdaLayerNorm",
            ):
                self.records.append((name, m, m.forward, "forward" in m.__dict__, kind))
        assert self.records

    def checked(self, kind, reference, candidate, args):
        out = candidate(*args)
        if self.validation:
            assert same(out, reference(*args)), kind
            self.calls[kind] += 1
            if kind not in self.fixtures:
                self.fixtures[kind] = tuple(
                    a.detach().clone() if torch.is_tensor(a) else a for a in args
                )
        return out

    def vision(self, *args):
        return self.checked(
            "vision_rope", self.q.apply_rotary_pos_emb_vision, ops.vision_rope, args
        )

    def text(self, *args, **kwargs):
        if kwargs:

            def ref(*a):
                return self.q.apply_rotary_pos_emb(*a, **kwargs)

            def cand(*a):
                return ops.text_rope(*a, **kwargs)

            return self.checked("text_rope", ref, cand, args)
        return self.checked(
            "text_rope", self.q.apply_rotary_pos_emb, ops.text_rope, args
        )

    def set(self, mode):
        assert mode in ("baseline", "rope_rms", "all")
        for name, m, original, had, kind in self.records:
            if had:
                m.forward = original
            elif "forward" in m.__dict__:
                del m.forward
            if mode == "baseline":
                continue
            if kind.endswith("Attention"):
                fn = clone_forward(
                    original.__func__,
                    {
                        "apply_rotary_pos_emb_vision": self.vision,
                        "apply_rotary_pos_emb": self.text,
                    },
                )
                m.forward = types.MethodType(fn, m)
            elif kind == "Qwen3VLTextRMSNorm":

                def forward(mod, x, _ref=original):
                    return self.checked(
                        "rmsnorm",
                        _ref,
                        lambda a: ops.rmsnorm(a, mod.weight, mod.variance_epsilon),
                        (x,),
                    )

                m.forward = types.MethodType(forward, m)
            elif mode == "all":

                def forward(mod, x, temb=None, _ref=original):
                    scale, shift = mod.linear(mod.silu(temb)).chunk(2, dim=-1)
                    normalized = mod.norm(x)
                    if scale.ndim == 3:
                        # The native specialization is shared-condition only.
                        # Reuse the same Triton arithmetic with each token as a row.
                        if scale.shape != normalized.shape:
                            raise ValueError("Per-token modulation must match [B,T,D]")
                        width = x.shape[-1]
                        out = modulation(
                            normalized.reshape(-1, 1, width),
                            scale.reshape(-1, width),
                            shift.reshape(-1, width),
                        ).reshape_as(x)
                    else:
                        out = self.modulation(normalized, scale, shift)
                    if self.validation:
                        assert same(out, _ref(x, temb=temb)), "adaln"
                        self.calls["adaln"] += 1
                    return out

                m.forward = types.MethodType(forward, m)

    def benchmark_operators(self, out):
        # End-to-end paired measurements determine promotion. Fixtures retained
        # for later isolated kernel tuning; no fabricated operator speedups.
        (out / "operator_validation.json").write_text(
            json.dumps(
                {
                    "calls": dict(self.calls),
                    "fixtures": list(self.fixtures),
                    "operator_speedup": "not measured",
                },
                indent=2,
            )
        )

    def close(self):
        self.set("baseline")
