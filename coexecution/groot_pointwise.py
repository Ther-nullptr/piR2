"""BF16 inference fusion for the measured GR00T layout; no request cache.

The native-activation lookup follows Speedup Paradox
robotics_kernels/common/fused.py (239b4a3); see THIRD_PARTY_NOTICES.md.
"""

import types
from collections import Counter

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from coexecution.fused_ops import _launch
from coexecution.groot_fusion import same


@triton.jit
def _silu_gate(G, U, LUT, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < N, 0)
    activated = tl.load(LUT + g.to(tl.uint16, bitcast=True).to(tl.int32)).to(tl.float32)
    u = tl.load(U + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, activated * u, i < N)


@triton.jit
def _prepare_condition(
    T,
    IMAGE,
    VALID,
    LUT,
    A,
    IM,
    NM,
    B: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = tl.load(T + i, i < B * D, 0)
    a = tl.load(LUT + t.to(tl.uint16, bitcast=True).to(tl.int32))
    tl.store(A + i, a, i < B * D)
    b = i // (H * P)
    s = i % P
    image = tl.load(IMAGE + b * S + s, (b < B) & (s < S), 0)
    valid = tl.load(VALID + b * S + s, (b < B) & (s < S), 0)
    im = tl.where(image & valid, 0.0, float("-inf"))
    nm = tl.where((~image) & valid, 0.0, float("-inf"))
    tl.store(IM + i, tl.where(s < S, im, 0.0), b < B)
    tl.store(NM + i, tl.where(s < S, nm, 0.0), b < B)


class PreparedAttention:
    """Measured 3D SDPA path, with prepared additive masks and unit rescale."""

    def __call__(
        self, attn, x, encoder_hidden_states=None, attention_mask=None, **kwargs
    ):
        context = x if encoder_hidden_states is None else encoder_hidden_states
        if (
            len(x.shape) != 3
            or len(context.shape) != 3
            or x.shape[0] != context.shape[0]
            or kwargs
        ):
            raise ValueError(
                "PreparedAttention supports 3D SDPA without processor extras"
            )
        b = x.shape[0]
        h = attn.heads
        q = attn.to_q(x)
        k = attn.to_k(context)
        v = attn.to_v(context)
        d = k.shape[-1] // h
        q = q.view(b, -1, h, d).transpose(1, 2)
        k = k.view(b, -1, h, d).transpose(1, 2)
        v = v.view(b, -1, h, d).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        y = y.transpose(1, 2).reshape(b, -1, h * d)
        return attn.to_out[1](attn.to_out[0](y))


class PointwiseFusion:
    def __init__(self, policy):
        self.model = policy.model.action_head.model
        if type(self.model).__name__ != "AlternateVLDiT":
            raise ValueError("Pointwise fusion supports GR00T AlternateVLDiT")
        self.blocks = list(self.model.transformer_blocks)
        self.text = [
            m for m in policy.model.modules() if type(m).__name__ == "Qwen3VLTextMLP"
        ]
        self.validation = False
        self.calls = Counter()
        device = next(policy.model.parameters()).device
        values = (
            torch.arange(65536, device=device, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        self.lut = F.silu(values)
        self.processors = [b.attn1.processor for b in self.blocks]
        # Scope checks once at installation, rather than in every kernel call.
        if policy.model.training or not self.model.config.interleave_self_attention:
            raise ValueError(
                "Pointwise fusion requires evaluation and interleaved self-attention"
            )
        if not all(
            a.rescale_output_factor == 1
            and not a.residual_connection
            and a.spatial_norm is None
            and a.group_norm is None
            and not a.norm_cross
            and a.norm_q is None
            and a.norm_k is None
            and type(a.processor).__name__ == "AttnProcessor2_0"
            and a.added_kv_proj_dim is None
            and a.to_q.out_features == a.to_k.out_features == a.to_v.out_features
            and a.scale == (a.to_k.out_features // a.heads) ** -0.5
            for a in (b.attn1 for b in self.blocks)
        ):
            raise ValueError(
                "PreparedAttention requires standard SDPA with equal Q/K/V heads"
            )
        if len({b.attn1.heads for b in self.blocks}) != 1:
            raise ValueError(
                "Prepared masks require identical head counts across DiT blocks"
            )
        self.saved = [
            (m, m.forward, "forward" in m.__dict__)
            for m in [self.model, *self.text, *(b.norm1.silu for b in self.blocks)]
        ]

    def gate(self, g, u):
        if not (
            g.shape == u.shape
            and g.dtype == u.dtype == torch.bfloat16
            and g.is_contiguous()
            and u.is_contiguous()
            and g.device == u.device == self.lut.device
        ):
            raise ValueError(
                "SiLU gate requires contiguous CUDA BF16 tensors with equal shape"
            )
        y = torch.empty_like(g)
        _launch(
            _silu_gate,
            (triton.cdiv(g.numel(), 256),),
            (g, u, self.lut, y),
            (g.numel(), 256),
        )
        if self.validation:
            assert same(y, F.silu(g) * u)
            self.calls["silu_gate"] += 1
        return y

    def prepare(self, temb, image, valid):
        if not (
            temb.ndim in (2, 3)
            and temb.dtype == torch.bfloat16
            and temb.is_contiguous()
            and image is not None
            and valid is not None
            and image.ndim == valid.ndim == 2
            and image.shape == valid.shape
            and image.shape[0] == temb.shape[0]
            and image.shape[1] > 0
            and image.dtype == valid.dtype == torch.bool
            and image.is_contiguous()
            and valid.is_contiguous()
            and temb.device == image.device == valid.device == self.lut.device
        ):
            raise ValueError(
                "Expected contiguous BF16 [B,D]/[B,T,D] and boolean [B,S] masks"
            )
        # The kernel's activation span is independent of the attention-mask batch.
        b = temb.shape[0]
        d = temb.numel() // b
        s = image.shape[-1]
        h = self.blocks[0].attn1.heads
        p = triton.cdiv(s, 8) * 8
        a = torch.empty_like(temb)
        im = torch.empty((b, h, 1, p), device=temb.device, dtype=temb.dtype)
        nm = torch.empty_like(im)
        _launch(
            _prepare_condition,
            (triton.cdiv(max(b * d, b * h * p), 256),),
            (temb, image, valid, self.lut, a, im, nm),
            (b, d, s, h, p, 256),
        )
        if self.validation:
            assert same(a, F.silu(temb))
            for output, mask in ((im, image & valid), (nm, (~image) & valid)):
                ref = (
                    torch.where(mask, 0.0, -float("inf"))
                    .to(temb.dtype)[:, None, None, :]
                    .expand(b, h, 1, s)
                )
                assert same(output, F.pad(ref, (0, p - s)))
            self.calls["prepare_condition"] += 1
        return a, im[..., :s], nm[..., :s]

    def dit(
        self,
        mod,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        temb = mod.timestep_encoder(timestep)
        activated, im, nm = self.prepare(temb, image_mask, backbone_attention_mask)
        x = hidden_states.contiguous()
        context = encoder_hidden_states.contiguous()
        history = [x]
        for i, block in enumerate(mod.transformer_blocks):
            cross = i % 2 == 0
            mask = nm if i % (2 * mod.attend_text_every_n_blocks) == 0 else im
            x = block(
                x,
                encoder_hidden_states=context if cross else None,
                encoder_attention_mask=mask if cross else None,
                temb=activated,
            )
            history.append(x)
        shift, scale = mod.proj_out_1(activated).chunk(2, dim=-1)
        if scale.ndim == 2:
            scale, shift = scale[:, None], shift[:, None]
        x = mod.norm_out(x) * (1 + scale) + shift
        result = mod.proj_out_2(x)
        return (result, history) if return_all_hidden_states else result

    def set(self, mode):
        if mode not in ("parent", "dit", "gate", "all"):
            raise ValueError("Pointwise mode must be parent, dit, gate or all")
        for m, forward, had in self.saved:
            if had:
                m.forward = forward
            elif "forward" in m.__dict__:
                del m.forward
        for block, processor in zip(self.blocks, self.processors):
            block.attn1.set_processor(processor)
        if mode == "parent":
            return
        if mode != "dit":
            for m in self.text:

                def forward(mod, x):
                    return mod.down_proj(self.gate(mod.gate_proj(x), mod.up_proj(x)))

                m.forward = types.MethodType(forward, m)
        if mode == "gate":
            return
        self.model.forward = types.MethodType(self.dit, self.model)
        for block in self.blocks:
            block.norm1.silu.forward = lambda x: x
            block.attn1.set_processor(PreparedAttention())

    def close(self):
        self.set("parent")
