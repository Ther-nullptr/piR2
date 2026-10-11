"""GR00T B1/LIBERO connections to the reference integer and graph implementations.

Reuses robotics-the-speedup-paradox revision
239b4a3ef2268048571c9f508ad5700f98398c2f without modifying its kernels.
"""

import torch
from robotics_kernels.ampere_ada.modulation import (
    PackedActivations,
    prepare_modulation,
    prepare_norm_modulation,
    prepare_residual_norm_modulation,
)

from coexecution.groot_graph import DitGraphs
from coexecution.quantization import PreparedLinear

__all__ = ["AdditionalConnections", "DitGraphs", "packed_input"]


def packed_input(packed, shape):
    result = PackedActivations({packed.bits: packed}).with_leading_shape(shape[:-1])
    result.shape = shape  # Attention reads the batch size before Q projection.
    return result


class AdditionalConnections:
    def __init__(self, policy, controller, categories):
        self.controller = controller
        self.categories = categories
        self.norms = [
            b.norm1 for b in policy.model.action_head.model.transformer_blocks
        ]
        if self.norms and controller is not None:
            quantized = {id(module) for module, _, _ in controller.sites.values()}
            self.norms = [
                block.norm1
                for block in policy.model.action_head.model.transformer_blocks
                if all(
                    id(getattr(block.attn1, name)) in quantized
                    for name in (
                        ("to_q",)
                        if block.attn1.is_cross_attention
                        else ("to_q", "to_k", "to_v")
                    )
                )
            ]
        self.all_blocks = policy.model.action_head.model.transformer_blocks
        self.blocks, self.saved_blocks = [], []
        self.saved = [
            (m, m.forward, "forward" in m.__dict__)
            for m in [*(m for m, _ in categories.values()), *self.norms]
        ]
        self.linears, self.quant = {}, {8: {}, 4: {}}
        for name, (module, category) in categories.items():
            if (
                module.W.ndim != 3
                or module.b.shape != (module.W.shape[0], module.W.shape[2])
                or not isinstance(category, int)
                or not 0 <= category < module.W.shape[0]
            ):
                raise ValueError(f"Invalid fixed category projection: {name}")
            # This adapter supports one explicit B1 embodiment, checked at each call.
            linear = torch.nn.Linear(
                module.W.shape[1],
                module.W.shape[2],
                device="meta",
                dtype=torch.bfloat16,
            )
            linear.weight = torch.nn.Parameter(
                module.W[category].t().contiguous(), requires_grad=False
            )
            linear.bias = torch.nn.Parameter(module.b[category], requires_grad=False)
            self.linears[name] = linear
            for bits in (8, 4):
                self.quant[bits][name] = PreparedLinear.from_linear(
                    linear, bits=bits, pack_reuse=True, biasless_epilogue=True
                )

    def set(
        self, mode, *, expanded=True, fuse=True, norm_quant=False, residual_quant=False
    ):
        self.close()
        if (norm_quant or residual_quant) and (not fuse or mode == "bf16"):
            raise ValueError(
                "Norm quantization fusion requires fusion and integer precision"
            )
        if residual_quant and not norm_quant:
            raise ValueError("Residual fusion requires norm quantization fusion")
        if mode == "bf16":
            return
        bits = {"w8a8": 8, "w4a4": 4}[mode]
        if residual_quant:
            quantized = {id(module) for module, _, _ in self.controller.sites.values()}
            self.blocks = [
                b for b in self.all_blocks if id(b.ff.net[0].proj) in quantized
            ]
        if norm_quant:
            if not self.norms:
                raise ValueError(
                    "Norm fusion selected no fully quantized attention input"
                )
            for norm in [
                *(n.norm for n in self.norms),
                *(b.norm3 for b in self.blocks if residual_quant),
            ]:
                if not isinstance(norm, torch.nn.LayerNorm) or norm.elementwise_affine:
                    raise ValueError("Norm fusion requires non-affine LayerNorm")
        if residual_quant and (
            not self.blocks or any(b.pos_embed is not None for b in self.blocks)
        ):
            raise ValueError(
                "Residual fusion requires quantized FFN inputs without position embeddings"
            )
        if expanded:
            for name, (module, category) in self.categories.items():
                q = self.quant[bits][name]

                def forward(x, cat_ids, q=q, category=category):
                    if (
                        x.shape[0] != 1
                        or cat_ids.numel() != 1
                        or cat_ids.dtype not in (torch.int32, torch.int64)
                    ):
                        raise ValueError(
                            "Fixed category quantization requires B1 and one integer category ID"
                        )
                    torch._assert_async(
                        cat_ids.reshape(-1)[0] == category,
                        f"Fixed category quantization requires category {category}",
                    )
                    return self.controller.dispatch(q, x)

                module.forward = forward
        if fuse:
            for norm in self.norms:

                def forward(x, temb=None, norm=norm):
                    scale, shift = norm.linear(norm.silu(temb)).chunk(2, dim=-1)
                    normalized = x if norm_quant else norm.norm(x)
                    prepare = (
                        prepare_norm_modulation if norm_quant else prepare_modulation
                    )
                    options = {"eps": norm.norm.eps} if norm_quant else {}
                    if scale.ndim == 3:
                        width = x.shape[-1]
                        packed = prepare(
                            normalized.reshape(-1, 1, width),
                            scale.reshape(-1, 1, width),
                            shift.reshape(-1, 1, width),
                            bits,
                            **options,
                        )
                    else:
                        packed = prepare(
                            normalized, scale[:, None], shift[:, None], bits, **options
                        )
                    return packed_input(packed, x.shape)

                norm.forward = forward
        if residual_quant:
            self._residual_blocks(bits)

    def _residual_blocks(self, bits):
        from gr00t.model.modules.dit import _sdpa_context

        for block in self.blocks:
            self.saved_blocks.append(
                (block, block.forward, "forward" in block.__dict__)
            )
            weight = block.ff.net[0].proj.weight
            zero = torch.zeros(
                (1, 1, block.dim), device=weight.device, dtype=weight.dtype
            )
            one = torch.ones_like(zero)

            def forward(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=None,
                block=block,
                zero=zero,
                one=one,
            ):
                x = hidden_states
                normalized = (
                    block.norm1(x, temb)
                    if block.norm_type == "ada_norm"
                    else block.norm1(x)
                )
                with _sdpa_context():
                    attention = block.attn1(
                        normalized,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask
                        if encoder_hidden_states is not None
                        else attention_mask,
                    )
                if block.final_dropout is not None:
                    attention = block.final_dropout(attention)
                # GR00T's attention residual is ungated. Unit gate and zero
                # modulation reuse the reference kernel for residual + LN + pack.
                shape = (x.shape[0], 1, x.shape[-1])
                residual, packed = prepare_residual_norm_modulation(
                    x,
                    attention,
                    one.expand(shape),
                    zero.expand(shape),
                    zero.expand(shape),
                    bits,
                    eps=block.norm3.eps,
                )
                return residual + block.ff(packed_input(packed, x.shape))

            block.forward = forward

    def close(self):
        for module, forward, had in [*self.saved, *self.saved_blocks]:
            if had:
                module.forward = forward
            elif "forward" in module.__dict__:
                del module.forward
        self.saved_blocks.clear()
        self.blocks = []
