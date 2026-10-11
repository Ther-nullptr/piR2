"""DiT normalization/residual producers for true floating packed projections.

The streaming block connection follows quantization_connections.py. Floating
buffers, scale layouts and producer kernels remain separate from integer data.
"""

import torch

from coexecution.floating_packing import prepare_norm


class FloatingConnections:
    def __init__(self, policy, controller, *, residual=False):
        self.saved = []
        self.norms, self.blocks = [], []
        precision = controller.precision
        selected = {id(module) for module, _, _ in controller.sites.values()}
        for block in policy.model.action_head.model.transformer_blocks:
            names = (
                ("to_q",)
                if block.attn1.is_cross_attention
                else ("to_q", "to_k", "to_v")
            )
            if all(id(getattr(block.attn1, n)) in selected for n in names):
                if block.norm_type != "ada_norm":
                    raise ValueError("Floating norm fusion requires AdaLayerNorm")
                if block.pos_embed is not None:
                    raise ValueError(
                        "Floating norm fusion requires no position embeddings"
                    )
                self.norms.append(block.norm1)
            if residual and id(block.ff.net[0].proj) in selected:
                if block.pos_embed is not None:
                    raise ValueError(
                        "Floating residual fusion requires no position embeddings"
                    )
                self.blocks.append(block)
        if not self.norms or (residual and not self.blocks):
            raise ValueError(
                "Floating norm fusion selected no compatible DiT projections"
            )
        for norm in [*(m.norm for m in self.norms), *(b.norm3 for b in self.blocks)]:
            if not isinstance(norm, torch.nn.LayerNorm) or norm.elementwise_affine:
                raise ValueError("Floating norm fusion requires non-affine LayerNorm")
        try:
            for norm in self.norms:

                def forward(x, temb=None, norm=norm):
                    scale, shift = norm.linear(norm.silu(temb)).chunk(2, dim=-1)
                    packed, _ = prepare_norm(
                        x, precision, eps=norm.norm.eps, scale=scale, shift=shift
                    )
                    return packed

                self._patch(norm, forward)
            if residual:
                from gr00t.model.modules.dit import _sdpa_context

                for block in self.blocks:

                    def forward(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        temb=None,
                        block=block,
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
                        packed, summed = prepare_norm(
                            x,
                            precision,
                            eps=block.norm3.eps,
                            residual=attention.contiguous(),
                        )
                        return summed + block.ff(packed)

                    self._patch(block, forward)
        except BaseException:
            self.close()
            raise
        controller.coverage["norm_quantization_fusion"] = {
            "attention_producers": len(self.norms),
            "residual_ffn_producers": len(self.blocks),
            "reduction": "FP32 Triton LayerNorm; differs from native reduction order",
            "rounding": "BF16 norm, modulation and residual boundaries retained",
        }

    def _patch(self, module, forward):
        self.saved.append((module, module.forward, "forward" in module.__dict__))
        module.forward = forward

    def close(self):
        for module, forward, had in reversed(self.saved):
            if had:
                module.forward = forward
            else:
                del module.forward
        self.saved.clear()
