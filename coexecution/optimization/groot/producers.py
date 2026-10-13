"""Reversible native norm and activation producer candidates for GR00T."""

import torch

from coexecution.kernels.ampere_ada.modulation import PackedActivations
from coexecution.kernels.ampere_ada.producers import (
    prepare_final_modulation,
    prepare_rmsnorm,
)
from coexecution.optimization.backends.integer.linear import prepare


class QuantizationProducers:
    """Install only before complete, selected integer consumer groups."""

    def __init__(self, policy, controller, pointwise):
        self.pointwise = pointwise
        self.original_final = None
        self.final_installed = False
        quantized = {id(site[0]) for site in controller.sites.values()}
        self.norms = []
        for name, layer in policy.model.named_modules():
            if type(layer).__name__ != "Qwen3VLTextDecoderLayer":
                continue
            if all(
                id(getattr(layer.self_attn, attr)) in quantized
                for attr in ("q_proj", "k_proj", "v_proj")
            ):
                self.norms.append((name + ".input_layernorm", layer.input_layernorm))
            if all(
                id(getattr(layer.mlp, attr)) in quantized
                for attr in ("gate_proj", "up_proj")
            ):
                self.norms.append(
                    (name + ".post_attention_layernorm", layer.post_attention_layernorm)
                )
        self.saved = []
        self.installed_norms = []
        self.final_selected = (
            pointwise is not None and id(pointwise.model.proj_out_2) in quantized
        )
        self.bits = None

    def set(self, bits=None, *, rmsnorm=True, final_modulation=True):
        self.close()
        if bits is None:
            return
        if bits not in (4, 8):
            raise ValueError("Producer precision must be 4 or 8")
        install_final = self.final_selected and final_modulation
        norms = self.norms if rmsnorm else []
        if not norms and not install_final:
            raise ValueError("Producer fusion selected no complete integer consumers")
        self.saved = [(m, m.forward, "forward" in m.__dict__) for _, m in norms]
        try:
            for _, norm in norms:

                def forward(x, module=norm, bits=bits):
                    packed = prepare_rmsnorm(
                        x, module.weight, module.variance_epsilon, bits
                    )
                    result = PackedActivations({bits: packed})
                    result.shape = x.shape
                    return result

                norm.forward = forward
            if install_final:
                self.original_final = self.pointwise.final_projection_prepare
                self.final_installed = True
                self.pointwise.final_projection_prepare = lambda x, scale, shift: (
                    prepare_final_modulation(x, scale, shift, bits)
                )
            self.installed_norms = list(norms)
            self.bits = bits
        except BaseException:
            self.close()
            raise

    def evidence(self):
        return {
            "bits": self.bits,
            "rmsnorm_sites": [n for n, _ in self.installed_norms],
            "final_modulation": self.final_installed,
            "norm_statistics": "exact native-order FP32 square mean; Ada fusion where eligible",
        }

    def close(self):
        for module, forward, had in self.saved:
            if had:
                module.forward = forward
            elif "forward" in module.__dict__:
                del module.forward
        self.saved.clear()
        self.installed_norms.clear()
        if self.final_installed:
            self.pointwise.final_projection_prepare = self.original_final
            self.final_installed = False
            self.original_final = None
        self.bits = None


class SharedContextQuantization:
    """One fresh activation pack per DiT invocation, without a feature cache."""

    def __init__(self, pointwise, controller):
        self.pointwise = pointwise
        self.original = pointwise.context_quantization_prepare
        self.bits = None
        self.blocks = [
            (i, block.attn1)
            for i, block in enumerate(pointwise.blocks)
            if block.attn1.is_cross_attention
        ]
        selected = {id(site[0]) for site in controller.sites.values()}
        self.complete = bool(self.blocks) and all(
            id(getattr(attn, attr)) in selected
            for _, attn in self.blocks
            for attr in ("to_k", "to_v")
        )
        widths = {
            getattr(attn, attr).in_features
            for _, attn in self.blocks
            for attr in ("to_k", "to_v")
        }
        self.width = next(iter(widths)) if len(widths) == 1 else None

    def set(self, bits=None):
        self.close()
        if bits is None:
            return
        if bits not in (8, 4):
            raise ValueError("Context quantization requires bits 8 or 4")
        if not self.complete or self.width is None:
            raise ValueError(
                "Context sharing requires complete cross-attention K/V consumers"
            )

        def prepare_context(context):
            if not (
                context.is_cuda
                and context.dtype == torch.bfloat16
                and context.ndim == 3
                and context.is_contiguous()
                and context.shape[-1] == self.width
            ):
                raise ValueError(
                    "Shared context requires matching contiguous CUDA BF16 B,S,D"
                )
            packed = PackedActivations({bits: prepare(context, bits)})
            packed.shape = context.shape
            return packed

        self.pointwise.context_quantization_prepare = prepare_context
        self.bits = bits

    def evidence(self):
        return {
            "bits": self.bits,
            "cross_attention_blocks": [i for i, _ in self.blocks],
            "scope": "fresh pack per DiT call; no reuse across observations or plans",
        }

    def close(self):
        self.pointwise.context_quantization_prepare = self.original
        self.bits = None


class LayerNormQuantization:
    """Fresh native-order LayerNorm packs for complete selected DiT consumers."""

    def __init__(self, controller, pointwise):
        self.pointwise = pointwise
        selected = {id(site[0]) for site in controller.sites.values()}
        self.attention, self.ffn = [], []
        for index, block in enumerate(pointwise.blocks):
            query_attrs = (
                ("to_q",)
                if block.attn1.is_cross_attention
                else ("to_q", "to_k", "to_v")
            )
            if (
                block.pos_embed is None
                and self.supported_norm(getattr(block.norm1, "norm", None))
                and all(
                    id(getattr(block.attn1, attr)) in selected for attr in query_attrs
                )
            ):
                self.attention.append((index, block.norm1))
            if (
                self.supported_norm(block.norm3)
                and id(block.ff.net[0].proj) in selected
            ):
                self.ffn.append((index, block.norm3))
        self.final_selected = id(
            pointwise.model.proj_out_2
        ) in selected and self.supported_norm(pointwise.model.norm_out)
        self.original_final = None
        self.final_installed = False
        self.saved = []
        self.bits, self.mode = None, "none"

    @staticmethod
    def supported_norm(norm):
        return (
            isinstance(norm, torch.nn.LayerNorm)
            and len(norm.normalized_shape) == 1
            and not norm.elementwise_affine
        )

    @staticmethod
    def wrap(packed, shape):
        result = PackedActivations({packed.bits: packed})
        result.shape = shape
        return result

    def set(self, bits=None, mode="none"):
        self.close()
        if mode == "none":
            return
        if bits not in (8, 4) or mode not in ("attention", "all"):
            raise ValueError(
                "LayerNorm packing requires bits4/8 and attention/all scope"
            )
        if not self.attention and not (
            mode == "all" and (self.ffn or self.final_selected)
        ):
            raise ValueError(
                "LayerNorm packing selected no complete non-affine consumers"
            )
        from coexecution.kernels.ampere_ada.norm import prepare_layernorm

        entries = [norm for _, norm in self.attention]
        if mode == "all":
            entries += [norm for _, norm in self.ffn]
        self.saved = [
            (module, module.forward, "forward" in module.__dict__) for module in entries
        ]
        try:
            for _, norm in self.attention:

                def forward(x, temb=None, norm=norm):
                    scale, shift = norm.linear(norm.silu(temb)).chunk(2, dim=-1)
                    if scale.ndim == 2:
                        scale, shift = scale[:, None], shift[:, None]
                    packed = prepare_layernorm(x, bits, norm.norm.eps, scale, shift)
                    return self.wrap(packed, x.shape)

                norm.forward = forward
            if mode == "all":
                for _, norm in self.ffn:

                    def forward(x, norm=norm):
                        return self.wrap(prepare_layernorm(x, bits, norm.eps), x.shape)

                    norm.forward = forward
                if self.final_selected:
                    self.original_final = self.pointwise.final_norm_prepare
                    self.final_installed = True
                    self.pointwise.final_norm_prepare = lambda x, scale, shift: (
                        self.wrap(
                            prepare_layernorm(
                                x, bits, self.pointwise.model.norm_out.eps, scale, shift
                            ),
                            x.shape,
                        )
                    )
            self.bits, self.mode = bits, mode
        except BaseException:
            self.close()
            raise

    def evidence(self):
        return {
            "bits": self.bits,
            "mode": self.mode,
            "attention_blocks": [index for index, _ in self.attention],
            "ffn_blocks": [index for index, _ in self.ffn]
            if self.mode == "all"
            else [],
            "final_norm": self.final_selected and self.mode == "all",
            "statistics": "PyTorch2.7.1 vectorized CUDA Welford order; packed BF16 boundaries retained",
            "scope": "fresh LayerNorm and activation pack per call; no feature cache",
        }

    def close(self):
        for module, forward, had in self.saved:
            if had:
                module.forward = forward
            elif "forward" in module.__dict__:
                del module.forward
        self.saved.clear()
        if self.final_installed:
            self.pointwise.final_norm_prepare = self.original_final
            self.final_installed = False
            self.original_final = None
        self.bits, self.mode = None, "none"
