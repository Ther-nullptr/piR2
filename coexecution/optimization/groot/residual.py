"""Reversible integer residual writers for the pinned GR00T DiT block.

The existing activation packer, integer dispatch policy, SDPA context, GELU
lookup and BF16 projection boundaries remain authoritative. Only native eval
Dropout identities may be omitted around the two residual projections.
"""

import importlib
import types

import torch
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from gr00t.model.modules.dit import AdaLayerNorm, BasicTransformerBlock
from coexecution.kernels.common.fused import prepare_gelu

from coexecution.groot_pointwise import PreparedAttention


def dispatch_residual(controller, q, x, residual, lut=None, up=None):
    """Keep controller dispatch precedence and the logical output row shape.

    Valid unsupported layouts use the unchanged dispatch followed by native
    BF16 addition. Eligible calls prepare at most one activation pack and pass
    only a contiguous aligned physical residual matrix to the existing writer.
    No activation, residual or output buffers are retained or modified.
    """
    candidate = getattr(controller, "candidates", None)
    if candidate is None:
        return controller.dispatch(q, x, lut, up) + residual
    return candidate.residual(q, x, residual, lut, up)


def _has_hooks(module):
    return bool(module._forward_pre_hooks or module._forward_hooks)


def _identity_dropout(module):
    return (
        type(module) is torch.nn.Dropout
        and not module.training
        and "forward" not in module.__dict__
        and not _has_hooks(module)
    )


def _supported(block):
    if any(module.training for module in block.modules()):
        return False
    if block.norm_type not in ("ada_norm", "layer_norm"):
        return False
    norm = block.norm1
    if block.norm_type == "ada_norm":
        if type(norm) is not AdaLayerNorm or type(norm.norm) is not torch.nn.LayerNorm:
            return False
    elif type(norm) is not torch.nn.LayerNorm:
        return False
    if type(block.norm3) is not torch.nn.LayerNorm:
        return False
    if (
        block.pos_embed is not None
        and type(block.pos_embed) is not SinusoidalPositionalEmbedding
    ):
        return False
    attn = block.attn1
    if not (
        type(attn.processor) is PreparedAttention
        and attn.rescale_output_factor == 1
        and not attn.residual_connection
        and attn.spatial_norm is None
        and attn.group_norm is None
        and not attn.norm_cross
        and attn.norm_q is None
        and attn.norm_k is None
        and attn.added_kv_proj_dim is None
        and attn.to_q.out_features == attn.to_k.out_features == attn.to_v.out_features
        and attn.scale == (attn.to_k.out_features // attn.heads) ** -0.5
        and len(attn.to_out) == 2
        and isinstance(attn.to_out[0], torch.nn.Linear)
        and _identity_dropout(attn.to_out[1])
        and "forward" not in attn.__dict__
        and not _has_hooks(attn)
        and not _has_hooks(attn.to_out[0])
    ):
        return False
    if block.final_dropout is not None and not _identity_dropout(block.final_dropout):
        return False
    ff = block.ff
    return (
        type(ff) is FeedForward
        and len(ff.net) in (3, 4)
        and type(ff.net[0]) is GELU
        and ff.net[0].approximate in ("none", "tanh")
        and _identity_dropout(ff.net[1])
        and isinstance(ff.net[2], torch.nn.Linear)
        and "forward" not in ff.__dict__
        and not _has_hooks(ff)
        and not _has_hooks(ff.net[2])
        and (len(ff.net) == 3 or _identity_dropout(ff.net[3]))
    )


class DitResidualAdapters:
    """Replace selected actual GR00T block instances, never the shared class."""

    def __init__(self, controller, pointwise):
        self.controller = controller
        self.pointwise = pointwise
        selected = {id(site[0]): name for name, site in controller.sites.items()}
        self.entries = []
        for index, block in enumerate(pointwise.blocks):
            if type(block) is not BasicTransformerBlock:
                continue
            outputs = getattr(getattr(block, "attn1", None), "to_out", None)
            net = getattr(getattr(block, "ff", None), "net", None)
            if not (
                isinstance(outputs, torch.nn.ModuleList)
                and len(outputs) == 2
                and isinstance(net, torch.nn.ModuleList)
                and len(net) in (3, 4)
            ):
                continue
            output = selected.get(id(block.attn1.to_out[0]))
            down = selected.get(id(block.ff.net[2]))
            if output is not None and down is not None:
                self.entries.append((index, block, output, down))
        self.definition = importlib.import_module(BasicTransformerBlock.__module__)
        self.saved = []
        self.bits = None

    def set(self, bits=None):
        self.close()
        if bits is None:
            return
        if type(bits) is not int or bits not in (4, 8):
            raise ValueError("Residual precision must be 4 or 8")
        if self.controller.mode != {4: "w4a4", 8: "w8a8"}[bits]:
            raise ValueError("Residual precision requires the matching integer mode")
        if not self.entries:
            raise ValueError("Residual fusion selected no complete GR00T block")
        if any(not _supported(block) for _, block, _, _ in self.entries):
            raise ValueError(
                "Residual fusion requires eval GR00T blocks, PreparedAttention and native Dropout"
            )
        projections = self.controller.quant[bits]
        if any(
            output not in projections or down not in projections
            for _, _, output, down in self.entries
        ):
            raise ValueError(
                "Residual fusion requires both selected integer projections"
            )
        approximates = {
            (id(activation), down): approximate
            for activation, attr, down, approximate in self.controller.activations
            if attr == "gelu"
        }
        if any(
            approximates.get((id(block.ff.net[0]), down)) != block.ff.net[0].approximate
            for _, block, _, down in self.entries
        ):
            raise ValueError(
                "Residual fusion requires the controller's captured GELU approximation"
            )
        self.saved = [
            (block, block.forward, "forward" in block.__dict__)
            for _, block, _, _ in self.entries
        ]
        try:
            for (_, block, output, down), (_, original, _) in zip(
                self.entries, self.saved
            ):
                self._install(
                    block,
                    original,
                    bits,
                    projections[output],
                    projections[down],
                    approximates[(id(block.ff.net[0]), down)],
                )
            self.bits = bits
        except BaseException:
            self.close()
            raise

    def _install(self, block, original, bits, output, down, approximate):
        attn, ff = block.attn1, block.ff
        norm1, norm3, position = block.norm1, block.norm3, block.pos_embed
        net = tuple(ff.net)
        processor = attn.processor
        norm_type = block.norm_type
        output_module, output_dropout = attn.to_out
        final_dropout = block.final_dropout
        output_forward, down_forward = output_module.forward, net[2].forward
        attention_forward, ffn_forward = attn.forward, ff.forward
        gelu = net[0].gelu
        dropouts = (output_dropout, net[1], *net[3:])
        if final_dropout is not None:
            dropouts += (final_dropout,)
        mode = {4: "w4a4", 8: "w8a8"}[bits]

        def forward(
            module,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            temb=None,
        ):
            # Structural changes and unsupported runtime layouts retain the
            # complete original block, including its dropout and error behavior.
            if not (
                self.controller.mode == mode
                and not module.training
                and not attn.training
                and not ff.training
                and not norm1.training
                and not norm3.training
                and module.norm_type == norm_type
                and module.attn1 is attn
                and module.ff is ff
                and module.norm1 is norm1
                and module.norm3 is norm3
                and module.pos_embed is position
                and attn.processor is processor
                and "forward" not in attn.__dict__
                and "forward" not in ff.__dict__
                and attn.forward == attention_forward
                and ff.forward == ffn_forward
                and tuple(ff.net) == net
                and tuple(attn.to_out) == (output_module, output_dropout)
                and module.final_dropout is final_dropout
                and output_module.forward is output_forward
                and net[2].forward is down_forward
                and net[0].gelu is gelu
                and net[0].approximate == approximate
                and all(_identity_dropout(dropout) for dropout in dropouts)
                and not _has_hooks(attn)
                and not _has_hooks(ff)
                and not _has_hooks(output_module)
                and not _has_hooks(net[2])
                and isinstance(hidden_states, torch.Tensor)
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.ndim == 3
                and hidden_states.is_contiguous()
                and not hidden_states.requires_grad
                and hidden_states.shape[-1] == output.out_features == down.out_features
                and hidden_states.device
                == output.packed_weight.device
                == down.packed_weight.device
            ):
                return original(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            normalized = (
                norm1(hidden_states, temb)
                if norm_type == "ada_norm"
                else norm1(hidden_states)
            )
            if position is not None:
                normalized = position(normalized)
            with self.definition._sdpa_context():
                attended = processor.attend(
                    attn,
                    normalized,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=(
                        encoder_attention_mask
                        if encoder_hidden_states is not None
                        else attention_mask
                    ),
                )
            hidden = dispatch_residual(self.controller, output, attended, hidden_states)
            activated = net[1](net[0](norm3(hidden)))
            lut = prepare_gelu(activated.device, activated.dtype, approximate)
            return dispatch_residual(self.controller, down, activated, hidden, lut)

        block.forward = types.MethodType(forward, block)

    def evidence(self):
        return {
            "bits": self.bits,
            "blocks": [index for index, _, _, _ in self.entries],
            "projections": [
                {"attention_output": output, "ffn_down": down}
                for _, _, output, down in self.entries
            ],
            "scope": "instance-local; fresh activation and residual inputs per call",
            "rounding": "native projection BF16 boundary before BF16 residual addition",
        }

    def close(self):
        for module, forward, had in self.saved:
            if had:
                module.forward = forward
            elif "forward" in module.__dict__:
                del module.forward
        self.saved.clear()
        self.bits = None
