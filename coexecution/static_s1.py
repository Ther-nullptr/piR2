"""Inference adapters preserving GR00T's streaming arithmetic and RNG ordering.

Adapted from the Apache-2.0 NVIDIA GR00T N1.7 action head/AlternateVLDiT.
Only static tensor construction and repeated SiLU are hoisted. No parameters,
precision, attention algorithm, dynamic state update or random draw are changed.
"""

from collections import Counter
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from coexecution.graph_dit import GraphCall


class StaticS1:
    """Scoped adapter for an eval-only, fixed-weight AlternateVLDiT policy."""

    def __init__(self, workload, graph=False):
        self.workload = workload
        self.head = workload.model.action_head
        self.dit = self.head.model
        self.use_graph = graph
        self.restore = []
        self.constants = {}
        self.counts = Counter()
        self.graph = None
        self.temb = None
        self.activated_temb = None

    def replace(self, target, name, value):
        self.restore.append((target, name, name in vars(target), getattr(target, name)))
        setattr(target, name, value)

    @torch.inference_mode()
    def __enter__(self):
        try:
            return self._install()
        except BaseException:
            self._restore_methods()
            raise

    def _install(self):
        assert not self.head.training and self.head.config.use_alternate_vl_dit
        assert self.head.config.streaming_schedule_mode == "pir2"
        assert self.dit.config.interleave_self_attention
        self.original_streaming = self.head._streaming_inference
        for block in self.dit.transformer_blocks:
            norm = block.norm1
            assert hasattr(norm, "silu"), "Expected AdaLayerNorm in every DiT block"
            original = norm.forward

            def normalized(module, x, temb=None, _original=original):
                if temb is not self.temb:
                    self.counts["silu_fallbacks"] += 1
                    return _original(x, temb)
                modulation = module.linear(self.activated_temb)
                scale, shift = modulation.chunk(2, dim=-1)
                if scale.dim() == 3:
                    return module.norm(x) * (1 + scale) + shift
                return module.norm(x) * (1 + scale[:, None]) + shift[:, None]

            self.replace(norm, "forward", MethodType(normalized, norm))
        if self.use_graph:
            self.graph = GraphCall(self.dit_forward)
            forward = self.graph
        else:
            forward = self.dit_forward
        self.replace(self.dit, "forward", forward)
        self.replace(self.head, "_streaming_inference", self.streaming)
        # Prepare deployed schedule/embeddings before either worker starts.
        features = self.workload.initial_features["backbone_features"]
        for slide in [1, 2, 3]:
            self.prepare_constants(features, slide)
        torch.cuda.synchronize()
        return self

    def __exit__(self, *exception):
        try:
            torch.cuda.synchronize()
        finally:
            self._restore_methods()

    def _restore_methods(self):
        for target, name, existed, original in reversed(self.restore):
            if existed:
                setattr(target, name, original)
            else:
                delattr(target, name)
        self.restore.clear()

    def dit_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        model = self.dit
        assert image_mask is not None
        temb = model.timestep_encoder(timestep)
        self.temb = temb
        self.activated_temb = F.silu(temb)
        self.counts["shared_silu_eager_calls"] += 1
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask
        all_hidden_states = [hidden_states]
        for index, block in enumerate(model.transformer_blocks):
            if index % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                mask = (
                    non_image_attention_mask
                    if index % (2 * model.attend_text_every_n_blocks) == 0
                    else image_attention_mask
                )
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)
        shift, scale = model.proj_out_1(self.activated_temb).chunk(2, dim=-1)
        if scale.dim() == 3:
            hidden_states = model.norm_out(hidden_states) * (1 + scale) + shift
        else:
            hidden_states = (
                model.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            )
        output = model.proj_out_2(hidden_states)
        return (output, all_hidden_states) if return_all_hidden_states else output

    @torch.inference_mode()
    def prepare_constants(self, vl_embeds, slide):
        head, config = self.head, self.head.config
        batch = vl_embeds.shape[0]
        horizon = config.action_horizon
        dtype, device = vl_embeds.dtype, vl_embeds.device
        key = (batch, horizon, slide, dtype, device, config.noise_s)
        if key in self.constants:
            return self.constants[key]
        ramp_width = max(horizon - 2 * slide, 1)
        positions = torch.arange(horizon, device=device, dtype=dtype)
        ramp = config.noise_s * (1.0 - (positions - slide + 0.5) / ramp_width)
        target = torch.where(
            positions < slide,
            torch.tensor(config.noise_s, device=device, dtype=dtype),
            torch.where(
                positions >= horizon - slide,
                torch.tensor(0.0, device=device, dtype=dtype),
                ramp,
            ),
        )
        target_per_position = torch.cat(
            [
                torch.full((slide,), config.noise_s, device=device, dtype=dtype),
                target[: horizon - slide],
            ]
        )
        constants = {
            "target": target_per_position,
            "state_t": torch.zeros(batch, 1, dtype=torch.long, device=device),
            "new_t": torch.zeros(batch, slide, dtype=dtype, device=device),
            "positions": None,
            "delays": {},
        }
        if config.add_pos_embed:
            pos_ids = torch.arange(horizon, dtype=torch.long, device=device)
            constants["positions"] = head.position_embedding(pos_ids).unsqueeze(0)
        if hasattr(head, "delay_embedding"):
            for delay in range(config.image_delay_max + 1):
                delay_t = torch.tensor([delay], dtype=torch.long, device=device)
                constants["delays"][delay] = (
                    head.delay_embedding(delay_t).expand(batch, -1).unsqueeze(1)
                )
        self.constants[key] = constants
        self.counts["constant_builds"] += 1
        return constants

    @torch.no_grad()
    def streaming(
        self, vl_embeds, state_features, embodiment_id, backbone_output, options=None
    ):
        head, config = self.head, self.head.config
        if options is None or "slide_steps" not in options or "inpaint" in options:
            self.counts["streaming_fallbacks"] += 1
            return self.original_streaming(
                vl_embeds, state_features, embodiment_id, backbone_output, options
            )
        batch = vl_embeds.shape[0]
        horizon = config.action_horizon
        buffer = head._stream_buf
        if buffer is None or buffer.shape[0] != batch or buffer.shape[1] != horizon:
            self.counts["streaming_fallbacks"] += 1
            return self.original_streaming(
                vl_embeds, state_features, embodiment_id, backbone_output, options
            )
        dtype, device = vl_embeds.dtype, vl_embeds.device
        slide = max(1, min(int(options["slide_steps"]), horizon))
        substeps = int(options.get("num_inference_steps_per_call", 1))
        constants = self.prepare_constants(vl_embeds, slide)
        dt_total = (constants["target"].unsqueeze(0) - head._stream_buf_t).clamp(min=0)
        dt_per_position = dt_total / max(substeps, 1)
        for _ in range(substeps):
            discrete_time = (
                (head._stream_buf_t / config.noise_s * config.num_timestep_buckets)
                .long()
                .clamp(0, config.num_timestep_buckets - 1)
            )
            action_features = head.action_encoder(
                head._stream_buf, discrete_time, embodiment_id
            )
            if hasattr(head, "delay_embedding") and "image_delay" in options:
                delay = max(0, min(int(options["image_delay"]), config.image_delay_max))
                action_features = action_features + constants["delays"][delay]
            if config.add_pos_embed:
                action_features = action_features + constants["positions"]
            combined = torch.cat((state_features, action_features), dim=1)
            dit_time = torch.cat([constants["state_t"], discrete_time], dim=1)
            model_output = head.model(
                hidden_states=combined,
                encoder_hidden_states=vl_embeds,
                timestep=dit_time,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
            predicted = head.action_decoder(model_output, embodiment_id)
            velocity = predicted[:, -horizon:]
            step = dt_per_position.to(velocity.dtype)
            head._stream_buf = head._stream_buf + step.unsqueeze(-1) * velocity
            head._stream_buf_t = head._stream_buf_t + step
        action_pred = head._stream_buf.clone()
        noise = torch.randn(batch, slide, head.action_dim, device=device, dtype=dtype)
        head._stream_buf = torch.cat([head._stream_buf[:, slide:], noise], dim=1)
        head._stream_buf_t = torch.cat(
            [head._stream_buf_t[:, slide:], constants["new_t"]], dim=1
        )
        self.counts["streaming_calls"] += 1
        return BatchFeature(
            {
                "action_pred": action_pred,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    def freeze(self):
        if self.graph is not None:
            self.graph.freeze()

    def evidence(self):
        result = dict(self.counts)
        result["graph"] = self.graph.evidence() if self.graph else None
        return result
