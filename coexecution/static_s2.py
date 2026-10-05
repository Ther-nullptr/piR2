# Adapted from Transformers 4.57.3 modeling_qwen3_vl.py.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. Local modifications cache static metadata;
# see licenses/TRANSFORMERS_LICENSE and THIRD_PARTY_NOTICES.md.

"""Scoped, value-checked static metadata for the resident Qwen3-VL S2 path.

The supported contract is image-only, SDPA, eval/inference, and one S2 caller.
Weights/configuration remain fixed while the scope is active. Image pixels are
never cached: patch embedding, every vision block, DeepStack, and the original
text forward execute on every call. The workload's pre-final-norm hook still
selects the released policy's exact feature tensor.

Input metadata are copied into ordinary version-tracked tensors at entry.
Their identities, versions, shape, dtype, and device are checked without GPU
reads in the hot path. Replacement/in-place changes invoke a cold full-content
signature and rebuild as needed. Inference-tensor replacements are first made
version-trackable. Mutating metadata through .data, an untracked foreign writer,
or an alias predating this scope is outside the contract.

The attention/vision arithmetic below preserves the installed Transformers
4.57.3 Qwen3-VL implementation and only replaces static metadata construction.
The dynamically resolved RoPE function preserves an outer FusionPatch choice.
"""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from types import MethodType

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

METADATA_KEYS = ("input_ids", "attention_mask", "image_grid_thw")


@dataclass
class VisionMetadata:
    pos_embeds: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    cu_seqlens: torch.Tensor
    lengths: tuple[int, ...]
    image_splits: tuple[int, ...]
    total_tokens: int
    ready: torch.cuda.Event


@dataclass
class InputMetadata:
    visual: VisionMetadata
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    cache_position: torch.Tensor
    attention_mask: torch.Tensor
    image_mask: torch.Tensor
    placeholder_image_mask: torch.Tensor
    placeholder_video_mask: torch.Tensor
    input_shape: tuple[int, ...]
    text_width: int
    image_tokens: int
    ready: torch.cuda.Event

    def tensors(self):
        return (
            self.visual.pos_embeds,
            *self.visual.position_embeddings,
            self.visual.cu_seqlens,
            self.position_ids,
            self.rope_deltas,
            self.cache_position,
            self.attention_mask,
            self.image_mask,
            self.placeholder_image_mask,
            self.placeholder_video_mask,
        )


@dataclass
class Registration:
    tensors: tuple[torch.Tensor, ...]
    stamps: tuple
    model_stamp: tuple
    metadata: InputMetadata


def tensor_stamp(tensor):
    """Host-side properties only; no .item(), .cpu(), or device synchronization."""
    version = None if torch.is_inference(tensor) else tensor._version
    return (
        id(tensor),
        version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def ordinary_clone(tensor):
    with torch.inference_mode(False), torch.no_grad():
        return tensor.detach().clone()


class StaticS2:
    def __init__(self, workload):
        self.workload = workload
        self.inner = workload.model.backbone.model.model
        self.visual = self.inner.visual
        self.counts = Counter()
        self.preparation_host_ms = 0.0
        self.registrations = {}
        self.metadata_cache = {}
        self.vision_cache = {}
        self.restore = []
        self._active = None
        self._active_inputs = None
        self._entered = False

    def _replace(self, target, name, replacement):
        existed = name in vars(target)
        original = getattr(target, name)
        self.restore.append((target, name, existed, original))
        setattr(target, name, replacement)

    def _check_supported(self, inputs):
        if not self.workload.features_only:
            raise ValueError("StaticS2 requires the features-only S2 workload")
        if self.inner.training or self.visual.training:
            raise ValueError("StaticS2 only supports eval/inference")
        if self.visual.config._attn_implementation != "sdpa":
            raise ValueError("StaticS2 only supports SDPA vision attention")
        if self.inner.language_model.config._attn_implementation != "sdpa":
            raise ValueError("StaticS2 only supports SDPA text attention")
        if any(
            inputs.get(key) is not None
            for key in ("pixel_values_videos", "video_grid_thw")
        ):
            raise ValueError("StaticS2 only supports image-only inputs")
        for key in (*METADATA_KEYS, "pixel_values"):
            if not torch.is_tensor(inputs.get(key)):
                raise ValueError(f"StaticS2 requires tensor {key}")
            if inputs[key].device != self.visual.pos_embed.weight.device:
                raise ValueError(f"StaticS2 requires resident metadata/pixels: {key}")

    def _model_stamp(self):
        visual, config = self.visual, self.inner.config
        return (
            config.image_token_id,
            config.video_token_id,
            config.vision_start_token_id,
            visual.spatial_merge_size,
            visual.config.spatial_merge_size,
            visual.num_grid_per_side,
            visual.config.hidden_size,
            visual.config.num_heads,
            tuple(visual.deepstack_visual_indexes),
            tensor_stamp(visual.pos_embed.weight),
            tensor_stamp(visual.rotary_pos_emb.inv_freq),
            self.inner.get_input_embeddings().embedding_dim,
            visual.dtype,
        )

    @torch.inference_mode()
    def _prepare_visual(self, grid, rows, stamp):
        key = (rows, grid.dtype, grid.device, stamp)
        if key in self.vision_cache:
            metadata = self.vision_cache[key]
            # A new token/mask signature can reuse vision metadata prepared on
            # another stream; its new readiness event must include that work.
            torch.cuda.current_stream(grid.device).wait_event(metadata.ready)
            return metadata
        merge = self.visual.spatial_merge_size
        if not rows or any(
            t <= 0 or h <= 0 or w <= 0 or h % merge or w % merge for t, h, w in rows
        ):
            raise ValueError("StaticS2 requires positive, merge-aligned image grids")
        lengths = tuple(h * w for t, h, w in rows for _ in range(t))
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        # Preserve the original floating-point interpolation/rotary arithmetic.
        pos_embeds = self.visual.fast_pos_embed_interpolate(grid)
        rotary = self.visual.rot_pos_emb(grid).reshape(cumulative[-1], -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        metadata = VisionMetadata(
            pos_embeds=pos_embeds,
            position_embeddings=(emb.cos(), emb.sin()),
            cu_seqlens=torch.tensor(cumulative, dtype=torch.int32, device=grid.device),
            lengths=lengths,
            image_splits=tuple(t * h * w // merge**2 for t, h, w in rows),
            total_tokens=cumulative[-1],
            ready=torch.cuda.Event(),
        )
        metadata.ready.record()
        self.vision_cache[key] = metadata
        self.counts["vision_preparations"] += 1
        return metadata

    @torch.inference_mode()
    def _register(self, inputs, stamp):
        started = perf_counter()
        for key in METADATA_KEYS:
            if torch.is_inference(inputs[key]):
                inputs[key] = ordinary_clone(inputs[key])
        tensors = tuple(inputs[key] for key in METADATA_KEYS)
        cpu = tuple(tensor.detach().cpu() for tensor in tensors)
        ids, mask, grid = cpu
        if (
            ids.ndim != 2
            or mask.shape != ids.shape
            or grid.ndim != 2
            or grid.shape[1] != 3
        ):
            raise ValueError("StaticS2 expects [B,T] token/mask and [images,3] grid")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("StaticS2 requires integer token IDs")
        if grid.dtype not in (torch.int32, torch.int64):
            raise ValueError("StaticS2 requires integer image grids")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("StaticS2 supports only binary 2-D attention masks")
        # Full contents participate, not merely a length/shape hash. This is cold
        # registration only and also makes duplicate replay inputs share metadata.
        content = tuple(
            (
                tuple(host.shape),
                host.dtype,
                device.device,
                tuple(host.reshape(-1).tolist()),
            )
            for host, device in zip(cpu, tensors)
        )
        key = (content, stamp)
        if key in self.metadata_cache:
            metadata = self.metadata_cache[key]
            self.counts["content_cache_hits"] += 1
        else:
            config = self.inner.config
            if bool((ids == config.video_token_id).any()):
                raise ValueError("StaticS2 only supports image-only token sequences")
            rows = tuple(tuple(row) for row in grid.tolist())
            visual = self._prepare_visual(inputs["image_grid_thw"], rows, stamp)
            image_tokens = int((ids == config.image_token_id).sum())
            if image_tokens != sum(visual.image_splits):
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens {image_tokens}, features {sum(visual.image_splits)}"
                )
            # This function is integer-only; cold CPU metadata avoids scalar GPU
            # reads while retaining the original position-index implementation.
            positions, deltas = self.inner.get_rope_index(
                input_ids=ids, image_grid_thw=grid, attention_mask=mask
            )
            token_device = inputs["input_ids"].device
            image_mask = inputs["input_ids"] == config.image_token_id
            video_mask = inputs["input_ids"] == config.video_token_id
            width = self.inner.get_input_embeddings().embedding_dim
            metadata = InputMetadata(
                visual=visual,
                position_ids=positions.to(token_device),
                rope_deltas=deltas.to(token_device),
                cache_position=torch.arange(ids.shape[1], device=token_device),
                attention_mask=inputs["attention_mask"] == 1,
                image_mask=image_mask,
                placeholder_image_mask=image_mask.unsqueeze(-1).expand(
                    *ids.shape, width
                ),
                placeholder_video_mask=video_mask.unsqueeze(-1).expand(
                    *ids.shape, width
                ),
                input_shape=tuple(ids.shape),
                text_width=width,
                image_tokens=image_tokens,
                ready=torch.cuda.Event(),
            )
            metadata.ready.record()
            self.metadata_cache[key] = metadata
            self.counts["preparations"] += 1
        registration = Registration(
            tensors=tensors,
            stamps=tuple(tensor_stamp(tensor) for tensor in tensors),
            model_stamp=stamp,
            metadata=metadata,
        )
        self.registrations[id(inputs)] = registration
        self.counts["registrations"] += 1
        self.preparation_host_ms += (perf_counter() - started) * 1000
        return metadata

    def _lookup(self, inputs):
        self._check_supported(inputs)
        stamp = self._model_stamp()
        record = self.registrations.get(id(inputs))
        stamps = tuple(tensor_stamp(inputs[key]) for key in METADATA_KEYS)
        if (
            record is not None
            and record.stamps == stamps
            and record.model_stamp == stamp
        ):
            self.counts["cache_hits"] += 1
            return record.metadata
        if record is not None:
            self.counts["invalidations"] += 1
        return self._register(inputs, stamp)

    def _visual_forward(self, hidden_states, grid_thw, **kwargs):
        if self._active is None:
            return self._original_visual(hidden_states, grid_thw, **kwargs)
        if grid_thw is not self._active_inputs["image_grid_thw"]:
            raise ValueError("StaticS2 visual call does not match registered metadata")
        metadata = self._active.visual
        expected_width = (
            self.visual.patch_embed.in_channels
            * self.visual.patch_embed.temporal_patch_size
            * self.visual.patch_embed.patch_size**2
        )
        if hidden_states.shape != (metadata.total_tokens, expected_width):
            raise ValueError("StaticS2 pixels do not match the registered image grid")
        self.counts["vision_forwards"] += 1
        hidden_states = self.visual.patch_embed(hidden_states)
        hidden_states = hidden_states + metadata.pos_embeds
        hidden_states = hidden_states.reshape(metadata.total_tokens, -1)
        deepstack = []
        for layer_num, block in enumerate(self.visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=metadata.cu_seqlens,
                position_embeddings=metadata.position_embeddings,
                **kwargs,
            )
            if layer_num in self.visual.deepstack_visual_indexes:
                merger_index = self.visual.deepstack_visual_indexes.index(layer_num)
                deepstack.append(
                    self.visual.deepstack_merger_list[merger_index](hidden_states)
                )
        return self.visual.merger(hidden_states), deepstack

    def _attention_forward(
        self,
        module,
        original,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        if self._active is None:
            return original(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if module.training or module.config._attn_implementation != "sdpa":
            raise ValueError("StaticS2 only supports eval SDPA attention")
        seq_length = hidden_states.shape[0]
        query, key, value = (
            module.qkv(hidden_states)
            .reshape(seq_length, 3, module.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query, key = qwen.apply_rotary_pos_emb_vision(query, key, cos, sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        splits = [
            torch.split(tensor, self._active.visual.lengths, dim=2)
            for tensor in (query, key, value)
        ]
        attention = qwen.ALL_ATTENTION_FUNCTIONS["sdpa"]
        outputs = [
            attention(
                module,
                q,
                k,
                v,
                attention_mask=None,
                scaling=module.scaling,
                dropout=0.0,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        output = torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous()
        return module.proj(output)

    def _image_features(self, pixel_values, image_grid_thw=None):
        if self._active is None:
            return self._original_images(pixel_values, image_grid_thw)
        image_embeds, deepstack = self.visual(
            pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw
        )
        return torch.split(image_embeds, self._active.visual.image_splits), deepstack

    def _placeholder_mask(
        self, input_ids, inputs_embeds, image_features=None, video_features=None
    ):
        if self._active is None:
            return self._original_placeholder(
                input_ids, inputs_embeds, image_features, video_features
            )
        metadata = self._active
        if (
            input_ids is not self._active_inputs["input_ids"]
            or video_features is not None
        ):
            raise ValueError(
                "StaticS2 placeholder call requires registered image-only IDs"
            )
        if inputs_embeds.shape != (*metadata.input_shape, metadata.text_width):
            raise ValueError("StaticS2 placeholder embedding shape changed")
        if image_features is None or image_features.shape != (
            metadata.image_tokens,
            metadata.text_width,
        ):
            raise ValueError("Image features and image tokens do not match")
        return metadata.placeholder_image_mask, metadata.placeholder_video_mask

    @torch.inference_mode()
    def _slow(self, index):
        if self._active is not None:
            raise RuntimeError("StaticS2 supports one S2 caller at a time")
        inputs = self.workload.backbone_inputs[
            index % len(self.workload.backbone_inputs)
        ]
        metadata = self._lookup(inputs)
        stream = torch.cuda.current_stream(inputs["input_ids"].device)
        stream.wait_event(metadata.ready)
        # Metadata are owned by this scope but may execute on an S2 worker stream.
        for tensor in (*metadata.tensors(), *(inputs[key] for key in METADATA_KEYS)):
            tensor.record_stream(stream)
        inputs["pixel_values"].record_stream(stream)
        self.counts["calls"] += 1
        self._active, self._active_inputs = metadata, inputs
        try:
            self.inner.rope_deltas = metadata.rope_deltas
            self.inner(
                **{key: inputs[key] for key in (*METADATA_KEYS, "pixel_values")},
                position_ids=metadata.position_ids,
                cache_position=metadata.cache_position,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            return BatchFeature(
                {
                    "backbone_features": self.workload._pre_final_norm,
                    "backbone_attention_mask": metadata.attention_mask,
                    "image_mask": metadata.image_mask,
                }
            )
        finally:
            self._active = self._active_inputs = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError("StaticS2 scopes cannot be entered twice")
        self._entered = True
        self._rope_before = self.inner.rope_deltas
        try:
            prepared_inputs = []
            for original in self.workload.backbone_inputs:
                self._check_supported(original)
                copied = dict(original)
                for key in METADATA_KEYS:
                    copied[key] = ordinary_clone(copied[key])
                self._register(copied, self._model_stamp())
                prepared_inputs.append(copied)
            self._replace(self.workload, "backbone_inputs", prepared_inputs)
            self._original_visual = self.visual.forward
            self._original_images = self.inner.get_image_features
            self._original_placeholder = self.inner.get_placeholder_mask
            self._replace(self.visual, "forward", self._visual_forward)
            self._replace(self.inner, "get_image_features", self._image_features)
            self._replace(self.inner, "get_placeholder_mask", self._placeholder_mask)
            for block in self.visual.blocks:
                original = block.attn.forward

                def forward(module, *args, _original=original, **kwargs):
                    return self._attention_forward(module, _original, *args, **kwargs)

                self._replace(block.attn, "forward", MethodType(forward, block.attn))
            self._replace(self.workload, "slow", self._slow)
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exception):
        for target, name, existed, original in reversed(self.restore):
            if existed:
                setattr(target, name, original)
            else:
                delattr(target, name)
        self.restore.clear()
        if self._entered:
            self.inner.rope_deltas = self._rope_before
        self._entered = False
        self._active = self._active_inputs = None

    def evidence(self):
        counters = {
            name: self.counts[name]
            for name in (
                "calls",
                "cache_hits",
                "content_cache_hits",
                "preparations",
                "vision_preparations",
                "registrations",
                "invalidations",
                "fallbacks",
                "vision_forwards",
            )
        }
        return {
            **counters,
            "preparation_host_ms": self.preparation_host_ms,
            "metadata_entries": len(self.metadata_cache),
            "vision_entries": len(self.vision_cache),
            "scope": "image-only SDPA eval; frozen weights; one S2 worker",
            "pixel_outputs_cached": False,
            "unsupported_policy": "explicit ValueError; no silent eager fallback",
            "metadata_validity": "identity/version/shape/dtype/device + cold full contents",
        }
