"""Opt-in Qwen3-VL whole-vision and fresh-prefill text block graphs.

One frozen policy, one submission stream. The enclosing optimization scope keeps
its serial guard. Cache updates remain eager and returned tensors are copied.
"""

import types

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from coexecution.kernels.common.graph import CudaGraphCall


class PrefillCache:
    """Capture K/V tensor production; publish copies into the real cache eagerly."""

    def __init__(self):
        self.keys = self.values = None

    def update(self, keys, values, layer_idx, cache_kwargs):
        self.keys, self.values = keys, values
        return keys, values


class VlmGraphs:
    def __init__(self, backbone):
        qwen = backbone.model.model
        self.vision = qwen.visual
        self.text = qwen.language_model
        self.saved = []
        self.graphs = {}
        self.metadata = None
        self.grid_key = None
        self.max_length = None
        self.text_metadata = {}
        self.mask_pointer = None
        self.closed = False
        self.installed = False
        self.stream = None
        self.prepares = 0
        self.retired_captures = self.retired_replays = 0
        self.requests = {"vision": 0, "text": 0}
        self.caches = [PrefillCache() for _ in self.text.layers]
        self.ropes = {}

    def patch(self, module, function):
        self.saved.append((module, "forward" in module.__dict__, module.forward))
        module.forward = types.MethodType(function, module)

    def check_call(self, tensor):
        if self.closed or torch.is_grad_enabled():
            raise RuntimeError("VLM Graph requires a live inference-only scope")
        if not tensor.is_cuda or tensor.dtype != torch.bfloat16:
            raise ValueError("VLM Graph requires BF16 CUDA activations")
        stream = (tensor.device, torch.cuda.current_stream(tensor.device).cuda_stream)
        if self.stream is not None and stream != self.stream:
            raise RuntimeError("VLM Graph requires one caller-owned CUDA stream")
        self.stream = stream

    def install(self):
        if self.installed or self.closed:
            raise RuntimeError("VLM Graph installation is one-shot")
        if self.vision.training or self.text.training:
            raise ValueError("VLM Graph requires eval-mode Qwen3-VL")
        from transformers.integrations.flash_attention import flash_attention_forward
        from transformers.modeling_flash_attention_utils import (
            _is_packed_sequence,
            prepare_fa_kwargs_from_position_ids,
        )

        # Resolve before patching: preserve the selected fusion implementation.
        for block in self.vision.blocks:
            attention = block.attn
            if attention.config._attn_implementation != "flash_attention_2":
                raise ValueError("VLM Graph requires Flash Attention 2")
            self.ropes[id(attention)] = attention.forward.__func__.__globals__[
                "apply_rotary_pos_emb_vision"
            ]
        if self.text.config._attn_implementation != "flash_attention_2":
            raise ValueError("VLM Graph requires Flash Attention 2")
        self.graphs["vision"] = CudaGraphCall(
            self.vision_gpu,
            retained_tensors=lambda: self.metadata,
            tuple_output=True,
            capture_key=lambda: self.grid_key,
        )

        def vision_root(module, pixels, grid_thw, **kwargs):
            self.check_call(pixels)
            if (
                kwargs
                or module.training
                or grid_thw.ndim != 2
                or grid_thw.shape[1] != 3
                or grid_thw.device != pixels.device
            ):
                raise ValueError("Unsupported Qwen3-VL vision call")
            grid = grid_thw.detach().cpu()
            if grid.dtype not in (torch.int32, torch.int64) or not bool(
                (grid > 0).all()
            ):
                raise ValueError("Vision grid must contain positive integer T/H/W")
            tokens = int(grid.prod(-1).sum().item())
            if pixels.ndim != 2 or pixels.shape[0] != tokens:
                raise ValueError("Pixel token count differs from vision grid")
            key = (tuple(grid.flatten().tolist()), tuple(grid.shape), pixels.device)
            self.max_length = int((grid[:, 1] * grid[:, 2]).max().item())
            if key != self.grid_key:
                pos = module.fast_pos_embed_interpolate(grid_thw)
                rot = module.rot_pos_emb(grid_thw).reshape(tokens, -1)
                emb = torch.cat((rot, rot), -1)
                cu = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).cumsum(0, dtype=torch.int32)
                self.metadata = (pos, emb.cos(), emb.sin(), F.pad(cu, (1, 0), value=0))
                self.grid_key = key
                self.prepares += 1
            self.requests["vision"] += 1
            result = self.graphs["vision"](pixels)
            return result[0], list(result[1:])

        def vision_attention(
            module,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=None,
            position_embeddings=None,
            **kwargs,
        ):
            if (
                module.training
                or position_embeddings is None
                or rotary_pos_emb is not None
            ):
                raise ValueError("Unsupported vision attention capture")
            seq = hidden_states.shape[0]
            q, k, v = (
                module.qkv(hidden_states)
                .reshape(seq, 3, module.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            q, k = self.ropes[id(module)](q, k, *position_embeddings)
            q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
            output, _ = flash_attention_forward(
                module,
                q,
                k,
                v,
                attention_mask=None,
                scaling=module.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=self.max_length,
                max_length_k=self.max_length,
                is_causal=False,
                **kwargs,
            )
            return module.proj(output.reshape(seq, -1).contiguous())

        original_text = self.text.forward

        def text_root(module, *args, **kwargs):
            if module.training:
                raise ValueError("VLM Graph requires eval-mode text prefill")
            if kwargs.get("past_key_values") is not None:
                raise ValueError("VLM Graph supports fresh prefill, not historical KV")
            if args:
                raise ValueError("VLM Graph text root requires named prefill inputs")
            hidden = kwargs.get("inputs_embeds")
            if hidden is None:
                raise ValueError("VLM Graph requires precomputed input embeddings")
            self.check_call(hidden)
            self.text_metadata = {}
            self.mask_pointer = None
            positions = kwargs.get("position_ids")
            if positions is not None:
                positions = positions[0] if positions.ndim == 3 else positions
                if bool(_is_packed_sequence(positions, positions.shape[0])):
                    (cuq, cuk), (mq, mk) = prepare_fa_kwargs_from_position_ids(
                        positions
                    )
                    self.text_metadata = {
                        "cu_seq_lens_q": cuq,
                        "cu_seq_lens_k": cuk,
                        "max_length_q": mq,
                        "max_length_k": mk,
                    }
            self.requests["text"] += 1
            return original_text(**kwargs)

        self.patch(self.vision, vision_root)
        for block in self.vision.blocks:
            self.patch(block.attn, vision_attention)
        self.patch(self.text, text_root)
        for index, block in enumerate(self.text.layers):
            self.wrap_text(block, index)
        self.installed = True

    def vision_gpu(self, pixels):
        pos, cos, sin, cu = self.metadata
        hidden = self.vision.patch_embed(pixels) + pos
        hidden = hidden.reshape(hidden.size(0), -1)
        deep = []
        for index, block in enumerate(self.vision.blocks):
            hidden = block(hidden, cu_seqlens=cu, position_embeddings=(cos, sin))
            if index in self.vision.deepstack_visual_indexes:
                merger = self.vision.deepstack_visual_indexes.index(index)
                deep.append(self.vision.deepstack_merger_list[merger](hidden))
        return (self.vision.merger(hidden), *deep)

    def wrap_text(self, block, index):
        original = block.forward

        def forward(module, *args, **kwargs):
            kwargs = dict(kwargs)
            cache = kwargs.get("past_key_values")
            if cache is not None and cache.get_seq_length(index) != 0:
                raise ValueError("VLM Graph requires empty per-layer prefill KV")
            mask = kwargs.get("attention_mask")
            if mask is not None:
                if self.mask_pointer is None:
                    if mask.ndim != 2 or not bool((mask == 1).all()):
                        raise ValueError("VLM Graph does not support padding masks")
                    self.mask_pointer = mask.data_ptr()
                    batch, length = mask.shape
                    cu = (
                        torch.arange(batch + 1, device=mask.device, dtype=torch.int32)
                        * length
                    )
                    self.text_metadata = {
                        "cu_seq_lens_q": cu,
                        "cu_seq_lens_k": cu,
                        "max_length_q": length,
                        "max_length_k": length,
                    }
                if self.mask_pointer != mask.data_ptr():
                    raise ValueError("Text mask changed between decoder layers")
                kwargs["attention_mask"] = None
            kwargs["position_ids"] = None
            kwargs.update(self.text_metadata)
            proxy = self.caches[index]
            if cache is not None:
                kwargs["past_key_values"] = proxy
            previous_kv = (proxy.keys, proxy.values)
            try:
                result = self.text_call(original, index, args, kwargs)
            except BaseException:
                # A failed warmup/capture must not replace the K/V pointers of
                # a previously working bucket that still owns its graph pool.
                proxy.keys, proxy.values = previous_kv
                raise
            if cache is not None:
                cos, sin = kwargs["position_embeddings"]
                cache.update(
                    proxy.keys.clone(),
                    proxy.values.clone(),
                    index,
                    {
                        "cos": cos,
                        "sin": sin,
                        "cache_position": kwargs.get("cache_position"),
                    },
                )
            return result

        self.patch(block, forward)

    def text_call(self, function, index, args, kwargs):
        flat, spec = tree_flatten((args, kwargs))
        tensor_indices = [
            i for i, value in enumerate(flat) if isinstance(value, torch.Tensor)
        ]
        constants = tuple((i, x) for i, x in enumerate(flat) if i not in tensor_indices)
        if not tensor_indices or any(
            not isinstance(x, (str, int, float, bool, type(None), PrefillCache))
            for _, x in constants
        ):
            raise ValueError("Unsupported text Graph input/control structure")
        control = (repr(spec), constants)
        key = ("text", index)
        entry = self.graphs.get(key)
        if entry is None or entry[0] != control:
            template = list(flat)
            for i in tensor_indices:
                template[i] = None

            def call(*values):
                rebuilt = list(template)
                for i, value in zip(tensor_indices, values):
                    rebuilt[i] = value
                aa, kk = tree_unflatten(rebuilt, spec)
                return function(*aa, **kk)

            graph = CudaGraphCall(call)
            # Capture before replacing a previously working bucket.
            result = graph(*(flat[i] for i in tensor_indices))
            if entry is not None:
                self.retired_captures += entry[1].captures
                self.retired_replays += entry[1].replays
                entry[1].close()
            self.graphs[key] = (control, graph)
            return result
        return entry[1](*(flat[i] for i in tensor_indices))

    def evidence(self):
        graphs = [g if k == "vision" else g[1] for k, g in self.graphs.items()]
        return {
            "installed": self.installed and not self.closed,
            "captures": self.retired_captures + sum(g.captures for g in graphs),
            "replays": self.retired_replays + sum(g.replays for g in graphs),
            "vision_metadata_prepares": self.prepares,
            "requests": dict(self.requests),
            "scope": "One frozen Flow policy/stream; whole vision and fresh text prefill",
        }

    def close(self):
        if self.closed:
            return
        for module, had, original in reversed(self.saved):
            if had:
                module.forward = original
            else:
                del module.forward
        self.saved.clear()
        for key, entry in self.graphs.items():
            graph = entry if key == "vision" else entry[1]
            self.retired_captures += graph.captures
            self.retired_replays += graph.replays
            graph.close()
        self.graphs.clear()
        self.metadata = None
        self.caches.clear()
        self.closed = True
