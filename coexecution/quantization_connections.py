"""GR00T B1/LIBERO connections to the reference integer and graph implementations.

Reuses robotics-the-speedup-paradox revision
239b4a3ef2268048571c9f508ad5700f98398c2f without modifying its kernels.
"""

import torch
from robotics_kernels.ampere_ada.modulation import PackedActivations, prepare_modulation
from robotics_kernels.common.graph import CudaGraphCall

from coexecution.quantization import PreparedLinear


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

    def set(self, mode, *, expanded=True, fuse=True):
        self.close()
        if mode == "bf16":
            return
        bits = {"w8a8": 8, "w4a4": 4}[mode]
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
                    normalized = norm.norm(x)
                    if scale.ndim == 3:
                        width = x.shape[-1]
                        packed = prepare_modulation(
                            normalized.reshape(-1, 1, width),
                            scale.reshape(-1, 1, width),
                            shift.reshape(-1, 1, width),
                            bits,
                        )
                        result = PackedActivations({bits: packed}).with_leading_shape(
                            x.shape[:-1]
                        )
                    else:
                        packed = prepare_modulation(
                            normalized, scale[:, None], shift[:, None], bits
                        )
                        result = PackedActivations({bits: packed})
                    result.shape = (
                        x.shape
                    )  # GR00T attention reads the batch size before Q projection.
                    return result

                norm.forward = forward

    def close(self):
        for module, forward, had in self.saved:
            if had:
                module.forward = forward
            elif "forward" in module.__dict__:
                del module.forward


class DitGraphs:
    def __init__(self, model):
        self.model = model
        self.forward = model.forward
        self.had_forward = "forward" in model.__dict__
        self.graphs = {}

    def set(self, key=None):
        if key is None:
            self._restore()
            return

        def forward(
            hidden_states,
            encoder_hidden_states,
            timestep=None,
            encoder_attention_mask=None,
            return_all_hidden_states=False,
            image_mask=None,
            backbone_attention_mask=None,
        ):
            values = (
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_attention_mask,
                image_mask,
                backbone_attention_mask,
            )
            names = (
                "timestep",
                "encoder_attention_mask",
                "image_mask",
                "backbone_attention_mask",
            )

            def call(*args):
                return self.forward(
                    *args[:2],
                    **dict(zip(names, args[2:])),
                    return_all_hidden_states=return_all_hidden_states,
                )

            # The reference graph helper returns one Tensor; preserve the tuple/list API eagerly.
            if return_all_hidden_states:
                return call(*values)
            present = tuple(value is not None for value in values)
            bucket = f"{key}:{int(return_all_hidden_states)}:{''.join(str(int(v)) for v in present)}"
            if bucket not in self.graphs:

                def function(*inputs):
                    iterator = iter(inputs)
                    return call(
                        *(
                            next(iterator) if has_value else None
                            for has_value in present
                        )
                    )

                self.graphs[bucket] = CudaGraphCall(function)
            try:
                return self.graphs[bucket](
                    *(value for value in values if value is not None)
                )
            except Exception:
                self.graphs.pop(bucket, None)
                self._restore()
                raise

        self.model.forward = forward

    def _restore(self):
        if self.had_forward:
            self.model.forward = self.forward
        elif "forward" in self.model.__dict__:
            del self.model.forward

    def close(self):
        self._restore()
        self.graphs.clear()
