"""Instance-local DiT CUDA Graph capture, independent of quantization format."""

from robotics_kernels.common.graph import CudaGraphCall


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
