"""Instance-local DiT CUDA Graph capture, independent of quantization format."""

from collections import OrderedDict

from coexecution.kernels.common.graph import CudaGraphCall
from coexecution.optimization.execution.submission import cold_capture


class DitGraphs:
    def __init__(self, model, max_entries=64):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Graph max_entries must be a positive integer")
        self.model = model
        self.forward = model.forward
        self.had_forward = "forward" in model.__dict__
        self.graphs = OrderedDict()
        self.max_entries = max_entries
        self.capture_count = self.replay_count = self.failure_count = (
            self.eviction_count
        ) = 0

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
            signature = tuple(
                None
                if value is None
                else (
                    tuple(value.shape),
                    value.dtype,
                    value.device,
                    tuple(value.stride()),
                )
                for value in values
            )
            bucket = (key, return_all_hidden_states, present, signature)
            graph = self.graphs.get(bucket)
            if graph is None:

                def function(*inputs):
                    iterator = iter(inputs)
                    return call(
                        *(
                            next(iterator) if has_value else None
                            for has_value in present
                        )
                    )

                graph = CudaGraphCall(function)
            before = graph.captures, graph.replays
            try:
                with cold_capture(graph.captures == 0):
                    result = graph(*(value for value in values if value is not None))
            except Exception:
                self.failure_count += 1
                self.graphs.pop(bucket, None)
                self._restore()
                raise
            finally:
                self.capture_count += graph.captures - before[0]
                self.replay_count += graph.replays - before[1]
            # Publish only after successful capture/replay, retaining existing
            # working shapes if a new shape fails. All outputs are fresh clones
            # from the pinned helper, independent of cache eviction.
            self.graphs[bucket] = graph
            self.graphs.move_to_end(bucket)
            while len(self.graphs) > self.max_entries:
                self.graphs.popitem(last=False)
                self.eviction_count += 1
            return result

        self.model.forward = forward

    def _restore(self):
        if self.had_forward:
            self.model.forward = self.forward
        elif "forward" in self.model.__dict__:
            del self.model.forward

    def close(self):
        self._restore()
        self.graphs.clear()

    def evidence(self):
        return {
            "capture_count": self.capture_count,
            "replay_count": self.replay_count,
            "failure_count": self.failure_count,
            "eviction_count": self.eviction_count,
            "resident_signatures": len(self.graphs),
            "max_entries": self.max_entries,
        }
