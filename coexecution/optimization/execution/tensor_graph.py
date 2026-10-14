"""Serial deterministic CUDA tuple execution with owned inputs and outputs.

The pinned single-Tensor helper cannot preserve the action-head's three-output
API. This variant keeps the same explicit-RNG/copy/replay contract for tuples,
preserves input strides, and binds each instance to its caller stream.
"""


class TensorGraphCall:
    def __init__(self, function, *, retained_tensors=()):
        self.function = function
        self.constants = tuple(retained_tensors)
        self.graph = None
        self.inputs = self.outputs = ()
        self.signature = self.stream_id = None
        self.captures = self.replays = 0

    def __call__(self, *inputs):
        import torch

        if not inputs or any(
            not isinstance(value, torch.Tensor)
            or not value.is_cuda
            or value.layout != torch.strided
            for value in inputs
        ):
            raise ValueError("Action graph inputs must be strided CUDA tensors")
        device = inputs[0].device
        if any(value.device != device for value in inputs):
            raise ValueError("Action graph inputs must share a device")
        signature = tuple(
            (tuple(value.shape), value.dtype, value.device, tuple(value.stride()))
            for value in inputs
        )
        with torch.cuda.device(device):
            caller = torch.cuda.current_stream(device)
            if self.graph is not None and (
                signature != self.signature or caller.cuda_stream != self.stream_id
            ):
                raise ValueError("Action graph signature or caller stream changed")
            if self.graph is None:
                buffers = tuple(
                    torch.empty_strided(
                        value.shape, value.stride(), dtype=value.dtype, device=device
                    ).copy_(value)
                    for value in inputs
                )
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(caller)
                for value in (*buffers, *self.constants):
                    value.record_stream(stream)
                try:
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self.function(*buffers)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = self.function(*buffers)
                    if (
                        not isinstance(outputs, tuple)
                        or not outputs
                        or any(
                            not isinstance(value, torch.Tensor)
                            or value.device != device
                            for value in outputs
                        )
                    ):
                        raise ValueError("Action graph must return a CUDA Tensor tuple")
                finally:
                    caller.wait_stream(stream)
                self.graph, self.inputs, self.outputs = graph, buffers, outputs
                self.signature, self.stream_id = signature, caller.cuda_stream
                self.captures += 1
            for target, source in zip(self.inputs, inputs):
                target.copy_(source)
            self.graph.replay()
            # Private graph allocations and packed closure constants must outlive
            # asynchronous replay and the owned output copies, including eviction.
            for value in (*self.inputs, *self.outputs, *self.constants):
                value.record_stream(caller)
            outputs = tuple(value.clone() for value in self.outputs)
            self.replays += 1
            return outputs
