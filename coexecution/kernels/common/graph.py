"""Instance-owned CUDA graph execution with explicit input refresh."""

import torch


class CudaGraphCall:
    """Capture a deterministic tensor function; RNG must be an explicit input.

    The caller owns serialization and weight lifetime. Returned outputs are
    copied so a later replay cannot overwrite a caller's previous action chunk.
    A changed input signature replaces the previous graph rather than growing
    an unbounded cache. Capture and warmup cost is separately observable.
    """

    def __init__(
        self, function, *, retained_tensors=None, tuple_output=False, capture_key=None
    ):
        self.function = function
        self.retained_tensors = retained_tensors
        self.tuple_output = tuple_output
        self.capture_key = capture_key
        self._constants = ()
        self.signature = None
        self.graph = None
        self.inputs = None
        self.output = None
        self.captures = 0
        self.replays = 0

    def __call__(self, *inputs):
        if not inputs or any(
            not isinstance(x, torch.Tensor) or not x.is_cuda for x in inputs
        ):
            raise ValueError("CUDA graph inputs must be CUDA tensors")
        device = inputs[0].device
        if any(x.device != device for x in inputs):
            raise ValueError("Graph inputs must share a device")
        signature = tuple(
            (tuple(x.shape), x.dtype, x.device, tuple(x.stride())) for x in inputs
        )
        signature = (signature, self.capture_key() if self.capture_key else None)
        with torch.cuda.device(device):
            if signature != self.signature:
                # Build a replacement transactionally. Failed warmup/capture
                # must not invalidate the previous working shape.
                new_inputs = tuple(x.clone() for x in inputs)
                constants = (
                    tuple(self.retained_tensors())
                    if self.retained_tensors is not None
                    else ()
                )
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                # Inputs were allocated on the caller stream. Even a failed
                # warmup must keep their storage alive until this stream exits.
                for tensor in (*new_inputs, *constants):
                    if isinstance(tensor, torch.Tensor):
                        tensor.record_stream(stream)
                try:
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self.function(*new_inputs)
                finally:
                    torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = self.function(*new_inputs)
                outputs = output if self.tuple_output else (output,)
                if (
                    not isinstance(outputs, tuple)
                    or not outputs
                    or any(
                        not isinstance(x, torch.Tensor) or x.device != device
                        for x in outputs
                    )
                ):
                    raise ValueError("Graph callable must return CUDA tensor outputs")
                self.graph, self.output = graph, output
                self.inputs = new_inputs
                self._constants = constants
                self.signature = signature
                self.captures += 1
            for target, source in zip(self.inputs, inputs):
                target.copy_(source)
            self.graph.replay()
            current_stream = torch.cuda.current_stream(device)
            for constant in self._constants:
                if isinstance(constant, torch.Tensor):
                    constant.record_stream(current_stream)
            self.replays += 1
            if self.tuple_output:
                return tuple(x.clone() for x in self.output)
            return self.output.clone()

    def close(self):
        """Release captured storage after the caller has drained its stream."""
        self.graph = self.inputs = self.output = self.signature = None
        self._constants = ()
