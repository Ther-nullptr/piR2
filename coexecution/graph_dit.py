"""Current-stream CUDA graph replay with owned inputs and non-aliasing outputs."""

from dataclasses import dataclass

import torch


def description(value):
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), value.stride(), value.dtype, value.device)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f"Unsupported graph argument: {type(value)}")


def clone_result(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_result(item) for item in value)
    if isinstance(value, list):
        return [clone_result(item) for item in value]
    raise TypeError(f"Unsupported graph output: {type(value)}")


@dataclass
class Capture:
    graph: object
    args: tuple
    kwargs: dict
    output: object
    done: object


class GraphCall:
    """One S1 submitter, frozen weights, and fixed-shape graph buckets.

    Call freeze() after preparation: unseen layouts then run eagerly instead of
    attempting a global CUDA capture while the S2 worker is running.
    """

    def __init__(self, function):
        self.function = function
        self.captures = {}
        self.allow_capture = True
        self.replays = 0
        self.fallbacks = 0
        self.capture_streams = []

    @torch.inference_mode()
    def _capture(self, args, kwargs):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fixed_args = tuple(
                value.clone() if torch.is_tensor(value) else value for value in args
            )
            fixed_kwargs = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
            for _ in range(3):
                self.function(*fixed_args, **fixed_kwargs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self.function(*fixed_args, **fixed_kwargs)
        done = torch.cuda.Event()
        done.record(stream)
        self.capture_streams.append(stream)
        return Capture(graph, fixed_args, fixed_kwargs, output, done)

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        key = (
            tuple(description(value) for value in args),
            tuple((name, description(value)) for name, value in sorted(kwargs.items())),
        )
        record = self.captures.get(key)
        if record is None:
            if not self.allow_capture:
                self.fallbacks += 1
                return self.function(*args, **kwargs)
            record = self._capture(args, kwargs)
            self.captures[key] = record
        stream = torch.cuda.current_stream()
        # Serialize reuse of fixed graph storage across stream changes without a
        # CPU/device barrier. The caller must order the source tensors' producer.
        stream.wait_event(record.done)
        for fixed, actual in zip(record.args, args):
            if torch.is_tensor(actual):
                fixed.copy_(actual)
        for name, actual in kwargs.items():
            if torch.is_tensor(actual):
                record.kwargs[name].copy_(actual)
        record.graph.replay()
        result = clone_result(record.output)
        record.done.record(stream)
        self.replays += 1
        return result

    def freeze(self):
        self.allow_capture = False

    def evidence(self):
        return {
            "captures": len(self.captures),
            "replays": self.replays,
            "fallbacks": self.fallbacks,
            "allow_capture": self.allow_capture,
        }
