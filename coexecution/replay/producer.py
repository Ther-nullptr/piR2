"""Optional metadata producers for a bounded replay writer; no model imports.

Sinks must implement the nonblocking ``BoundedEventWriter.emit`` contract. These
helpers neither serialize files nor start inference. Occurrence timestamps are
host monotonic seconds; a GPU completion event is only an explicit host-observed
completion, never a fabricated GPU timestamp or kernel duration.
"""

import time
from contextlib import contextmanager

HOST_CLOCK = {"unit": "s", "clock_id": "host-monotonic", "domain": "host_monotonic"}
_NOW = object()


def validate_sink(sink):
    if sink is not None and not callable(sink):
        raise TypeError("event_sink must be callable or None")


def emit_event(sink, kind, *, t_s=_NOW, sources=None, payload=None, clock=None):
    if sink is None:
        return False
    validate_sink(sink)
    return sink(
        kind,
        occurrence={
            **HOST_CLOCK,
            "value": (clock or time.monotonic)() if t_s is _NOW else t_s,
        },
        sources={} if sources is None else sources,
        payload={} if payload is None else payload,
    )


def source_id(namespace, kind, value):
    if value is None:
        return None
    prefix = f"{namespace}:" if namespace else ""
    return f"{prefix}{kind}-{value}"


def queue_sources(record, namespace):
    executed = record.get("executed_slot") or {}
    observation = record.get("source_tick", record.get("state_tick"))
    if observation is None and record.get("kind") == "control_tick":
        observation = record.get("tick")
    return {
        "episode_id": namespace or None,
        "observation_id": source_id(
            namespace,
            "observation",
            observation,
        ),
        "request_id": source_id(
            namespace,
            "request",
            record.get("request_tick", executed.get("producer_request_tick")),
        ),
        "feature_observation_id": source_id(
            namespace, "observation", record.get("feature_source_tick")
        ),
        "feature_id": source_id(namespace, "feature", record.get("feature_sequence")),
        "action_id": source_id(namespace, "action", record.get("tick")),
    }


def emit_sample(sink, row, *, completed_s=None, actions=None):
    """Emit only after a clean timer/decode; never invent internal stage spans."""
    if sink is None:
        return False
    from .model import _output, encode_action_arrays

    encoded = (
        {} if actions is None else encode_action_arrays(actions, allow_nonfinite=True)
    )
    output = _output(
        {f"{row['request_id']}/{key}": value for key, value in encoded.items()},
        row["request_id"],
    )
    return emit_event(
        sink,
        "model.request.completed",
        t_s=completed_s,
        sources={
            "observation_id": row["observation_id"],
            "request_id": row["request_id"],
        },
        payload={
            **row,
            "sample": dict(row),
            "gpu_completion_observed": True,
            "internal_stages_available": False,
            "output": output,
        },
    )


class ModelRequestEvents:
    """Instrumented host launch scopes around the original engine's components."""

    def __init__(
        self, event_sink, *, request_id, observation_id, mode=None, clock=None
    ):
        validate_sink(event_sink)
        self.sink, self.clock, self.mode = event_sink, clock or time.monotonic, mode
        self.sources = {
            "request_id": request_id,
            "observation_id": observation_id,
            "feature_id": request_id + "/features",
        }

    def _emit(self, kind, t_s, **payload):
        return emit_event(
            self.sink,
            kind,
            t_s=t_s,
            sources=dict(self.sources),
            payload={"mode": self.mode, "instrumented": True, **payload},
        )

    @contextmanager
    def stage(self, name, scope):
        if self.sink is None:
            with scope:
                yield
            return
        start = self.clock()
        self._emit(
            "model.stage.started",
            start,
            stage=name,
            timing_boundary="host_submission",
            gpu_completion_observed=False,
        )
        if name == "action":
            self._emit(
                "model.feature.read",
                start,
                stage=name,
                timing_boundary="host_submission",
                dependency="same request VLM output; ordered CUDA stream",
            )
        try:
            with scope:
                yield
        except BaseException as error:
            self._emit(
                "model.stage.failed",
                self.clock(),
                stage=name,
                error=repr(error),
                gpu_completion_observed=False,
            )
            raise
        stop = self.clock()
        self._emit(
            "model.stage.submitted",
            stop,
            stage=name,
            timing_boundary="host_submission",
            start_s=start,
            duration_s=stop - start,
            gpu_completion_observed=False,
        )
        if name == "vlm":
            self._emit(
                "model.feature.submitted",
                stop,
                stage=name,
                timing_boundary="host_submission",
                gpu_completion_observed=False,
            )

    def completed(self):
        if self.sink is not None:
            self._emit(
                "model.request.completed",
                self.clock(),
                gpu_completion_observed=True,
                timing_boundary="host_observed_after_cuda_synchronize",
                output_available=False,
            )
