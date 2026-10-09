"""In-memory lifecycle evidence for the actual LIBERO controller queues."""

import copy
import json
import threading
import time


class QueueTrace:
    """Serialize metadata from controller and worker threads; defer all file I/O."""

    def __init__(self, clock=None):
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._records = []

    def emit(self, kind, t_s=None, **fields):
        with self._lock:
            record = {
                "schema_version": 1,
                "seq": len(self._records),
                "kind": kind,
                "t_s": self.clock() if t_s is None else t_s,
                **copy.deepcopy(fields),
            }
            self._records.append(record)
            return record["seq"]

    def records(self):
        with self._lock:
            return copy.deepcopy(self._records)

    def flush(self, stream):
        for record in self.records():
            stream.write(json.dumps(record) + "\n")
        stream.flush()


class LatestFrameMailbox:
    """One waiting frame, one in-flight RPC, and the last published feature.

    VisionWorker holds its condition lock for every operation on this mailbox.
    Keeping the actual objects here makes replacement events describe ownership,
    rather than inferring a queue from independently collected timestamps.
    """

    def __init__(self, events):
        self.events = events
        self.pending = None
        self.inflight = None
        self.latest = None

    def snapshot(self):
        meta = self.latest["meta"] if self.latest else {}
        return {
            "waiting_tick": self.pending.tick if self.pending else None,
            "waiting_capture_s": self.pending.capture_s if self.pending else None,
            "inflight_tick": self.inflight.tick if self.inflight else None,
            "inflight_capture_s": self.inflight.capture_s if self.inflight else None,
            "feature_sequence": self.latest["sequence"] if self.latest else None,
            "feature_source_tick": meta.get("source_tick"),
            "feature_capture_s": meta.get("capture_s"),
        }

    def offer(self, observation):
        replaced = self.pending
        self.pending = observation
        if replaced is not None:
            self.events.emit(
                "camera_replaced",
                source_tick=replaced.tick,
                capture_s=replaced.capture_s,
                replacement_source_tick=observation.tick,
                camera_queue=self.snapshot(),
            )
        self.events.emit(
            "camera_offered",
            source_tick=observation.tick,
            capture_s=observation.capture_s,
            camera_queue=self.snapshot(),
        )
        return replaced

    def take(self):
        if self.pending is None or self.inflight is not None:
            raise RuntimeError(
                "A VLM RPC requires one waiting frame and an idle worker"
            )
        self.inflight = self.pending
        self.pending = None
        self.events.emit(
            "vision_started",
            source_tick=self.inflight.tick,
            capture_s=self.inflight.capture_s,
            camera_queue=self.snapshot(),
        )
        return self.inflight

    def publish(self, result, rpc_completed_s=None):
        meta = result["meta"]
        if self.inflight is None or (
            meta["source_tick"] != self.inflight.tick
            or meta["capture_s"] != self.inflight.capture_s
        ):
            raise ValueError(
                "Published VLM features must belong to the in-flight capture"
            )
        self.latest = result
        self.inflight = None
        self.events.emit(
            "feature_published",
            rpc_completed_s=rpc_completed_s,
            source_tick=meta["source_tick"],
            capture_s=meta["capture_s"],
            feature_sequence=result["sequence"],
            server_vlm_seconds=meta.get("vlm_seconds"),
            camera_queue=self.snapshot(),
        )

    def drop_pending(self):
        dropped = self.pending
        self.pending = None
        if dropped is not None:
            self.events.emit(
                "camera_dropped",
                source_tick=dropped.tick,
                capture_s=dropped.capture_s,
                reason="control_ended",
                camera_queue=self.snapshot(),
            )
        return dropped


def feature_dependency(visual):
    """Small metadata view; never put image tensors or embeddings into the trace."""
    meta = visual["meta"] if visual else {}
    return {
        "feature_sequence": visual.get("sequence") if visual else None,
        "feature_source_tick": meta.get("source_tick"),
        "feature_capture_s": meta.get("capture_s"),
    }
