"""Framework-free Flow4 request scheduling and numerical evidence contracts."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

from .common import response_ms
from .scheduling import CooperativeGate

MODES = (
    "vlm_only",
    "dit_only",
    "serial",
    "concurrent_equal",
    "concurrent_dit_priority",
    "vlm_only_fenced",
    "serial_fenced",
    "concurrent_equal_fenced",
    "concurrent_dit_priority_fenced",
    "cooperative_dit_priority",
)


@dataclass(frozen=True)
class FlowCase:
    mode: str
    offset_ms: float = 0.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown Flow4 scheduling mode: {self.mode}")
        if not math.isfinite(self.offset_ms) or self.offset_ms < 0:
            raise ValueError("A finite nonnegative offset is required")
        if not self.concurrent and self.offset_ms:
            raise ValueError("Offsets apply only to concurrent requests")

    @property
    def concurrent(self):
        return self.mode.startswith(("concurrent_", "cooperative_"))

    @property
    def cooperative(self):
        return self.mode == "cooperative_dit_priority"

    @property
    def fenced(self):
        return self.cooperative or self.mode.endswith("_fenced")

    @property
    def priority(self):
        return "dit_priority" in self.mode

    @property
    def has_vlm(self):
        return self.mode != "dit_only"

    @property
    def has_dit(self):
        return not self.mode.startswith("vlm_only")

    @property
    def identity(self):
        return f"{self.mode}@{self.offset_ms:.17g}ms"


def validate_flow_head(head):
    if (
        head.config.streaming
        or head.config.action_horizon != 40
        or head.action_horizon != 40
        or head.num_inference_timesteps != 4
    ):
        raise ValueError("Official diagnostic requires non-streaming H40 Flow4")


def difference(reference, candidate):
    """Strict structural/finite admission; numerical drift is measured, not vetoed."""
    import numpy as np

    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("Output keys differ or are empty")
    total, absolute, square, norm, maximum, equal = 0, 0.0, 0.0, 0.0, 0.0, True
    for key in reference:
        left, right = np.asarray(reference[key]), np.asarray(candidate[key])
        if left.shape != right.shape:
            raise ValueError(f"Output shape changed for {key}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Nonfinite output for {key}")
        equal = equal and left.dtype == right.dtype and np.array_equal(left, right)
        delta = right.astype(np.float64) - left.astype(np.float64)
        total += delta.size
        absolute += float(np.abs(delta).sum())
        square += float(np.square(delta).sum())
        norm += float(np.square(left.astype(np.float64)).sum())
        if delta.size:
            maximum = max(maximum, float(np.abs(delta).max()))
    return {
        "passes": True,
        "bitwise_equal": bool(equal),
        "elements": total,
        "mae": absolute / max(total, 1),
        "max_abs": maximum,
        "relative_l2": math.sqrt(square / norm)
        if norm
        else (0.0 if not square else None),
        "reference_l2": math.sqrt(norm),
        "contract": "Keys, shapes and finiteness are strict; numerical drift is reported without an equality threshold.",
    }


class FlowSchedule:
    """Two persistent host workers; each executor observes its own CUDA end event.

    ``execute(role, case, fixture, gate, request_id)`` returns output, host submit
    timestamps, host-observed ready timestamp and its local CUDA event span.
    Preparation/RNG restore and output snapshots belong outside ``pair``. Every
    submitted worker finishes, including on failure, before ``pair`` returns.
    """

    def __init__(self, execute, *, timeout_s=60.0):
        self.execute, self.timeout_s = execute, timeout_s
        self.pools = {
            role: ThreadPoolExecutor(1, thread_name_prefix=f"Flow4-{role}")
            for role in ("vlm", "dit")
        }
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        for pool in self.pools.values():
            pool.shutdown(wait=True)

    def warm(self, role, case, index):
        """Serial cold execution on the eventual owning host worker/stream."""
        with self.lock:
            return (
                self.pools[role]
                .submit(
                    self.execute,
                    role,
                    case,
                    index,
                    CooperativeGate(self.timeout_s),
                    f"warm/{role}/{index}",
                )
                .result()
            )

    def _job(self, role, case, index, gate, barrier, request_id):
        try:
            if barrier is not None:
                barrier.wait(timeout=self.timeout_s)
            if role == "vlm":
                release = gate.start_vlm()
            elif case.concurrent:
                release = gate.release_dit(case.offset_ms, case.cooperative)
            else:
                release = time.perf_counter_ns()
            result = self.execute(role, case, index, gate, request_id)
            result["release_ns"] = release
            result["request_id"] = request_id
            if (
                not release
                <= result["host_submit_start_ns"]
                <= result["host_submit_end_ns"]
                <= result["ready_ns"]
            ):
                raise RuntimeError(
                    "Request timestamps violate release/submission/readiness order"
                )
            if role == "vlm":
                gate.complete_vlm()
            else:
                gate.complete_dit()
            return result
        except BaseException:
            gate.abort()
            if barrier is not None:
                barrier.abort()
            raise

    def pair(self, case, index, iteration, *, old_generation, next_generation):
        if old_generation >= next_generation:
            raise ValueError(
                "S1 must consume a previously completed feature generation"
            )
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Pairs cannot overlap preparation or output lifetimes")
        try:
            return self._pair(case, index, iteration, old_generation, next_generation)
        finally:
            self.lock.release()

    def _pair(self, case, index, iteration, old_generation, next_generation):
        gate = CooperativeGate(self.timeout_s)
        barrier = threading.Barrier(2) if case.concurrent else None
        pending, results = {}, {}
        start = time.perf_counter_ns()

        def submit(role):
            identity = f"{case.identity}/i{iteration}/f{index}/{role}"
            pending[role] = self.pools[role].submit(
                self._job, role, case, index, gate, barrier, identity
            )

        try:
            if case.has_dit:
                submit("dit")
                if not case.concurrent:
                    results["dit"] = pending["dit"].result()
            if case.has_vlm:
                submit("vlm")
            for role, future in pending.items():
                results[role] = future.result()
        except BaseException:
            gate.abort()
            if barrier is not None:
                barrier.abort()
            # Do not let the caller restore RNG or reuse storage while a peer runs.
            wait(pending.values())
            raise
        finish = max(value["ready_ns"] for value in results.values())
        row = {
            "mode": case.mode,
            "case": case.identity,
            "offset_ms": case.offset_ms,
            "fixture": index,
            "iteration": iteration,
            "pair_release_ns": start,
            "pair_ready_ns": finish,
            "pair_makespan_ms": response_ms(start, finish),
            "boundary_fences": case.fenced,
            "cooperative": case.cooperative,
            "s1_feature_generation": old_generation if case.has_dit else None,
            "s2_feature_generation": next_generation if case.has_vlm else None,
            "gate_events": gate.events,
        }
        outputs = {}
        for role, value in results.items():
            prefix = "s1" if role == "dit" else "s2"
            outputs[role] = value["output"]
            for name in (
                "release_ns",
                "ready_ns",
                "request_id",
                "host_submit_start_ns",
                "host_submit_end_ns",
                "gpu_span_ms",
            ):
                row[f"{prefix}_{name}"] = value[name]
            row[f"{prefix}_response_ms"] = response_ms(
                value["release_ns"], value["ready_ns"]
            )
            row[f"{prefix}_pre_submit_wait_ms"] = response_ms(
                value["release_ns"], value["host_submit_start_ns"]
            )
            row[f"{prefix}_host_submit_ms"] = response_ms(
                value["host_submit_start_ns"], value["host_submit_end_ns"]
            )
            row[f"{prefix}_completion_wait_ms"] = response_ms(
                value["host_submit_end_ns"], value["ready_ns"]
            )
        return row, outputs
