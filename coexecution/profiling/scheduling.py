"""Host protocol for offset arrival and cooperative VLM layer-boundary yield."""

import threading
import time


class CooperativeGate:
    def __init__(self, timeout_s=60.0):
        self.timeout_s = timeout_s
        self.vlm_started = threading.Event()
        self.requested = threading.Event()
        self.permit = threading.Event()
        self.dit_done = threading.Event()
        self.aborted = threading.Event()
        self.events = []
        self.vlm_release_ns = None

    def record(self, event, **fields):
        self.events.append(
            {"event": event, "host_ns": time.perf_counter_ns(), **fields}
        )

    def wait(self, event, label):
        deadline = time.monotonic() + self.timeout_s
        while not event.wait(0.02):
            if self.aborted.is_set():
                raise RuntimeError(f"Worker aborted while waiting for {label}")
            if time.monotonic() >= deadline:
                self.abort()
                raise TimeoutError(f"Timed out waiting for {label}")
        if self.aborted.is_set():
            raise RuntimeError(f"Worker aborted at {label}")

    def start_vlm(self):
        self.vlm_release_ns = time.perf_counter_ns()
        self.record("vlm_release")
        self.vlm_started.set()
        return self.vlm_release_ns

    def release_dit(self, offset_ms, cooperative, on_release=None):
        self.wait(self.vlm_started, "VLM release")
        target_ns = self.vlm_release_ns + round(offset_ms * 1e6)
        remaining = (target_ns - time.perf_counter_ns()) / 1e9
        if remaining > 0 and self.aborted.wait(remaining):
            raise RuntimeError("Worker aborted before DiT release")
        release_ns = time.perf_counter_ns()
        self.record("dit_release", requested_offset_ms=offset_ms)
        if on_release is not None:
            on_release()
        if cooperative:
            self.requested.set()
            self.wait(self.permit, "completed VLM layer boundary")
        return release_ns

    def boundary(self, layer):
        self.record("vlm_boundary_complete", layer=layer)
        if self.requested.is_set() and not self.dit_done.is_set():
            self.record("vlm_yield", layer=layer)
            self.permit.set()
            self.wait(self.dit_done, "DiT CUDA completion")
            self.record("vlm_resume", layer=layer)

    def complete_vlm(self):
        self.record("vlm_complete")
        # A request arriving after VLM completion must never deadlock.
        self.permit.set()

    def complete_dit(self):
        self.record("dit_complete")
        self.dit_done.set()

    def abort(self):
        self.aborted.set()
        self.vlm_started.set()
        self.permit.set()
        self.dit_done.set()
