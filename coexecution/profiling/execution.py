"""Independent CUDA submissions with versioned feature ownership and timing."""

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .common import observe_ready, response_ms
from .scheduling import CooperativeGate


class PairRunner:
    def __init__(self, workload, args, priorities):
        from coexecution.cache import FeatureCache

        self.w, self.args = workload, args
        self.vlm_stream = torch.cuda.Stream(priority=0)
        self.dit_stream = torch.cuda.Stream(priority=0)
        self.priority_stream = torch.cuda.Stream(priority=priorities[1])
        if self.priority_stream.priority >= self.vlm_stream.priority:
            raise RuntimeError(
                "GPU has no higher-priority stream for the requested comparison"
            )
        self.vlm_pool = ThreadPoolExecutor(1, thread_name_prefix="S2-VLM")
        self.dit_pool = ThreadPoolExecutor(1, thread_name_prefix="S1-DiT")
        with torch.inference_mode():
            rng_before = torch.cuda.get_rng_state()
            self.reference_features = [
                {
                    key: value.detach().clone()
                    for key, value in workload.vlm(index).items()
                }
                for index in range(args.frames)
            ]
            torch.cuda.synchronize()
            if not torch.equal(rng_before, torch.cuda.get_rng_state()):
                raise RuntimeError("VLM unexpectedly consumes shared CUDA RNG")
            self.cache = FeatureCache(self.reference_features[0])
        self.layer_modules = [
            (name, module)
            for name, module in workload.model.backbone.named_modules()
            if type(module).__name__
            in ("Qwen3VLVisionBlock", "Qwen3VLTextDecoderLayer")
        ]
        if not self.layer_modules:
            raise RuntimeError("VLM cooperative boundaries not found")
        self.stream_metadata = {
            "vlm": {
                "id": self.vlm_stream.cuda_stream,
                "priority": self.vlm_stream.priority,
            },
            "dit_equal": {
                "id": self.dit_stream.cuda_stream,
                "priority": self.dit_stream.priority,
            },
            "dit_priority": {
                "id": self.priority_stream.cuda_stream,
                "priority": self.priority_stream.priority,
            },
            "supported_priority_range": priorities,
            "priority_semantics": "Scheduling hint; never a claim that a running kernel is forcibly interrupted",
        }

    def prepare(self, index):
        self.w.restore()
        ticket = self.cache.reserve_write(time.perf_counter_ns())
        if ticket is None:
            raise RuntimeError("No cache slot available while both workers are idle")
        with torch.cuda.stream(self.vlm_stream), torch.inference_mode():
            self.cache.copy_into(ticket, self.reference_features[index])
            ready = torch.cuda.Event()
            ready.record()
        self.cache.finish_write(ticket, ready)
        ready.synchronize()
        if not self.cache.publish_ready():
            raise RuntimeError("Prepared old generation did not publish")
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        origin.synchronize()
        return origin

    @contextlib.contextmanager
    def chunk_hooks(self, gate, enabled):
        handles = []
        counter = 0
        if enabled:

            def complete(_module, _inputs, _output, name):
                nonlocal counter
                counter += 1
                if counter % self.args.vlm_chunk_blocks == 0 or counter == len(
                    self.layer_modules
                ):
                    with torch.cuda.nvtx.range(f"S2/VLM/cooperative_boundary/{name}"):
                        self.vlm_stream.synchronize()
                        gate.boundary(name)

            for name, module in self.layer_modules:
                handles.append(
                    module.register_forward_hook(
                        lambda module, inputs, output, name=name: complete(
                            module, inputs, output, name
                        )
                    )
                )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def dit_job(self, lease, index, mode, iteration, gate, barrier):
        stream = (
            self.priority_stream
            if mode in ("concurrent_dit_priority", "cooperative_dit_priority")
            else self.dit_stream
        )
        try:
            with (
                torch.cuda.device(0),
                torch.inference_mode(),
                torch.cuda.stream(stream),
            ):
                if barrier is not None:
                    barrier.wait(timeout=60)

                def mark_release():
                    torch.cuda.nvtx.mark(f"S1_RELEASE/{mode}/iteration={iteration}")

                if barrier is not None:
                    release = gate.release_dit(
                        self.args.dit_offset_ms,
                        mode == "cooperative_dit_priority",
                        mark_release,
                    )
                else:
                    release = time.perf_counter_ns()
                    mark_release()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                submit_start = time.perf_counter_ns()
                with torch.cuda.nvtx.range(
                    f"S1/DIT/{mode}/iteration={iteration}/input={index}/cache={lease.generation}"
                ):
                    stream.wait_event(lease.ready)
                    for tensor in lease.tensors.values():
                        tensor.record_stream(stream)
                    start.record()
                    output = self.w.dit(lease.tensors, index)
                    end.record()
                submit_end = time.perf_counter_ns()
                self.cache.release(lease, end)
                # All modes observe the same response boundary on this job's stream.
                ready_ns = observe_ready(end)
                torch.cuda.nvtx.mark(f"S1_READY/{mode}/iteration={iteration}")
                gate.complete_dit()
                return {
                    "start": start,
                    "end": end,
                    "output": output,
                    "release_ns": release,
                    "ready_ns": ready_ns,
                    "host_submit_start_ns": submit_start,
                    "host_submit_end_ns": submit_end,
                    "generation": lease.generation,
                }
        except BaseException:
            gate.abort()
            raise

    def vlm_job(self, ticket, index, mode, iteration, gate, barrier, dependency):
        try:
            with (
                torch.cuda.device(0),
                torch.inference_mode(),
                torch.cuda.stream(self.vlm_stream),
            ):
                if barrier is not None:
                    barrier.wait(timeout=60)
                if dependency is not None:
                    self.vlm_stream.wait_event(dependency)
                release = gate.start_vlm()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                submit_start = time.perf_counter_ns()
                cooperative = mode in (
                    "cooperative_dit_priority",
                    "vlm_only_cooperative",
                )
                with (
                    torch.cuda.nvtx.range(
                        f"S2/VLM/{mode}/iteration={iteration}/input={index}/publish={ticket.generation}"
                    ),
                    self.chunk_hooks(gate, cooperative),
                ):
                    start.record()
                    features = self.w.vlm(index)
                    self.cache.copy_into(ticket, features)
                    end.record()
                submit_end = time.perf_counter_ns()
                self.cache.finish_write(ticket, end)
                # Only this VLM completion event is synchronized, never the DiT stream.
                ready_ns = observe_ready(end)
                torch.cuda.nvtx.mark(f"S2_READY/{mode}/iteration={iteration}")
                gate.complete_vlm()
                return {
                    "start": start,
                    "end": end,
                    "output": features,
                    "release_ns": release,
                    "ready_ns": ready_ns,
                    "host_submit_start_ns": submit_start,
                    "host_submit_end_ns": submit_end,
                    "generation": ticket.generation,
                }
        except BaseException:
            gate.abort()
            raise

    def pair(self, mode, index, iteration, capture=False):
        with torch.cuda.nvtx.range(
            f"RESTORE/{mode}/iteration={iteration}/fixture={index}"
        ):
            origin = self.prepare(index)
        next_index = (index + 1) % self.args.frames
        has_dit = mode not in ("vlm_only", "vlm_only_cooperative")
        has_vlm = mode != "dit_only"
        lease = self.cache.acquire() if has_dit else None
        ticket = self.cache.reserve_write(time.perf_counter_ns()) if has_vlm else None
        if has_vlm and ticket is None:
            raise RuntimeError("Previous generation still has a reader")
        gate = CooperativeGate()
        before_counts = dict(self.w.counts)
        fast, slow = None, None
        capture_started = False
        try:
            if capture:
                rc = torch.cuda.cudart().cudaProfilerStart()
                if int(rc) != 0:
                    raise RuntimeError(f"cudaProfilerStart failed: {rc}")
                capture_started = True
            with torch.cuda.nvtx.range(f"PAIR/{mode}/{iteration}"):
                host_start = time.perf_counter_ns()
                if mode in (
                    "concurrent_equal",
                    "concurrent_dit_priority",
                    "cooperative_dit_priority",
                ):
                    barrier = threading.Barrier(2)
                    slow_future = self.vlm_pool.submit(
                        self.vlm_job,
                        ticket,
                        next_index,
                        mode,
                        iteration,
                        gate,
                        barrier,
                        None,
                    )
                    fast_future = self.dit_pool.submit(
                        self.dit_job, lease, index, mode, iteration, gate, barrier
                    )
                    fast, slow = fast_future.result(), slow_future.result()
                else:
                    if has_dit:
                        fast = self.dit_pool.submit(
                            self.dit_job, lease, index, mode, iteration, gate, None
                        ).result()
                    if has_vlm:
                        slow = self.vlm_pool.submit(
                            self.vlm_job,
                            ticket,
                            next_index,
                            mode,
                            iteration,
                            gate,
                            None,
                            fast["end"] if fast else None,
                        ).result()
                for job in (fast, slow):
                    if job:
                        job["end"].synchronize()
                host_end = time.perf_counter_ns()
        except BaseException:
            gate.abort()
            raise
        finally:
            if capture_started:
                torch.cuda.cudart().cudaProfilerStop()
        if slow and not self.cache.publish_ready():
            raise RuntimeError(
                "New generation published before completion or not at all"
            )
        counts = {key: self.w.counts[key] - before_counts[key] for key in before_counts}
        if counts != {"vlm": int(has_vlm), "dit": int(has_dit)}:
            raise RuntimeError(f"Unexpected forward count: {counts}")
        row = {
            "mode": mode,
            "iteration": iteration,
            "fixture": index,
            "wall_ms": (host_end - host_start) / 1e6,
            "host_start_ns": host_start,
            "host_end_ns": host_end,
            "forward_counts": counts,
            "requested_dit_offset_ms": self.args.dit_offset_ms,
            "scheduling_events": gate.events,
        }
        for label, job, frame in (("s1", fast, index), ("s2", slow, next_index)):
            if job:
                row.update(
                    {
                        f"{label}_response_ms": response_ms(
                            job["release_ns"], job["ready_ns"]
                        ),
                        f"{label}_gpu_span_ms": job["start"].elapsed_time(job["end"]),
                        f"{label}_start_ms": origin.elapsed_time(job["start"]),
                        f"{label}_end_ms": origin.elapsed_time(job["end"]),
                        f"{label}_host_submit_ms": (
                            job["host_submit_end_ns"] - job["host_submit_start_ns"]
                        )
                        / 1e6,
                        f"{label}_generation": job["generation"],
                        f"{label}_feature_dataset_step": self.w.fixtures[frame][
                            "image_dataset_step"
                        ],
                    }
                )
                for field in (
                    "release_ns",
                    "ready_ns",
                    "host_submit_start_ns",
                    "host_submit_end_ns",
                ):
                    row[f"{label}_{field}"] = job[field]
        if fast and slow:
            if fast["generation"] >= slow["generation"]:
                raise RuntimeError(
                    "DiT did not consume a strictly older feature generation"
                )
            row["actual_dit_release_offset_ms"] = (
                fast["release_ns"] - slow["release_ns"]
            ) / 1e6
            row["event_span_intersection_ms"] = max(
                0.0,
                min(row["s1_end_ms"], row["s2_end_ms"])
                - max(row["s1_start_ms"], row["s2_start_ms"]),
            )
        return row, {
            "dit": fast["output"] if fast else None,
            "vlm": slow["output"] if slow else None,
        }

    def close(self):
        self.vlm_pool.shutdown(wait=True, cancel_futures=True)
        self.dit_pool.shutdown(wait=True, cancel_futures=True)
