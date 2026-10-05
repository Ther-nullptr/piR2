"""Single physical GPU benchmark: actual S1/S2, versioned cache and CUDA streams."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import torch

from coexecution.cache import FeatureCache
from coexecution.workload import PiR2Workload

MODES = ["s1_only", "s2_only", "serial", "concurrent"]


def stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99_empirical": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


class Runner:
    def __init__(self, workload):
        self.w = workload
        self.fast_stream = torch.cuda.Stream(priority=0)
        self.slow_stream = torch.cuda.Stream(priority=0)
        self.fast_pool = ThreadPoolExecutor(1, thread_name_prefix="S1")
        self.slow_pool = ThreadPoolExecutor(1, thread_name_prefix="S2")
        # Equal priority keeps this first baseline independent of scheduling policy changes.
        with torch.inference_mode():
            self.features = [
                dict(workload.slow(i)) for i in range(len(workload.backbone_inputs))
            ]
        torch.cuda.synchronize()

    def fast_job(self, cache, lease, index, mode, barrier=None):
        with torch.inference_mode(), torch.cuda.stream(self.fast_stream):
            if barrier is not None:
                barrier.wait(timeout=30)
            host_start = time.perf_counter_ns()
            torch.cuda.nvtx.range_push(
                f"S1/{mode}/input={index}/cache={lease.generation}"
            )
            self.fast_stream.wait_event(
                lease.ready
            )  # The published event is already complete.
            for tensor in lease.tensors.values():
                tensor.record_stream(self.fast_stream)
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            output = self.w.fast(lease.tensors, index)
            end.record()
            torch.cuda.nvtx.range_pop()
            cache.release(lease, end)
            return {
                "start": start,
                "end": end,
                "output": output,
                "host_start_ns": host_start,
                "host_submit_end_ns": time.perf_counter_ns(),
                "generation": lease.generation,
                "capture_ns": lease.capture_ns,
                "slot": lease.slot,
            }

    def slow_job(self, cache, ticket, index, mode, barrier=None, dependency=None):
        with torch.inference_mode(), torch.cuda.stream(self.slow_stream):
            if barrier is not None:
                barrier.wait(timeout=30)
            if dependency is not None:
                self.slow_stream.wait_event(dependency)
            host_start = time.perf_counter_ns()
            torch.cuda.nvtx.range_push(
                f"S2/{mode}/input={index}/publish={ticket.generation}"
            )
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            features = self.w.slow(index)
            cache.copy_into(ticket, features)
            end.record()
            torch.cuda.nvtx.range_pop()
            cache.finish_write(ticket, end)
            return {
                "start": start,
                "end": end,
                "host_start_ns": host_start,
                "host_submit_end_ns": time.perf_counter_ns(),
                "generation": ticket.generation,
            }

    def install_reference(self, cache, index):
        ticket = cache.reserve_write(time.perf_counter_ns())
        assert ticket is not None
        with torch.cuda.stream(self.slow_stream):
            cache.copy_into(ticket, self.features[index % len(self.features)])
            complete = torch.cuda.Event()
            complete.record()
        cache.finish_write(ticket, complete)
        complete.synchronize()
        assert cache.publish_ready()

    def paired(self, mode, iterations, warmup, keep_outputs=False, record_audit=False):
        self.w.reset()
        cache = FeatureCache(self.w.initial_features)
        rows, outputs = [], []
        for iteration in range(warmup + iterations):
            current = (iteration + 1) % len(self.features)
            if mode == "s1_only" and iteration:
                # Identical feature values to the prior S2 result in paired modes.
                # A precomputed producer is installed before starting the S1 timer.
                self.install_reference(cache, iteration)
            lease = cache.acquire() if mode != "s2_only" else None
            ticket = (
                cache.reserve_write(time.perf_counter_ns())
                if mode != "s1_only"
                else None
            )
            if mode != "s1_only":
                assert ticket is not None, (
                    "A completed pair must release the retired slot"
                )
            origin = torch.cuda.Event(enable_timing=True)
            origin.record()
            origin.synchronize()  # Measurement boundary: neither worker has started this pair.
            host_start = time.perf_counter_ns()
            fast, slow = None, None
            torch.cuda.nvtx.range_push(f"PAIR/{mode}/{iteration}")
            if mode == "concurrent":
                barrier = Barrier(2)
                f = self.fast_pool.submit(
                    self.fast_job, cache, lease, current, mode, barrier
                )
                s = self.slow_pool.submit(
                    self.slow_job, cache, ticket, current, mode, barrier
                )
                fast, slow = f.result(), s.result()
            else:
                if lease is not None:
                    fast = self.fast_pool.submit(
                        self.fast_job, cache, lease, current, mode
                    ).result()
                if ticket is not None:
                    slow = self.slow_pool.submit(
                        self.slow_job,
                        cache,
                        ticket,
                        current,
                        mode,
                        None,
                        fast["end"] if fast else None,
                    ).result()
            # These waits occur only after both workers have submitted their GPU work.
            for job in [fast, slow]:
                if job:
                    job["end"].synchronize()
            wall_ms = (time.perf_counter_ns() - host_start) / 1e6
            if slow:
                assert cache.publish_ready()
            torch.cuda.nvtx.range_pop()
            row = {"mode": mode, "iteration": iteration - warmup, "wall_ms": wall_ms}
            for label, job in [("s1", fast), ("s2", slow)]:
                if job:
                    row[label + "_gpu_span_ms"] = job["start"].elapsed_time(job["end"])
                    row[label + "_start_ms"] = origin.elapsed_time(job["start"])
                    row[label + "_end_ms"] = origin.elapsed_time(job["end"])
                    row[label + "_generation"] = job["generation"]
            if fast and slow:
                row["event_span_intersection_ms"] = max(
                    0.0,
                    min(row["s1_end_ms"], row["s2_end_ms"])
                    - max(row["s1_start_ms"], row["s2_start_ms"]),
                )
                row["note"] = (
                    "Event spans include gaps; actual kernel overlap requires the profiler trace"
                )
            if iteration >= warmup:
                rows.append(row)
                if keep_outputs and fast:
                    outputs.append(fast["output"].detach().clone())
        torch.cuda.synchronize()
        return rows, outputs, cache.audit if record_audit else []

    def close(self):
        self.fast_pool.shutdown(wait=True)
        self.slow_pool.shutdown(wait=True)


def main(args):
    assert torch.cuda.device_count() == 1, "Exactly one physical GPU must be visible"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    workload = PiR2Workload(
        args.checkpoint, samples=args.samples, features_only=not args.original_s2
    )
    runner = Runner(workload)
    result = {
        "checkpoint": str(args.checkpoint),
        "physical_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "scope": "GPU-resident real-input replay: excludes preprocessing, CPU action decoding and robot I/O",
        "quantization": False,
        "cuda_graphs": False,
        "profiled": args.profile,
        "workload": workload.metadata(),
        "stream_handles": {
            "s1": runner.fast_stream.cuda_stream,
            "s2": runner.slow_stream.cuda_stream,
        },
        "rows": [],
        "cache_audit": [],
        "summary": {},
    }
    # Fixed input order and identical RNG/buffer reset: concurrency must preserve actions.
    serial, reference, _ = runner.paired("serial", 6, 2, keep_outputs=True)
    parallel, actual, audit = runner.paired(
        "concurrent", 6, 2, keep_outputs=True, record_audit=True
    )
    max_abs = 0.0
    for a, b, r, s in zip(reference, actual, serial, parallel):
        assert r["s1_generation"] == s["s1_generation"]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        max_abs = max(max_abs, (a.float() - b.float()).abs().max().item())
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
    result["correctness"] = {
        "compared_actions": len(reference),
        "serial_concurrent_max_abs": max_abs,
        "matching_cache_versions": True,
        "all_finite": True,
    }
    result["cache_audit"] = audit
    print("CORRECTNESS", json.dumps(result["correctness"]), flush=True)
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(args.repeats):
        modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
        for mode in modes:
            torch.cuda.nvtx.range_push(f"MODE/{mode}/repeat={repeat}")
            rows, _, _ = runner.paired(mode, args.iterations, args.warmup)
            torch.cuda.nvtx.range_pop()
            for row in rows:
                row["repeat"] = repeat
            result["rows"].extend(rows)
            print(
                "FINISHED",
                mode,
                repeat,
                stats([r["wall_ms"] for r in rows]),
                flush=True,
            )
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    for mode in args.modes:
        rows = [row for row in result["rows"] if row["mode"] == mode]
        result["summary"][mode] = {
            key: stats([r[key] for r in rows])
            for key in [
                "wall_ms",
                "s1_gpu_span_ms",
                "s2_gpu_span_ms",
                "event_span_intersection_ms",
            ]
            if key in rows[0]
        }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    runner.close()
    print("SAVED", args.output / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/pir2-so100-smoke/checkpoint-10"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--original-s2", action="store_true")
    main(parser.parse_args())
