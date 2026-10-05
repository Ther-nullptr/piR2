"""Single-GPU periodic replay with independent S1 and camera arrival clocks."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from coexecution.cache import FeatureCache
from coexecution.run import Runner, stats
from coexecution.workload import PiR2Workload


def run_periodic(workload, runner, args, return_outputs=False):
    """Run the established arrival/drop protocol on an already prepared runtime."""
    runner.paired("concurrent", 5, 5)
    workload.reset()
    cache = FeatureCache(workload.initial_features)
    period_ns = int(args.period_ms * 1e6)
    camera_period_ns = int(1e9 / args.camera_hz)
    origin = time.perf_counter_ns()
    cache.slots[0].capture_ns = origin
    end_at = origin + int(args.seconds * 1e9)
    next_release, release_index, last_camera_frame = origin, 0, -1
    fast, slow = None, None
    calls, updates, drops, outputs = [], [], [], []
    sources = {0: 0}
    blocked_writes = 0
    torch.cuda.nvtx.range_push("PERIODIC/concurrent")
    while next_release < end_at or fast is not None or slow is not None:
        now = time.perf_counter_ns()
        if fast is not None:
            if fast["job"] is None and fast["future"].done():
                fast["job"] = fast["future"].result()
            job = fast["job"]
            if job is not None and job["end"].query():
                ready = time.perf_counter_ns()
                latency_ms = (ready - fast["release_ns"]) / 1e6
                calls.append(
                    {
                        "release_index": fast["index"],
                        "release_ns": fast["release_ns"],
                        "host_start_ns": job["host_start_ns"],
                        "ready_ns": ready,
                        "latency_ms": latency_ms,
                        "deadline_miss": latency_ms > args.period_ms,
                        "gpu_span_ms": job["start"].elapsed_time(job["end"]),
                        "cache_generation": job["generation"],
                        "cache_slot": job["slot"],
                        "cache_source_index": sources[job["generation"]],
                        "feature_capture_ns": job["capture_ns"],
                        "feature_age_at_start_ms": (
                            job["host_start_ns"] - job["capture_ns"]
                        )
                        / 1e6,
                        "feature_age_at_ready_ms": (ready - job["capture_ns"]) / 1e6,
                    }
                )
                outputs.append(job["output"].detach())
                fast = None
        if slow is not None:
            if slow["job"] is None and slow["future"].done():
                slow["job"] = slow["future"].result()
            job = slow["job"]
            if job is not None and job["end"].query():
                assert cache.publish_ready()
                sources[job["generation"]] = slow["index"]
                updates.append(
                    {
                        "generation": job["generation"],
                        "input_index": slow["index"],
                        "capture_ns": slow["capture_ns"],
                        "publish_ns": time.perf_counter_ns(),
                        "gpu_span_ms": job["start"].elapsed_time(job["end"]),
                    }
                )
                slow = None
        while next_release <= now and next_release < end_at:
            if fast is not None or now - next_release >= period_ns:
                drops.append(
                    {
                        "release_index": release_index,
                        "release_ns": next_release,
                        "reason": "s1_busy" if fast is not None else "expired",
                    }
                )
            else:
                lease = cache.acquire()
                fast = {
                    "future": runner.fast_pool.submit(
                        runner.fast_job, cache, lease, release_index, "periodic"
                    ),
                    "job": None,
                    "release_ns": next_release,
                    "index": release_index,
                }
            release_index += 1
            next_release += period_ns
        if now < end_at and slow is None:
            camera_frame = (now - origin) // camera_period_ns
            if camera_frame > last_camera_frame:
                capture = origin + camera_frame * camera_period_ns
                ticket = cache.reserve_write(capture)
                if ticket is not None:
                    slow = {
                        "future": runner.slow_pool.submit(
                            runner.slow_job, cache, ticket, camera_frame, "periodic"
                        ),
                        "job": None,
                        "index": camera_frame,
                        "capture_ns": capture,
                    }
                    last_camera_frame = camera_frame
                else:
                    blocked_writes += 1
        time.sleep(0.0005)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    planned_releases = len(range(origin, end_at, period_ns))
    recorded_ids = [row["release_index"] for row in calls + drops]
    assert release_index == planned_releases
    assert sorted(recorded_ids) == list(range(planned_releases)), (
        "Missing or duplicated release"
    )
    # Replay every observed cache version/state input in serial after measurement.
    # This checks the async runtime's data semantics independently of completion order.
    workload.reset()
    max_error = 0.0
    with torch.inference_mode():
        for row, asynchronous in zip(calls, outputs):
            source = runner.features[row["cache_source_index"] % len(runner.features)]
            reference = workload.fast(source, row["release_index"])
            error = float((asynchronous.float() - reference.float()).abs().max())
            max_error = max(max_error, error)
            torch.testing.assert_close(asynchronous, reference, rtol=1e-3, atol=1e-3)
            assert torch.isfinite(asynchronous).all()
    missed = len(drops) + sum(row["deadline_miss"] for row in calls)
    result = {
        "physical_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": 1,
        "duration_seconds": args.seconds,
        "period_ms": args.period_ms,
        "camera_hz": args.camera_hz,
        "scope": "GPU-resident replay, fixed slide_steps=1 and image_delay embedding=1; excludes sensor I/O/preprocessing/action decode",
        "scheduled_releases": release_index,
        "planned_releases": planned_releases,
        "completed_s1": len(calls),
        "dropped_releases": len(drops),
        "deadline_misses_including_drops": missed,
        "deadline_miss_fraction": missed / release_index,
        "s2_updates": len(updates),
        "blocked_write_polls": blocked_writes,
        "latency_ms_completed": stats([row["latency_ms"] for row in calls]),
        "feature_age_at_start_ms": stats(
            [row["feature_age_at_start_ms"] for row in calls]
        ),
        "feature_age_at_ready_ms": stats(
            [row["feature_age_at_ready_ms"] for row in calls]
        ),
        "async_serial_max_abs": max_error,
        "calls": calls,
        "updates": updates,
        "drops": drops,
        "cache_audit": cache.audit,
    }
    return (result, outputs) if return_outputs else result


def main(args):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    workload = PiR2Workload(args.checkpoint, samples=8)
    runner = Runner(workload)
    result = run_periodic(workload, runner, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    runner.close()
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["calls", "updates", "drops", "cache_audit"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/pir2-so100-smoke/checkpoint-10"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--period-ms", type=float, default=40)
    parser.add_argument("--camera-hz", type=float, default=30)
    main(parser.parse_args())
