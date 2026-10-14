"""Real LIBERO piR2 VLM/DiT contention diagnostic; explicit CUDA entrypoint.

Importing this module and requesting CLI help never loads a tensor framework.
The campaign owns GPU admission and independent hardware-counter passes.
"""

import argparse
import contextlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from .common import (
    create_output,
    cuda_capture_range,
    digest,
    launcher_state,
    save,
    stats,
)
from .runtime import ComponentBoundaries, executed_projections

ROOT = Path(__file__).resolve().parents[2]
MODES = (
    "vlm_only",
    "dit_only",
    "serial",
    "concurrent_equal",
    "concurrent_dit_priority",
    "cooperative_dit_priority",
    "vlm_only_cooperative",
)


def run(args):
    validate_arguments(args)
    create_output(args)
    save(
        args.output / "status.json",
        {
            "state": "loading",
            "pid": os.getpid(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
    )
    sys.path.insert(0, str(args.groot_root.resolve()))
    import torch

    from coexecution.optimization.config import OptimizationConfig
    from coexecution.optimization.execution.cache import SerialForwardBoundary
    from coexecution.optimization.session import GrootOptimizations

    from .execution import PairRunner
    from .validation import validate
    from .workload import LiberoWorkload

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one physical GPU must be visible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # PyTorch returns the CUDA (leastPriority, greatestPriority) pair.
    priorities = list(torch.cuda.current_stream().priority_range())
    if not priorities[1] <= -1 <= priorities[0]:
        raise RuntimeError(f"Priority -1 unavailable: {priorities}")
    # Use -1 explicitly rather than silently selecting a more aggressive priority.
    selected_priorities = [priorities[0], -1]
    config = OptimizationConfig(
        precision=args.precision,
        fusion=True,
        scope="transformer",
        component=args.component,
    )
    with torch.inference_mode():
        workload = LiberoWorkload(args)
        with GrootOptimizations(workload.policy, config) as optimized:
            adapter = ComponentBoundaries(optimized, SerialForwardBoundary)
            runner = None
            try:
                runner = PairRunner(workload, args, selected_priorities)
                runner.stream_metadata["supported_priority_range"] = priorities
                save(args.output / "runtime-adapter.json", adapter.evidence)
                save(args.output / "streams.json", runner.stream_metadata)
                with executed_projections(adapter.controllers) as executed:
                    save(
                        args.output / "status.json",
                        {"state": "validating", "pid": os.getpid()},
                    )
                    equivalence = validate(runner, adapter, args)
                    with torch.cuda.nvtx.range("pir2/phase=warmup"):
                        for iteration in range(max(args.warmup, args.frames)):
                            runner.pair(args.mode, iteration % args.frames, iteration)
                assignments = {name: f"w{bits}a{bits}" for name, bits in executed}
                expected = set(optimized.coverage.get("linears", []))
                if expected != set(assignments):
                    raise RuntimeError(
                        f"Executed quantization coverage differs: missing={sorted(expected - set(assignments))}, extra={sorted(set(assignments) - expected)}"
                    )
                actual_coverage = [
                    {
                        "name": name,
                        "precision": f"w{bits}a{bits}",
                        "dispatch_calls": count,
                    }
                    for (name, bits), count in sorted(executed.items())
                ]
                save(args.output / "executed-coverage.json", actual_coverage)
                frozen_launchers = launcher_state()
                save(
                    args.output / "status.json",
                    {"state": "measuring", "pid": os.getpid()},
                )
                samples = []
                resident_memory = {
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                }
                torch.cuda.reset_peak_memory_stats()
                profiler = contextlib.nullcontext()
                if args.capture == "torch":
                    profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    )
                with (
                    profiler as trace,
                    cuda_capture_range(
                        args.capture == "cuda" and args.capture_window == "whole"
                    ),
                    torch.cuda.nvtx.range("pir2/phase=measurement"),
                ):
                    for iteration in range(args.iterations):
                        row, _ = runner.pair(
                            args.mode,
                            iteration % args.frames,
                            iteration,
                            capture=args.capture == "cuda"
                            and args.capture_window == "pair",
                        )
                        samples.append(row)
                        with (args.output / "samples.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                if args.capture == "torch":
                    trace.export_chrome_trace(str(args.output / "trace.chrome.json"))
                adapter.assert_clear()
                if launcher_state() != frozen_launchers:
                    raise RuntimeError(
                        "Lazy kernel launcher state changed during measurement; discard timing"
                    )
                summary = {
                    key: stats([row[key] for row in samples])
                    for key in (
                        "wall_ms",
                        "s1_response_ms",
                        "s2_response_ms",
                        "s1_gpu_span_ms",
                        "s2_gpu_span_ms",
                        "s1_host_submit_ms",
                        "s2_host_submit_ms",
                        "event_span_intersection_ms",
                    )
                    if all(key in row for row in samples)
                }
                properties = torch.cuda.get_device_properties(0)
                result = {
                    "schema_version": 2,
                    "worker_revision": 3,
                    "response_semantics": "s1/s2_response_ms = (ready_ns - release_ns)/1e6 on the same host perf_counter clock. Ready is observed immediately after synchronizing only that job's CUDA end event. Includes cooperative permit wait, CPU submission, CUDA execution/queueing and host wake-up; excludes action decoding/transport and is not GPU kernel busy time.",
                    "offset_ms": args.dit_offset_ms,
                    "mode": args.mode,
                    "precision": args.precision,
                    "component": args.component,
                    "gpu": args.gpu,
                    "pid": os.getpid(),
                    "protocol": {
                        "action_horizon": 40,
                        "slide_steps": 1,
                        "image_delay": 3,
                        "dit_nfe": 1,
                        "bootstrap_nfe": 4,
                        "frames": args.frames,
                        "seed": args.seed,
                        "capture": args.capture,
                        "capture_window": args.capture_window,
                        "warmup": max(args.warmup, args.frames),
                        "iterations": args.iterations,
                        "dit_offset_ms": args.dit_offset_ms,
                        "vlm_chunk_blocks": args.vlm_chunk_blocks,
                        "workload": "Real LIBERO prepared GPU input fixtures; each independent pair restores the common BF16 bootstrap state. S1 consumes the previous completed feature generation; S2 computes the next dataset image. Modulo wrap is an explicit fixture reset, not a sequential environment rollout.",
                        "boundary": "GPU-resident model forward plus S2 GPU cache copy; excludes image/token/state preparation, RPC, action decode, MuJoCo and pre-pair restore.",
                        "serial_order": "DiT with old generation, then VLM producing new generation; identical inputs and features to concurrent modes.",
                    },
                    "memory": {
                        **resident_memory,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    },
                    "hardware": {
                        "name": properties.name,
                        "sm_count": properties.multi_processor_count,
                        "total_memory": properties.total_memory,
                        "compute_capability": [properties.major, properties.minor],
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                    },
                    "optimization": optimized.evidence(),
                    "runtime_adapter": adapter.evidence,
                    "streams": runner.stream_metadata,
                    "equivalence": equivalence,
                    "executed_coverage": actual_coverage,
                    "samples": samples,
                    "summary": summary,
                    "measured_region": {
                        "capture": args.capture,
                        "iterations": args.iterations,
                        "uninstrumented": args.capture == "none",
                        "capture_window": args.capture_window,
                        "cuda_profiler_ranges": (
                            args.iterations if args.capture_window == "pair" else 1
                        )
                        if args.capture == "cuda"
                        else 0,
                        "capture_includes_restore": args.capture in ("cuda", "torch")
                        and (args.capture_window == "whole" or args.capture == "torch"),
                        "restore_range": "RESTORE/{mode}/iteration={i}/fixture={index}",
                        "counter_scope_note": "Whole-window device counters include fixture restore, RNG restore and cache copies between PAIR ranges; per-model counters require pair capture or explicit activity-range filtering. All sample wall_ms exclude pre-pair restore.",
                    },
                    "provenance": {
                        "output": str(args.output),
                        "nominal_output": str(args.nominal_output),
                        "replay_output_root": str(args.replay_output_root)
                        if args.replay_output_root is not None
                        else None,
                        "checkpoint": str(args.checkpoint),
                        "dataset": str(args.dataset),
                        "repository": str(ROOT),
                        "worker_sha256": digest(__file__),
                        "worker_modules_sha256": {
                            name: digest(Path(__file__).with_name(name))
                            for name in (
                                "worker.py",
                                "common.py",
                                "runtime.py",
                                "scheduling.py",
                                "workload.py",
                                "execution.py",
                                "validation.py",
                            )
                        },
                        "groot_root": str(args.groot_root.resolve()),
                        "adapter_sha256": digest(
                            Path(__file__).with_name("runtime.py")
                        ),
                        "scheduling_sha256": digest(
                            Path(__file__).with_name("scheduling.py")
                        ),
                        "fixtures": workload.fixtures,
                        "bootstrap": workload.bootstrap,
                        "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    },
                    "limitations": [
                        "CUDA event span excludes pre-start-event waiting (including DiT cooperative permit wait), but includes later submission gaps and execution waits; it is not request response latency, GPU busy time or kernel overlap.",
                        "Response latency uses host-observed GPU-output readiness and includes thread wake-up; it is not the exact device timestamp of the final kernel or a decoded robot-action completion.",
                        "Stream priority is a scheduling hint; cooperative mode yields only at explicit completed VLM layer boundaries.",
                        "Same-precision bitwise replay agreement does not establish closed-loop task quality. W4A4 is hardware diagnostic only.",
                        "The ordinary integer controller retains BF16 parameters and prepares both 8-bit and 4-bit weights. Resident allocation is diagnostic-process memory, not the footprint of a stripped single-precision deployment.",
                        "Hardware counters must come from independent Nsight passes; instrumented wall times are not unprofiled latency.",
                    ],
                }
                save(args.output / "measurement.json", result)
                save(args.output / "cache-audit.json", runner.cache.audit)
                save(
                    args.output / "status.json",
                    {"state": "complete", "pid": os.getpid(), "samples": len(samples)},
                )
                print(
                    json.dumps(
                        {
                            "worker_revision": 3,
                            "response_semantics": "release to host-observed CUDA-output-ready, including cooperative permit wait and CPU submission; not kernel busy time",
                            "mode": args.mode,
                            "precision": args.precision,
                            "component": args.component,
                            "capture": args.capture,
                            "summary": summary,
                        }
                    ),
                    flush=True,
                )
            finally:
                if runner is not None:
                    runner.close()
                adapter.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=MODES, required=True)
    result.add_argument("--precision", choices=("bf16", "w8a8", "w4a4"), default="bf16")
    result.add_argument("--component", choices=("all", "vlm", "dit"), default="all")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=30)
    result.add_argument("--frames", type=int, default=8)
    result.add_argument("--capture", choices=("none", "cuda", "torch"), default="none")
    result.add_argument("--capture-window", choices=("pair", "whole"), default="pair")
    result.add_argument("--dit-offset-ms", type=float, default=0.0)
    result.add_argument(
        "--vlm-chunk-blocks",
        "--slice-blocks",
        dest="vlm_chunk_blocks",
        type=int,
        choices=(1, 2),
        default=1,
    )
    result.add_argument("--episode-position", type=int, default=0)
    result.add_argument("--dataset-step", type=int, default=10)
    result.add_argument("--seed", type=int, default=1000)
    result.add_argument("--torch-threads", type=int, default=2)
    result.add_argument("--gpu", type=nonnegative_int, required=True)
    result.add_argument(
        "--gpu-uuid", help="Optional exact UUID mask, checked against --gpu"
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--replay-output-root",
        type=Path,
        help="Use a fresh pass-PID-time_ns output under this root for every application replay; --output is retained only as nominal provenance",
    )
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument(
        "--groot-root", type=Path, default=ROOT / "upstream/learning/Isaac-GR00T"
    )
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def validate_arguments(args):
    mask = args.gpu_uuid or str(args.gpu)
    if args.gpu < 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != mask:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must exactly match --gpu-uuid or the physical --gpu index"
        )
    if args.gpu_uuid:
        if re.fullmatch(r"GPU-[0-9a-fA-F-]+", args.gpu_uuid) is None:
            raise ValueError("--gpu-uuid must be a complete NVIDIA GPU UUID")
        identity = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
                "-i",
                str(args.gpu),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        if identity != args.gpu_uuid:
            raise RuntimeError("Physical GPU index and UUID no longer match")
    if (
        args.frames < 1
        or args.iterations < 1
        or args.warmup < 0
        or not math.isfinite(args.dit_offset_ms)
        or args.dit_offset_ms < 0
        or args.episode_position < 0
        or args.dataset_step < 0
        or args.torch_threads < 1
    ):
        raise ValueError("Invalid fixture counts, threads, positions or offset")
    for name in ("checkpoint", "dataset", "groot_root"):
        if not getattr(args, name).is_dir():
            raise ValueError(
                f"--{name.replace('_', '-')} must be an existing directory"
            )
    if not (args.groot_root / "gr00t").is_dir():
        raise ValueError("--groot-root must contain the gr00t package")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except BaseException as error:
        if getattr(args, "output_created", False):
            save(
                args.output / "failure.json",
                {"error": repr(error), "pid": os.getpid(), "measurements_valid": False},
            )
        raise


if __name__ == "__main__":
    main()
