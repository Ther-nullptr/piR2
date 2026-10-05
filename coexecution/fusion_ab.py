"""Re-measure eager and fused recipes in one process on one physical GPU."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from coexecution.fusion import VARIANTS, FusionPatch
from coexecution.run import Runner, stats
from coexecution.workload import PiR2Workload


def gpu_state():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def exact(first, second):
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(second).all()
    return float((first.float() - second.float()).abs().max())


@torch.inference_mode()
def validate(workload, runner, variants, action_calls):
    results = {}
    _, references, _ = runner.paired("serial", action_calls, 2, keep_outputs=True)
    reference_buffer = workload.model.action_head._stream_buf.clone()
    reference_times = workload.model.action_head._stream_buf_t.clone()
    torch.cuda.synchronize()
    for variant in variants:
        if variant == "baseline":
            continue
        with FusionPatch(workload, variant, validate=True) as patch:
            for index, reference in enumerate(runner.features):
                candidate = workload.slow(index)
                for key in reference:
                    exact(reference[key], candidate[key])
            # Every layer was checked on all distinct frames. Continuous action
            # replay checks final outputs/state without repeated local syncs.
            patch.validate = False
            for mode in ["serial", "concurrent"]:
                _, actual, _ = runner.paired(mode, action_calls, 2, keep_outputs=True)
                for first, second in zip(references, actual):
                    exact(first, second)
                exact(reference_buffer, workload.model.action_head._stream_buf)
                exact(reference_times, workload.model.action_head._stream_buf_t)
            results[variant] = {
                **patch.evidence(),
                "frames": len(runner.features),
                "actions_per_mode": action_calls,
                "max_abs": 0.0,
                "buffer_max_abs": 0.0,
                "all_finite": True,
            }
        print("VALIDATED", variant, json.dumps(results[variant]), flush=True)
    return results


def main(args):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "source"
    source_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for name in ["fused_ops.py", "fusion.py", "fusion_ab.py", "run.py", "workload.py"]:
        source = Path(__file__).parent / name
        shutil.copyfile(source, source_dir / name)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    workload = PiR2Workload(args.checkpoint)
    runner = Runner(workload)
    result = {
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "physical_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "source_sha256": source_hashes,
        "cuda_device_count": 1,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "workload": workload.metadata(),
        "profiled": args.profile,
        "scope": "GPU-resident replay, original features-only eager BF16 baseline; same inputs/reset/seed/NFE/cache/streams",
        "excluded": [
            "loading",
            "JIT",
            "warmup",
            "preprocessing",
            "CPU action decode",
            "robot I/O",
        ],
        "variants": args.variants,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "rows": [],
        "trials": [],
        "summary": {},
    }
    # Numerical checks and all production JIT specializations precede timing.
    result["correctness"] = validate(
        workload, runner, args.variants, args.check_actions
    )
    (args.output / "correctness.json").write_text(
        json.dumps(result["correctness"], indent=2)
    )
    for variant in args.variants:
        with FusionPatch(workload, variant):
            runner.paired("concurrent", 2, 3)
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(args.repeats):
        # A cyclic Latin ordering puts every recipe in every order position
        # across five rounds. Modes reverse on odd rounds, independently.
        offset = repeat % len(args.variants)
        order = args.variants[offset:] + args.variants[:offset]
        for variant in order:
            with FusionPatch(workload, variant, profile=args.profile) as patch:
                before = gpu_state() if not args.profile else "profiler"
                torch.cuda.nvtx.range_push(f"VARIANT/{variant}/repeat={repeat}")
                modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
                for mode in modes:
                    rows, _, _ = runner.paired(mode, args.iterations, args.warmup)
                    for row in rows:
                        row.update(variant=variant, repeat=repeat)
                    result["rows"].extend(rows)
                    print(
                        "TRIAL",
                        variant,
                        mode,
                        repeat,
                        json.dumps(stats([row["wall_ms"] for row in rows])),
                        flush=True,
                    )
                torch.cuda.nvtx.range_pop()
                result["trials"].append(
                    {
                        "repeat": repeat,
                        **patch.evidence(),
                        "hardware_before": before,
                        "hardware_after": gpu_state()
                        if not args.profile
                        else "profiler",
                    }
                )
            (args.output / "progress.json").write_text(json.dumps(result, indent=2))
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    for variant in args.variants:
        result["summary"][variant] = {}
        for mode in args.modes:
            rows = [
                row
                for row in result["rows"]
                if row["variant"] == variant and row["mode"] == mode
            ]
            result["summary"][variant][mode] = {
                key: stats([row[key] for row in rows])
                for key in ["wall_ms", "s1_gpu_span_ms", "s2_gpu_span_ms"]
                if key in rows[0]
            }
    runner.close()
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print("SAVED", args.output / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/pir2-so100-smoke/checkpoint-10"),
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check-actions", type=int, default=32)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["s1_only", "s2_only", "serial", "concurrent"],
        default=["s2_only", "concurrent"],
    )
    parser.add_argument("--profile", action="store_true")
    main(parser.parse_args())
