"""Single-card revalidation of static S1, DiT graphs and static S2 metadata."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch

from coexecution.fusion import FusionPatch
from coexecution.fusion_ab import exact, gpu_state
from coexecution.periodic import run_periodic
from coexecution.run import Runner, stats
from coexecution.static_s1 import StaticS1
from coexecution.static_s2 import StaticS2
from coexecution.workload import PiR2Workload

VARIANTS = ["eager", "rope", "s1_static", "s1_graph", "s2_metadata", "combined"]


class RuntimeVariant:
    def __init__(self, workload, variant):
        self.workload, self.variant = workload, variant
        self.stack = ExitStack()
        self.parts = {}

    def __enter__(self):
        try:
            if self.variant != "eager":
                self.parts["rope"] = self.stack.enter_context(
                    FusionPatch(self.workload, "rope")
                )
            if self.variant in {"s1_static", "s1_graph", "combined"}:
                self.parts["s1"] = self.stack.enter_context(
                    StaticS1(self.workload, graph=self.variant != "s1_static")
                )
            if self.variant in {"s2_metadata", "combined"}:
                self.parts["s2"] = self.stack.enter_context(StaticS2(self.workload))
            # Captures/constant construction must happen before concurrent work.
            self.workload.reset()
            for index in range(len(self.workload.action_inputs)):
                self.workload.fast(self.workload.initial_features, index)
            torch.cuda.synchronize()
            if "s1" in self.parts:
                self.parts["s1"].freeze()
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exception):
        return self.stack.__exit__(*exception)

    def evidence(self):
        return {name: part.evidence() for name, part in self.parts.items()}


@torch.inference_mode()
def validate(workload, runner, variants, calls):
    _, expected, _ = runner.paired("serial", calls, 2, keep_outputs=True)
    state = workload.model.action_head._stream_buf.clone()
    times = workload.model.action_head._stream_buf_t.clone()
    rng = torch.cuda.get_rng_state()
    checks = {}
    for variant in variants:
        with RuntimeVariant(workload, variant) as runtime:
            for index, reference in enumerate(runner.features):
                actual = workload.slow(index)
                for key in reference:
                    exact(reference[key], actual[key])
            for mode in ["serial", "concurrent"]:
                _, actual, _ = runner.paired(mode, calls, 2, keep_outputs=True)
                assert len(expected) == len(actual)
                for first, second in zip(expected, actual):
                    exact(first, second)
                exact(state, workload.model.action_head._stream_buf)
                exact(times, workload.model.action_head._stream_buf_t)
                assert torch.equal(rng, torch.cuda.get_rng_state())
            checks[variant] = {
                "frames": len(runner.features),
                "actions_per_mode": calls,
                "max_abs": 0.0,
                "state_max_abs": 0.0,
                "rng_equal": True,
                "runtime": runtime.evidence(),
            }
        print("VALIDATED", variant, json.dumps(checks[variant]), flush=True)
    return checks


@torch.inference_mode()
def baseline_replay(workload, runner, result, outputs):
    """After removing adapters, replay the exact observed states/cache versions."""
    workload.reset()
    assert len(result["calls"]) == len(outputs)
    for call, output in zip(result["calls"], outputs):
        source = runner.features[call["cache_source_index"] % len(runner.features)]
        reference = workload.fast(source, call["release_index"])
        exact(reference, output)
    result["unmodified_eager_replay_max_abs"] = 0.0


def snapshot_sources(output):
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    paths = [
        Path(__file__).parent / name
        for name in [
            "runtime_ab.py",
            "static_s1.py",
            "static_s2.py",
            "graph_dit.py",
            "run.py",
            "periodic.py",
            "workload.py",
            "fusion.py",
            "fused_ops.py",
        ]
    ]
    root = Path(__file__).resolve().parents[1]
    paths.extend(
        [
            root / "upstream/learning/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
            root / "upstream/learning/Isaac-GR00T/gr00t/model/modules/dit.py",
        ]
    )
    hashes = {}
    for path in paths:
        shutil.copyfile(path, source_dir / path.name)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main(args):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources(args.output)
    workload = PiR2Workload(args.checkpoint)
    runner = Runner(workload)
    report = {
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "physical_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": 1,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "profiled": args.profile,
        "source_sha256": sources,
        "workload": workload.metadata(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "variants": args.variants,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "rows": [],
        "trials": [],
        "summary": {},
        "periodic": [],
        "scope": "Single-card GPU-resident real-input replay; excludes preparation, graph capture, preprocessing and robot I/O",
    }
    report["correctness"] = validate(
        workload, runner, args.variants, args.check_actions
    )
    (args.output / "correctness.json").write_text(
        json.dumps(report["correctness"], indent=2)
    )
    if args.validate_only:
        runner.close()
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        return
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(args.repeats):
        offset = repeat % len(args.variants)
        order = args.variants[offset:] + args.variants[:offset]
        for variant in order:
            with RuntimeVariant(workload, variant) as runtime:
                before = gpu_state()
                prepared = runtime.evidence()
                torch.cuda.nvtx.range_push(f"VARIANT/{variant}/repeat={repeat}")
                modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
                for mode in modes:
                    rows, _, _ = runner.paired(mode, args.iterations, args.warmup)
                    for row in rows:
                        row.update(variant=variant, repeat=repeat)
                    report["rows"].extend(rows)
                    print(
                        "TRIAL",
                        variant,
                        mode,
                        repeat,
                        json.dumps(stats([row["wall_ms"] for row in rows])),
                        flush=True,
                    )
                torch.cuda.nvtx.range_pop()
                report["trials"].append(
                    {
                        "variant": variant,
                        "repeat": repeat,
                        "hardware_before": before,
                        "hardware_after": gpu_state(),
                        "prepared_runtime": prepared,
                        "runtime": runtime.evidence(),
                    }
                )
            (args.output / "progress.json").write_text(json.dumps(report, indent=2))
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    if args.periodic_seconds:
        periodic_args = SimpleNamespace(
            seconds=args.periodic_seconds, period_ms=args.period_ms, camera_hz=30
        )
        for repeat in range(args.periodic_repeats):
            order = (
                args.periodic_variants
                if repeat % 2 == 0
                else list(reversed(args.periodic_variants))
            )
            for variant in order:
                with RuntimeVariant(workload, variant) as runtime:
                    result, outputs = run_periodic(
                        workload, runner, periodic_args, return_outputs=True
                    )
                    result.update(
                        variant=variant, repeat=repeat, runtime=runtime.evidence()
                    )
                baseline_replay(workload, runner, result, outputs)
                report["periodic"].append(result)
                path = args.output / f"periodic-{variant}-{repeat}.json"
                path.write_text(json.dumps(result, indent=2))
                print(
                    "PERIODIC",
                    variant,
                    repeat,
                    json.dumps(
                        {
                            key: result[key]
                            for key in [
                                "completed_s1",
                                "dropped_releases",
                                "deadline_miss_fraction",
                                "latency_ms_completed",
                                "feature_age_at_ready_ms",
                                "unmodified_eager_replay_max_abs",
                            ]
                        }
                    ),
                    flush=True,
                )
    for variant in args.variants:
        report["summary"][variant] = {}
        for mode in args.modes:
            rows = [
                row
                for row in report["rows"]
                if row["variant"] == variant and row["mode"] == mode
            ]
            report["summary"][variant][mode] = {
                key: stats([row[key] for row in rows])
                for key in ["wall_ms", "s1_gpu_span_ms", "s2_gpu_span_ms"]
                if key in rows[0]
            }
    report["source_unchanged_during_run"] = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in sources.items()
    )
    runner.close()
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print("SAVED", args.output / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/pir2-so100-smoke/checkpoint-10"),
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument(
        "--modes", nargs="+", default=["s1_only", "s2_only", "concurrent"]
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check-actions", type=int, default=32)
    parser.add_argument("--periodic-seconds", type=float, default=0)
    parser.add_argument("--periodic-repeats", type=int, default=3)
    parser.add_argument("--period-ms", type=float, default=40)
    parser.add_argument(
        "--periodic-variants", nargs="+", choices=VARIANTS, default=["rope", "combined"]
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    main(parser.parse_args())
