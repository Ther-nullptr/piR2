"""Compare optimization under causal S2->S1 serial and cached concurrent execution."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from coexecution.cache import FeatureCache
from coexecution.fusion_ab import exact, gpu_state
from coexecution.run import Runner, stats
from coexecution.runtime_ab import RuntimeVariant, snapshot_sources
from coexecution.workload import PiR2Workload


def causal_serial(runner, iterations, warmup, keep_outputs=False):
    """Publish this observation's S2 output before acquiring its S1 read lease."""
    runner.w.reset()
    cache = FeatureCache(runner.w.initial_features)
    rows, outputs = [], []
    for iteration in range(iterations + warmup):
        current = (iteration + 1) % len(runner.features)
        ticket = cache.reserve_write(time.perf_counter_ns())
        assert ticket is not None
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        origin.synchronize()
        started = time.perf_counter_ns()
        slow = runner.slow_pool.submit(
            runner.slow_job, cache, ticket, current, "causal_serial"
        ).result()
        slow["end"].synchronize()
        assert cache.publish_ready()
        lease = cache.acquire()
        assert lease.generation == slow["generation"]
        fast = runner.fast_pool.submit(
            runner.fast_job, cache, lease, current, "causal_serial"
        ).result()
        fast["end"].synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        row = {
            "mode": "causal_serial",
            "iteration": iteration - warmup,
            "wall_ms": elapsed,
        }
        for label, job in [("s1", fast), ("s2", slow)]:
            row[label + "_start_ms"] = origin.elapsed_time(job["start"])
            row[label + "_end_ms"] = origin.elapsed_time(job["end"])
            row[label + "_gpu_span_ms"] = job["start"].elapsed_time(job["end"])
            row[label + "_generation"] = job["generation"]
        assert row["s1_start_ms"] >= row["s2_end_ms"], "Causal serial order violated"
        if iteration >= warmup:
            rows.append(row)
            if keep_outputs:
                outputs.append(fast["output"].detach().clone())
    torch.cuda.synchronize()
    return rows, outputs


def execute(runner, mode, iterations, warmup, keep_outputs=False):
    if mode == "causal_serial":
        return causal_serial(runner, iterations, warmup, keep_outputs)
    rows, outputs, _ = runner.paired(
        mode, iterations, warmup, keep_outputs=keep_outputs
    )
    return rows, outputs


@torch.inference_mode()
def validate(workload, runner):
    evidence = {}
    for mode in ["causal_serial", "concurrent"]:
        references = None
        for variant in ["eager", "combined"]:
            with RuntimeVariant(workload, variant):
                rows, outputs = execute(runner, mode, 32, 2, keep_outputs=True)
                state = workload.model.action_head._stream_buf.clone()
                times = workload.model.action_head._stream_buf_t.clone()
                rng = torch.cuda.get_rng_state()
                if references is None:
                    references = outputs, state, times, rng
                else:
                    for expected, actual in zip(references[0], outputs):
                        exact(expected, actual)
                    exact(references[1], state)
                    exact(references[2], times)
                    assert torch.equal(references[3], rng)
        evidence[mode] = {
            "compared_actions": 32,
            "max_abs": 0.0,
            "state_max_abs": 0.0,
            "rng_equal": True,
        }
        if mode == "causal_serial":
            evidence[mode]["fresh_cache_each_call"] = all(
                r["s1_generation"] == r["s2_generation"] for r in rows
            )
        print("CORRECTNESS", mode, json.dumps(evidence[mode]), flush=True)
    return evidence


def main(args):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources(args.output)
    source = Path(__file__).resolve()
    shutil.copyfile(source, args.output / "source" / source.name)
    sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    workload = PiR2Workload("outputs/pir2-so100-smoke/checkpoint-10")
    runner = Runner(workload)
    result = {
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "physical_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": 1,
        "profiled": False,
        "workload": workload.metadata(),
        "source_sha256": sources,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "rows": [],
        "trials": [],
        "summary": {},
        "scope": "Same GR00T/piR2 BF16 model, NFE1; compare execution protocols, not a pi0.5 model benchmark",
        "protocols": {
            "causal_serial": "S2(current observation) finishes and publishes before S1 starts",
            "concurrent": "S1 reads previous completed cache while S2 processes the current observation",
        },
    }
    result["correctness"] = validate(workload, runner)
    for repeat in range(args.repeats):
        variants = ["eager", "combined"] if repeat % 2 == 0 else ["combined", "eager"]
        modes = (
            ["causal_serial", "concurrent"]
            if (repeat // 2) % 2 == 0
            else ["concurrent", "causal_serial"]
        )
        for variant in variants:
            with RuntimeVariant(workload, variant) as runtime:
                before = gpu_state()
                for mode in modes:
                    rows, _ = execute(runner, mode, args.iterations, args.warmup)
                    for row in rows:
                        row.update(variant=variant, repeat=repeat)
                    result["rows"].extend(rows)
                    print(
                        "TRIAL",
                        variant,
                        mode,
                        repeat,
                        json.dumps(stats([r["wall_ms"] for r in rows])),
                        flush=True,
                    )
                result["trials"].append(
                    {
                        "variant": variant,
                        "repeat": repeat,
                        "hardware_before": before,
                        "hardware_after": gpu_state(),
                        "runtime": runtime.evidence(),
                    }
                )
            (args.output / "progress.json").write_text(json.dumps(result, indent=2))
    for variant in ["eager", "combined"]:
        result["summary"][variant] = {}
        for mode in ["causal_serial", "concurrent"]:
            selected = [
                row
                for row in result["rows"]
                if row["variant"] == variant and row["mode"] == mode
            ]
            result["summary"][variant][mode] = stats(
                [row["wall_ms"] for row in selected]
            )
    result["source_unchanged_during_run"] = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in sources.items()
    )
    runner.close()
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print("SAVED", args.output / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    main(parser.parse_args())
