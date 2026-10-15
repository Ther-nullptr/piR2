"""CUDA adapters for the Flow4 diagnostic, imported lazily by its explicit CLI."""

import contextlib
import json
import logging
import os
import subprocess
import threading
import time

from .common import (
    cuda_capture_range,
    digest,
    launcher_state,
    observe_ready,
    save,
    stats,
)
from .flow_protocol import FlowCase, FlowSchedule, difference, wait_start_gate

LOGGER = logging.getLogger(__name__)


class FlowExecutor:
    def __init__(self, workload):
        self.w = workload
        self.instrumented = False

    def __call__(self, role, case, index, gate, request_id):
        torch = self.w.torch
        stream = self.w.streams[
            "vlm" if role == "vlm" else "dit_priority" if case.priority else "dit_equal"
        ]
        device = self.w.devices[role]
        next_index = (index + 1) % len(self.w.fixtures)
        with (
            torch.cuda.device(device),
            torch.inference_mode(),
            torch.cuda.stream(stream),
        ):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stage = "S2/VLM" if role == "vlm" else "S1/Flow4"
            label = f"Flow4/{stage}/request={request_id}/old={index}/next={next_index}"
            self.w.context.current = (case, gate, [0])
            submit_start = time.perf_counter_ns()
            try:
                with (
                    torch.cuda.nvtx.range(label),
                    torch.profiler.record_function("GR00T_BENCH/request/" + request_id)
                    if self.instrumented
                    else contextlib.nullcontext(),
                    torch.profiler.record_function("GR00T_BENCH/stage/" + stage)
                    if self.instrumented
                    else contextlib.nullcontext(),
                ):
                    start.record(stream)
                    if role == "vlm":
                        output = self.w.engines[role].vlm(
                            self.w.prepared[role][next_index].backbone
                        )
                    else:
                        output = self.w.engines[role].action(
                            self.w.reference_features[index],
                            self.w.prepared[role][next_index].action,
                            {"num_inference_timesteps": 4},
                        )
                    end.record(stream)
                submit_end = time.perf_counter_ns()
                ready = observe_ready(end)
            except BaseException:
                # A Python exception can occur after earlier CUDA work was
                # enqueued, before the end event exists. Unblock a cooperative
                # peer first, then drain this worker's stream before ownership
                # returns to the scheduler and model teardown can proceed.
                gate.abort()
                try:
                    stream.synchronize()
                except BaseException:
                    LOGGER.exception(
                        "CUDA stream drain failed while preserving the original request exception"
                    )
                raise
            finally:
                self.w.context.current = None
            return {
                "output": output,
                "host_submit_start_ns": submit_start,
                "host_submit_end_ns": submit_end,
                "ready_ns": ready,
                "gpu_span_ms": start.elapsed_time(end),
                "worker_tid": threading.get_native_id(),
            }


def snapshots(outputs):
    """Own outputs after all workers completed, before any graph replays again."""
    return {
        role: {
            key: value.detach().to(device="cpu").float().numpy().copy()
            for key, value in mapping.items()
        }
        for role, mapping in outputs.items()
    }


def graph_state(workload):
    from coexecution.libero.measurement import graph_snapshot

    return {
        role: graph_snapshot(session) for role, session in workload.sessions.items()
    }


def graph_audit(before, after, *, vlm_calls, dit_calls):
    from coexecution.libero.measurement import graph_interval

    result = {}
    for role, calls in (("vlm", vlm_calls), ("dit", dit_calls)):
        result[role] = {
            name: graph_interval(
                before[role][name],
                after[role][name],
                expected_replays=calls * (4 if name == "dit" else 1),
            )
            for name in ("dit", "vlm", "action")
        }
    if not all(
        item["passes"] for component in result.values() for item in component.values()
    ):
        raise RuntimeError(
            f"Cold capture, graph bypass or K/V accounting changed: {result}"
        )
    return result


def source_evidence(args):
    from .flow import ROOT

    def revision(path):
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "repository_revision": revision(ROOT),
        "groot_revision": revision(args.groot_root),
        "sources_lock_sha256": digest(ROOT / "sources.lock.json"),
        "coexecution_source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in sorted((ROOT / "coexecution").rglob("*.py"))
        },
        "groot_model_sha256": {
            str(path.relative_to(args.groot_root)): digest(path)
            for path in sorted(
                (args.groot_root / "gr00t/model/gr00t_n1d7").glob("*.py")
            )
        },
        "checkpoint_config_sha256": digest(args.checkpoint / "config.json"),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
    }


def run_group(args):
    import torch

    from .flow import cases
    from .flow_workload import FlowWorkload

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for device in (args.action_device, args.vlm_device):
        if int(device.split(":")[1]) >= torch.cuda.device_count():
            raise ValueError(f"Requested device is not visible: {device}")
    plan = cases(args)

    def status(state, **fields):
        value = {"state": state, "pid": os.getpid(), **fields}
        save(args.output / "status.json", value)
        print(json.dumps(value), flush=True)

    status("loading", precision=args.precision, group=args.group)
    with torch.inference_mode(), FlowWorkload(args) as workload:
        executor = FlowExecutor(workload)
        with FlowSchedule(executor) as schedule:
            status("warming", fixtures=len(workload.fixtures))
            # All graph captures happen serially on each eventual owning thread
            # and stream, before the first concurrent submission.
            for _ in range(args.warmup):
                for index in range(len(workload.fixtures)):
                    schedule.warm("vlm", FlowCase("vlm_only"), index)
                    for case in (
                        FlowCase("dit_only"),
                        FlowCase("concurrent_dit_priority"),
                    ):
                        workload.restore(index)
                        schedule.warm("dit", case, index)
            reference, expected_rng = {}, {}
            for index in range(len(workload.fixtures)):
                workload.restore(index)
                _, output = schedule.pair(
                    FlowCase("serial"),
                    index,
                    -1,
                    old_generation=index,
                    next_generation=len(workload.fixtures) + index,
                )
                reference[index] = snapshots(output)
                for value in reference[index].values():
                    difference(value, value)
                expected_rng[index] = torch.cuda.get_rng_state(args.action_device)
            # Warm every fixture under every hook/fence/priority configuration.
            for case in plan:
                for index in range(len(workload.fixtures)):
                    workload.restore(index)
                    _, output = schedule.pair(
                        case,
                        index,
                        -2,
                        old_generation=index,
                        next_generation=len(workload.fixtures) + index,
                    )
                    workload.check_rng(index, case.has_dit, expected_rng)
                    for role, value in snapshots(output).items():
                        difference(reference[index][role], value)
            coverage = {
                role: observer.evidence()
                for role, observer in workload.coverage.items()
            }
            save(args.output / "executed-coverage.json", coverage)
            for observer in workload.coverage.values():
                observer.assert_clear()
                observer.active = False
            if not all(value["passes"] for value in coverage.values()):
                raise RuntimeError(
                    f"Selected projection coverage was not observed; inspect {args.output / 'executed-coverage.json'}"
                )
            workload.assert_features_unchanged()
            frozen_launchers = launcher_state()
            resident_memory = {}
            for device in sorted(set(workload.devices.values())):
                torch.cuda.synchronize(device)
                resident_memory[device] = {
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                }
                torch.cuda.reset_peak_memory_stats(device)
            samples, audits = [], {}
            generation = len(workload.fixtures)

            def run_pair(case, iteration, *, instrumented=False):
                nonlocal generation
                index = iteration % len(workload.fixtures)
                workload.restore(index)
                row, outputs = schedule.pair(
                    case,
                    index,
                    iteration,
                    old_generation=index,
                    next_generation=generation,
                )
                generation += 1
                owned = snapshots(outputs)
                workload.check_rng(index, case.has_dit, expected_rng)
                for observer in workload.coverage.values():
                    observer.assert_clear()
                old = workload.feature_records[index]
                if case.has_dit and old["ready_ns"] >= row["s1_release_ns"]:
                    raise RuntimeError("S1 read an unfinished feature generation")
                row.update(
                    seed=args.seed + index,
                    old_observation_id=old["observation_id"],
                    next_observation_id=workload.fixtures[
                        (index + 1) % len(workload.fixtures)
                    ]["observation_id"],
                    old_feature_ready_ns=old["ready_ns"],
                    numerical={
                        role: difference(reference[index][role], value)
                        for role, value in owned.items()
                    },
                    instrumented=instrumented,
                )
                return row

            start_coordination = None
            if args.start_gate is not None:

                def publish_ready(record):
                    save(args.output / "start-ready.json", record)
                    status("ready", **record)

                start_coordination = wait_start_gate(
                    args.start_gate, args.gate_timeout_s, on_ready=publish_ready
                )
                save(args.output / "start-gate.json", start_coordination)

            for case in plan:
                status("measuring", case=case.identity)
                before = graph_state(workload)
                for iteration in range(args.iterations):
                    row = run_pair(case, iteration)
                    samples.append(row)
                    with (args.output / "samples.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                audits[case.identity] = graph_audit(
                    before,
                    graph_state(workload),
                    vlm_calls=args.iterations if case.has_vlm else 0,
                    dit_calls=args.iterations if case.has_dit else 0,
                )
                if launcher_state() != frozen_launchers:
                    raise RuntimeError(
                        "Lazy CUDA launcher keys changed during steady measurement"
                    )
                workload.assert_features_unchanged()
            profiles = []
            if args.profile_iterations:
                status("profiling")
                executor.instrumented = True
                for case_index, case in enumerate(plan):
                    before = graph_state(workload)
                    rows = []
                    trace_path = args.output / f"trace-{case_index:02d}.chrome.json"
                    with (
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profiler,
                        cuda_capture_range(args.cuda_profiler_range),
                    ):
                        for iteration in range(args.profile_iterations):
                            with torch.profiler.record_function(
                                "Flow4/pair/" + case.identity
                            ):
                                rows.append(
                                    run_pair(case, iteration, instrumented=True)
                                )
                    profiler.export_chrome_trace(str(trace_path))
                    audit = graph_audit(
                        before,
                        graph_state(workload),
                        vlm_calls=args.profile_iterations if case.has_vlm else 0,
                        dit_calls=args.profile_iterations if case.has_dit else 0,
                    )
                    profiles.append(
                        {
                            "case": case.identity,
                            "trace": trace_path.name,
                            "samples": rows,
                            "graph_audit": audit,
                            "instrumented": True,
                        }
                    )
                executor.instrumented = False
            hardware = {}
            for device in sorted(set(workload.devices.values())):
                prop = torch.cuda.get_device_properties(device)
                hardware[device] = {
                    "name": prop.name,
                    "uuid": str(getattr(prop, "uuid", "unavailable")),
                    "sm_count": prop.multi_processor_count,
                    "total_memory": prop.total_memory,
                    "compute_capability": [prop.major, prop.minor],
                }
                resident_memory[device].update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                )
            summaries = {}
            for case in plan:
                selected = [row for row in samples if row["case"] == case.identity]
                names = [
                    name
                    for name in selected[0]
                    if name.endswith("_ms") and name != "offset_ms"
                ]
                summaries[case.identity] = {
                    name: stats([row[name] for row in selected]) for name in names
                }
            result = {
                "schema": "libero-flow4-contention",
                "schema_version": 1,
                "precision": args.precision,
                "group": args.group,
                "protocol": {
                    "action_horizon": 40,
                    "nfe": 4,
                    "streaming": False,
                    "fixtures": len(workload.fixtures),
                    "iterations_per_case": args.iterations,
                    "warmup_passes_per_stream": args.warmup,
                    "input_scope": "Three real image/state/text fixtures by default. S1 uses a previously completed frozen feature generation and the next fixture's state; S2 computes the next fixture's image. Each pair is an independent fixture reset, not a closed-loop rollout or same-frame VLM-to-DiT request.",
                    "timer_scope": "Prepared GPU-resident model forward; release-to-own-CUDA-output-ready includes cooperative wait, CPU submission and host observation. Pair makespan includes host dispatch and serial handoff. Input preparation, old-feature copies, RNG reset, snapshots and numeric checks are outside timers.",
                    "feature_transfer": "Old features copied once into owned action-device storage before all timed pairs. S2 output readiness is on the VLM device; new-feature transport is excluded, including in dual-device pairs.",
                    "graph_group": "Whole VLM graph enabled"
                    if args.group == "optimized"
                    else "Whole VLM graph disabled for every control. Permanent layer hooks present in every control; completed CUDA boundary fences only in explicitly fenced and cooperative modes.",
                    "serial_order": "S1(old features) completes, then S2(next image); same inputs/seeds as parallel cases.",
                    "numerical_reference": "Same-precision, same-placement, same-graph-group serial per fixture. Drift metrics are evidence, not an exact-equality gate.",
                    "rng": "Only selected action device state restored while workers idle; VLM non-consumption and matched serial DiT consumption checked.",
                },
                "ownership": {
                    "policies": "Two full resident policies with independent mutable projection caches, graph instances, static bindings and submission guards; unused component weights remain resident.",
                    "parameter_object_aliases": 0,
                    "outputs": "Both workers complete before CPU snapshots; the next graph replay cannot overwrite a retained result.",
                    "features": workload.feature_records,
                },
                "placement": {
                    **workload.devices,
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cross_device": args.action_device != args.vlm_device,
                },
                "hardware": hardware,
                "runtime": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                },
                "memory": resident_memory,
                "streams": workload.stream_evidence,
                "optimization": {
                    role: session.evidence()
                    for role, session in workload.sessions.items()
                },
                "executed_coverage": coverage,
                "graph_audits": audits,
                "fixtures": workload.fixtures,
                "samples": samples,
                "summary": summaries,
                "profiles": profiles,
                "provenance": {
                    **source_evidence(args),
                    "start_coordination": start_coordination,
                },
                "limitations": [
                    "CUDA event spans are per-request queue/submission/execution spans, not kernel busy time. No event-span intersection is claimed as kernel overlap.",
                    "Stream priority is a scheduling hint. Cooperative handoff occurs after synchronizing a completed VLM layer boundary, never inside whole-VLM graph replay.",
                    "Dual-device results require their own same-placement alone/serial baselines and recorded device clocks; mixed physical-device clocks cannot establish an isolation gain.",
                    "Optional profiler passes are separate from clean samples; NVTX request/stage ranges are emitted by the actual S1/S2 workers. Inspect actual CUDA activities for overlap.",
                    "Prepared fixture diagnostics and numerical drift do not establish LIBERO closed-loop success.",
                ],
            }
            save(args.output / "measurement.json", result)
            status(
                "complete",
                samples=len(samples),
                output=str(args.output / "measurement.json"),
            )
            return result
