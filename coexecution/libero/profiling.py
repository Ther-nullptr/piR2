"""Measured leaf-device activity classification, independent of model execution.

Reusable accounting extracted from the prior measured LIBERO benchmark; no model,
Torch or CUDA dependency is imported. Profiler GPU work is distinct from wall time.
"""

import json
import math
import re
from collections import defaultdict

STAGE_PREFIX = "GR00T_BENCH/stage/"
REQUEST_PREFIX = "GR00T_BENCH/request/"


def collect_model_profile(
    policy,
    observations,
    session,
    output,
    *,
    rounds,
    seed,
    device,
    model_only=False,
    observation_ids=None,
    max_prepared_observations=16,
    event_sink=None,
    mode=None,
):
    """Separate instrumented model probe after priming; never a clean wall sample."""
    import torch

    from coexecution.libero.inference import (
        PreparedInferenceEngine,
        PreparedInputs,
        prepare_observations,
    )
    from coexecution.libero.measurement import (
        _prepare,
        _seed_reset,
        graph_interval,
        graph_snapshot,
    )
    from coexecution.replay.producer import (
        ModelRequestEvents,
        emit_event,
        validate_sink,
    )

    validate_sink(event_sink)

    if type(rounds) is not int or rounds < 1:
        raise ValueError("Profile rounds must be positive")
    if observation_ids is None:
        observation_ids = [f"obs-{index}" for index in range(len(observations))]
    if len(observations) != len(observation_ids) or len(set(observation_ids)) != len(
        observation_ids
    ):
        raise ValueError("One unique profile identity is required per observation")
    prepared = None
    engine = PreparedInferenceEngine(policy.model)
    if model_only:
        prepared = prepare_observations(
            policy, observations, observation_ids, limit=max_prepared_observations
        )
    for index, observation in enumerate(observations):
        _seed_reset(policy, seed + index, device)
        if model_only:
            engine.full(prepared[index])
        else:
            policy.get_action(observation)
    torch.cuda.synchronize(device)
    if event_sink is not None and model_only:
        for identity in observation_ids:
            emit_event(
                event_sink,
                "input.ready",
                sources={"observation_id": identity},
                payload={
                    "mode": mode,
                    "instrumented": True,
                    "gpu_completion_observed": True,
                    "scope": "prepared cohort after warmup synchronization",
                },
            )
    active_request = {}

    def stage_scope(name):
        context = torch.profiler.record_function(STAGE_PREFIX + name)
        return (
            active_request["events"].stage(name, context)
            if event_sink is not None
            else context
        )

    engine = PreparedInferenceEngine(
        policy.model,
        stage=stage_scope,
    )
    before = graph_snapshot(session)
    output.parent.mkdir(parents=True, exist_ok=True)
    requests = []
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for round_index in range(rounds):
            for index, observation in enumerate(observations):
                _seed_reset(policy, seed + index, device)
                request_id = f"profile-r{round_index}-o{index}"
                annotation = REQUEST_PREFIX + request_id
                if event_sink is not None:
                    active_request["events"] = ModelRequestEvents(
                        event_sink,
                        request_id=request_id,
                        observation_id=observation_ids[index],
                        mode=mode,
                    )
                    emit_event(
                        event_sink,
                        "model.request.started",
                        sources={
                            "request_id": request_id,
                            "observation_id": observation_ids[index],
                        },
                        payload={
                            "mode": mode,
                            "seed": seed + index,
                            "instrumented": True,
                            "measurement_boundary": "profile",
                            "output_available": False,
                        },
                    )
                requests.append(
                    {
                        "request_id": request_id,
                        "annotation": annotation,
                        "round": round_index,
                        "observation": index,
                        "observation_id": observation_ids[index],
                        "seed": seed + index,
                    }
                )
                with torch.profiler.record_function(annotation):
                    if model_only:
                        inputs = prepared[index]
                    else:
                        with stage_scope("preparation_transfer"):
                            backbone, action, states = _prepare(policy, observation)
                            torch.cuda.synchronize(device)
                        inputs = PreparedInputs(
                            backbone, action, states, observation_ids[index]
                        )
                    engine.full(inputs)
                    torch.cuda.synchronize(device)
                    if event_sink is not None:
                        active_request["events"].completed()
    profiler.export_chrome_trace(str(output))
    after = graph_snapshot(session)
    calls = rounds * len(observations)
    # The same full model body executes once per profiler call.
    audit = {
        name: {
            "capture_delta": None
            if before[name] is None
            else after[name]["capture_count"] - before[name]["capture_count"],
            "replay_delta": None
            if before[name] is None
            else after[name]["replay_count"] - before[name]["replay_count"],
        }
        for name in ("dit", "vlm", "action")
    }
    if not all(
        graph_interval(before[name], after[name], expected_replays=count * calls)[
            "passes"
        ]
        for name, count in (("dit", 4), ("vlm", 1), ("action", 1))
    ):
        raise RuntimeError(f"Profiler probe recaptured or bypassed a graph: {audit}")
    result = summarize_trace(json.loads(output.read_text()))
    result.update(
        calls=calls,
        trace=str(output),
        graph_audit=audit,
        scope=(
            "Instrumented prepared full VLM and Flow4"
            if model_only
            else "Instrumented preparation + full VLM and Flow4"
        )
        + "; output decoding excluded; separate from clean wall/event spans",
        instrumented=True,
        input_preparation_included=not model_only,
        requests=requests,
    )
    return result


def _union_duration(intervals):
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _category(name, chain, cat):
    """Ordered, exclusive classification; parent CPU totals are never added."""
    kernel = name.lower()
    ancestors = " ".join(chain).lower()
    if cat == "gpu_memcpy":
        return "transfer"
    if cat == "gpu_memset":
        return "layout_copy"
    if "_prepare_condition" in kernel:
        return "pointwise"
    if any(
        token in kernel
        for token in (
            "layernorm_pack",
            "rms_tail_pack",
            "_prepare_norm_modulation",
            "_prepare_residual_norm_modulation",
        )
    ):
        return "producer_quantization"
    if any(
        token in kernel
        for token in (
            "prepare_integer",
            "_prepare",
            "quantize",
            "quantization",
            "pack_activation",
            "prepare_modulation",
            "rms_tail_pack",
            "rmsnorm_pack",
            "layernorm_pack",
        )
    ):
        return "quantization_prepare"
    if any(
        token in kernel + " " + ancestors
        for token in (
            "scaled_dot_product",
            "flash_att",
            "flash_fwd",
            "fmha",
            "efficient_attention",
            "cudnn_attention",
        )
    ):
        return "attention"
    leaf = next(
        (
            name.lower().split(".", 1)[0]
            for name in reversed(chain)
            if name.startswith("aten::")
        ),
        "",
    )
    if re.search(
        r"(?<![a-z0-9])cast(?:_kernel)?(?=$|[^a-z0-9])", kernel
    ) or leaf.startswith("aten::_cast_"):
        return "dtype_conversion"
    conversions = {"aten::to", "aten::_to_copy", "aten::type_as"}
    if leaf in conversions or (
        leaf == "aten::copy_"
        and any(name.split(".", 1)[0] in conversions for name in chain)
    ):
        # to/copy can change device, dtype, layout or only copy storage. Names
        # alone do not establish a dtype transition, even for a CUDA kernel.
        return "conversion_copy_ambiguous"
    if any(
        token in kernel
        for token in (
            "nchwtonhwc",
            "nhwctonchw",
            "nchw2nhwc",
            "nhwc2nchw",
            "transpose_kernel",
        )
    ):
        return "layout_copy"
    if "copy" in kernel or leaf == "aten::copy_":
        # Graph replay hides ATen input dtype metadata. A generic copy kernel
        # can also convert dtype; only clone/contiguous proves preservation.
        return (
            "layout_copy"
            if leaf in {"aten::clone", "aten::contiguous"}
            else "conversion_copy_ambiguous"
        )
    if any(
        token in kernel + " " + ancestors
        for token in ("convolution", "conv3d", "conv2d", "conv_fwd")
    ) or (
        "cudnn" in kernel
        and any(
            token in kernel
            for token in ("implicit_gemm", "_fprop_", "_dgrad_", "_wgrad_")
        )
    ):
        return "convolution"
    if any(
        token in kernel
        for token in ("gemm", "gemv", "matmul", "cutlass", "cublas", "::reduce_scale<")
    ) or any(
        token in ancestors
        for token in (
            "aten::mm",
            "aten::addmm",
            "aten::bmm",
            "aten::matmul",
            "robotics_integer::gemm",
            "pir2_integer_split_k::gemm",
        )
    ):
        return "gemm"
    if any(
        token in kernel + " " + leaf
        for token in (
            "copy",
            "contiguous",
            "clone",
            "index",
            "gather",
            "scatter",
            "concat",
            "aten::cat",
            "transpose",
        )
    ):
        return "layout_copy"
    if any(
        token in kernel + " " + ancestors
        for token in (
            "layer_norm",
            "rms_norm",
            "layernorm",
            "rmsnorm",
            "_rms",
            "square_fp32",
            "squaremeanops",
            "normalization",
            "modulation",
            "adaln",
        )
    ):
        return "normalization_modulation"
    if (
        any(
            token in ancestors
            for token in ("aten::rand", "aten::normal", "aten::uniform")
        )
        or "philox" in kernel
    ):
        return "random"
    if any(
        token in kernel + " " + ancestors
        for token in (
            "reduce",
            "softmax",
            "aten::sum",
            "aten::mean",
            "aten::amax",
            "scan",
        )
    ):
        return "reduction"
    if any(
        token in kernel
        for token in (
            "elementwise",
            "vectorized",
            "_silu_gate",
            "silu",
            "gelu",
            "pointwise",
        )
    ) or any(
        token in ancestors
        for token in (
            "aten::add",
            "aten::mul",
            "aten::sub",
            "aten::div",
            "aten::silu",
            "aten::gelu",
            "aten::sin",
            "aten::cos",
            "aten::exp",
        )
    ):
        return "pointwise"
    return "unknown"


def classify_trace(trace):
    """Classify only leaf Chrome device activities, each into exactly one bucket.

    CUDA runtime/driver correlation and CPU parent chains identify the launch
    stage. Graph replay can lose inner ATen names, so actual kernel names remain
    recorded and unknown work is retained. Durations are Chrome microseconds.
    """
    events = [
        event
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("dur", 0) > 0
    ]
    device_categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    hosts = defaultdict(list)
    device = []
    for event in events:
        if event.get("cat") in device_categories:
            device.append(event)
        elif event.get("cat") in {
            "cpu_op",
            "user_annotation",
            "cuda_runtime",
            "cuda_driver",
            "python_function",
        }:
            hosts[(event.get("pid"), event.get("tid"))].append(event)
    chains, runtime = defaultdict(list), defaultdict(list)
    host_stages = defaultdict(list)
    for thread, entries in hosts.items():
        active = []
        for event in sorted(entries, key=lambda row: (row["ts"], -row["dur"])):
            start, stop = event["ts"], event["ts"] + event["dur"]
            while active and start >= active[-1]["ts"] + active[-1]["dur"]:
                active.pop()
            # Real synchronous CPU scopes are nested; unrelated overlapping
            # annotations are retained without inventing an ancestry relation.
            parents = [row for row in active if stop <= row["ts"] + row["dur"]]
            chain = [row["name"] for row in parents] + [event["name"]]
            stage = next(
                (
                    name.removeprefix(STAGE_PREFIX)
                    for name in reversed(chain)
                    if name.startswith(STAGE_PREFIX)
                ),
                "unattributed",
            )
            external = event.get("args", {}).get("External id")
            if external is not None:
                chains[str(external)].append((thread, chain, stage))
            if event.get("cat") in {"cuda_runtime", "cuda_driver"}:
                correlation = event.get("args", {}).get("correlation")
                if correlation is not None:
                    runtime[str(correlation)].append((thread, chain, stage))
            if event["name"].startswith(STAGE_PREFIX):
                host_stages[event["name"].removeprefix(STAGE_PREFIX)].append(
                    (start, stop)
                )
            active.append(event)
    device_events = []
    categories, stages, kernels = (
        defaultdict(lambda: [0, 0.0]),
        defaultdict(lambda: [0, 0.0]),
        defaultdict(lambda: [0, 0.0]),
    )
    intervals, unattributed = [], 0
    for event in device:
        args = event.get("args", {})
        candidates = runtime.get(str(args.get("correlation")), [])
        if not candidates:
            candidates = chains.get(str(args.get("External id")), [])
        if candidates:
            possible_stages = {row[2] for row in candidates}
            stage = (
                possible_stages.pop() if len(possible_stages) == 1 else "unattributed"
            )
            chain = max((row[1] for row in candidates), key=len)
        else:
            stage, chain = "unattributed", []
        unattributed += int(stage == "unattributed")
        category = _category(event["name"], chain, event["cat"])
        request_ids = {
            next(
                (
                    name.removeprefix(REQUEST_PREFIX)
                    for name in reversed(row[1])
                    if name.startswith(REQUEST_PREFIX)
                ),
                None,
            )
            for row in candidates
        }
        request_id = next(iter(request_ids)) if len(request_ids) == 1 else None
        device_events.append(
            {
                "ts_us": event["ts"],
                "duration_us": event["dur"],
                "stage": stage,
                "category": category,
                "kernel": event["name"],
                "request_id": request_id,
                "stream": args.get("stream"),
                "device": args.get("device"),
            }
        )
        elapsed = event["dur"] / 1000.0
        for bucket, key in (
            (categories, category),
            (stages, stage),
            (kernels, (stage, category, event["name"])),
        ):
            bucket[key][0] += 1
            bucket[key][1] += elapsed
        intervals.append((event["ts"], event["ts"] + event["dur"]))
    work = sum(row["dur"] for row in device) / 1000.0
    return {
        "device_events": device_events,
        "request_scopes": [
            {
                "request_id": event["name"].removeprefix(REQUEST_PREFIX),
                "ts_us": event["ts"],
                "duration_us": event["dur"],
            }
            for event in events
            if event.get("cat") == "user_annotation"
            and event["name"].startswith(REQUEST_PREFIX)
        ],
        "device_event_count": len(device),
        "gpu_work_ms": work,
        "gpu_active_union_ms": _union_duration(intervals) / 1000.0,
        "unattributed_device_event_count": unattributed,
        "categories": [
            {"category": key, "event_count": value[0], "gpu_ms": value[1]}
            for key, value in sorted(categories.items())
        ],
        "by_stage": [
            {"stage": key, "event_count": value[0], "gpu_ms": value[1]}
            for key, value in sorted(stages.items())
        ],
        "kernels": [
            {
                "stage": key[0],
                "category": key[1],
                "kernel": key[2],
                "event_count": value[0],
                "gpu_ms": value[1],
            }
            for key, value in sorted(kernels.items())
        ],
        "host_stage_inclusive_ms": {
            key: _union_duration(value) / 1000.0
            for key, value in sorted(host_stages.items())
        },
        "kernel_stage_attribution": "CPU launch correlation; never GPU/host range overlap",
        "accounting": "Each leaf device activity appears once. Work sums may overlap across streams. Host stage scopes can be nested; never add host and GPU times to reconstruct wall latency.",
    }


def summarize_trace(trace):
    """Aggregate the shared device classifier without exposing a fake timeline."""
    result = classify_trace(trace)
    result.pop("device_events")
    result.pop("request_scopes")
    return result
