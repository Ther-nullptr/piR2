"""LIBERO-Dynamic import: physical time, request lineage and unscaled wall evidence."""

import math
from pathlib import Path

from .schema import (
    VERSION,
    digest,
    interval,
    optional_number,
    read_json,
    read_jsonl,
    source,
    under_root,
)


def _path(root, value):
    """The evaluator writes absolute paths; admit only files within its root."""
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Dynamic artifact must remain under evaluation root"
            ) from error
    return under_root(root, path)


def _number(value):
    value = optional_number(value)
    if value is None or value < 0:
        raise ValueError("Dynamic timing needs a nonnegative recorded number")
    return value


def _stage(audit, name, start_key, end_key):
    start = optional_number(audit.get(start_key))
    end = optional_number(audit.get(end_key))
    if start is not None and end is not None:
        interval(start, end)
    return {
        "name": name,
        "wall_started_s": start,
        "wall_completed_s": end,
        "duration_ms": None if start is None or end is None else (end - start) * 1000,
    }


def _native_equal(left, right, label, *, tolerance=1e-9):
    if abs(_number(left) - _number(right)) > tolerance:
        raise ValueError(f"Native Flow {label} disagrees with recorded evidence")


def _native_mode(manifest, config, audit):
    native = (
        config.get("native_latency") is True
        or manifest.get("timing_mode") == "native"
        or audit.get("timing_mode") == "native"
    )
    if not native:
        return False
    if (
        config.get("native_latency") is not True
        or manifest.get("timing_mode") != "native"
        or manifest.get("latency_kind") != "measured_per_request"
        or manifest.get("timing_rule") != "release_continuous_sync"
        or any(
            manifest.get(key) is not None
            for key in (
                "applied_latency_s",
                "raw_calibration_latency_s",
                "calibration_kind",
                "calibration_protocol",
                "calibration_retained_requests",
            )
        )
        or any(
            config.get(key) is not None for key in ("calibration", "latency_seconds")
        )
        or config.get("calibrate", False)
    ):
        raise ValueError("Native Flow requires measured requests, not a scalar latency")
    if audit and (
        audit.get("timing_mode") != "native"
        or audit.get("latency_kind") != "measured_per_request"
        or audit.get("calibration") is not None
        or not isinstance(manifest.get("latency_boundary"), str)
        or not manifest["latency_boundary"]
        or audit.get("latency_boundary") != manifest["latency_boundary"]
    ):
        raise ValueError("Native Flow audit must identify its measured timing boundary")
    return True


def _native_request(record, *, boundary, prefix):
    duration = _number(record.get("controller_policy_elapsed_s"))
    if (
        duration <= 0
        or record.get("controller_timing_boundary") != boundary
        or record.get("predicted_horizon") != 40
        or record.get("decoded_horizon") != 16
        or record.get("execution_prefix_steps") != prefix
        or record.get("server_audit", {}).get("forward_counts") != {"vlm": 1, "dit": 4}
    ):
        raise ValueError(
            "Native Flow needs full fresh-VLM/Flow4 request timing and H16 prefix"
        )
    start, end = (
        _number(record.get("request_started_s")),
        _number(record.get("request_completed_s")),
    )
    interval(start, end)
    if end - start > duration + 1e-6:
        raise ValueError(
            "Native Flow full policy duration cannot exclude part of its adapter call"
        )
    rpc_start, rpc_end = (
        _number(record.get("rpc_started_s")),
        _number(record.get("rpc_completed_s")),
    )
    interval(rpc_start, rpc_end)
    if rpc_start < start - 1e-6 or rpc_end > end + 1e-6:
        raise ValueError("Native Flow adapter timing must enclose its RPC")
    return duration


def _requests(audit, item_id, episode_id, *, native=False, boundary=None, prefix=None):
    requests = {}
    for record in audit.get("records", []):
        if record.get("item_id") != item_id:
            raise ValueError("Request belongs to another dynamic episode")
        if record.get("phase") != "scored":
            continue
        sequence = record.get("request_sequence")
        if type(sequence) is not int or sequence < 0 or sequence in requests:
            raise ValueError("Dynamic request sequence must be unique")
        server = record.get("server_audit", {})
        if server.get("source_tick") is not None and server[
            "source_tick"
        ] != record.get("source_tick"):
            raise ValueError("Dynamic server observation source disagrees with request")
        stages = [
            _stage(server, "Fresh VLM", "vlm_started_s", "vlm_completed_s"),
            _stage(server, "Flow DiT", "action_started_s", "action_completed_s"),
        ]
        vlm_end, dit_start = stages[0]["wall_completed_s"], stages[1]["wall_started_s"]
        if vlm_end is not None and dit_start is not None and dit_start < vlm_end:
            raise ValueError("Fresh VLM must complete before dependent Flow DiT")
        duration = (
            _native_request(record, boundary=boundary, prefix=prefix)
            if native and record.get("status") == "complete"
            else None
        )
        if duration is not None:
            stages.insert(
                0,
                {
                    "name": "Full policy call + normalization",
                    # Source perf_counter exposes a duration, not absolute endpoints.
                    # Do not manufacture endpoints from the adapter's monotonic clock.
                    "wall_started_s": None,
                    "wall_completed_s": None,
                    "duration_ms": duration * 1000,
                    "timing_scope": boundary,
                },
            )
        requests[sequence] = {
            **record,
            "service_duration_ms": None if duration is None else duration * 1000,
            "id": f"{episode_id}-request-{sequence}",
            "wall_stages": stages,
            "rpc_duration_ms": _stage(
                record, "RPC", "rpc_started_s", "rpc_completed_s"
            )["duration_ms"],
        }
    return requests


def _chunks(audit, requests, *, native=False):
    chunks, used_requests = {}, set()
    for chunk in audit.get("chunks", []):
        sequence = chunk.get("metadata", {}).get("policy_request_sequence")
        request = requests.get(sequence)
        chunk_id = chunk.get("chunk_id")
        if (
            type(chunk_id) is not int
            or chunk_id in chunks
            or request is None
            or sequence in used_requests
        ):
            raise ValueError("Dynamic chunk needs one unique scored request")
        if (
            request.get("status") != "complete"
            or request.get("source_tick_kind") != "action_step"
        ):
            raise ValueError(
                "Dynamic action source must be a complete action-step request"
            )
        if chunk.get("obs_step_idx") != request.get("source_tick"):
            raise ValueError("Dynamic chunk and observation action step disagree")
        if native:
            metadata = chunk["metadata"]
            duration = request["controller_policy_elapsed_s"]
            if (
                metadata.get("timing_mode") != "native"
                or metadata.get("latency_kind") != "measured_per_request"
                or metadata.get("latency_boundary")
                != request["controller_timing_boundary"]
                or metadata.get("timing_rule") != "release_continuous_sync"
                or metadata.get("raw_calibration_latency_s") is not None
                or metadata.get("physics_time_quantized_to_control_steps") is not False
            ):
                raise ValueError("Native Flow chunk has inconsistent timing metadata")
            _native_equal(chunk.get("policy_elapsed_s"), duration, "chunk duration")
            for key in ("applied_latency_s", "sim_latency_s"):
                _native_equal(metadata.get(key), duration, key)
            _native_equal(
                chunk.get("rollout_sim_time_s"),
                request.get("sim_time_s"),
                "observation time",
            )
        used_requests.add(sequence)
        chunks[chunk_id] = {**chunk, "request": request}
    return chunks


def _same_action(left, right):
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right) == 7
        and all(
            type(a) in (int, float)
            and type(b) in (int, float)
            and math.isfinite(a)
            and math.isfinite(b)
            and abs(a - b) <= 1e-5
            for a, b in zip(left, right)
        )
    )


def _events(
    audit, chunks, episode_id, control_freq, trace, *, native=False, complete=False
):
    events, pending = [], {}
    seen_holds = set()
    previous_time = 0.0
    actions = {}
    for motion in audit.get("motion_events", []):
        kind = motion.get("kind")
        if kind not in ("hold_started", "hold_completed", "action_completed"):
            continue
        end = _number(motion.get("rollout_sim_time_s"))
        if end + 1e-8 < previous_time:
            raise ValueError("Nonmonotonic dynamic physics event")
        chunk_id = motion.get("active_chunk_id")
        if chunk_id not in chunks:
            raise ValueError("Dynamic physics event has no recorded chunk source")
        request = chunks[chunk_id]["request"]
        if kind == "hold_started":
            if chunk_id in pending:
                raise ValueError("Duplicate inference hold start")
            if native:
                if (
                    chunk_id in seen_holds
                    or motion.get("action_step") != request["source_tick"]
                ):
                    raise ValueError(
                        "Native Flow must apply each request hold exactly once"
                    )
                _native_equal(end, previous_time, "hold progression", tolerance=1e-8)
                _native_equal(
                    end,
                    request.get("sim_time_s"),
                    "hold observation time",
                    tolerance=1e-8,
                )
                _native_equal(
                    motion.get("applied_latency_s"),
                    request["controller_policy_elapsed_s"],
                    "hold start duration",
                )
                seen_holds.add(chunk_id)
            pending[chunk_id] = motion
            previous_time = end
            continue
        if kind == "hold_completed":
            if chunk_id not in pending:
                raise ValueError("Inference hold completion has no start")
            start_motion = pending.pop(chunk_id)
            start = _number(start_motion["rollout_sim_time_s"])
            if native:
                duration = request["controller_policy_elapsed_s"]
                _native_equal(
                    motion.get("applied_latency_s"),
                    duration,
                    "hold completion duration",
                    tolerance=1e-8,
                )
                _native_equal(
                    end - start, duration, "physical hold duration", tolerance=1e-8
                )
                if motion.get("action_step") != request["source_tick"]:
                    raise ValueError(
                        "Native Flow hold completion has a different action source"
                    )
            kind = "inference_hold"
        else:
            if native and (chunk_id not in seen_holds or chunk_id in pending):
                raise ValueError("Native Flow action cannot precede its measured hold")
            start = end - 1 / control_freq
            if abs(start - previous_time) > 1e-6:
                raise ValueError(
                    "Action physics boundary does not match recorded progression"
                )
            step = motion.get("action_step")
            if type(step) is not int or step in actions:
                raise ValueError("Action steps must be unique integers")
            offset = step - request["source_tick"]
            source_actions = request.get("actions", [])
            if not 0 <= offset < len(source_actions) or not _same_action(
                motion.get("action"), source_actions[offset]
            ):
                raise ValueError(
                    "Executed action does not match its recorded request prefix"
                )
            actions[step] = motion
            kind = "action"
        interval(start, end)
        events.append(
            {
                "id": f"{episode_id}-event-{len(events)}",
                "kind": kind,
                "start_s": start,
                "end_s": end,
                "chunk_id": chunk_id,
                "request_id": request["id"],
                "action_step": motion.get("action_step"),
                "source": motion,
            }
        )
        previous_time = end
    if native and complete and seen_holds != set(chunks):
        raise ValueError(
            "Native Flow complete episode requires every measured chunk hold"
        )
    if pending:
        # A failed episode may retain a start without inventing its completion.
        for chunk_id, motion in pending.items():
            events.append(
                {
                    "id": f"{episode_id}-event-{len(events)}",
                    "kind": "inference_hold",
                    "start_s": motion["rollout_sim_time_s"],
                    "end_s": None,
                    "chunk_id": chunk_id,
                    "request_id": chunks[chunk_id]["request"]["id"],
                    "action_step": motion.get("action_step"),
                    "source": motion,
                }
            )
    for row in trace:
        motion = actions.get(row.get("step_idx"))
        if motion is None or row.get("chunk_id") != motion.get("active_chunk_id"):
            raise ValueError("Trace action has no matching episode motion source")
        if abs(
            _number(row.get("sim_time_s")) - motion["rollout_sim_time_s"]
        ) > 1e-6 or not _same_action(row.get("executed_action"), motion.get("action")):
            raise ValueError("Trace action disagrees with recorded physics event")
    return events


def _frames(audit, episode_id, duration, events):
    frames = []
    previous_sim = previous_media = -1.0
    for index, sample in enumerate(audit.get("video_samples", [])):
        sim = _number(sample.get("sample_sim_time_s"))
        media = _number(sample.get("media_time_s"))
        if (
            sample.get("frame_index") != index
            or sim <= previous_sim
            or media <= previous_media
        ):
            raise ValueError(
                "Dynamic frame indices and timestamps must increase monotonically"
            )
        if duration is not None and sim > duration + 1e-6:
            raise ValueError("Dynamic frame lies after recorded episode duration")
        matching = [
            event
            for event in events
            if event["kind"] == sample.get("phase")
            and event["chunk_id"] == sample.get("active_chunk_id")
            and event.get("action_step") == sample.get("action_step")
            and event["start_s"] - 1e-8 <= sim
            and (event["end_s"] is None or sim <= event["end_s"] + 1e-8)
        ]
        if len(matching) != 1:
            raise ValueError(
                "Dynamic frame must match its own recorded physics phase and action source"
            )
        frames.append(
            {**sample, "frame_id": f"{episode_id}-frame-{index}", "relative_s": sim}
        )
        previous_sim, previous_media = sim, media
    return frames


def _episode(root, manifest, config, media, run_id):
    item_id = manifest["item_id"]
    episode_id = f"{run_id}-episode-{digest(item_id)[:12]}"
    provenance, warnings = [], []
    paths = {
        key: _path(root, manifest[key])
        for key in (
            "policy_audit",
            "step_trace",
            "model_input_video",
            "wrist_input_video",
        )
        if manifest.get(key)
    }

    def read_optional(key, lines=False):
        path = paths.get(key)
        if path is None or not path.is_file():
            warnings.append(f"{key}: evidence unavailable")
            return [] if lines else {}
        provenance.append(source(path))
        return (read_jsonl if lines else read_json)(path)

    audit = read_optional("policy_audit")
    if audit and audit.get("item_id") != item_id:
        raise ValueError("Dynamic audit belongs to another episode")
    trace = read_optional("step_trace", lines=True)
    native = _native_mode(manifest, config, audit)
    requests = _requests(
        audit,
        item_id,
        episode_id,
        native=native,
        boundary=manifest.get("latency_boundary"),
        prefix=manifest.get("replan_steps_requested"),
    )
    chunks = _chunks(audit, requests, native=native)
    freq = _number(manifest.get("control_freq"))
    if freq <= 0:
        raise ValueError("Dynamic control frequency must be positive")
    duration = optional_number(manifest.get("policy_rollout_sim_time_s"))
    events = (
        _events(
            audit,
            chunks,
            episode_id,
            freq,
            trace,
            native=native,
            complete=manifest.get("status") == "ok",
        )
        if audit
        else []
    )
    if (
        audit
        and manifest.get("status") == "ok"
        and manifest.get("steps") is not None
        and sum(event["kind"] == "action" for event in events) != manifest["steps"]
    ):
        raise ValueError("Dynamic manifest action count differs from physics evidence")
    if audit and manifest.get("status") == "ok":
        if any(event["end_s"] is None for event in events):
            raise ValueError(
                "Complete dynamic episode contains incomplete physics phase"
            )
        last_time = events[-1]["end_s"] if events else 0
        if duration is None or abs(last_time - duration) > 1e-6:
            raise ValueError("Complete episode duration differs from recorded physics")
    frames = _frames(audit, episode_id, duration, events)
    if (
        audit
        and manifest.get("video_frame_count") is not None
        and len(frames) != manifest["video_frame_count"]
    ):
        raise ValueError("Dynamic frame count differs from recorded manifest")
    assets = {}
    browser_times = None
    for key, camera in (
        ("model_input_video", "image"),
        ("wrist_input_video", "wrist_image"),
    ):
        path = paths.get(key)
        if path is not None and path.is_file() and frames:
            derivative = media.browser_video(
                root,
                path.relative_to(root),
                source_media_times=[frame["media_time_s"] for frame in frames],
            )
            if (
                browser_times is not None
                and browser_times != derivative["media_times_s"]
            ):
                raise ValueError("Dynamic camera derivative timestamps disagree")
            browser_times = derivative["media_times_s"]
            assets[camera] = {
                "media_id": derivative["media_id"],
                "mime": derivative["mime"],
            }
        elif path is not None:
            warnings.append(f"{key}: media or exact frame mapping unavailable")
    if browser_times is not None:
        frames = [
            {
                **frame,
                "source_media_time_s": frame["media_time_s"],
                "media_time_s": timestamp,
            }
            for frame, timestamp in zip(frames, browser_times)
        ]
    return {
        "id": episode_id,
        **{
            key: manifest.get(key)
            for key in (
                "item_id",
                "condition_id",
                "suite",
                "task_id",
                "level",
                "traj_index",
                "speed_mps",
                "precision",
                "protocol",
                "status",
                "error",
                "success",
                "language",
                "steps",
                "diagnostics",
                "timing_rule",
                "timing_mode",
                "latency_kind",
                "latency_boundary",
                "policy_identity",
                "video_view_contract",
                "setup",
            )
        },
        "seed": config.get("seed"),
        "init_state_id": config.get("init_state_id"),
        "duration_s": duration,
        "clock": "rollout_sim_time_s",
        "control_freq": freq,
        "execution_prefix_steps": manifest.get("replan_steps_requested"),
        "predicted_horizon": manifest.get("policy_identity", {}).get(
            "predicted_horizon"
        ),
        "decoded_horizon": manifest.get("policy_identity", {}).get("decoded_horizon"),
        "calibration": {
            "kind": manifest.get("calibration_kind"),
            "protocol": manifest.get("calibration_protocol"),
            "retained_requests": manifest.get("calibration_retained_requests"),
            "warmup_requests": manifest.get("calibration_warmup_requests"),
            "raw_latency_s": manifest.get("raw_calibration_latency_s"),
            "applied_latency_s": manifest.get("applied_latency_s"),
        },
        "episode_warmup_requests": manifest.get("warmup_requests_per_episode"),
        "episode_warmup_recorded": sum(
            record.get("phase") == "warmup" for record in audit.get("records", [])
        )
        if audit
        else None,
        "video_fps": manifest.get("video_fps"),
        "media": assets,
        "frames": frames,
        "events": events,
        "requests": list(requests.values()),
        "trace": trace,
        "motion_events": audit.get("motion_events", []),
        "provenance": provenance,
        "warnings": warnings,
    }


def dynamic_run(path, media):
    root = Path(path).resolve()
    evaluation_path, manifest_path = root / "evaluation.json", root / "manifest.jsonl"
    evaluation = read_json(evaluation_path)
    is_streaming = (
        evaluation.get("schema") == "pir2.libero_dynamic.streaming_evaluation.v1"
    )
    if (
        not is_streaming
        and evaluation.get("schema") != "pir2.libero_dynamic.default.v1"
    ):
        raise ValueError("Unsupported dynamic evaluation schema")
    manifests = read_jsonl(manifest_path)
    if not manifests or len({m["item_id"] for m in manifests}) != len(manifests):
        raise ValueError("Dynamic evaluation needs unique recorded episodes")
    precisions = {m.get("precision") for m in manifests}
    if len(precisions) != 1:
        raise ValueError("Dynamic evaluation precision must be unambiguous")
    provenance = [source(evaluation_path), source(manifest_path)]
    run_id = "dynamic-" + digest(provenance)[:24]
    config = evaluation.get("configuration", {})
    if is_streaming:
        from .dynamic_streaming import streaming_episode

        episodes = [
            streaming_episode(root, m, config, media, run_id) for m in manifests
        ]
    else:
        episodes = [_episode(root, m, config, media, run_id) for m in manifests]
    return {
        "schema": "pir2.replay.run",
        "version": VERSION,
        "id": run_id,
        "kind": "dynamic",
        "label": root.name,
        "precision": precisions.pop(),
        "mode": "libero_dynamic",
        "protocol": "pir2_dynamic" if is_streaming else evaluation.get("protocol"),
        "configuration": config,
        "upstream": evaluation.get("upstream"),
        "episodes": episodes,
        "provenance": provenance,
        "warnings": [
            f"{e['item_id']}: {warning}" for e in episodes for warning in e["warnings"]
        ],
        "time_contract": {
            "timeline": "rollout simulation seconds; streaming logical timestamps subtract logical_epoch_s only"
            if is_streaming
            else "rollout simulation seconds, including physical inference holds",
            "wall_panel": "measured request/server monotonic time; never scaled onto simulation time",
            "video": "exact video_samples sample_sim_time_s to media_time_s mapping",
            "source_tick": "executed action count, excluding inference hold substeps",
        },
    }
