"""Import recorded πR² dynamic scheduling and its separate measured RPC evidence."""

from .dynamic import _frames, _number, _path, _same_action, _stage
from .schema import digest, interval, optional_number, read_json, read_jsonl, source


def _equal(left, right, label):
    if abs(_number(left) - _number(right)) > 1e-6:
        raise ValueError(f"Streaming {label} disagrees with recorded evidence")


def _service_duration(record, expected):
    service = record.get("service_wall") or {}
    if (
        service.get("clock") != "host_monotonic"
        or service.get("boundary") != "full_adapter_call"
    ):
        raise ValueError(
            "Native streaming needs full adapter monotonic service evidence"
        )
    start, end = _number(service.get("started_s")), _number(service.get("completed_s"))
    interval(start, end)
    duration = _number(service.get("duration_s"))
    _equal(end - start, duration, "native service interval")
    _equal(duration, expected, "native logical duration")
    rpc = record.get("wall", record)
    rpc_start, rpc_end = rpc.get("wall_started_s"), rpc.get("wall_completed_s")
    if rpc_start is not None and (
        _number(rpc_start) < start - 1e-6 or _number(rpc_end) > end + 1e-6
    ):
        raise ValueError("Native adapter service must enclose its own RPC interval")
    return duration


def _wall(record, lane):
    rpc = _stage(record, "RPC", "wall_started_s", "wall_completed_s")
    server = record.get("server_audit", {})
    if lane == "s1":
        started = optional_number(server.get("server_plan_started_s"))
        duration = optional_number(server.get("server_seconds"))
        label = "DiT server plan"
    else:
        started = None
        duration = optional_number(
            server.get("vlm_seconds" if lane == "s2" else "bootstrap_seconds")
        )
        label = "S2 VLM" if lane == "s2" else "Bootstrap VLM + Flow4"
    # A duration without an absolute start remains unpositioned on the wall axis.
    stage = {
        "name": label,
        "wall_started_s": started,
        "wall_completed_s": started + duration
        if started is not None and duration is not None
        else None,
        "duration_ms": duration * 1000 if duration is not None else None,
        "timing_scope": "server reported interval; not isolated GPU kernel time",
    }
    service = record.get("service_wall")
    full = (
        []
        if not service
        else [
            {
                "name": "Full adapter call",
                "wall_started_s": service["started_s"],
                "wall_completed_s": service["completed_s"],
                "duration_ms": service["duration_s"] * 1000,
                "timing_scope": "measured full adapter call; contains RPC, not additive",
            }
        ]
    )
    return [rpc, *full, stage]


def _requests(stream, episode_id, epoch, latencies, *, native=False):
    records, by_tick, features = [], {}, {}
    bootstrap = stream.get("bootstrap")
    if bootstrap:
        start = _number(bootstrap["capture_s"]) - epoch
        end = _number(bootstrap["logical_completed_s"]) - epoch
        interval(start, end)
        _equal(end - start, bootstrap.get("logical_duration_s"), "bootstrap duration")
        if native:
            _service_duration(bootstrap, end - start)
        else:
            _equal(end - start, latencies.get("bootstrap_s"), "bootstrap latency label")
        record = {
            **bootstrap,
            **bootstrap.get("wall", {}),
            "lane": "bootstrap",
            "id": f"{episode_id}-bootstrap",
            "request_sequence": "bootstrap",
            "start_s": start,
            "end_s": end,
            "parent_request_id": None,
            "source_tick": 0,
            "sim_time_s": start,
        }
        record["wall_stages"] = _wall(record, "bootstrap")
        record["rpc_duration_ms"] = record["wall_stages"][0]["duration_ms"]
        record["service_duration_ms"] = (
            bootstrap["service_wall"]["duration_s"] * 1000 if native else None
        )
        records.append(record)
        features[0] = record
    seen = set()
    for raw in stream.get("requests", []):
        seq, lane = raw.get("request_id"), raw.get("lane")
        if type(seq) is not int or seq < 0 or seq in seen or lane not in ("s1", "s2"):
            raise ValueError("Streaming requests need unique IDs and known lanes")
        seen.add(seq)
        start = _number(raw["logical_started_s"]) - epoch
        end = _number(raw["logical_completed_s"]) - epoch
        interval(start, end)
        _equal(end - start, raw.get("logical_duration_s"), "logical request duration")
        latency_key = (
            "s2_s"
            if lane == "s2"
            else "recovery_s"
            if raw.get("normal_or_recovery") == "recovery5"
            else "s1_s"
        )
        if native:
            _service_duration(raw, end - start)
        else:
            _equal(end - start, latencies.get(latency_key), "stage latency label")
        if start < 0:
            raise ValueError("Streaming request starts before logical epoch")
        record = {
            **raw,
            "id": f"{episode_id}-request-{seq}",
            "request_sequence": seq,
            "start_s": start,
            "end_s": end,
            "sim_time_s": start,
            "parent_request_id": None,
            "wall_stages": _wall(raw, lane),
        }
        record["rpc_duration_ms"] = record["wall_stages"][0]["duration_ms"]
        record["service_duration_ms"] = (
            raw["service_wall"]["duration_s"] * 1000 if native else None
        )
        if lane == "s2":
            sequence = raw.get("feature_sequence")
            if type(sequence) is not int or sequence <= 0 or sequence in features:
                raise ValueError("Streaming feature sequence must be unique")
            features[sequence] = record
        else:
            tick = raw.get("request_tick")
            if type(tick) is not int or tick < 0 or tick in by_tick:
                raise ValueError("Streaming S1 request tick must be unique")
            by_tick[tick] = record
        records.append(record)
    for record in by_tick.values():
        parent = features.get(record.get("feature_sequence"))
        if parent is None or parent["end_s"] > record["start_s"] + 1e-8:
            raise ValueError("Streaming S1 needs an already completed feature source")
        if parent["lane"] != "bootstrap" and parent.get("status") != "completed":
            raise ValueError("Streaming S1 cannot use an unpublished feature")
        _equal(
            record.get("feature_capture_s"), parent.get("capture_s"), "feature capture"
        )
        if record.get("feature_source_tick") != parent.get("source_tick"):
            raise ValueError("Streaming S1 feature source tick disagrees")
        _validate_cache(record, parent)
        record["parent_request_id"] = parent["id"]
    _publications(stream, by_tick)
    return records, by_tick


def _validate_cache(request, parent):
    server = request.get("server_audit", {})
    cache = server.get("cache") or {}
    meta = (
        parent.get("meta")
        if parent["lane"] == "bootstrap"
        else parent.get("server_audit")
    )
    meta = meta or {}
    _equal(cache.get("capture_s"), meta.get("capture_s"), "server cache capture")
    if cache.get("source_tick") != meta.get("source_tick") or cache.get(
        "frame_hashes"
    ) != meta.get("frame_hashes"):
        raise ValueError("Streaming server cache source or frame hashes disagree")
    _equal(
        request.get("feature_age_s"),
        _number(request.get("state_capture_s"))
        - _number(request.get("feature_capture_s")),
        "request feature age",
    )
    recovery = request.get("normal_or_recovery")
    if recovery not in ("normal1", "recovery5") or server.get(
        "buffer_restarted"
    ) is not (recovery == "recovery5"):
        raise ValueError(
            "Streaming recovery label disagrees with server buffer restart"
        )
    counts = server.get("forward_counts", {})
    if counts.get("vlm") != 0 or counts.get("dit") != (
        5 if recovery == "recovery5" else 1
    ):
        raise ValueError(
            "Streaming recovery label disagrees with actual forward counts"
        )


def _publications(stream, by_tick):
    """Adoption is proved by the recorded post-publication queue, not readiness."""
    for request in by_tick.values():
        request["installed_ticks"] = []
        publication = request.get("publication")
        matching = [
            e
            for e in stream.get("events", [])
            if e.get("kind") == "logical_request_completed"
            and e.get("lane") == "s1"
            and e.get("request_id") == request["request_id"]
        ]
        if request.get("status") != "completed":
            if publication or matching:
                raise ValueError(
                    "Unpublished streaming request has publication evidence"
                )
            continue
        if not publication or len(matching) != 1:
            raise ValueError("Completed streaming S1 needs one publication queue event")
        event = matching[0]
        for key in (
            "installed_slots",
            "expired_slots",
            "protected_slots",
            "next_unexecuted_tick",
        ):
            value = publication.get(key)
            if type(value) is not int or value < 0 or value != event.get(key):
                raise ValueError(
                    "Streaming publication counts disagree with queue event"
                )
        _equal(
            publication.get("logical_published_s"),
            event.get("logical_published_s"),
            "publication time",
        )
        _equal(
            publication.get("logical_published_s"),
            event.get("t_s"),
            "publication event time",
        )
        _equal(
            publication.get("logical_published_s"),
            request.get("logical_completed_s"),
            "publication completion",
        )
        queue = event.get("action_buffer", {})
        slots = queue.get("slots", [])
        installed = [
            slot
            for slot in slots
            if slot.get("producer_kind") == "request"
            and slot.get("producer_request_tick") == request["request_tick"]
        ]
        ticks = [slot.get("tick") for slot in installed]
        if (
            len(ticks) != len(set(ticks))
            or len(ticks) != publication["installed_slots"]
        ):
            raise ValueError(
                "Streaming installed count disagrees with exact queue slots"
            )
        valid = request.get("server_audit", {})
        low, high = valid.get("valid_start_tick"), valid.get("valid_end_tick")
        committed = queue.get("committed_until")
        if (
            type(low) is not int
            or type(high) is not int
            or high - low != len(request.get("model_actions", []))
            or type(committed) is not int
        ):
            raise ValueError(
                "Streaming publication lacks a valid recorded action window"
            )
        for slot in installed:
            tick = slot.get("tick")
            if (
                type(tick) is not int
                or not low <= tick < high
                or tick < publication["next_unexecuted_tick"]
                or tick < committed
                or slot.get("committed") is not False
                or slot.get("status") != "future"
            ):
                raise ValueError("Streaming installed tick is expired or protected")
        request["installed_ticks"] = ticks


def _action_ages(action, request):
    if request is None:
        if any(
            action.get(k) is not None
            for k in (
                "feature_capture_s",
                "state_capture_s",
                "feature_age_s",
                "state_age_s",
            )
        ):
            raise ValueError("Streaming fallback has no model capture or AoI")
        return
    for kind in ("feature", "state"):
        expected = (
            request.get("capture_s")
            if request["lane"] == "bootstrap"
            else request.get(kind + "_capture_s")
        )
        capture = action.get(kind + "_capture_s")
        _equal(capture, expected, "action " + kind + " capture")
        _equal(
            action.get(kind + "_age_s"),
            _number(action.get("logical_started_s")) - _number(capture),
            "action " + kind + " AoI",
        )


def _events(
    stream, motions, records, by_tick, episode_id, epoch, period, duration, complete
):
    bootstrap = next((r for r in records if r["lane"] == "bootstrap"), None)
    actions = {a["tick"]: a for a in stream.get("actions", [])}
    if len(actions) != len(stream.get("actions", [])):
        raise ValueError("Streaming action ticks must be unique")
    events, used, pending = [], set(), None
    previous = 0.0
    for motion in motions:
        kind = motion.get("kind")
        if kind not in ("hold_started", "hold_completed", "action_completed"):
            continue
        end = _number(motion["rollout_sim_time_s"])
        if end < previous - 1e-6:
            raise ValueError("Streaming physical time must be monotonic")
        tick = motion.get("action_step")
        if kind == "hold_started":
            if (
                pending is not None
                or events
                or motion.get("active_chunk_id") is not None
            ):
                raise ValueError("Streaming startup hold must precede actions")
            pending = motion
            previous = end
            continue
        if kind == "hold_completed":
            if pending is None or bootstrap is None:
                raise ValueError("Streaming startup hold has no bootstrap source")
            start = _number(pending["rollout_sim_time_s"])
            _equal(start, bootstrap["start_s"], "startup start")
            _equal(end, bootstrap["end_s"], "startup end")
            event = {
                "kind": "bootstrap_hold",
                "start_s": start,
                "end_s": end,
                "request_id": bootstrap["id"],
                "action_step": 0,
                "chunk_id": None,
                "source": motion,
                "label": "Bootstrap · moving world hold",
            }
            pending = None
        else:
            action = actions.get(tick)
            if action is None:
                if complete:
                    raise ValueError(
                        "Streaming physics action lacks a controller record"
                    )
                # Failed env.step can finish physics before the core records its result.
                event = {
                    "kind": "action",
                    "start_s": end - period,
                    "end_s": end,
                    "request_id": None,
                    "action_step": tick,
                    "chunk_id": motion.get("active_chunk_id"),
                    "source": motion,
                    "lineage_missing": True,
                }
            else:
                if tick in used or motion.get("active_chunk_id") != tick:
                    raise ValueError("Streaming motion action tick disagrees")
                used.add(tick)
                start = _number(action["logical_started_s"]) - epoch
                _equal(start, previous, "action start")
                _equal(action["logical_completed_s"] - epoch, end, "action end")
                _equal(end - start, period, "action period")
                if not _same_action(action.get("raw_action"), motion.get("action")):
                    raise ValueError("Streaming executed action differs from physics")
                model_action = action.get("model_action")
                if (
                    not isinstance(model_action, list)
                    or len(model_action) != 7
                    or not _same_action(
                        action.get("raw_action"),
                        [*model_action[:6], 1 - 2 * model_action[6]],
                    )
                ):
                    raise ValueError("Streaming model/raw gripper mapping disagrees")
                source_info = action.get("source", {})
                if source_info.get("tick") != tick:
                    raise ValueError("Streaming source slot differs from executed tick")
                producer = source_info.get("producer_kind")
                request = None
                if producer == "bootstrap":
                    request = bootstrap
                    offset = tick
                    source_actions = (bootstrap or {}).get("actions", [])
                elif producer == "request":
                    request = by_tick.get(source_info.get("producer_request_tick"))
                    if request is None:
                        raise ValueError("Streaming action has no S1 producer")
                    server = request.get("server_audit", {})
                    low, high = (
                        server.get("valid_start_tick"),
                        server.get("valid_end_tick"),
                    )
                    if (
                        type(low) is not int
                        or type(high) is not int
                        or not low <= tick < high
                    ):
                        raise ValueError(
                            "Streaming action is outside producer valid window"
                        )
                    if (
                        request["end_s"] > start + 1e-8
                        or request.get("status") != "completed"
                    ):
                        raise ValueError(
                            "Streaming action precedes producer completion"
                        )
                    if (
                        tick not in request.get("installed_ticks", [])
                        or _number(
                            request.get("publication", {}).get("logical_published_s")
                        )
                        > action["logical_started_s"] + 1e-8
                    ):
                        raise ValueError(
                            "Streaming executed tick was not installed before execution"
                        )
                    source_actions = request.get("model_actions", [])
                    offset = tick - low
                elif producer == "fallback" and action.get("fallback") is True:
                    source_actions, offset = [action.get("model_action")], 0
                else:
                    raise ValueError("Streaming action producer is unknown")
                _action_ages(action, request)
                if not 0 <= offset < len(source_actions) or not _same_action(
                    action.get("model_action"), source_actions[offset]
                ):
                    raise ValueError("Streaming action differs from its producer")
                event = {
                    "kind": "action",
                    "start_s": start,
                    "end_s": end,
                    "request_id": request["id"] if request else None,
                    "parent_request_id": request.get("parent_request_id")
                    if request
                    else None,
                    "action_step": tick,
                    "chunk_id": tick,
                    "source": action,
                    "feature_age_s": action.get("feature_age_s"),
                    "state_age_s": action.get("state_age_s"),
                    "fallback": action.get("fallback"),
                    "producer_kind": producer,
                }
        event["id"] = f"{episode_id}-event-{len(events)}"
        events.append(event)
        previous = end
    if pending:
        if complete:
            raise ValueError(
                "Complete streaming episode has an unfinished bootstrap hold"
            )
        events.append(
            {
                "id": f"{episode_id}-event-{len(events)}",
                "kind": "bootstrap_hold",
                "start_s": pending["rollout_sim_time_s"],
                "end_s": None,
                "request_id": bootstrap["id"] if bootstrap else None,
                "action_step": 0,
                "chunk_id": None,
                "source": pending,
            }
        )
    if complete:
        if used != set(actions):
            raise ValueError("Streaming controller action has no physical evidence")
        _equal(previous, duration, "episode duration")
    for record in records:
        if record["lane"] == "bootstrap":
            continue
        events.append(
            {
                "id": f"{record['id']}-event",
                "kind": record["lane"],
                "start_s": record["start_s"],
                "end_s": record["end_s"],
                "request_id": record["id"],
                "parent_request_id": record.get("parent_request_id"),
                "action_step": record.get("request_tick"),
                "chunk_id": None,
            }
        )
    return events


def _partial_physics(audit, manifest, events, episode_id, period):
    if manifest.get("status") == "ok":
        return
    recorded = {e["action_step"] for e in events if e["kind"] == "action"}
    for attempt in audit.get("action_attempts", []):
        tick = attempt.get("tick")
        if tick in recorded:
            continue
        if type(tick) is not int or tick < 0 or attempt.get("status") != "error":
            raise ValueError(
                "Unrecorded physical action needs an explicit failed attempt"
            )
        origin = _number(manifest.get("physical_started_s"))
        start = _number(attempt.get("physics_started_s")) - origin
        end = _number(attempt.get("physics_completed_s")) - origin
        interval(start, end)
        _equal(
            end - start, attempt.get("physics_elapsed_s"), "partial physical elapsed"
        )
        if (
            end > _number(manifest.get("rollout_sim_time_s")) + 1e-8
            or end - start > period + 1e-8
        ):
            raise ValueError(
                "Partial physical action exceeds its recorded episode/step"
            )
        completed = abs(end - start - period) <= 1e-8
        if attempt.get("physics_complete") is not completed:
            raise ValueError(
                "Partial physical completion flag disagrees with measured time"
            )
        events.append(
            {
                "id": f"{episode_id}-partial-{tick}",
                "kind": "action",
                "start_s": start,
                "end_s": end,
                "request_id": None,
                "action_step": tick,
                "chunk_id": tick,
                "source": attempt,
                "partial": not completed,
                "lineage_missing": True,
            }
        )
        recorded.add(tick)


def _media(root, manifest, audit, episode_id, duration, events, registry, warnings):
    samples = audit.get("video_samples", [])
    explicit = any("streams" in s for s in samples)
    cameras, assets = {}, {}
    counts = manifest.get("video_frame_counts", {})
    for stream, camera, path_key in [
        ("dual", "image", "model_input_video"),
        ("wrist", "wrist_image", "wrist_input_video"),
    ]:
        mapping = []
        for sample in samples:
            info = sample.get("streams", {}).get(stream)
            if explicit:
                if info is None or info.get("status") != "written":
                    continue
                sample = {
                    **sample,
                    "frame_index": info["frame_index"],
                    "media_time_s": info["media_time_s"],
                }
            mapping.append(sample)
        frames = _frames(
            {"video_samples": mapping}, episode_id + "-" + camera, duration, events
        )
        expected = counts.get(stream) if isinstance(counts, dict) else None
        if expected is None and not explicit:
            expected = manifest.get("video_frame_count")
        if expected is not None and len(frames) != expected:
            raise ValueError(
                "Streaming camera frame count differs from recorded mapping"
            )
        path = _path(root, manifest[path_key]) if manifest.get(path_key) else None
        if path is not None and path.is_file() and frames:
            derivative = registry.browser_video(
                root,
                path.relative_to(root),
                source_media_times=[f["media_time_s"] for f in frames],
            )
            frames = [
                {**f, "source_media_time_s": f["media_time_s"], "media_time_s": t}
                for f, t in zip(frames, derivative["media_times_s"], strict=True)
            ]
            assets[camera] = {
                "media_id": derivative["media_id"],
                "mime": derivative["mime"],
            }
        elif frames:
            warnings.append(
                f"{camera}: exact frame mapping retained; playable media unavailable"
            )
        cameras[camera] = frames
    return assets, cameras


def streaming_episode(root, manifest, config, media, run_id):
    episode_id = f"{run_id}-episode-{digest(manifest['item_id'])[:12]}"
    provenance, warnings = [], []
    audit_path = _path(root, manifest["policy_audit"])
    audit = read_json(audit_path)
    provenance.append(source(audit_path))
    if (
        audit.get("schema") != "pir2.libero_dynamic.streaming_audit.v1"
        or audit.get("item_id") != manifest["item_id"]
    ):
        raise ValueError("Streaming audit identity/schema mismatch")
    stream = dict(audit.get("streaming") or {})
    # Runner partial results omit bootstrap/config. Its paid startup receipt is
    # retained at audit top level even if the steady-state controller failed.
    if not stream.get("bootstrap") and audit.get("bootstrap"):
        stream["bootstrap"] = audit["bootstrap"]
    mode = manifest.get("timing_mode", "fixed")
    native = mode == "native"
    if mode not in ("native", "fixed") or stream.get("timing_mode", mode) != mode:
        raise ValueError("Streaming timing mode disagrees with its audit")
    if native:
        for evidence in (manifest, stream):
            if evidence and (
                evidence.get("latency_provenance") != "measured_per_request"
                or evidence.get("timing_domain") != "measured_request_discrete_event"
            ):
                raise ValueError("Native streaming timing provenance is inconsistent")
        if any(
            evidence.get("stage_latencies") is not None
            for evidence in (manifest, stream, stream.get("config", {}), audit)
        ):
            raise ValueError("Native streaming cannot declare fixed stage latencies")
    epoch = (
        _number(manifest["logical_epoch_s"])
        if manifest.get("logical_epoch_s") is not None
        else None
    )
    duration = _number(manifest["rollout_sim_time_s"])
    complete = manifest.get("status") == "ok"
    period = _number(stream.get("config", {}).get("period_s", 0.05))
    if period <= 0:
        raise ValueError("Streaming control period must be positive")
    if epoch is None:
        if complete or stream.get("requests") or stream.get("actions"):
            raise ValueError("Streaming logical epoch missing")
        records, by_tick, events = [], {}, []
    else:
        if stream.get("logical_epoch_s") is not None:
            _equal(epoch, stream["logical_epoch_s"], "logical epoch")
        latencies = manifest.get("stage_latencies") or {}
        bootstrap = stream.get("bootstrap")
        if bootstrap or not native:
            startup = (
                _number(bootstrap.get("logical_duration_s"))
                if native
                else _number(latencies.get("bootstrap_s"))
            )
            _equal(manifest.get("control_origin_s"), epoch + startup, "control origin")
            if native:
                _equal(manifest.get("startup_seconds"), startup, "native startup")
        elif complete or stream.get("requests") or stream.get("actions"):
            raise ValueError("Native streaming requires a completed bootstrap")
        records, by_tick = _requests(
            stream, episode_id, epoch, latencies, native=native
        )
        events = _events(
            stream,
            audit.get("motion_events", []),
            records,
            by_tick,
            episode_id,
            epoch,
            period,
            duration,
            complete,
        )
    if complete and sum(e["kind"] == "action" for e in events) != manifest.get("steps"):
        raise ValueError("Streaming manifest action count differs from physics")
    _partial_physics(audit, manifest, events, episode_id, period)
    assets, cameras = _media(
        root, manifest, audit, episode_id, duration, events, media, warnings
    )
    trace = stream.get("events", [])
    if manifest.get("step_trace"):
        trace_path = _path(root, manifest["step_trace"])
        if trace_path.is_file():
            trace = read_jsonl(trace_path)
            provenance.append(source(trace_path))
            if trace != stream.get("events", []):
                raise ValueError(
                    "Streaming trace disagrees with controller event evidence"
                )
    identity = (audit.get("identities") or {}).get("action") or {}
    return {
        **manifest,
        "id": episode_id,
        "seed": config.get("seed"),
        "init_state_id": config.get("init_state_id"),
        "duration_s": duration,
        "clock": "rollout_sim_time_s",
        "control_freq": 1 / period,
        "execution_prefix_steps": 1,
        "predicted_horizon": identity.get("predicted_horizon"),
        "decoded_horizon": identity.get("decoded_horizon"),
        "policy_identity": identity,
        "calibration": {
            "kind": "measured_per_request" if native else "counterfactual",
            "protocol": "independent_stage_service",
            "retained_requests": 0,
            "warmup_requests": 0,
            "applied_latency_s": None,
        },
        "episode_warmup_requests": manifest.get("warmup_rounds"),
        "episode_warmup_recorded": len(audit.get("warmup", [])),
        "warmup_rpc_recorded": sum(
            r.get("phase") == "warmup" for r in audit.get("rpc_records", [])
        ),
        "delay_ticks_observed": sorted(
            {
                r["delay_ticks"]
                for r in records
                if r["lane"] == "s1" and type(r.get("delay_ticks")) is int
            }
        ),
        "media": assets,
        "frames": cameras["image"],
        "frames_by_camera": {
            key: value for key, value in cameras.items() if key != "image"
        },
        "events": events,
        "requests": records,
        "trace": trace,
        "motion_events": audit.get("motion_events", []),
        "provenance": provenance,
        "warnings": warnings,
        "inflight_at_end": stream.get("inflight_at_end", []),
    }
