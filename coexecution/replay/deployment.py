"""Closed-loop evidence from the canonical controller and its own robot media."""

from pathlib import Path

from coexecution.queue_report import summarize_episode

from .schema import (
    PRECISIONS,
    VERSION,
    digest,
    interval,
    optional_number,
    read_json,
    read_jsonl,
    source,
    under_root,
)


def deployment_directories(path):
    path = Path(path).resolve()
    if (path / "config.json").is_file():
        return [path]
    if (path / "deployment" / "config.json").is_file():
        return [path / "deployment"]
    return sorted(config.parent for config in path.glob("*/deployment/config.json"))


def _precision(config):
    values = []
    for name in ("checkpoint", "slow_checkpoint"):
        precision = (
            config.get(name, {})
            .get("inference_optimization", {})
            .get("config", {})
            .get("precision")
        )
        if precision is not None:
            values.append(precision)
    if not values:
        return "unknown"
    if len(set(values)) != 1 or values[0] not in PRECISIONS:
        raise ValueError("Deployment action/VLM precision identities disagree")
    return values[0]


def _robot(root, metadata, start, ticks, media):
    record = metadata.get("replay_recording", {})
    if not record.get("path"):
        return {
            "status": "unavailable",
            "reason": "This episode did not record timestamped robot camera/state replay",
            "frames": [],
            "media": {},
        }, []
    path = under_root(root, record["path"])
    data = read_json(path)
    if data.get("schema") != "pir2.robot-recording" or data.get("version") != VERSION:
        raise ValueError("Unsupported robot recording schema")
    frames, previous_wall, previous_media = [], None, None
    by_tick = {row["tick"]: row for row in ticks}
    for index, frame in enumerate(data.get("frames", [])):
        wall, movie = (
            optional_number(frame.get("wall_s")),
            optional_number(frame.get("media_time_s")),
        )
        if (
            wall is None
            or movie is None
            or frame.get("frame_index") != index
            or frame.get("frame_id") != f"frame-{index}"
        ):
            raise ValueError("Invalid robot frame identity or timestamp")
        if previous_wall is not None:
            interval(previous_wall, wall)
            interval(previous_media, movie)
        if movie < 0:
            raise ValueError("Negative media timestamp")
        if frame.get("phase") == "post_step":
            tick = by_tick.get(frame.get("tick"))
            if tick is None or tick.get("next_observation_computed_s") != wall:
                raise ValueError(
                    "Robot frame must map to its actual post-step observation time"
                )
        elif frame.get("phase") != "initial" or index != 0:
            raise ValueError("Unknown robot frame phase")
        frames.append({**frame, "relative_s": wall - start})
        previous_wall, previous_media = wall, movie
    assets = {}
    for camera, asset in data.get("media", {}).items():
        if asset.get("path"):
            assets[camera] = {
                key: value for key, value in asset.items() if key != "path"
            } | {"media_id": media.add(root, asset["path"], mime=asset["mime"])}
        else:
            assets[camera] = asset
    return {**data, "frames": frames, "media": assets}, [source(path)]


def _stage_matches(row, expected):
    if not isinstance(row, dict) or any(
        key not in row or row[key] != value for key, value in expected.items()
    ):
        raise ValueError(
            "Synchronous stage events disagree with cycle audit or source identity"
        )


def _synchronous_stage_events(groups, cycle, inference, execution, ticks):
    """Bind every displayed stage and lineage join to the same measured cycle."""
    tick, sequence, audit = cycle["request_tick"], cycle["cycle"], cycle["audit"]

    def one(kind):
        rows = [
            row
            for row in groups[kind]
            if (
                row.get("source_tick")
                if kind in ("vision_request", "feature_published")
                else row.get("audit", {}).get("request_tick")
                if kind == "request_completed"
                else row.get("request_tick")
            )
            == tick
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Synchronous stage {kind} requires one matching source per cycle"
            )
        return rows[0]

    capture = audit["capture_s"]
    va, vb, aa, ab = (
        audit[key]
        for key in (
            "vlm_started_s",
            "vlm_completed_s",
            "action_started_s",
            "action_completed_s",
        )
    )
    _stage_matches(audit, {"source_tick": tick, "forward_counts": {"vlm": 1, "dit": 4}})
    _stage_matches(
        cycle,
        {
            "submitted_s": inference["start_s"],
            "completed_s": inference["end_s"],
            "executed_steps": len(ticks),
        },
    )
    _stage_matches(
        inference,
        {
            "request_tick": tick,
            "physics_start_tick": tick,
            "source_capture_s": capture,
            "frame_hashes": audit["frame_hashes"],
        },
    )
    _stage_matches(execution, {"request_tick": tick, "physics_start_tick": tick})
    if vb > aa or capture > va:
        raise ValueError("Synchronous stage source or VLM/DiT ordering is invalid")
    source = {"source_tick": tick, "capture_s": capture}
    dependency = {
        "feature_sequence": sequence,
        "feature_source_tick": tick,
        "feature_capture_s": capture,
        "state_tick": tick,
        "state_capture_s": capture,
    }
    _stage_matches(
        one("vision_request"),
        {
            **source,
            "t_s": vb,
            "started_s": va,
            "completed_s": vb,
            "rpc_seconds": vb - va,
            "timing_scope": "server_vlm_stage",
        },
    )
    _stage_matches(
        one("feature_published"),
        {
            **source,
            "t_s": vb,
            "rpc_completed_s": vb,
            "feature_sequence": sequence,
        },
    )
    _stage_matches(
        one("action_submitted"),
        {
            **dependency,
            "t_s": aa,
            "request_tick": tick,
            "deadline_s": None,
            "delay_ticks": 0,
            "timing_scope": "server_action_stage",
        },
    )
    _stage_matches(one("action_started"), {"t_s": aa, "request_tick": tick})
    adopted = one("action_adopted")
    accepted = adopted.get("installed_slots")
    native = cycle["native_decoded_actions"]
    if (
        type(accepted) is not int
        or not len(ticks) <= accepted <= len(native)
        or audit["decoded_horizon"] != len(native)
        or cycle["unused_plan_suffix_slots"] != len(native) - len(ticks)
    ):
        raise ValueError("Synchronous stage action counts disagree with the cycle")
    _stage_matches(
        adopted,
        {
            "t_s": cycle["completed_s"],
            "request_tick": tick,
            "expired_slots": 0,
            "protected_slots": 0,
            "ready_to_adopt_seconds": cycle["completed_s"] - ab,
        },
    )
    completed = one("action_completed")
    _stage_matches(
        completed,
        {
            **dependency,
            "t_s": ab,
            "request_tick": tick,
            "request_elapsed_s": ab - aa,
            "valid_start_tick": tick,
            "valid_end_tick": tick + accepted,
        },
    )
    consumed = one("request_completed")
    _stage_matches(
        consumed,
        {"t_s": cycle["completed_s"], "audit": {**audit, "request_tick": tick}},
    )
    for timing in (completed, consumed["client_timing"]):
        _stage_matches(
            timing,
            {
                **dependency,
                "submitted_s": aa,
                "completed_s": ab,
                "server_plan_started_s": aa,
                "timing_version": 2,
                "cache_age_at_server_plan_ms": (aa - capture) * 1000,
            },
        )
        if "synchronous_rpc_started_s" in timing:
            _stage_matches(
                timing,
                {
                    "synchronous_rpc_started_s": cycle["submitted_s"],
                    "synchronous_rpc_completed_s": cycle["completed_s"],
                    "timing_scope": "server_action_stage",
                },
            )
            if "plan_rpc_started_s" in timing:
                raise ValueError("Synchronous stage contains conflicting RPC scopes")
        else:
            # The first frozen recordings used this legacy name for the whole
            # call. Its meaning is verified here; the raw evidence stays intact.
            _stage_matches(timing, {"plan_rpc_started_s": cycle["submitted_s"]})
            if (
                timing.get("timing_scope", "server_action_stage")
                != "server_action_stage"
            ):
                raise ValueError("Synchronous stage contains an invalid timing scope")
    for offset, row in enumerate(ticks):
        _stage_matches(
            row,
            {
                "feature_sequence": sequence,
                "feature_source_tick": tick,
                "feature_capture_s": capture,
                "state_capture_s": capture,
                "action": native[offset],
            },
        )
        _stage_matches(
            row["executed_slot"],
            {"producer_kind": "request", "producer_request_tick": tick},
        )


def _synchronous_phases(events, ticks, start, stop, episode_id):
    """Accept actual mutually exclusive phases; never derive them from video FPS."""
    phases = sorted(
        (row for row in events if row.get("kind") == "cycle_phase"),
        key=lambda row: row["start_s"],
    )
    cycles = [row for row in events if row.get("kind") == "synchronous_cycle"]
    if not phases or len(phases) != 2 * len(cycles):
        raise ValueError("Synchronous episode requires two actual phases per cycle")
    stage_groups = {
        kind: [row for row in events if row.get("kind") == kind]
        for kind in (
            "vision_request",
            "feature_published",
            "action_submitted",
            "action_started",
            "action_completed",
            "request_completed",
            "action_adopted",
        )
    }
    if any(len(rows) != len(cycles) for rows in stage_groups.values()):
        raise ValueError("Synchronous stage event count must match the cycle count")
    previous, next_tick, normalized = start, 0, []
    for index, cycle in enumerate(cycles):
        inference, execution = phases[2 * index : 2 * index + 2]
        if (
            cycle.get("cycle") != index
            or inference.get("cycle") != index
            or execution.get("cycle") != index
            or inference.get("phase") != "inference"
            or execution.get("phase") != "execution"
            or inference.get("simulation_paused") is not True
            or execution.get("simulation_paused") is not False
        ):
            raise ValueError("Invalid synchronous phase order or identity")
        for phase in (inference, execution):
            a, b = interval(phase["start_s"], phase["end_s"])
            if a < previous or b > stop:
                raise ValueError(
                    "Synchronous phases overlap or escape the control window"
                )
            previous = b
            normalized.append(
                {
                    **phase,
                    "id": f"{episode_id}/cycle-{index}/{phase['phase']}",
                    "start_s": a - start,
                    "end_s": b - start,
                }
            )
        selected = [row for row in ticks if row.get("cycle") == index]
        if (
            inference.get("physics_start_tick") != next_tick
            or inference.get("physics_end_tick") != next_tick
            or execution.get("physics_start_tick") != next_tick
            or execution.get("physics_end_tick") != next_tick + len(selected)
            or not selected
            or [row["tick"] for row in selected]
            != list(range(next_tick, next_tick + len(selected)))
        ):
            raise ValueError(
                "Synchronous phase physics progression disagrees with actual ticks"
            )
        for row in selected:
            a, b = interval(row["applied_s"], row["next_observation_computed_s"])
            if a < execution["start_s"] or b > execution["end_s"]:
                raise ValueError(
                    "Physical step lies outside its synchronous execution phase"
                )
        audit = cycle["audit"]
        for a, b in (
            ("vlm_started_s", "vlm_completed_s"),
            ("action_started_s", "action_completed_s"),
        ):
            low, high = interval(audit[a], audit[b])
            if low < inference["start_s"] or high > inference["end_s"]:
                raise ValueError(
                    "Neural stage lies outside its synchronous inference phase"
                )
        _synchronous_stage_events(stage_groups, cycle, inference, execution, selected)
        next_tick += len(selected)
    if next_tick != len(ticks):
        raise ValueError(
            "Synchronous phase coverage does not include every physics step"
        )
    return normalized, cycles


def deployment_run(root, media):
    root = Path(root).resolve()
    config = read_json(root / "config.json")
    if config.get("protocol") != "deployment":
        raise ValueError("Robot replay requires actual deployment traces")
    execution_clock = config.get("execution_clock", "paced")
    if execution_clock not in ("paced", "ready"):
        raise ValueError("Unknown deployment execution clock")
    execution_coupling = config.get("execution_coupling", "asynchronous")
    if execution_coupling not in (
        "asynchronous",
        "synchronous",
        "one_step_async",
        "chunked_async",
    ):
        raise ValueError("Unknown deployment execution coupling")
    metadata = read_jsonl(root / "episodes.jsonl")
    run_id = "deployment-" + digest(str(root))[:20]
    episodes, identities = [], set()
    provenance = [source(root / "config.json"), source(root / "episodes.jsonl")]
    for episode in metadata:
        if episode.get("execution_clock", execution_clock) != execution_clock:
            raise ValueError("Episode and configuration execution clocks disagree")
        if episode.get("execution_coupling", execution_coupling) != execution_coupling:
            raise ValueError("Episode and configuration execution coupling disagree")
        task, number = episode.get("task_id"), episode.get("episode_id")
        if type(task) is not int or task not in range(10):
            raise ValueError("Deployment task_id must be an integer in 0..9")
        if type(number) is not int or number < 0:
            raise ValueError("Deployment episode_id must be a nonnegative integer")
        key = task, number
        if key in identities:
            raise ValueError("Duplicate deployment episode identity")
        identities.add(key)
        episode_id = f"{run_id}/task-{key[0]}/episode-{key[1]}"
        path = under_root(root, f"task{key[0]}-episode{key[1]}-trace.jsonl")
        if not path.is_file():
            episodes.append(
                {
                    "id": episode_id,
                    "metadata": episode,
                    "status": "unavailable",
                    "reason": "Missing actual controller trace",
                    "ticks": [],
                    "queue": None,
                    "robot": {"status": "unavailable", "frames": [], "media": {}},
                }
            )
            continue
        events = read_jsonl(path)
        for row in events:
            if optional_number(row.get("t_s")) is None:
                raise ValueError(
                    "Controller event requires a finite monotonic timestamp"
                )
        begin = [row for row in events if row.get("kind") == "control_start"]
        end = [row for row in events if row.get("kind") == "control_end"]
        provenance.append(source(path))
        if len(begin) != 1 or len(end) != 1:
            episodes.append(
                {
                    "id": episode_id,
                    "metadata": episode,
                    "status": "incomplete",
                    "reason": "Controller trace lacks one complete start/end window (possibly recorder overflow)",
                    "ticks": [],
                    "queue": None,
                    "robot": {"status": "unavailable", "frames": [], "media": {}},
                }
            )
            continue
        start, stop = interval(begin[0]["t_s"], end[0]["t_s"])
        ticks = sorted(
            (dict(row) for row in events if row.get("kind") == "control_tick"),
            key=lambda row: row["tick"],
        )
        previous_applied = start
        seen_ticks = set()
        for row in ticks:
            if row["tick"] in seen_ticks:
                raise ValueError("Duplicate physical control tick")
            seen_ticks.add(row["tick"])
            applied = row.get("applied_s", row["t_s"])
            interval(previous_applied, applied)
            if row.get("next_observation_computed_s") is not None:
                interval(applied, row["next_observation_computed_s"])
            interval(applied, stop)
            row["id"] = f"{episode_id}/tick-{row['tick']}"
            row["observation_id"] = f"{episode_id}/observation-{row['tick']}"
            row["relative_s"] = applied - start
            producer = (row.get("executed_slot") or {}).get("producer_request_tick")
            model_request_id = (
                f"{episode_id}/S1-{producer}" if producer is not None else None
            )
            if execution_coupling in ("one_step_async", "chunked_async"):
                row["model_request_id"] = model_request_id
            else:
                row["request_id"] = model_request_id
            previous_applied = applied
        queue = summarize_episode(events, episode)
        if queue["execution_clock"] != execution_clock:
            raise ValueError(
                "Controller trace and configuration execution clocks disagree"
            )
        if queue["execution_coupling"] != execution_coupling:
            raise ValueError(
                "Controller trace and configuration execution coupling disagree"
            )
        phases, cycles = [], []
        if execution_coupling == "synchronous":
            phases, cycles = _synchronous_phases(events, ticks, start, stop, episode_id)
        elif execution_coupling in ("one_step_async", "chunked_async"):
            from .async_deployment import async_replay

            phases, queue["replay_requests"] = async_replay(events, queue, episode_id)
        for request in queue["requests"]:
            sequence = (
                request.get("request_tick")
                if request["role"] == "S1"
                else request.get("feature_sequence")
                if execution_coupling in ("one_step_async", "chunked_async")
                else request.get("source_tick")
            )
            request["id"] = f"{episode_id}/{request['role']}-{sequence}"
            sequence = request.get("feature_sequence")
            request["feature_id"] = (
                f"{episode_id}/feature-{sequence}" if sequence is not None else None
            )
            observation = (
                request.get("state_tick")
                if request["role"] == "S1"
                else request.get("source_tick")
            )
            request["observation_id"] = (
                f"{episode_id}/observation-{observation}"
                if observation is not None
                else None
            )
        robot, robot_sources = _robot(root, episode, start, ticks, media)
        provenance.extend(robot_sources)
        fixture_fields = (
            "seed",
            "task_id",
            "episode_id",
            "initial_state_sha256",
            "fixture_model_sha256",
            "settled_sim_state_sha256",
            "initial_rgb_sha256",
        )
        fixture = {name: episode.get(name) for name in fixture_fields}
        episodes.append(
            {
                "id": episode_id,
                "task_id": key[0],
                "episode_id": key[1],
                "seed": episode.get("seed"),
                "metadata": episode,
                "status": "recorded",
                "controlled_start_s": start,
                "controlled_end_s": stop,
                "work_end_s": queue.get("work_end_s", stop),
                "paired_fixture_key": digest(fixture)
                if all(value is not None for value in fixture.values())
                else None,
                "same_input_comparison": False,
                "queue": queue,
                "ticks": ticks,
                "phases": phases,
                "cycles": cycles,
                "robot": robot,
            }
        )
    schedule = _deployment_schedule(config, episodes, execution_coupling)
    return {
        "schema": "pir2.replay.run",
        "version": VERSION,
        "id": run_id,
        "kind": "deployment",
        "execution_clock": execution_clock,
        "execution_coupling": execution_coupling,
        **schedule,
        "label": root.parent.name,
        "precision": _precision(config),
        "mode": root.parent.name,
        "source_root": str(root),
        "same_input_comparison": False,
        "configuration": config,
        "task_ids": config.get("task_ids"),
        "episodes": episodes,
        "provenance": provenance,
        "warnings": [
            "Closed-loop trajectories diverge after paired initial conditions; robot frames and actions belong only to their own precision run"
        ],
        "timing_note": "Actual host/controller/RPC timestamps; these lanes are not GPU kernel activity. Debug camera recording can affect control timing.",
    }


def _deployment_schedule(config, episodes, coupling):
    """Declare only recorded K/D, and reject configuration/evidence disagreement."""
    keys = ("execution_prefix_steps", "async_overlap_steps")
    if coupling == "asynchronous":
        return dict.fromkeys(keys)
    fixed = (
        (1, 1)
        if coupling == "one_step_async"
        else (None, 0 if coupling == "synchronous" else None)
    )
    schedule = []
    for index, key in enumerate(keys):
        configured = config.get(key)
        if key == "execution_prefix_steps":
            alias = config.get("synchronous_chunk_steps")
            if configured is not None and alias is not None and configured != alias:
                raise ValueError("Conflicting configured execution prefixes")
            configured = configured if configured is not None else alias
        values = [
            e["queue"][key]
            for e in episodes
            if e.get("queue") is not None and e["queue"].get(key) is not None
        ]
        values += [v for v in (configured, fixed[index]) if v is not None]
        if any(type(v) is not int or v < (1 if index == 0 else 0) for v in values):
            raise ValueError("Invalid recorded execution schedule")
        if values and any(v != values[0] for v in values):
            raise ValueError("Deployment schedule disagrees with recorded evidence")
        schedule.append(values[0] if values else None)
    k, d = schedule
    if coupling == "chunked_async" and (k is None or d is None or not 1 <= d <= k):
        raise ValueError("Chunked async requires a measured valid K/D schedule")
    return dict(zip(keys, schedule))
