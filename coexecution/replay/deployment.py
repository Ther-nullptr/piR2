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


def deployment_run(root, media):
    root = Path(root).resolve()
    config = read_json(root / "config.json")
    if config.get("protocol") != "deployment":
        raise ValueError("Robot replay requires actual deployment traces")
    execution_clock = config.get("execution_clock", "paced")
    if execution_clock not in ("paced", "ready"):
        raise ValueError("Unknown deployment execution clock")
    metadata = read_jsonl(root / "episodes.jsonl")
    run_id = "deployment-" + digest(str(root))[:20]
    episodes, identities = [], set()
    provenance = [source(root / "config.json"), source(root / "episodes.jsonl")]
    for episode in metadata:
        if episode.get("execution_clock", execution_clock) != execution_clock:
            raise ValueError("Episode and configuration execution clocks disagree")
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
            (row for row in events if row.get("kind") == "control_tick"),
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
            row["request_id"] = (
                f"{episode_id}/S1-{producer}" if producer is not None else None
            )
            previous_applied = applied
        queue = summarize_episode(events, episode)
        if queue["execution_clock"] != execution_clock:
            raise ValueError(
                "Controller trace and configuration execution clocks disagree"
            )
        for request in queue["requests"]:
            sequence = (
                request.get("request_tick")
                if request["role"] == "S1"
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
                "paired_fixture_key": digest(fixture)
                if all(value is not None for value in fixture.values())
                else None,
                "same_input_comparison": False,
                "queue": queue,
                "ticks": ticks,
                "robot": robot,
            }
        )
    return {
        "schema": "pir2.replay.run",
        "version": VERSION,
        "id": run_id,
        "kind": "deployment",
        "execution_clock": execution_clock,
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
