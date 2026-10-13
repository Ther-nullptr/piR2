"""Replay evidence contracts: host metadata only, never CPU neural inference."""

import importlib
import json
import subprocess
import sys

import pytest


def test_import_and_cli_do_not_load_neural_or_array_frameworks():
    code = "import sys; import coexecution.replay; import coexecution.replay.catalog; assert not {'torch', 'numpy', 'av'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_schema_rejects_nonfinite_and_reversed_times_but_preserves_missing():
    schema = importlib.import_module("coexecution.replay.schema")
    assert schema.optional_number(None) is None
    assert schema.interval(4.0, 4.2) == pytest.approx((4.0, 4.2))
    for value in [float("nan"), float("inf"), True, "4"]:
        with pytest.raises(ValueError):
            schema.optional_number(value)
    with pytest.raises(ValueError, match="order"):
        schema.interval(4, 3)


def test_same_input_key_requires_hash_seed_config_and_clock_and_never_closed_loop():
    schema = importlib.import_module("coexecution.replay.schema")
    values = {
        "kind": "model",
        "observation_sha256": "a" * 64,
        "seed": 7,
        "shared_config_sha256": "b" * 64,
        "clock_cohort": "gpu3-session1",
    }
    key = schema.comparison_key(**values)
    assert key and key == schema.comparison_key(**values)
    for field in ("observation_sha256", "seed", "shared_config_sha256", "clock_cohort"):
        assert schema.comparison_key(**(values | {field: None})) is None
    assert schema.comparison_key(**(values | {"kind": "deployment"})) is None
    assert schema.comparison_key(**(values | {"seed": 8})) != key


def test_bounded_recording_owns_frames_records_post_step_times_and_overflow():
    import numpy as np

    recorder = importlib.import_module("coexecution.replay.recording").ReplayRecorder(
        max_frames=2, max_bytes=100000
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    def observation(x):
        return {
            "state.x": [x],
            "state.y": [2],
            "state.z": [3],
            "state.roll": [0.1],
            "state.pitch": [0.2],
            "state.yaw": [0.3],
            "state.gripper": [0.04, -0.04],
            "video.image": image,
            "video.wrist_image": image,
        }

    recorder.initial(observation(1), capture_s=9.0, computed_s=None)
    before = observation(1)
    after = observation(2)
    tick = {
        "tick": 0,
        "sensor_capture_s": 10.0,
        "sensor_release_s": 10.0,
        "applied_s": 10.002,
        "next_observation_computed_s": 10.03,
        "action": [0] * 7,
        "executed_slot": {"producer_kind": "bootstrap"},
    }
    state = recorder.step(before, after, tick)
    image[:] = 255
    assert state["before"]["position_xyz_m"] == [1.0, 2.0, 3.0]
    assert state["after"]["rotation_rotvec_rad"] == [0.1, 0.2, 0.3]
    assert recorder.frames["image"][0].max() == 0
    assert recorder.mapping[1]["wall_s"] == 10.03
    assert recorder.mapping[1]["phase"] == "post_step"
    assert recorder.mapping[1]["tick"] == 0
    assert recorder.mapping[0]["wall_s"] == 9.0
    assert (
        recorder.step(
            after,
            after,
            tick
            | {"tick": 1, "applied_s": 10.05, "next_observation_computed_s": 10.08},
        )
        is None
    )
    status = recorder.status()
    assert status["status"] == "overflow" and status["dropped_frames"] == 1
    assert len(recorder.mapping) == 2
    assert status["rotation_representation"] == "axis_angle_rotation_vector_rad"


def test_recorder_encodes_after_control_and_uses_separate_camera_outputs(tmp_path):
    import numpy as np

    recorder = importlib.import_module("coexecution.replay.recording").ReplayRecorder(
        max_frames=2
    )
    observation = {
        "video.image": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
    }
    recorder.initial(observation, capture_s=2.0)
    calls = []

    def encode(path, frames, **options):
        calls.append((path.name, int(frames[0].max()), options))
        path.write_bytes(b"video")

    result = recorder.finish(tmp_path, "task0-episode0", encoder=encode)
    assert len(calls) == 2
    assert {value for _, value, _ in calls} == {0, 1}
    assert all(options == {"codec": "libvpx-vp9", "fps": 20} for _, _, options in calls)
    data = json.loads((tmp_path / result["path"]).read_text())
    assert data["frames"][0]["media_time_s"] == 0
    assert data["status"] == "recorded"
    assert data["media"]["image"]["path"].endswith(".webm")


def test_profile_device_timeline_uses_correlation_not_gpu_host_overlap():
    module = importlib.import_module("coexecution.libero.profiling")
    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": module.REQUEST_PREFIX + "profile-r0-o0",
            "ts": 100,
            "dur": 1000,
            "pid": 1,
            "tid": 1,
        },
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": module.STAGE_PREFIX + "vlm",
            "ts": 110,
            "dur": 800,
            "pid": 1,
            "tid": 1,
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaGraphLaunch",
            "ts": 200,
            "dur": 10,
            "pid": 1,
            "tid": 1,
            "args": {"correlation": 8},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm",
            "ts": 1500,
            "dur": 400,
            "args": {"correlation": 8},
        },
        {"ph": "X", "cat": "kernel", "name": "unknown", "ts": 300, "dur": 50},
    ]
    result = module.classify_trace({"traceEvents": events})
    linked = next(row for row in result["device_events"] if row["kernel"] == "gemm")
    assert linked["request_id"] == "profile-r0-o0" and linked["stage"] == "vlm"
    assert linked["ts_us"] == 1500 and linked["duration_us"] == 400
    assert (
        next(row for row in result["device_events"] if row["kernel"] == "unknown")[
            "request_id"
        ]
        is None
    )


def model_fixture(tmp_path):
    import numpy as np

    root = tmp_path / "model"
    root.mkdir()
    results = {
        "checkpoint": {"sha256": {"weights": "abc"}},
        "device": {"name": "Test GPU", "torch": "test", "cuda": "test"},
        "runtime_sources": {"engine": "hash"},
        "arguments": {
            "measurement_boundary": "model-only",
            "torch_threads": 2,
            "operator_fusion": True,
            "dit_cuda_graph": True,
            "action_cuda_graph": False,
            "vision_channels_last": False,
            "cuda_events": False,
        },
        "observations": [{"observation_id": "obs-a", "sha256": "a" * 64}],
        "profiles": {},
        "validation": [{"mode": "bf16-compiled", "prepared_model": {"passes": True}}],
        "graph_timing_audit": [],
        "boundaries": {"prepared_model": "full model"},
    }
    rows, actions = [], {}
    for precision, value in (("bf16", 1), ("w8a8", 2)):
        request = f"r0/{precision}-compiled/o0/prepared_model"
        rows.append(
            {
                "request_id": request,
                "precision": precision,
                "mode": precision + "-compiled",
                "scope": "prepared_model",
                "observation_id": "obs-a",
                "seed": 17,
                "round": 0,
                "wall_ms": 12,
                "instrumented": False,
            }
        )
        actions[request + "/action.x"] = np.array(
            [[[value], [value + 1]]], dtype=np.float32
        )
    (root / "results.json").write_text(json.dumps(results))
    (root / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)))
    np.savez(root / "actions.npz", **actions)
    return root, results


def test_catalog_keeps_clean_outputs_separate_from_unmatched_profile_seeds(tmp_path):
    root, result = model_fixture(tmp_path)
    profile_dir = root / "profiles"
    profile_dir.mkdir()
    from coexecution.libero.profiling import REQUEST_PREFIX, STAGE_PREFIX

    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": REQUEST_PREFIX + "profile-r0-o0",
            "ts": 100,
            "dur": 1000,
            "pid": 1,
            "tid": 1,
        },
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": REQUEST_PREFIX + "profile-r0-o0",
            "ts": 140,
            "dur": 700,
            "pid": 2,
            "tid": 2,
        },
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": STAGE_PREFIX + "vlm",
            "ts": 110,
            "dur": 800,
            "pid": 1,
            "tid": 1,
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaGraphLaunch",
            "ts": 200,
            "dur": 10,
            "pid": 1,
            "tid": 1,
            "args": {"correlation": 8},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm",
            "ts": 300,
            "dur": 400,
            "args": {"correlation": 8},
        },
    ]
    path = profile_dir / "bf16-compiled.trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    result["profiles"]["bf16-compiled"] = {
        "trace": str(path),
        "requests": [
            {"request_id": "profile-r0-o0", "observation_id": "obs-a", "seed": 127}
        ],
        "instrumented": True,
    }
    (root / "results.json").write_text(json.dumps(result))
    (root / "validation.json").write_text("[]")
    module = importlib.import_module("coexecution.replay.catalog")
    output = tmp_path / "catalog"
    catalog = module.build_catalog(
        model_runs=[root], deployments=[], output=output, clock_cohort="audited-cohort"
    )
    assert catalog["schema"] == "pir2.replay.catalog" and catalog["version"] == 1
    runs = [json.loads((output / row["path"]).read_text()) for row in catalog["runs"]]
    bf16 = next(run for run in runs if run["precision"] == "bf16")
    lowbit = next(run for run in runs if run["precision"] == "w8a8")
    clean = bf16["clean_requests"][0]
    profile = bf16["profile_requests"][0]
    assert clean["output"]["actions"]["action.x"]["values"] == [[[1.0], [2.0]]]
    assert "timeline" not in clean
    assert profile["output"]["status"] == "unavailable" and profile["seed"] == 127
    assert profile["comparison_key"] is not None
    assert profile["comparison_key"] != clean["comparison_key"]
    assert profile["host_wall_ms"] == 1
    assert profile["timeline"][0]["start_ms"] == pytest.approx(0.2)
    assert profile["timeline"][0]["duration_ms"] == pytest.approx(0.4)
    assert lowbit["clean_requests"][0]["difference_from_bf16"]["max_abs"] == 1
    assert any("validation.json" in issue for issue in bf16["warnings"])


def test_no_clock_cohort_disables_paired_output_claim_and_duplicate_ids_rejected(
    tmp_path,
):
    root, _ = model_fixture(tmp_path)
    module = importlib.import_module("coexecution.replay.catalog")
    output = tmp_path / "catalog"
    catalog = module.build_catalog(model_runs=[root], deployments=[], output=output)
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    assert run["clean_requests"][0]["comparison_key"] is None
    rows = (root / "samples.jsonl").read_text()
    (root / "samples.jsonl").write_text(rows + "\n" + rows.splitlines()[0])
    with pytest.raises(ValueError, match="Duplicate"):
        module.build_catalog(model_runs=[root], deployments=[], output=tmp_path / "bad")


def test_catalog_rejects_precision_conflict(tmp_path):
    root, _ = model_fixture(tmp_path)
    rows = [
        json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()
    ]
    rows[0]["precision"] = "w4a4"
    (root / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="precision"):
        importlib.import_module("coexecution.replay.catalog").build_catalog(
            model_runs=[root], deployments=[], output=tmp_path / "bad"
        )


def test_asset_export_is_optional_and_records_actual_two_cameras_and_state(tmp_path):
    import numpy as np

    module = importlib.import_module("coexecution.replay.assets")
    obs = {
        "video": {
            "image": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
            "wrist_image": np.ones((1, 1, 4, 4, 3), dtype=np.uint8),
        },
        "state": {"x": np.array([[[0.12]]], dtype=np.float32)},
    }
    written = []

    def save(path, value):
        written.append((path, int(value.max())))
        path.write_bytes(b"png")

    records = module.export_observations(
        tmp_path,
        [obs],
        [{"observation_id": "obs-a", "sha256": "sha"}],
        image_writer=save,
    )
    assert [value for _, value in written] == [0, 1]
    assert records[0]["state"]["x"][0][0][0] == pytest.approx(0.12)
    assert {asset["camera"] for asset in records[0]["assets"]} == {
        "image",
        "wrist_image",
    }
    assert all((tmp_path / asset["path"]).is_file() for asset in records[0]["assets"])
    parser = importlib.import_module("coexecution.libero.measurement").build_parser()
    args = parser.parse_args(
        ["--checkpoint", "x", "--dataset", "y", "--output", "z", "--replay-assets"]
    )
    assert args.replay_assets


def deployment_fixture(tmp_path):
    root = tmp_path / "bf16-debug" / "deployment"
    root.mkdir(parents=True)
    config = {
        "protocol": "deployment",
        "checkpoint": {"inference_optimization": {"config": {"precision": "bf16"}}},
        "task_ids": [0],
    }
    (root / "config.json").write_text(json.dumps(config))
    metadata = {
        "task_id": 0,
        "episode_id": 0,
        "seed": 1000,
        "success": False,
        "control_rate_valid": True,
    }
    (root / "episodes.jsonl").write_text(json.dumps(metadata) + "\n")
    events = [
        {"kind": "control_start", "t_s": 10, "period_s": 0.05},
        {
            "kind": "control_tick",
            "t_s": 10.002,
            "tick": 0,
            "applied_s": 10.002,
            "scheduled_s": 10,
            "sensor_capture_s": 10,
            "sensor_release_s": 10,
            "next_observation_computed_s": 10.03,
            "action": [0.1] * 7,
            "robot_response": {
                "before": {"position_xyz_m": [0, 0, 0]},
                "after": {"position_xyz_m": [0.1, 0, 0]},
                "after_frame_id": "frame-1",
            },
        },
        {"kind": "control_end", "t_s": 10.05, "control_steps": 1},
    ]
    (root / "task0-episode0-trace.jsonl").write_text("\n".join(map(json.dumps, events)))
    return root, metadata


def test_deployment_is_divergent_closed_loop_with_own_frame_timestamps(tmp_path):
    root, metadata = deployment_fixture(tmp_path)
    (root / "own.webm").write_bytes(b"webm")
    recording = {
        "schema": "pir2.robot-recording",
        "version": 1,
        "status": "recorded",
        "frames": [
            {
                "frame_id": "frame-0",
                "frame_index": 0,
                "tick": None,
                "phase": "initial",
                "wall_s": 9.8,
                "media_time_s": 0,
            },
            {
                "frame_id": "frame-1",
                "frame_index": 1,
                "tick": 0,
                "phase": "post_step",
                "wall_s": 10.03,
                "media_time_s": 0.05,
            },
        ],
        "media": {"image": {"path": "own.webm", "mime": "video/webm"}},
    }
    (root / "task0-episode0-replay.json").write_text(json.dumps(recording))
    metadata["replay_recording"] = {"path": "task0-episode0-replay.json"}
    (root / "episodes.jsonl").write_text(json.dumps(metadata) + "\n")
    output = tmp_path / "built"
    module = importlib.import_module("coexecution.replay.catalog")
    catalog = module.build_catalog(
        model_runs=[], deployments=[root], output=output, clock_cohort="same"
    )
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    episode = run["episodes"][0]
    assert run["same_input_comparison"] is False
    assert episode["robot"]["frames"][0]["relative_s"] == pytest.approx(-0.2)
    assert episode["robot"]["frames"][1]["relative_s"] == pytest.approx(0.03)
    assert episode["robot"]["frames"][1]["media_time_s"] == 0.05
    assert episode["ticks"][0]["action"] == [0.1] * 7
    assert episode["ticks"][0]["robot_response"]["after"]["position_xyz_m"] == [
        0.1,
        0,
        0,
    ]
    media = json.loads((output / "media.json").read_text())
    assert (output / media["assets"][0]["path"]).read_bytes() == b"webm"


def test_old_robot_evidence_is_explicitly_unavailable_and_invalid_time_rejected(
    tmp_path,
):
    root, _ = deployment_fixture(tmp_path)
    output = tmp_path / "built"
    module = importlib.import_module("coexecution.replay.catalog")
    catalog = module.build_catalog(model_runs=[], deployments=[root], output=output)
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    assert run["episodes"][0]["robot"]["status"] == "unavailable"
    path = root / "task0-episode0-trace.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1]["t_s"] = 9
    path.write_text("\n".join(map(json.dumps, events)))
    with pytest.raises(ValueError, match="order"):
        module.build_catalog(
            model_runs=[], deployments=[root], output=tmp_path / "invalid"
        )


def test_media_path_cannot_escape_explicit_input_root(tmp_path):
    root, metadata = deployment_fixture(tmp_path)
    metadata["replay_recording"] = {"path": "../../../secret.json"}
    (root / "episodes.jsonl").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="root"):
        importlib.import_module("coexecution.replay.catalog").build_catalog(
            model_runs=[], deployments=[root], output=tmp_path / "invalid"
        )


def test_recorder_reports_codec_failure_and_byte_overflow(tmp_path):
    import numpy as np

    recorder_type = importlib.import_module(
        "coexecution.replay.recording"
    ).ReplayRecorder
    observation = {"video.image": np.ones((4, 4, 3), dtype=np.uint8)}
    bounded = recorder_type(max_frames=10, max_bytes=1)
    bounded.initial(observation, capture_s=1)
    assert bounded.status()["status"] == "overflow"
    assert not bounded.frames
    recorder = recorder_type()
    recorder.initial(observation, capture_s=1)

    def broken(*args, **kwargs):
        raise OSError("codec unavailable")

    result = recorder.finish(tmp_path, "episode", encoder=broken)
    assert result["status"] == "failed"
    assert "codec unavailable" in result["error"]
    stored = json.loads((tmp_path / result["path"]).read_text())
    assert stored["media"]["image"]["status"] == "failed"


def test_catalog_rejects_frame_tick_guessing_and_cross_precision_deployment(tmp_path):
    root, metadata = deployment_fixture(tmp_path)
    recording = {
        "schema": "pir2.robot-recording",
        "version": 1,
        "status": "recorded",
        "frames": [
            {
                "frame_id": "frame-0",
                "frame_index": 0,
                "tick": 0,
                "phase": "post_step",
                "wall_s": 10.0,
                "media_time_s": 0,
            }
        ],
        "media": {},
    }
    (root / "recording.json").write_text(json.dumps(recording))
    metadata["replay_recording"] = {"path": "recording.json"}
    (root / "episodes.jsonl").write_text(json.dumps(metadata) + "\n")
    module = importlib.import_module("coexecution.replay.catalog")
    with pytest.raises(ValueError, match="post-step"):
        module.build_catalog(
            model_runs=[], deployments=[root], output=tmp_path / "badframe"
        )
    metadata.pop("replay_recording")
    (root / "episodes.jsonl").write_text(json.dumps(metadata) + "\n")
    config = json.loads((root / "config.json").read_text())
    config["slow_checkpoint"] = {
        "inference_optimization": {"config": {"precision": "w8a8"}}
    }
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="precision"):
        module.build_catalog(
            model_runs=[], deployments=[root], output=tmp_path / "badprecision"
        )


def test_nonfinite_exponent_timestamp_is_rejected(tmp_path):
    root, _ = deployment_fixture(tmp_path)
    path = root / "task0-episode0-trace.jsonl"
    path.write_text(path.read_text().replace('"t_s": 10,', '"t_s": 1e309,'))
    with pytest.raises(ValueError, match="finite"):
        importlib.import_module("coexecution.replay.catalog").build_catalog(
            model_runs=[], deployments=[root], output=tmp_path / "bad"
        )


def test_missing_shared_runtime_metadata_disables_same_input_claim(tmp_path):
    root, result = model_fixture(tmp_path)
    result["arguments"].pop("torch_threads")
    (root / "results.json").write_text(json.dumps(result))
    output = tmp_path / "built"
    catalog = importlib.import_module("coexecution.replay.catalog").build_catalog(
        model_runs=[root], deployments=[], output=output, clock_cohort="audited"
    )
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    assert run["same_input_comparison"] is False
    assert run["clean_requests"][0]["comparison_key"] is None


def test_second_camera_copy_failure_keeps_only_prior_complete_frames(
    monkeypatch, tmp_path
):
    import numpy as np

    recorder = importlib.import_module("coexecution.replay.recording").ReplayRecorder()
    observation = {
        "video.image": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
    }
    recorder.initial(observation, capture_s=1.0)
    original_array = np.array
    copies = []

    def copy(value, *args, **kwargs):
        copies.append(value)
        if len(copies) == 2:
            raise MemoryError("second camera allocation")
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(np, "array", copy)
    tick = {
        "tick": 0,
        "sensor_capture_s": 1.1,
        "sensor_release_s": 1.1,
        "applied_s": 1.1,
        "next_observation_computed_s": 1.13,
    }
    assert recorder.step(observation, observation, tick) is None
    assert recorder.status()["status"] == "failed"
    assert "MemoryError" in recorder.status()["error"]
    assert len(recorder.mapping) == 1
    assert {len(frames) for frames in recorder.frames.values()} == {1}
    assert recorder.used_bytes == 96
    assert (
        recorder.step(
            observation,
            observation,
            tick | {"tick": 1, "applied_s": 1.15, "next_observation_computed_s": 1.18},
        )
        is None
    )
    assert len(copies) == 2
    encoded = []

    def encode(path, frames, **kwargs):
        encoded.append(len(frames))
        path.write_bytes(b"video")

    recorder.finish(tmp_path, "episode", encoder=encode)
    assert encoded == [1, 1]
    assert (
        json.loads((tmp_path / "episode-replay.json").read_text())["status"] == "failed"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "0/../../../outside"),
        ("task_id", True),
        ("task_id", 10),
        ("episode_id", -1),
        ("episode_id", False),
        ("episode_id", "0"),
    ],
)
def test_deployment_fixture_ids_are_strict_integers(tmp_path, field, value):
    root, metadata = deployment_fixture(tmp_path)
    metadata[field] = value
    (root / "episodes.jsonl").write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValueError, match="task|episode"):
        importlib.import_module("coexecution.replay.catalog").build_catalog(
            model_runs=[], deployments=[root], output=tmp_path / "bad"
        )


def test_deployment_trace_symlink_cannot_escape_input_root(tmp_path):
    root, _ = deployment_fixture(tmp_path)
    trace = root / "task0-episode0-trace.jsonl"
    outside = tmp_path / "outside.jsonl"
    trace.rename(outside)
    trace.symlink_to(outside)
    with pytest.raises(ValueError, match="root"):
        importlib.import_module("coexecution.replay.catalog").build_catalog(
            model_runs=[], deployments=[root], output=tmp_path / "bad"
        )


@pytest.mark.parametrize("failure", ["rng", "graph", "nonfinite"])
def test_failed_model_sidecars_remain_visible_without_completed_result(
    tmp_path, failure
):
    import numpy as np

    root = tmp_path / "failed-model"
    root.mkdir()
    mode = "w4a4-compiled"
    (root / "observations.json").write_text(
        json.dumps([{"observation_id": "obs-a", "sha256": "a" * 64}])
    )
    gate = {
        "mode": mode,
        "round": 0,
        "observation": 0,
        "prepared_model": {"passes": failure != "nonfinite"},
        "rng_equal": failure != "rng",
    }
    (root / "validation.json").write_text(
        json.dumps([gate] if failure != "graph" else [])
    )
    (root / "graph-timing-audit.json").write_text(
        json.dumps(
            [
                {
                    "mode": mode,
                    "round": 0,
                    "observation": 0,
                    "prepared_model": {"passes": failure != "graph"},
                }
            ]
        )
    )
    if failure != "graph":
        np.savez(
            root / "actions.npz",
            **{
                f"r0/{mode}/o0/prepared_model/action": np.array(
                    [float("nan") if failure == "nonfinite" else 1.0]
                )
            },
        )
    output = tmp_path / "built"
    catalog = importlib.import_module("coexecution.replay.catalog").build_catalog(
        model_runs=[root], deployments=[], output=output, clock_cohort="declared"
    )
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    assert run["status"] == "failed"
    assert run["clean_requests"] == [] and run["same_input_comparison"] is False
    assert run["configuration"] == {} and run["checkpoint"] is None
    assert (
        run["validation"]["graph_status" if failure == "graph" else "status"]
        == "failed"
    )
    if failure == "graph":
        assert run["retained_outputs"] == []
    else:
        recorded = run["retained_outputs"][0]
        assert recorded["wall_ms"] is None and recorded["seed"] is None
        assert recorded["output"]["status"] == (
            "invalid" if failure == "nonfinite" else "recorded"
        )
        if failure == "nonfinite":
            assert recorded["output"]["actions"]["action"]["values"] is None
            assert recorded["output"]["actions"]["action"]["nonfinite_count"] == 1
