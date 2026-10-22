"""Dynamic replay keeps physical simulation and measured model clocks separate."""

import json
from copy import deepcopy

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "evaluation"
    root.mkdir()
    item = "libero_spatial_task00__l1__traj00"
    action = [0.1, 0, 0, 0, 0, 0, -1]
    record = {
        "item_id": item,
        "request_sequence": 8,
        "phase": "scored",
        "status": "complete",
        "source_tick": 0,
        "source_tick_kind": "action_step",
        "sim_time_s": 0.0,
        "request_started_s": 100.0,
        "request_completed_s": 100.4,
        "rpc_started_s": 100.01,
        "rpc_completed_s": 100.39,
        "execution_prefix_steps": 8,
        "predicted_horizon": 40,
        "decoded_horizon": 16,
        "actions": [action] * 8,
        "server_audit": {
            "source_tick": 0,
            "vlm_started_s": 100.02,
            "vlm_completed_s": 100.12,
            "action_started_s": 100.12,
            "action_completed_s": 100.38,
            "forward_counts": {"vlm": 1, "dit": 4},
            "timing_scope": "stream-complete stages",
        },
    }
    audit = {
        "schema": "pir2.libero_dynamic.policy_audit.v1",
        "item_id": item,
        "records": [dict(record, request_sequence=7, phase="warmup"), record],
        "chunks": [
            {
                "chunk_id": 1,
                "obs_step_idx": 0,
                "metadata": {"policy_request_sequence": 8},
            }
        ],
        "motion_events": [
            {
                "kind": "hold_started",
                "rollout_sim_time_s": 0.0,
                "active_chunk_id": 1,
                "action_step": 0,
            },
            {
                "kind": "hold_completed",
                "rollout_sim_time_s": 0.1,
                "active_chunk_id": 1,
                "action_step": 0,
            },
            {
                "kind": "action_completed",
                "rollout_sim_time_s": 0.15,
                "active_chunk_id": 1,
                "action_step": 0,
                "action": action,
            },
        ],
        "video_samples": [
            {
                "frame_index": 0,
                "media_time_s": 0.0,
                "sample_sim_time_s": 0.02,
                "phase": "inference_hold",
                "active_chunk_id": 1,
                "action_step": 0,
            },
            {
                "frame_index": 1,
                "media_time_s": 1 / 60,
                "sample_sim_time_s": 0.04,
                "phase": "inference_hold",
                "active_chunk_id": 1,
                "action_step": 0,
            },
            {
                "frame_index": 2,
                "media_time_s": 2 / 60,
                "sample_sim_time_s": 0.15,
                "phase": "action",
                "active_chunk_id": 1,
                "action_step": 0,
            },
        ],
    }
    manifest = {
        "item_id": item,
        "condition_id": item + "__s006",
        "suite": "libero_spatial",
        "task_id": 0,
        "level": "l1",
        "traj_index": 0,
        "speed_mps": 0.06,
        "precision": "bf16",
        "protocol": "default_vla_flow4",
        "status": "ok",
        "success": True,
        "language": "pick up the bowl",
        "steps": 1,
        "control_freq": 20,
        "policy_rollout_sim_time_s": 0.15,
        "applied_latency_s": 0.1,
        "raw_calibration_latency_s": 0.1,
        "calibration_kind": "counterfactual",
        "calibration_retained_requests": 0,
        "timing_rule": "release_continuous_sync",
        "replan_steps_requested": 8,
        "policy_identity": {"predicted_horizon": 40, "decoded_horizon": 16},
        "policy_audit": "audit.json",
        "step_trace": "trace.jsonl",
        "model_input_video": None,
        "wrist_input_video": None,
        "video_frame_count": 3,
        "video_fps": 60,
    }
    write(
        root / "evaluation.json",
        {"schema": "pir2.libero_dynamic.default.v1", "configuration": {"seed": 0}},
    )
    write(root / "calibration.json", {"kind": "counterfactual", "retained_requests": 0})
    write(root / "audit.json", audit)
    write(
        root / "trace.jsonl",
        {"step_idx": 0, "sim_time_s": 0.15, "chunk_id": 1, "executed_action": action},
    )
    write(root / "manifest.jsonl", manifest)
    return root, manifest, audit


def build(root, tmp_path):
    from coexecution.replay.catalog import build_catalog

    output = tmp_path / "catalog"
    catalog = build_catalog(
        model_runs=[], deployments=[], dynamic_evaluations=[root], output=output
    )
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    return catalog, run, output


def test_dynamic_catalog_uses_recorded_sim_phases_and_independent_wall_stages(
    evidence, tmp_path
):
    root, _, _ = evidence
    catalog, run, _ = build(root, tmp_path)
    assert catalog["runs"][0]["kind"] == "dynamic"
    episode = run["episodes"][0]
    assert episode["calibration"]["kind"] == "counterfactual"
    assert episode["calibration"]["retained_requests"] == 0
    assert [
        (event["kind"], event["start_s"], event["end_s"]) for event in episode["events"]
    ] == [("inference_hold", 0, 0.1), ("action", pytest.approx(0.1), 0.15)]
    request = episode["requests"][0]
    assert request["request_sequence"] == 8
    assert request["wall_stages"][0]["duration_ms"] == pytest.approx(100)
    assert request["wall_stages"][1]["duration_ms"] == pytest.approx(260)
    assert all(event["request_id"] == request["id"] for event in episode["events"])
    assert episode["frames"][0]["relative_s"] == 0.02
    assert episode["frames"][0]["media_time_s"] == 0
    assert episode["frames"][0]["phase"] == "inference_hold"
    assert any(
        s["path"].endswith("audit.json") and len(s["sha256"]) == 64
        for s in episode["provenance"]
    )
    assert episode["media"] == {}


@pytest.mark.parametrize("field", ["policy_audit", "model_input_video", "step_trace"])
def test_dynamic_paths_cannot_escape_evaluation_root(evidence, tmp_path, field):
    root, manifest, _ = evidence
    manifest[field] = "../outside.json"
    write(root / "manifest.jsonl", manifest)
    with pytest.raises(ValueError, match="root"):
        build(root, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["video_samples"][1].update(sample_sim_time_s=0.01),
        lambda a: a["video_samples"][1].update(media_time_s=0),
        lambda a: a["video_samples"][1].update(frame_index=0),
        lambda a: a["chunks"][0]["metadata"].update(policy_request_sequence=7),
        lambda a: a["records"][1].update(item_id="another_episode"),
        lambda a: a["records"][1].update(source_tick=1),
        lambda a: a["motion_events"][2].update(active_chunk_id=99),
        lambda a: a["motion_events"][2].update(action=[0] * 7),
    ],
)
def test_dynamic_rejects_untrustworthy_frame_or_action_lineage(
    evidence, tmp_path, mutation
):
    root, _, audit = evidence
    mutation(audit)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError):
        build(root, tmp_path)


def test_missing_audit_and_failure_remain_unavailable(evidence, tmp_path):
    root, manifest, _ = evidence
    manifest.update(status="error", error="RPC timeout", success=None, steps=0)
    (root / "audit.json").unlink()
    manifest.update(step_trace=None, model_input_video=None, video_frame_count=0)
    write(root / "manifest.jsonl", manifest)
    _, run, _ = build(root, tmp_path)
    episode = run["episodes"][0]
    assert episode["success"] is None
    assert episode["error"] == "RPC timeout"
    assert episode["requests"] == []
    assert episode["warnings"]
    assert episode["media"] == {}


def test_missing_wall_stage_stays_unknown(evidence, tmp_path):
    root, _, audit = evidence
    del audit["records"][1]["server_audit"]["vlm_completed_s"]
    write(root / "audit.json", audit)
    _, run, _ = build(root, tmp_path)
    request = run["episodes"][0]["requests"][0]
    assert request["wall_stages"][0]["duration_ms"] is None


def test_episode_request_ids_do_not_collide(evidence, tmp_path):
    root, manifest, audit = evidence
    second = deepcopy(manifest)
    second.update(
        item_id="another",
        condition_id="another_condition",
        policy_audit="another.json",
        step_trace=None,
        model_input_video=None,
        video_frame_count=0,
    )
    audit = deepcopy(audit)
    audit["item_id"] = "another"
    for record in audit["records"]:
        record["item_id"] = "another"
    audit["video_samples"] = []
    write(root / "another.json", audit)
    (root / "manifest.jsonl").write_text(
        json.dumps(manifest) + "\n" + json.dumps(second)
    )
    _, run, _ = build(root, tmp_path)
    a, b = run["episodes"]
    assert a["requests"][0]["id"] != b["requests"][0]["id"]


def test_dynamic_rejects_reused_request_chunk(evidence, tmp_path):
    root, _, audit = evidence
    audit["chunks"].append({**deepcopy(audit["chunks"][0]), "chunk_id": 2})
    write(root / "audit.json", audit)
    with pytest.raises(ValueError, match="request"):
        build(root, tmp_path)


def test_dynamic_rejects_manifest_action_count_mismatch(evidence, tmp_path):
    root, manifest, _ = evidence
    manifest["steps"] = 2
    write(root / "manifest.jsonl", manifest)
    with pytest.raises(ValueError, match="action count"):
        build(root, tmp_path)


def test_dynamic_wall_stages_cannot_disagree_with_observation(evidence, tmp_path):
    root, _, audit = evidence
    audit["records"][1]["server_audit"]["source_tick"] = 4
    write(root / "audit.json", audit)
    with pytest.raises(ValueError, match="source"):
        build(root, tmp_path)


@pytest.mark.parametrize(
    "change", [{"active_chunk_id": 999999}, {"phase": "action"}, {"action_step": 599}]
)
def test_dynamic_frames_require_own_physical_phase_and_action_source(
    evidence, tmp_path, change
):
    root, _, audit = evidence
    audit["video_samples"][0].update(change)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError, match="frame"):
        build(root, tmp_path)


def test_complete_episode_cannot_end_with_open_hold(evidence, tmp_path):
    root, _, audit = evidence
    audit["motion_events"].append(
        {
            "kind": "hold_started",
            "rollout_sim_time_s": 0.15,
            "active_chunk_id": 1,
            "action_step": 1,
        }
    )
    write(root / "audit.json", audit)
    with pytest.raises(ValueError, match="incomplete"):
        build(root, tmp_path)


def test_complete_episode_duration_matches_recorded_physics(evidence, tmp_path):
    root, manifest, _ = evidence
    manifest["policy_rollout_sim_time_s"] = 1.5
    write(root / "manifest.jsonl", manifest)
    with pytest.raises(ValueError, match="duration"):
        build(root, tmp_path)


def test_episode_warmup_is_distinct_from_counterfactual_calibration(evidence, tmp_path):
    root, manifest, _ = evidence
    manifest.update(warmup_requests_per_episode=1, calibration_warmup_requests=0)
    write(root / "manifest.jsonl", manifest)
    _, run, _ = build(root, tmp_path)
    episode = run["episodes"][0]
    assert episode["episode_warmup_requests"] == 1
    assert episode["episode_warmup_recorded"] == 1
    assert episode["calibration"]["warmup_requests"] == 0


def test_failed_episode_preserves_open_hold_without_inventing_end(evidence, tmp_path):
    root, manifest, audit = evidence
    audit["motion_events"].append(
        {
            "kind": "hold_started",
            "rollout_sim_time_s": 0.15,
            "active_chunk_id": 1,
            "action_step": 1,
        }
    )
    manifest.update(status="error", success=None, error="fixture failed")
    write(root / "audit.json", audit)
    write(root / "manifest.jsonl", manifest)
    _, run, _ = build(root, tmp_path)
    assert run["episodes"][0]["events"][-1]["end_s"] is None


def test_frame_on_closed_phase_endpoint_is_valid(evidence, tmp_path):
    root, _, audit = evidence
    audit["video_samples"][1]["sample_sim_time_s"] = 0.1
    write(root / "audit.json", audit)
    _, run, _ = build(root, tmp_path)
    assert run["episodes"][0]["frames"][1]["relative_s"] == 0.1


def test_browser_derivative_preserves_frames_and_uses_actual_container_pts(
    evidence, tmp_path
):
    import shutil
    import subprocess

    from coexecution.replay.http import ReplayFiles
    from coexecution.replay.media import MediaRegistry
    from coexecution.replay.schema import source

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Optional video encoding requires ffmpeg and ffprobe")
    root, manifest, _ = evidence
    original = root / "scene.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=60",
            "-frames:v",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(original),
        ],
        check=True,
    )
    before = source(original)
    manifest.update(model_input_video="scene.mp4", wrist_input_video="scene.mp4")
    write(root / "manifest.jsonl", manifest)
    _, run, output = build(root, tmp_path)
    episode = run["episodes"][0]
    frames = episode["frames"]
    assert [f["source_media_time_s"] for f in frames] == [0, 1 / 60, 2 / 60]
    assert [f["media_time_s"] for f in frames] == [0, 0.017, 0.033]
    assert [f["sample_sim_time_s"] for f in frames] == [0.02, 0.04, 0.15]
    assert episode["media"]["image"]["mime"] == "video/webm"
    assert (
        episode["media"]["image"]["media_id"]
        == episode["media"]["wrist_image"]["media_id"]
    )
    assert source(original) == before
    files = ReplayFiles(output)
    media_id = episode["media"]["image"]["media_id"]
    derived, mime = files.media[media_id]
    assert mime == "video/webm"
    asset = json.loads((output / "media.json").read_text())["assets"][0]
    assert asset["source"] == before
    assert asset["derivation"]["codec"] == "vp9"
    assert asset["derivation"]["frame_count"] == 3
    assert asset["derivation"]["frame_order"] == "passthrough"

    def checksums(path):
        output = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "framemd5",
                "-",
            ]
        ).decode()
        return [
            row.rsplit(",", 1)[-1].strip()
            for row in output.splitlines()
            if row and not row.startswith("#")
        ]

    assert checksums(output / derived) == checksums(original)
    with pytest.raises(ValueError, match="frame count"):
        MediaRegistry(tmp_path / "bad-count").browser_video(
            root, "scene.mp4", source_media_times=[0, 1 / 60]
        )
    with pytest.raises(ValueError, match="timestamp"):
        MediaRegistry(tmp_path / "bad-pts").browser_video(
            root, "scene.mp4", source_media_times=[0, 1 / 50, 2 / 50]
        )


def test_dynamic_rejects_different_camera_derivative_pts(
    evidence, tmp_path, monkeypatch
):
    from coexecution.replay.media import MediaRegistry

    root, manifest, _ = evidence
    manifest.update(model_input_video="scene.mp4", wrist_input_video="wrist.mp4")
    (root / "scene.mp4").touch()
    (root / "wrist.mp4").touch()
    write(root / "manifest.jsonl", manifest)

    def browser_video(self, root, relative, *, source_media_times):
        return {
            "media_id": str(relative),
            "mime": "video/webm",
            "media_times_s": [
                0,
                0.017 if str(relative) == "scene.mp4" else 0.018,
                0.033,
            ],
        }

    monkeypatch.setattr(MediaRegistry, "browser_video", browser_video)
    with pytest.raises(ValueError, match="camera.*timestamps"):
        build(root, tmp_path)


@pytest.fixture(params=[1, 2])
def native_evidence(evidence, request):
    root, manifest, audit = evidence
    prefix = request.param
    durations = [0.03123456789, 0.08345678912]
    boundary = (
        "upstream perf_counter: full get_action_chunk + NumPy result normalization; "
        "preprocessing/RPC/postprocessing included; warmup/reset/physics excluded"
    )
    mode = {
        "timing_mode": "native",
        "latency_kind": "measured_per_request",
        "latency_boundary": boundary,
    }
    base = audit["records"][1]
    audit.update(
        **mode, calibration=None, chunks=[], motion_events=[], video_samples=[]
    )
    audit["records"] = [audit["records"][0]]
    traces = []
    sim_time = 0.0
    for index, duration in enumerate(durations):
        step, sequence, chunk_id = index * prefix, 8 + index, 1 + index
        wall = 100.0 + index
        record = deepcopy(base)
        record.update(
            request_sequence=sequence,
            source_tick=step,
            sim_time_s=sim_time,
            request_started_s=wall,
            request_completed_s=wall + duration - 1e-7,
            rpc_started_s=wall + duration * 0.1,
            rpc_completed_s=wall + duration * 0.9,
            controller_policy_elapsed_s=duration,
            controller_timing_boundary=boundary,
            execution_prefix_steps=prefix,
            actions=record["actions"][:prefix],
        )
        record["server_audit"].update(
            source_tick=step,
            vlm_started_s=wall + duration * 0.2,
            vlm_completed_s=wall + duration * 0.6,
            action_started_s=wall + duration * 0.6,
            action_completed_s=wall + duration * 0.8,
        )
        audit["records"].append(record)
        audit["chunks"].append(
            {
                "chunk_id": chunk_id,
                "obs_step_idx": step,
                "rollout_sim_time_s": sim_time,
                "policy_elapsed_s": duration,
                "metadata": {
                    **mode,
                    "timing_rule": "release_continuous_sync",
                    "physics_time_quantized_to_control_steps": False,
                    "policy_request_sequence": sequence,
                    "raw_calibration_latency_s": None,
                    "applied_latency_s": duration,
                    "sim_latency_s": duration,
                },
            }
        )
        for kind, timestamp in [
            ("hold_started", sim_time),
            ("hold_completed", sim_time + duration),
        ]:
            audit["motion_events"].append(
                {
                    "kind": kind,
                    "rollout_sim_time_s": timestamp,
                    "applied_latency_s": duration,
                    "active_chunk_id": chunk_id,
                    "action_step": step,
                }
            )
        sim_time += duration
        for offset in range(prefix):
            sim_time += 0.05
            action = record["actions"][offset]
            audit["motion_events"].append(
                {
                    "kind": "action_completed",
                    "rollout_sim_time_s": sim_time,
                    "active_chunk_id": chunk_id,
                    "action_step": step + offset,
                    "action": action,
                }
            )
            traces.append(
                {
                    "step_idx": step + offset,
                    "sim_time_s": sim_time,
                    "chunk_id": chunk_id,
                    "executed_action": action,
                }
            )
    manifest.update(
        **mode,
        steps=2 * prefix,
        policy_rollout_sim_time_s=sim_time,
        applied_latency_s=None,
        raw_calibration_latency_s=None,
        calibration_kind=None,
        calibration_protocol=None,
        calibration_retained_requests=None,
        replan_steps_requested=prefix,
        video_frame_count=0,
    )
    write(root / "audit.json", audit)
    write(root / "manifest.jsonl", manifest)
    (root / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in traces))
    (root / "calibration.json").unlink()
    write(
        root / "evaluation.json",
        {
            "schema": "pir2.libero_dynamic.default.v1",
            "configuration": {"seed": 0, "native_latency": True},
            **mode,
        },
    )
    return root, manifest, audit


def test_native_replay_preserves_distinct_request_holds_and_short_prefix(
    native_evidence, tmp_path
):
    root, manifest, audit = native_evidence
    _, run, _ = build(root, tmp_path)
    episode = run["episodes"][0]
    assert episode["timing_mode"] == "native"
    assert episode["latency_kind"] == "measured_per_request"
    assert episode["calibration"]["kind"] is None
    assert episode["execution_prefix_steps"] == manifest["replan_steps_requested"]
    holds = [event for event in episode["events"] if event["kind"] == "inference_hold"]
    for hold, request in zip(holds, episode["requests"]):
        assert hold["end_s"] - hold["start_s"] == pytest.approx(
            request["service_duration_ms"] / 1000
        )
        assert hold["start_s"] == request["sim_time_s"]
    assert holds[0]["end_s"] == 0.03123456789
    assert holds[1]["start_s"] == pytest.approx(
        0.03123456789 + 0.05 * episode["execution_prefix_steps"]
    )
    assert (
        episode["requests"][0]["wall_stages"][0]["name"]
        == "Full policy call + normalization"
    )
    assert episode["requests"][0]["wall_stages"][0]["wall_started_s"] is None
    assert (
        episode["requests"][0]["wall_stages"][0]["duration_ms"]
        == audit["chunks"][0]["policy_elapsed_s"] * 1000
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m, a: a["chunks"][1]["metadata"].update(
            sim_latency_s=a["chunks"][0]["policy_elapsed_s"]
        ),
        lambda m, a: a["chunks"][1].update(policy_elapsed_s=0.05),
        lambda m, a: a["records"][2].update(controller_policy_elapsed_s=0.05),
        lambda m, a: a["records"][2].update(sim_time_s=0),
        lambda m, a: a["motion_events"][1].update(applied_latency_s=0.05),
        lambda m, a: a["motion_events"][0].update(applied_latency_s=0.05),
        lambda m, a: m.update(applied_latency_s=0.05),
        lambda m, a: m.update(timing_rule="paper_quantized_sync"),
        lambda m, a: a.update(calibration={"kind": "measured", "raw_latency_s": 0.05}),
        lambda m, a: a["records"][1]["server_audit"].update(
            forward_counts={"vlm": 0, "dit": 1}
        ),
    ],
)
def test_native_replay_rejects_stale_or_altered_delay_evidence(
    native_evidence, tmp_path, mutation
):
    root, manifest, audit = native_evidence
    mutation(manifest, audit)
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError, match="[Nn]ative"):
        build(root, tmp_path)


def test_native_failed_episode_retains_untimed_failed_request(
    native_evidence, tmp_path
):
    root, manifest, audit = native_evidence
    failed = deepcopy(audit["records"][-1])
    failed.update(request_sequence=10, status="error", error="RPC failed")
    failed.pop("controller_policy_elapsed_s")
    failed.pop("controller_timing_boundary")
    audit["records"].append(failed)
    manifest.update(status="error", success=None, error="RPC failed")
    write(root / "audit.json", audit)
    write(root / "manifest.jsonl", manifest)
    _, run, _ = build(root, tmp_path)
    episode = run["episodes"][0]
    assert episode["requests"][-1]["status"] == "error"
    assert episode["requests"][-1]["service_duration_ms"] is None
    assert (
        len([event for event in episode["events"] if event["kind"] == "inference_hold"])
        == 2
    )


def test_native_physical_duration_must_equal_this_request(native_evidence, tmp_path):
    root, _, audit = native_evidence
    audit["motion_events"][1]["rollout_sim_time_s"] += 0.001
    write(root / "audit.json", audit)
    with pytest.raises(ValueError, match="Native Flow physical hold"):
        build(root, tmp_path)
