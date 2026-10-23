"""πR² dynamic replay validates physical lineage without mixing logical/wall clocks."""

import json
from copy import deepcopy

import pytest

from coexecution.replay.catalog import build_catalog


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def streaming_evidence(tmp_path):
    root = tmp_path / "streaming"
    root.mkdir()
    action = [0.1, 0, 0, 0, 0, 0, 1]
    raw = [0.1, 0, 0, 0, 0, 0, -1]
    bootstrap = {
        "capture_s": 1000.0,
        "logical_completed_s": 1000.2,
        "logical_duration_s": 0.2,
        "actions": [action] * 40,
        "meta": {
            "capture_s": 1000.0,
            "source_tick": 0,
            "frame_hashes": {"image": "bootstrap"},
        },
        "wall": {"wall_started_s": 5000.0, "wall_completed_s": 5000.05},
        "server_audit": {
            "bootstrap_seconds": 0.04,
            "forward_counts": {"vlm": 1, "dit": 4},
        },
    }
    s2 = {
        "request_id": 0,
        "lane": "s2",
        "logical_started_s": 1000.2,
        "logical_completed_s": 1000.2,
        "logical_duration_s": 0.0,
        "wall_started_s": 5001.0,
        "wall_completed_s": 5001.04,
        "source_tick": 0,
        "capture_s": 1000.2,
        "feature_sequence": 1,
        "status": "completed",
        "publication": {"feature_sequence": 1},
        "server_audit": {
            "vlm_seconds": 0.03,
            "vlm_forward_calls": 1,
            "capture_s": 1000.2,
            "source_tick": 0,
            "frame_hashes": {"image": "s2"},
        },
    }
    s1 = {
        "request_id": 1,
        "lane": "s1",
        "request_tick": 0,
        "logical_started_s": 1000.2,
        "logical_completed_s": 1000.24,
        "logical_duration_s": 0.04,
        "wall_started_s": 5001.1,
        "wall_completed_s": 5001.13,
        "status": "completed",
        "feature_sequence": 1,
        "feature_source_tick": 0,
        "feature_capture_s": 1000.2,
        "state_capture_s": 1000.2,
        "feature_age_s": 0.0,
        "delay_ticks": 1,
        "model_actions": [action],
        "normal_or_recovery": "normal1",
        "server_audit": {
            "buffer_restarted": False,
            "cache": {
                "capture_s": 1000.2,
                "source_tick": 0,
                "frame_hashes": {"image": "s2"},
            },
            "valid_start_tick": 1,
            "valid_end_tick": 2,
            "server_seconds": 0.02,
            "server_plan_started_s": 5001.101,
            "forward_counts": {"vlm": 0, "dit": 1},
        },
        "publication": {
            "installed_slots": 1,
            "expired_slots": 0,
            "protected_slots": 0,
            "logical_published_s": 1000.24,
            "next_unexecuted_tick": 1,
        },
    }
    actions = [
        {
            "tick": i,
            "logical_started_s": 1000.2 + i * 0.05,
            "logical_completed_s": 1000.25 + i * 0.05,
            "raw_action": raw,
            "model_action": action,
            "source": {
                "tick": i,
                "producer_kind": "bootstrap" if i == 0 else "request",
                "producer_request_tick": None if i == 0 else 0,
            },
            "fallback": False,
            "queue_age_ticks": i,
            "feature_age_s": 0.2 if i == 0 else 0.05,
            "state_age_s": 0.2 if i == 0 else 0.05,
            "feature_capture_s": 1000.0 if i == 0 else 1000.2,
            "state_capture_s": 1000.0 if i == 0 else 1000.2,
        }
        for i in range(2)
    ]
    motions = [
        {
            "kind": "hold_started",
            "rollout_sim_time_s": 0.0,
            "action_step": 0,
            "active_chunk_id": None,
        },
        {
            "kind": "hold_completed",
            "rollout_sim_time_s": 0.2,
            "action_step": 0,
            "active_chunk_id": None,
        },
    ]
    motions += [
        {
            "kind": "action_completed",
            "rollout_sim_time_s": 0.25 + i * 0.05,
            "action_step": i,
            "active_chunk_id": i,
            "action": raw,
        }
        for i in range(2)
    ]
    samples = [
        {
            "frame_index": i,
            "media_time_s": i / 60,
            "sample_sim_time_s": t,
            "phase": phase,
            "action_step": tick,
            "active_chunk_id": chunk,
        }
        for i, (t, phase, tick, chunk) in enumerate(
            [
                (0.02, "bootstrap_hold", 0, None),
                (0.2, "bootstrap_hold", 0, None),
                (0.22, "action", 0, 0),
                (0.27, "action", 1, 1),
            ]
        )
    ]
    audit = {
        "schema": "pir2.libero_dynamic.streaming_audit.v1",
        "item_id": "task0",
        "identities": {"action": {"predicted_horizon": 40, "decoded_horizon": 40}},
        "warmup": [],
        "streaming": {
            "logical_epoch_s": 1000.0,
            "control_origin_s": 1000.2,
            "startup_seconds": 0.2,
            "physical_seconds": 0.3,
            "action_steps": 2,
            "requests": [s2, s1],
            "actions": actions,
            "bootstrap": bootstrap,
            "events": [
                {
                    "kind": "logical_request_completed",
                    "t_s": 1000.24,
                    "request_id": 1,
                    "lane": "s1",
                    **s1["publication"],
                    "action_buffer": {
                        "committed_until": 1,
                        "slots": [
                            {
                                "tick": 1,
                                "status": "future",
                                "committed": False,
                                "producer_kind": "request",
                                "producer_request_tick": 0,
                            }
                        ],
                    },
                }
            ],
            "config": {"period_s": 0.05},
        },
        "motion_events": motions,
        "video_samples": samples,
    }
    manifest = {
        "item_id": "task0",
        "suite": "libero_spatial",
        "task_id": 0,
        "level": "l1",
        "traj_index": 0,
        "speed_mps": 0.03,
        "precision": "bf16",
        "protocol": "pir2_dynamic",
        "status": "ok",
        "success": False,
        "steps": 2,
        "policy_audit": "audit.json",
        "logical_epoch_s": 1000.0,
        "control_origin_s": 1000.2,
        "rollout_sim_time_s": 0.3,
        "stage_latencies": {
            "bootstrap_s": 0.2,
            "s2_s": 0.0,
            "s1_s": 0.04,
            "recovery_s": 0.2,
        },
        "latency_provenance": "fixed_counterfactual",
        "resource_model": "independent_stage_service",
        "compute_coupling": "serialized_gpu_simulation",
        "timing_domain": "counterfactual_discrete_event",
        "warmup_rounds": 0,
        "video_frame_count": 4,
        "video_fps": 60,
    }
    write(
        root / "evaluation.json",
        {
            "schema": "pir2.libero_dynamic.streaming_evaluation.v1",
            "status": "complete",
            "configuration": {"seed": 0, "init_state_id": 0},
        },
    )
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    return root, manifest, audit


def build(root, tmp_path):
    output = tmp_path / "catalog"
    catalog = build_catalog(
        model_runs=[], deployments=[], dynamic_evaluations=[root], output=output
    )
    return json.loads((output / catalog["runs"][0]["path"]).read_text())


def native_evidence(evidence):
    root, manifest, audit = evidence
    stream = audit["streaming"]
    mode = {
        "timing_mode": "native",
        "timing_domain": "measured_request_discrete_event",
        "latency_provenance": "measured_per_request",
    }
    manifest.update(**mode, stage_latencies=None, startup_seconds=0.2)
    stream.update(mode)
    stream["config"]["stage_latencies"] = None
    bootstrap, (s2, s1) = stream["bootstrap"], stream["requests"]
    s2.update(
        logical_completed_s=1000.21, logical_duration_s=0.01, wall_completed_s=5001.005
    )
    # S2 is not ready at the S1 admission, so S1 uses the bootstrap feature.
    s1.update(feature_sequence=0, feature_capture_s=1000.0, feature_age_s=0.2)
    s1["server_audit"]["cache"].update(
        capture_s=1000.0, frame_hashes={"image": "bootstrap"}
    )
    stream["actions"][1].update(feature_capture_s=1000.0, feature_age_s=0.25)
    for request, start, duration in [
        (bootstrap, 5000.0, 0.2),
        (s2, 5001.0, 0.01),
        (s1, 5001.1, 0.04),
    ]:
        request["service_wall"] = {
            "clock": "host_monotonic",
            "started_s": start,
            "completed_s": start + duration,
            "duration_s": duration,
            "boundary": "full_adapter_call",
        }
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    return root, manifest, audit


def test_native_replay_preserves_each_measured_call_and_separate_rpc_boundary(
    streaming_evidence, tmp_path
):
    root, _, _ = native_evidence(streaming_evidence)
    episode = build(root, tmp_path)["episodes"][0]
    assert episode["calibration"]["kind"] == "measured_per_request"
    assert episode["stage_latencies"] is None
    requests = {r["lane"]: r for r in episode["requests"]}
    assert requests["s2"]["end_s"] - requests["s2"]["start_s"] == pytest.approx(0.01)
    assert requests["s1"]["end_s"] - requests["s1"]["start_s"] == pytest.approx(0.04)
    assert requests["s2"]["rpc_duration_ms"] == pytest.approx(5)
    assert requests["s2"]["service_duration_ms"] == pytest.approx(10)
    assert requests["s1"]["parent_request_id"] == requests["bootstrap"]["id"]


@pytest.mark.parametrize("mismatch", [False, True])
def test_streaming_replay_checks_custom_seed_against_run_configuration(
    streaming_evidence, tmp_path, mismatch
):
    root, manifest, _ = native_evidence(streaming_evidence)
    fields = {"seed": 8000, "init_state_id": 0, "seed_scope": "custom"}
    manifest.update(fields)
    evaluation = json.loads((root / "evaluation.json").read_text())
    evaluation["configuration"].update(fields)
    if mismatch:
        evaluation["configuration"]["seed"] = 8001
    write(root / "evaluation.json", evaluation)
    write(root / "manifest.jsonl", manifest)
    if mismatch:
        with pytest.raises(ValueError, match="seed"):
            build(root, tmp_path)
    else:
        episode = build(root, tmp_path)["episodes"][0]
        assert {key: episode[key] for key in fields} == fields


def aligned_native_evidence(evidence):
    root, manifest, audit = native_evidence(evidence)
    stream = audit["streaming"]
    startup = {
        "requested_startup_release_seconds": 0.4,
        "bootstrap_service_seconds": 0.2,
        "startup_barrier_seconds": 0.2,
        "startup_seconds": 0.4,
        "control_origin_s": 1000.4,
        "startup_alignment_met": True,
        "startup_policy": "common_startup_release_moving_world_hold",
    }
    for record in (manifest, stream, stream["config"], stream["bootstrap"]):
        record.update(startup)
    manifest["rollout_sim_time_s"] = stream["physical_seconds"] = 0.5
    manifest["startup_seconds_applied"] = 0.4
    s2, s1 = stream["requests"]
    for request in (s2, s1):
        request["logical_started_s"] += 0.2
        request["logical_completed_s"] += 0.2
    s2["capture_s"] += 0.2
    s2["server_audit"]["capture_s"] += 0.2
    s1["state_capture_s"] += 0.2
    s1["feature_age_s"] += 0.2
    s1["publication"]["logical_published_s"] += 0.2
    for action in stream["actions"]:
        action["logical_started_s"] += 0.2
        action["logical_completed_s"] += 0.2
        action["feature_age_s"] += 0.2
        if action["tick"] == 0:
            action["state_age_s"] += 0.2
        else:
            action["state_capture_s"] += 0.2
    for event in stream["events"]:
        event["t_s"] += 0.2
        event["logical_published_s"] += 0.2
    for motion in audit["motion_events"][1:]:
        motion["rollout_sim_time_s"] += 0.2
    for sample in audit["video_samples"][2:]:
        sample["sample_sim_time_s"] += 0.2
    audit["video_samples"].insert(
        2,
        {**audit["video_samples"][1], "sample_sim_time_s": 0.3},
    )
    for i, sample in enumerate(audit["video_samples"]):
        sample.update(frame_index=i, media_time_s=i / 60)
    manifest["video_frame_count"] = len(audit["video_samples"])
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    return root, manifest, audit


def test_startup_alignment_splits_wait_without_stretching_service_or_frames(
    streaming_evidence, tmp_path
):
    root, _, audit = aligned_native_evidence(streaming_evidence)
    episode = build(root, tmp_path)["episodes"][0]
    holds = [
        e
        for e in episode["events"]
        if e["kind"] in ("bootstrap_hold", "startup_barrier")
    ]
    assert [(h["kind"], h["start_s"], h["end_s"]) for h in holds] == [
        ("bootstrap_hold", 0.0, pytest.approx(0.2)),
        ("startup_barrier", pytest.approx(0.2), pytest.approx(0.4)),
    ]
    requests = {r["lane"]: r for r in episode["requests"]}
    assert requests["bootstrap"]["service_duration_ms"] == pytest.approx(200)
    assert requests["bootstrap"]["end_s"] == pytest.approx(0.2)
    assert requests["s1"]["service_duration_ms"] == pytest.approx(40)
    assert requests["s2"]["service_duration_ms"] == pytest.approx(10)
    assert requests["s1"]["state_capture_s"] == pytest.approx(1000.4)
    assert requests["s1"]["feature_capture_s"] == 1000.0
    assert [f["relative_s"] for f in episode["frames"]] == [
        f["sample_sim_time_s"] for f in audit["video_samples"]
    ]
    assert episode["frames"][2]["phase"] == "bootstrap_hold"
    assert episode["frames"][2]["relative_s"] == 0.3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m, s: m.update(startup_barrier_seconds=0.0),
        lambda m, s: m.update(requested_startup_release_seconds=0.5),
        lambda m, s: m.pop("requested_startup_release_seconds"),
        lambda m, s: s["bootstrap"].update(bootstrap_service_seconds=0.4),
        lambda m, s: s["bootstrap"].update(startup_seconds=0.2),
        lambda m, s: s["config"].update(control_origin_s=1000.2),
        lambda m, s: s.update(startup_alignment_met=False),
    ],
)
def test_startup_alignment_rejects_inconsistent_accounting(
    streaming_evidence, tmp_path, mutation
):
    root, manifest, audit = aligned_native_evidence(streaming_evidence)
    mutation(manifest, audit["streaming"])
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError):
        build(root, tmp_path)


def test_interrupted_startup_preserves_observed_barrier_without_inventing_end(
    streaming_evidence, tmp_path
):
    root, manifest, audit = aligned_native_evidence(streaming_evidence)
    manifest.update(
        status="error",
        steps=0,
        rollout_sim_time_s=0.3,
        startup_seconds_applied=0.3,
        video_frame_count=3,
    )
    audit["streaming"].update(requests=[], actions=[], events=[])
    audit["motion_events"] = audit["motion_events"][:1]
    audit["video_samples"] = audit["video_samples"][:3]
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    episode = build(root, tmp_path)["episodes"][0]
    assert len(episode["events"]) == 2
    assert episode["events"][0]["end_s"] == pytest.approx(0.2)
    assert episode["events"][1]["kind"] == "startup_barrier"
    assert episode["events"][1]["start_s"] == pytest.approx(0.2)
    assert episode["events"][1]["end_s"] is None
    assert episode["startup_seconds_applied"] == 0.3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m, s: s["requests"][0]["service_wall"].update(duration_s=0.15),
        lambda m, s: s["requests"][1]["service_wall"].update(completed_s=5001.13),
        lambda m, s: s["bootstrap"].pop("service_wall"),
        lambda m, s: m.update(stage_latencies={"s1_s": 0.04}),
        lambda m, s: s.update(stage_latencies={"s1_s": 0.04}),
        lambda m, s: s["requests"][0].update(wall_completed_s=5001.05),
        lambda m, s: m.update(latency_provenance="fixed_counterfactual"),
    ],
)
def test_native_replay_rejects_aligned_or_inconsistent_service_evidence(
    streaming_evidence, tmp_path, mutation
):
    root, manifest, audit = native_evidence(streaming_evidence)
    mutation(manifest, audit["streaming"])
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError):
        build(root, tmp_path)


def test_streaming_import_preserves_independent_clocks_and_dependency_chain(
    streaming_evidence, tmp_path
):
    root, _, _ = streaming_evidence
    run = build(root, tmp_path)
    e = run["episodes"][0]
    assert run["protocol"] == "pir2_dynamic"
    assert e["duration_s"] == 0.3
    requests = {r["lane"]: r for r in e["requests"]}
    assert requests["s2"]["start_s"] == pytest.approx(0.2)
    assert requests["s1"]["parent_request_id"] == requests["s2"]["id"]
    assert requests["s1"]["rpc_duration_ms"] == pytest.approx(30)
    assert requests["s2"]["wall_stages"][1]["duration_ms"] == 30
    assert requests["s2"]["wall_stages"][1]["wall_started_s"] is None
    actions = [v for v in e["events"] if v["kind"] == "action"]
    assert actions[0]["request_id"] == requests["bootstrap"]["id"]
    assert actions[1]["request_id"] == requests["s1"]["id"]
    assert e["frames"][0]["phase"] == "bootstrap_hold"
    assert e["calibration"]["kind"] == "counterfactual"
    assert e["compute_coupling"] == "serialized_gpu_simulation"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["streaming"]["requests"][1].update(feature_sequence=999),
        lambda a: a["streaming"]["requests"][1].update(feature_capture_s=5001.0),
        lambda a: a["streaming"]["requests"][1].update(logical_started_s=1000.205),
        lambda a: a["streaming"]["actions"][1]["source"].update(
            producer_request_tick=99
        ),
        lambda a: a["streaming"]["actions"][1].update(model_action=[0] * 7),
        lambda a: a["motion_events"][3].update(action=[0] * 7),
        lambda a: a["video_samples"][0].update(phase="action"),
        lambda a: a["video_samples"][2].update(active_chunk_id=999),
        lambda a: a["video_samples"][2].update(action_step=1),
        lambda a: a["streaming"]["requests"].append(
            deepcopy(a["streaming"]["requests"][0])
        ),
    ],
)
def test_streaming_rejects_wrong_dependency_or_frame_lineage(
    streaming_evidence, tmp_path, mutation
):
    root, _, audit = streaming_evidence
    mutation(audit)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError):
        build(root, tmp_path)


def test_per_camera_partial_maps_do_not_invent_the_missing_wrist_frame(
    streaming_evidence, tmp_path
):
    root, manifest, audit = streaming_evidence
    manifest.update(
        status="error",
        error="wrist encoder failed",
        video_frame_counts={"dual": 4, "wrist": 3},
        video_frame_count=3,
    )
    for i, sample in enumerate(audit["video_samples"]):
        sample["streams"] = {
            name: {
                "frame_index": i,
                "media_time_s": i / 60,
                "status": "error" if name == "wrist" and i == 3 else "written",
            }
            for name in ("dual", "wrist")
        }
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    e = build(root, tmp_path)["episodes"][0]
    assert len(e["frames"]) == 4
    assert len(e["frames_by_camera"]["wrist_image"]) == 3
    assert e["frames_by_camera"]["wrist_image"][-1]["sample_sim_time_s"] == 0.22


def test_fallback_has_no_model_request_and_expired_request_is_retained(
    streaming_evidence, tmp_path
):
    root, _, audit = streaming_evidence
    stream = audit["streaming"]
    action = stream["actions"][1]
    action.update(
        fallback=True,
        model_action=[0] * 7,
        raw_action=[0, 0, 0, 0, 0, 0, 1],
        feature_age_s=None,
        state_age_s=None,
    )
    action["source"].update(producer_kind="fallback", producer_request_tick=None)
    audit["motion_events"][3]["action"] = action["raw_action"]
    stream["requests"][1]["publication"].update(installed_slots=0, expired_slots=1)
    stream["events"][0].update(installed_slots=0, expired_slots=1)
    stream["events"][0]["action_buffer"]["slots"] = []
    action.update(feature_capture_s=None, state_capture_s=None)
    write(root / "audit.json", audit)
    e = build(root, tmp_path)["episodes"][0]
    a = next(v for v in e["events"] if v["kind"] == "action" and v["action_step"] == 1)
    assert a["request_id"] is None
    assert a["fallback"] is True
    assert (
        next(r for r in e["requests"] if r["lane"] == "s1")["publication"][
            "expired_slots"
        ]
        == 1
    )


def test_s1_can_use_bootstrap_feature_zero_without_claiming_it_is_s2_request(
    streaming_evidence, tmp_path
):
    root, _, audit = streaming_evidence
    audit["streaming"]["requests"][1].update(
        feature_sequence=0, feature_capture_s=1000.0, feature_age_s=0.2
    )
    audit["streaming"]["requests"][1]["server_audit"]["cache"] = deepcopy(
        audit["streaming"]["bootstrap"]["meta"]
    )
    audit["streaming"]["actions"][1].update(
        feature_capture_s=1000.0, feature_age_s=0.25
    )
    write(root / "audit.json", audit)
    e = build(root, tmp_path)["episodes"][0]
    request = next(r for r in e["requests"] if r["lane"] == "s1")
    parent = next(r for r in e["requests"] if r["id"] == request["parent_request_id"])
    assert parent["lane"] == "bootstrap"
    assert parent["start_s"] == 0
    assert parent["end_s"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m, a: m.update(steps=9),
        lambda m, a: m.update(rollout_sim_time_s=0.4),
        lambda m, a: a["streaming"]["requests"][0].update(status="inflight"),
        lambda m, a: a["streaming"]["requests"][1]["server_audit"].update(
            valid_end_tick=1
        ),
        lambda m, a: a["streaming"]["actions"][1]["source"].update(tick=99),
        lambda m, a: m.update(policy_audit="../outside.json"),
    ],
)
def test_streaming_rejects_unpublished_sources_and_manifest_mismatch(
    streaming_evidence, tmp_path, mutation
):
    root, manifest, audit = streaming_evidence
    mutation(manifest, audit)
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError):
        build(root, tmp_path)


def test_a_partial_finished_physics_action_does_not_invent_a_controller_source(
    streaming_evidence, tmp_path
):
    root, manifest, audit = streaming_evidence
    manifest.update(status="error", error="model evidence callback failed")
    audit["streaming"]["actions"].pop()
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    e = build(root, tmp_path)["episodes"][0]
    action = next(
        v for v in e["events"] if v["kind"] == "action" and v["action_step"] == 1
    )
    assert action["lineage_missing"] is True
    assert action["request_id"] is None
    assert action["end_s"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m, a: m["stage_latencies"].update(s2_s=0.15),
        lambda m, a: m.update(control_origin_s=1000.3),
        lambda m, a: a["streaming"]["requests"][1].update(logical_duration_s=0.05),
        lambda m, a: (
            a["streaming"]["actions"][1].update(raw_action=[0.1, 0, 0, 0, 0, 0, 1])
            or a["motion_events"][3].update(action=[0.1, 0, 0, 0, 0, 0, 1])
        ),
    ],
)
def test_streaming_rejects_mislabeled_timing_or_gripper_mapping(
    streaming_evidence, tmp_path, mutation
):
    root, manifest, audit = streaming_evidence
    mutation(manifest, audit)
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError):
        build(root, tmp_path)


@pytest.mark.parametrize("legacy_partial", [False, True])
def test_real_runner_partial_result_retains_startup_evidence(
    streaming_evidence, tmp_path, legacy_partial
):
    from types import SimpleNamespace

    from coexecution.libero_dynamic.streaming import StageLatencies
    from coexecution.libero_dynamic.streaming_evaluation import _partial_streaming

    root, manifest, audit = streaming_evidence
    full = audit["streaming"]
    startup = {
        "requested_startup_release_seconds": None,
        "startup_policy": "paid_moving_world_hold",
        "bootstrap_service_seconds": 0.2,
        "startup_barrier_seconds": 0.0,
        "startup_seconds": 0.2,
        "control_origin_s": 1000.2,
        "startup_alignment_met": None,
    }
    controller = SimpleNamespace(
        timing_mode="fixed",
        latencies=StageLatencies(bootstrap_s=0.2, s1_s=0.04, s2_s=0, recovery_s=0.2),
        logical_time_s=1000.3,
        epoch_s=1000.0,
        startup_metadata=lambda: dict(startup),
        bootstrap_evidence=deepcopy(full["bootstrap"]),
        requests=full["requests"],
        actions=full["actions"],
        trace=SimpleNamespace(records=lambda: full["events"]),
    )
    audit["bootstrap"] = full["bootstrap"]
    audit["streaming"] = _partial_streaming(controller)
    assert audit["streaming"]["bootstrap"] == full["bootstrap"]
    assert audit["streaming"]["config"]["startup_seconds"] == 0.2
    if legacy_partial:
        audit["streaming"].pop("bootstrap")
        audit["streaming"].pop("config")
    manifest.update(status="error", error="step callback failed")
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    e = build(root, tmp_path)["episodes"][0]
    assert e["events"][0]["kind"] == "bootstrap_hold"
    assert e["events"][0]["end_s"] == 0.2
    assert e["status"] == "error"


def test_real_runner_early_error_has_no_identities_or_bootstrap(
    streaming_evidence, tmp_path
):
    from coexecution.libero_dynamic.streaming_evaluation import _partial_streaming

    root, manifest, audit = streaming_evidence
    audit.update(
        streaming=_partial_streaming(None),
        bootstrap=None,
        identities=None,
        motion_events=[],
        video_samples=[],
    )
    manifest.update(
        status="error",
        error="identity endpoint failed",
        logical_epoch_s=None,
        control_origin_s=None,
        steps=0,
        rollout_sim_time_s=0,
        video_frame_count=0,
    )
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    e = build(root, tmp_path)["episodes"][0]
    assert e["requests"] == []
    assert e["predicted_horizon"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["streaming"]["actions"][1].update(feature_capture_s=5001.0),
        lambda a: a["streaming"]["actions"][1].update(state_capture_s=5001.0),
        lambda a: a["streaming"]["actions"][1].update(feature_age_s=9999),
        lambda a: a["streaming"]["actions"][0].update(state_age_s=9999),
        lambda a: a["streaming"]["requests"][1]["server_audit"]["cache"].update(
            capture_s=5001.0
        ),
        lambda a: a["streaming"]["requests"][1]["server_audit"]["cache"].update(
            source_tick=99
        ),
        lambda a: a["streaming"]["requests"][1]["server_audit"]["cache"].update(
            frame_hashes={"image": "wrong"}
        ),
        lambda a: a["streaming"]["requests"][1]["publication"].update(
            installed_slots=0, expired_slots=1
        ),
        lambda a: a["streaming"]["requests"][1]["publication"].update(
            logical_published_s=1000.4
        ),
        lambda a: a["streaming"]["requests"][1]["publication"].update(
            next_unexecuted_tick=99
        ),
        lambda a: a["streaming"]["events"][0]["action_buffer"]["slots"][0].update(
            committed=True
        ),
        lambda a: a["streaming"]["events"][0]["action_buffer"]["slots"][0].update(
            producer_request_tick=999
        ),
        lambda a: a["streaming"]["requests"][1]["server_audit"].update(
            buffer_restarted=True
        ),
        lambda a: a["streaming"]["requests"][1]["server_audit"][
            "forward_counts"
        ].update(dit=5),
    ],
)
def test_streaming_rejects_false_age_or_uninstalled_producer(
    streaming_evidence, tmp_path, mutation
):
    root, _, audit = streaming_evidence
    mutation(audit)
    write(root / "audit.json", audit)
    with pytest.raises(ValueError):
        build(root, tmp_path)


def test_partial_physics_step_retains_written_frames_without_completing_the_action(
    streaming_evidence, tmp_path
):
    root, manifest, audit = streaming_evidence
    manifest.update(
        status="error",
        error="wrist encoder failed",
        steps=1,
        rollout_sim_time_s=0.28,
        physical_started_s=4.5,
    )
    audit["streaming"]["actions"].pop()
    audit["motion_events"].pop()
    audit["action_attempts"] = [
        {
            "tick": 1,
            "status": "error",
            "raw_action": [0.1, 0, 0, 0, 0, 0, -1],
            "physics_started_s": 4.75,
            "physics_completed_s": 4.78,
            "physics_elapsed_s": 0.03,
            "physics_complete": False,
        }
    ]
    write(root / "manifest.jsonl", manifest)
    write(root / "audit.json", audit)
    e = build(root, tmp_path)["episodes"][0]
    a = next(v for v in e["events"] if v["kind"] == "action" and v["action_step"] == 1)
    assert a["partial"] is True
    assert a["lineage_missing"] is True
    assert a["start_s"] == pytest.approx(0.25)
    assert a["end_s"] == pytest.approx(0.28)
    assert len(e["frames"]) == 4


def test_streaming_normalization_retains_one_canonical_copy_of_requests_frames_and_trace(
    streaming_evidence, tmp_path
):
    root, _, audit = streaming_evidence
    e = build(root, tmp_path)["episodes"][0]
    assert "controller_events" not in e
    assert e["trace"] == audit["streaming"]["events"]
    assert "image" not in e["frames_by_camera"]
    assert len(e["frames"]) == len(audit["video_samples"])
    for event in e["events"]:
        if event["kind"] in ("s1", "s2"):
            assert "source" not in event
            request = next(r for r in e["requests"] if r["id"] == event["request_id"])
            assert request["lane"] == event["kind"]
