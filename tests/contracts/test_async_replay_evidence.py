"""Whole-work corruption gates, including unused terminal inference."""

import copy
import json

import pytest

from coexecution.async_evidence import validate_one_step_async_work
from coexecution.feature_usage import audit_raw
from coexecution.queue_report import summarize_episode
from coexecution.replay.catalog import build_catalog
from tests.contracts.test_one_step_async import scenario


@pytest.fixture
def terminal_work():
    result, events, *_ = scenario(success_at=1)
    queue = summarize_episode(events, {**result, "task_id": 0, "episode_id": 0})
    return result, events, queue


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(source_tick=999),
        lambda r: r.update(target_tick=999),
        lambda r: r.update(request_id=0),
        lambda r: r.update(frame_hashes={"image": "f" * 64}),
        lambda r: r["server_stages"].update(vlm_started_s=-1),
        lambda r: r["server_stages"].update(
            action_completed_s=r["server_stages"]["action_started_s"] - 1
        ),
        lambda r: r.update(accepted_prefix_steps=1),
        lambda r: r.update(forward_counts={"vlm": 0, "dit": 4}),
        lambda r: r.update(unused_plan_suffix_slots=0),
    ],
)
def test_unused_terminal_normalized_corruption_fails_raw_audit(
    tmp_path, terminal_work, mutation
):
    _, events, queue = terminal_work
    queue["id"] = "task0-episode0"
    directory = tmp_path / "one-step" / "deployment"
    directory.mkdir(parents=True)
    (directory / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    condition = {
        "layout": "one-step",
        "condition": "deployment",
        "id": "one-step/deployment",
    }
    assert audit_raw(condition, queue, tmp_path)
    mutation(queue["work_requests"][-1])
    with pytest.raises(ValueError, match="One-step async evidence"):
        audit_raw(condition, queue, tmp_path)


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("one_step_async_request", "source_tick", 999),
        ("policy_request_submitted", "frame_hashes", {"image": "f" * 64}),
        ("policy_rpc_completed", "rpc_completed_s", -1),
        ("vision_request", "started_s", -1),
        ("feature_published", "feature_sequence", 999),
        ("action_completed", "valid_end_tick", 999),
        ("one_step_async_execution", "executed_steps", 2),
        ("async_work_end", "policy_calls", 999),
    ],
)
def test_raw_full_work_corruption_rejected(terminal_work, kind, field, value):
    _, events, _ = terminal_work
    events = copy.deepcopy(events)
    [r for r in events if r["kind"] == kind][-1][field] = value
    with pytest.raises(ValueError, match="One-step async evidence"):
        summarize_episode(events, {})


def test_missing_and_duplicate_work_rejected(terminal_work):
    _, events, queue = terminal_work
    for rows in (
        queue["work_requests"][:-1],
        queue["work_requests"] + [queue["work_requests"][-1]],
    ):
        with pytest.raises(ValueError):
            validate_one_step_async_work(events, normalized=rows)
    for kind in (
        "one_step_async_request",
        "async_work_end",
        "policy_rpc_started",
        "cycle_phase",
    ):
        reduced = copy.deepcopy(events)
        reduced.remove([r for r in reduced if r["kind"] == kind][-1])
        with pytest.raises(ValueError):
            summarize_episode(reduced, {})


def test_catalog_preserves_unique_sources_overlapping_phases_and_terminal_work(
    tmp_path, terminal_work
):
    result, events, queue = terminal_work
    deployment = tmp_path / "input" / "deployment"
    deployment.mkdir(parents=True)
    (deployment / "config.json").write_text(
        json.dumps(
            {
                "protocol": "deployment",
                "execution_clock": "ready",
                "execution_coupling": "one_step_async",
                "checkpoint": {
                    "inference_optimization": {"config": {"precision": "bf16"}}
                },
            }
        )
    )
    (deployment / "episodes.jsonl").write_text(
        json.dumps({**result, "task_id": 0, "episode_id": 0}) + "\n"
    )
    (deployment / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in events)
    )
    output = tmp_path / "replay"
    catalog = build_catalog(
        model_runs=[],
        deployments=[deployment],
        output=output,
        default_execution_clock="ready",
        default_execution_coupling="one_step_async",
    )
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    episode = run["episodes"][0]
    tasks = episode["queue"]["replay_requests"]
    vlm = [r for r in tasks if r["role"] == "S2"]
    assert [r["source_tick"] for r in vlm] == [0, 0]
    assert len({r["id"] for r in vlm}) == 2
    assert [r["feature_sequence"] for r in vlm] == [0, 1]
    controlled_vlm = [r for r in episode["queue"]["requests"] if r["role"] == "S2"]
    assert all(r["observation_id"].endswith("/observation-0") for r in controlled_vlm)
    assert episode["ticks"][0]["request_id"] == 0
    assert episode["ticks"][0]["model_request_id"].endswith("/S1-0")
    assert tasks[-1]["status"] == "unused"
    assert tasks[-1]["end_s"] > queue["duration_s"]
    assert episode["work_end_s"] > episode["controlled_end_s"]
    inference = [r for r in episode["phases"] if r["phase"] == "inference"]
    assert inference[1]["simulation_paused"] is False
    assert all(
        r["raw"]["start_s"] - episode["controlled_start_s"] == r["start_s"]
        for r in episode["phases"]
    )


@pytest.fixture
def chunked_work():
    from tests.contracts.test_chunked_async import scenario as chunked_scenario

    result, events, *_ = chunked_scenario(chunk=4, depth=2, terminal=7)
    return result, events


def chunked_catalog(tmp_path, result, events, **defaults):
    deployment = tmp_path / "input" / "deployment"
    deployment.mkdir(parents=True)
    (deployment / "config.json").write_text(
        json.dumps(
            {
                "protocol": "deployment",
                "execution_clock": "ready",
                "execution_coupling": "chunked_async",
                "execution_prefix_steps": 4,
                "async_overlap_steps": 2,
                "checkpoint": {
                    "inference_optimization": {"config": {"precision": "bf16"}}
                },
            }
        )
    )
    (deployment / "episodes.jsonl").write_text(
        json.dumps({**result, "task_id": 0, "episode_id": 0}) + "\n"
    )
    (deployment / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    output = tmp_path / "replay"
    catalog = build_catalog(
        model_runs=[],
        deployments=[deployment],
        output=output,
        default_execution_clock="ready",
        default_execution_coupling="chunked_async",
        **defaults,
    )
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    return catalog, run


def test_chunked_catalog_preserves_native_offsets_partial_prefix_and_full_drain(
    tmp_path, chunked_work
):
    result, events = chunked_work
    catalog, run = chunked_catalog(
        tmp_path,
        result,
        events,
        default_execution_prefix_steps=4,
        default_async_overlap_steps=2,
    )
    assert catalog["default_execution_prefix_steps"] == 4
    assert catalog["default_async_overlap_steps"] == 2
    assert catalog["runs"][0]["execution_prefix_steps"] == 4
    assert catalog["runs"][0]["async_overlap_steps"] == 2
    episode = run["episodes"][0]
    ticks, phases, queue = episode["ticks"], episode["phases"], episode["queue"]
    assert [r["request_id"] for r in ticks] == [0, 0, 0, 0, 4, 4, 4]
    assert [r["executed_offset"] for r in ticks] == [0, 1, 2, 3, 0, 1, 2]
    assert [r["source_tick"] for r in ticks] == [0, 0, 0, 0, 2, 2, 2]
    assert [r["observation_lag_ticks"] for r in ticks] == [0, 1, 2, 3, 2, 3, 4]
    assert all(r["model_request_id"].endswith(f"/S1-{r['request_id']}") for r in ticks)
    assert len({p["id"] for p in phases}) == len(phases)
    execution = [p for p in phases if p["phase"] == "execution"]
    assert [p["execution_id"] for p in execution] == list(range(7))
    assert [p["request_tick"] for p in execution] == [0, 0, 0, 0, 4, 4, 4]
    assert all(p["id"].endswith(f"/execution-{p['execution_id']}") for p in execution)
    assert all(
        p["raw"]["start_s"] - episode["controlled_start_s"] == p["start_s"]
        for p in phases
    )
    requests = queue["work_requests"]
    assert [r["request_id"] for r in requests] == [0, 4, 8]
    partial, unused = requests[1:]
    assert partial["accepted_prefix_steps"] == 4
    assert partial["executed_prefix_steps"] == 3
    assert partial["unexecuted_accepted_prefix_steps"] == 1
    assert partial["adopted_s"] < partial["finalized_s"]
    assert unused["target_tick"] > len(ticks)
    assert unused["accepted_prefix_steps"] == unused["executed_prefix_steps"] == 0
    assert unused["discarded_native_suffix_steps"] == 16
    assert episode["work_end_s"] > episode["controlled_end_s"]
    tasks = queue["replay_requests"]
    assert [t["feature_sequence"] for t in tasks if t["role"] == "S2"] == [0, 4, 8]
    assert tasks[-1]["status"] == "unused"
    assert tasks[-1]["end_s"] == unused["server_stages"]["action_completed_s"]
    assert tasks[-1]["end_s"] != unused["finalized_s"]
    assert queue["metrics"]["amortized_wall_seconds_per_action"] == pytest.approx(
        queue["duration_s"] / 7
    )
    assert queue["metrics"]["actual_control_hz"] == pytest.approx(
        7 / queue["duration_s"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(
            next(i for i, r in enumerate(rows) if r["kind"] == "chunked_async_request")
        ),
        lambda rows: [r for r in rows if r["kind"] == "chunked_async_execution"][
            -1
        ].update(executed_offset=0),
        lambda rows: [
            r for r in rows if r["kind"] == "cycle_phase" and r["phase"] == "execution"
        ][-1].update(execution_id=0),
        lambda rows: [r for r in rows if r["kind"] == "chunked_async_request"][
            -1
        ].update(accepted_prefix_steps=4),
        lambda rows: [r for r in rows if r["kind"] == "chunked_async_request"][
            1
        ].update(finalized_s=0),
    ],
)
def test_chunked_replay_rejects_missing_and_corrupt_raw_evidence(
    tmp_path, chunked_work, mutation
):
    result, events = chunked_work
    mutation(events)
    with pytest.raises(ValueError):
        chunked_catalog(tmp_path, result, events)


@pytest.mark.parametrize(
    "defaults",
    [
        {"default_execution_prefix_steps": 4},
        {"default_execution_prefix_steps": 4, "default_async_overlap_steps": 8},
        {"default_execution_prefix_steps": 8, "default_async_overlap_steps": 1},
        {"default_execution_prefix_steps": True, "default_async_overlap_steps": 1},
    ],
)
def test_catalog_rejects_missing_or_unmeasured_schedule_defaults(
    tmp_path, chunked_work, defaults
):
    with pytest.raises(ValueError, match="default K/D"):
        chunked_catalog(tmp_path, *chunked_work, **defaults)


def test_replay_schedule_rejects_configuration_evidence_disagreement():
    from coexecution.replay.deployment import _deployment_schedule

    episodes = [{"queue": {"execution_prefix_steps": 4, "async_overlap_steps": 2}}]
    with pytest.raises(ValueError, match="disagrees"):
        _deployment_schedule({"execution_prefix_steps": 8}, episodes, "chunked_async")
    with pytest.raises(ValueError, match="valid K/D"):
        _deployment_schedule({}, [], "chunked_async")
    assert _deployment_schedule({}, [], "asynchronous") == {
        "execution_prefix_steps": None,
        "async_overlap_steps": None,
    }
    assert _deployment_schedule({}, [], "synchronous") == {
        "execution_prefix_steps": None,
        "async_overlap_steps": 0,
    }


def test_chunked_replay_max_cap_preserves_short_accepted_final_prefix(tmp_path):
    from tests.contracts.test_chunked_async import scenario as chunked_scenario

    result, events, *_ = chunked_scenario(chunk=4, depth=2, max_steps=10)
    _, run = chunked_catalog(tmp_path, result, events)
    episode = run["episodes"][0]
    final = episode["queue"]["work_requests"][-1]
    assert final["request_id"] == 8
    assert final["accepted_prefix_steps"] == final["executed_prefix_steps"] == 2
    assert final["unexecuted_accepted_prefix_steps"] == 0
    assert final["discarded_native_suffix_steps"] == 14
    assert [t["executed_offset"] for t in episode["ticks"][-2:]] == [0, 1]
    assert not any(
        r["disposition"] == "unused" for r in episode["queue"]["work_requests"]
    )
