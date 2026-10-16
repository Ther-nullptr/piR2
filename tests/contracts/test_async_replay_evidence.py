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
