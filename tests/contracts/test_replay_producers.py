"""Real producer event contracts without model or GPU execution."""

from contextlib import nullcontext

import numpy as np
import pytest


def collector():
    records = []

    def emit(kind, **fields):
        records.append({"type": kind, **fields})
        return True

    return records, emit


def test_profile_stage_records_actual_host_scope_and_causal_feature_ids():
    from coexecution.replay.producer import ModelRequestEvents

    rows, sink = collector()
    clock = iter([1.0, 1.2, 1.3, 1.5, 1.7]).__next__
    event = ModelRequestEvents(
        sink,
        request_id="profile-r0-o0",
        observation_id="obs-actual",
        mode="w8a8-compiled",
        clock=clock,
    )
    with event.stage("vlm", nullcontext()):
        pass
    with event.stage("action", nullcontext()):
        pass
    event.completed()
    assert [row["type"] for row in rows] == [
        "model.stage.started",
        "model.stage.submitted",
        "model.feature.submitted",
        "model.stage.started",
        "model.feature.read",
        "model.stage.submitted",
        "model.request.completed",
    ]
    assert rows[2]["sources"]["feature_id"] == rows[4]["sources"]["feature_id"]
    assert rows[1]["occurrence"]["value"] == rows[2]["occurrence"]["value"] == 1.2
    assert rows[3]["occurrence"]["value"] == rows[4]["occurrence"]["value"] == 1.3
    assert rows[0]["payload"]["timing_boundary"] == "host_submission"
    assert rows[1]["payload"]["gpu_completion_observed"] is False
    assert rows[-1]["payload"]["gpu_completion_observed"] is True
    assert rows[-1]["occurrence"] == {
        "value": 1.7,
        "unit": "s",
        "clock_id": "host-monotonic",
        "domain": "host_monotonic",
    }


def test_disabled_model_events_do_not_read_clock_and_error_is_not_completion():
    from coexecution.replay.producer import ModelRequestEvents

    def forbidden():
        raise AssertionError("Clean path requested an event timestamp")

    event = ModelRequestEvents(
        None, request_id="r", observation_id="o", clock=forbidden
    )
    with event.stage("vlm", nullcontext()):
        pass
    rows, sink = collector()
    event = ModelRequestEvents(sink, request_id="r", observation_id="o")
    with (
        pytest.raises(RuntimeError, match="model failed"),
        event.stage("action", nullcontext()),
    ):
        raise RuntimeError("model failed")
    assert rows[-1]["type"] == "model.stage.failed"
    assert not any(row["type"] == "model.request.completed" for row in rows)


def test_queue_sink_preserves_emission_order_and_backdated_occurrence():
    from scripts.libero_queue_trace import QueueTrace

    rows, sink = collector()
    trace = QueueTrace(event_sink=sink, source_namespace="task0-episode0")
    trace.emit("feature_published", t_s=2.0, source_tick=3, feature_sequence=7)
    trace.emit("action_started", t_s=1.9, request_tick=4, feature_sequence=7)
    assert [row["payload"]["seq"] for row in rows] == [0, 1]
    assert [row["occurrence"]["value"] for row in rows] == [2.0, 1.9]
    assert rows[0]["sources"]["feature_id"] == rows[1]["sources"]["feature_id"]
    assert rows[1]["sources"]["request_id"] == "task0-episode0:request-4"
    assert trace.records()[1]["t_s"] == 1.9


def test_recorder_only_announces_owned_frames_and_real_media_after_encoding(tmp_path):
    from coexecution.replay.recording import ReplayRecorder

    rows, sink = collector()
    recorder = ReplayRecorder(
        max_frames=1, event_sink=sink, source_namespace="task0-episode0"
    )
    observation = {"video.image": np.zeros((4, 4, 3), dtype=np.uint8)}
    recorder.initial(observation, capture_s=1.0)
    assert rows[-1]["type"] == "robot.frame.recorded"
    assert rows[-1]["payload"]["media_status"] == "pending_encoding"
    assert not any(row["type"] == "robot.media.ready" for row in rows)

    def encode(path, frames, **kwargs):
        assert not any(row["type"] == "robot.media.ready" for row in rows)
        path.write_bytes(b"encoded-actual-fixture")

    recorder.finish(tmp_path, "episode", encoder=encode)
    assert rows[-1]["type"] == "robot.media.ready"
    assert rows[-1]["sources"]["episode_id"] == "task0-episode0"
    assert "path" not in rows[-1]["payload"]
    assert rows[-1]["payload"]["cameras"] == ["image"]


def test_robot_source_ids_distinguish_executed_request_and_post_step_observation():
    from coexecution.replay.producer import queue_sources
    from coexecution.replay.recording import ReplayRecorder

    source = queue_sources(
        {
            "kind": "control_tick",
            "tick": 7,
            "executed_slot": {"producer_request_tick": 3},
        },
        "episode",
    )
    assert source["request_id"] == "episode:request-3"
    assert source["observation_id"] == "episode:observation-7"
    rows, sink = collector()
    recorder = ReplayRecorder(event_sink=sink, source_namespace="episode")
    observation = {"video.image": np.zeros((4, 4, 3), dtype=np.uint8)}
    recorder.initial(observation, capture_s=1)
    recorder.step(
        observation,
        observation,
        {
            "tick": 7,
            "sensor_capture_s": 2,
            "sensor_release_s": 2,
            "applied_s": 2,
            "next_observation_computed_s": 3,
        },
    )
    assert rows[0]["sources"]["observation_id"] == "episode:observation-0"
    assert rows[1]["sources"]["observation_id"] == "episode:observation-8"


def test_recorder_without_real_frames_does_not_announce_ready_media(tmp_path):
    from coexecution.replay.recording import ReplayRecorder

    rows, sink = collector()
    recorder = ReplayRecorder(event_sink=sink)
    recorder.finish(
        tmp_path,
        "empty",
        encoder=lambda *_args, **_kwargs: pytest.fail("No frame exists"),
    )
    assert rows[-1]["type"] == "robot.media.unavailable"
    assert rows[-1]["payload"]["media_status"] == "unavailable"


def test_profile_collector_streams_real_engine_stages_and_completion(
    monkeypatch, tmp_path
):
    import json
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    from coexecution.libero import measurement, profiling

    rows, sink = collector()
    model_calls = []

    class Profile:
        def __enter__(self):
            return self

        def __exit__(self, *_error):
            pass

        def export_chrome_trace(self, filename):
            filename = __import__("pathlib").Path(filename)
            filename.write_text(json.dumps({"traceEvents": []}))

    @contextmanager
    def annotation(_name):
        yield

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda _device: model_calls.append("sync")),
        profiler=SimpleNamespace(
            profile=lambda **_kwargs: Profile(),
            record_function=annotation,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(measurement, "_seed_reset", lambda *_args: None)
    monkeypatch.setattr(measurement, "_prepare", lambda *_args: ("pixels", "state", {}))
    monkeypatch.setattr(
        measurement,
        "graph_snapshot",
        lambda _session: {key: None for key in ("vlm", "dit", "action")},
    )
    model = SimpleNamespace(
        backbone=lambda value: model_calls.append("vlm") or "features",
        action_head=SimpleNamespace(
            get_action=lambda *_args: model_calls.append("action") or "prediction"
        ),
    )
    policy = SimpleNamespace(model=model, get_action=lambda _obs: None)
    result = profiling.collect_model_profile(
        policy,
        [{}],
        object(),
        tmp_path / "trace.json",
        rounds=1,
        seed=3,
        device="fake-cuda",
        observation_ids=["actual-observation"],
        event_sink=sink,
        mode="bf16-compiled",
    )
    assert result["calls"] == 1
    assert model_calls.count("vlm") == model_calls.count("action") == 1
    starts = [row for row in rows if row["type"] == "model.stage.started"]
    assert [row["payload"]["stage"] for row in starts] == [
        "preparation_transfer",
        "vlm",
        "action",
    ]
    assert rows[-1]["type"] == "model.request.completed"
    assert rows[-1]["payload"]["output_available"] is False
    assert rows[-1]["sources"]["request_id"] == result["requests"][0]["request_id"]


@pytest.mark.parametrize("enabled", [False, True])
def test_model_only_event_sink_emits_receipt_only_when_enabled(
    monkeypatch, tmp_path, enabled
):
    import sys
    from io import StringIO
    from types import SimpleNamespace

    from coexecution.libero import measurement, runtime_receipt

    monkeypatch.setitem(sys.modules, "torch", object())
    receipt = {
        "identity_sha256": "metadata-fixture",
        "validation": {"status": "pending"},
    }
    monkeypatch.setattr(
        runtime_receipt,
        "capture_runtime_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        measurement,
        "_model_only_mode",
        lambda *_args, **_kwargs: {
            "samples": [],
            "gates": [],
            "graph_audits": [],
            "outputs": {},
            "retained": {},
        },
    )
    args = SimpleNamespace(
        modes=["graph"],
        measurement_boundary="model-only",
        runtime_receipt=enabled,
        output=tmp_path,
    )
    rows, sink = collector()
    result = measurement._model_only_precision(
        object(),
        SimpleNamespace(set_runtime=lambda **_kwargs: None, evidence=dict),
        args,
        "bf16",
        [],
        [],
        [],
        StringIO(),
        event_sink=sink,
    )
    if enabled:
        assert result[3]["bf16-graph"]["runtime_receipt"] == receipt
        assert [row["type"] for row in rows] == ["runtime.receipt"]
        assert rows[0]["payload"]["runtime_receipt"] == receipt
    else:
        assert result[3]["bf16-graph"] == {}
        assert rows == []
