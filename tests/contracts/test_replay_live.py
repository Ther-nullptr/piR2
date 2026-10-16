"""Live registration and real metadata routing, with no model or GPU calls."""

import json
import subprocess
import sys

import numpy as np
import pytest


def arguments(tmp_path, *, precisions=("bf16", "w8a8")):
    from coexecution.libero.measurement import build_parser

    return build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--dataset",
            str(tmp_path / "dataset"),
            "--output",
            str(tmp_path / "model"),
            "--modes",
            "kv_reuse",
            "--action-cuda-graph",
            "--precisions",
            *precisions,
            "--observations",
            "1",
            "--rounds",
            "1",
            "--measurement-boundary",
            "model-only",
        ]
    )


def fake_measurement(args, *, event_sink):
    from coexecution.replay.producer import emit_event, emit_sample

    args.output.mkdir()
    observation = {"observation_id": "obs-real", "sha256": "a" * 64}
    emit_event(
        event_sink,
        "observation.loaded",
        t_s=1.0,
        sources={"observation_id": "obs-real"},
        payload=observation,
    )
    samples, actions, evidence = [], {}, {}
    for precision in args.precisions:
        mode = precision + "-kv_reuse"
        request = f"r0/{mode}/o0/prepared_model"
        row = {
            "request_id": request,
            "observation_id": "obs-real",
            "precision": precision,
            "mode": mode,
            "scope": "prepared_model",
            "round": 0,
            "seed": 17,
            "wall_ms": 12.5,
            "instrumented": False,
        }
        actual = {"action.x": np.array([[[1.0], [2.0]]], dtype=np.float32)}
        emit_sample(event_sink, row, completed_s=2.0, actions=actual)
        samples.append(row)
        actions.update({request + "/" + key: value for key, value in actual.items()})
        evidence[mode] = {
            "runtime_receipt": {
                "schema": "pir2.runtime.receipt",
                "version": 1,
                "identity_sha256": precision,
                "validation": {"status": "pending"},
            }
        }
        emit_event(
            event_sink, "runtime.receipt", payload={"mode": mode, **evidence[mode]}
        )
    result = {
        "observations": [observation],
        "profiles": {},
        "arguments": {},
        "mode_evidence": evidence,
    }
    (args.output / "results.json").write_text(json.dumps(result))
    (args.output / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    np.savez(args.output / "actions.npz", **actions)
    return result


def test_registers_before_execution_then_canonical_outputs_and_pending_receipt(
    tmp_path, monkeypatch
):
    from coexecution.libero import measurement
    from coexecution.replay import live
    from coexecution.replay.model import model_run_id

    args, catalog_root = arguments(tmp_path), tmp_path / "catalog"
    caller_args = args

    def run(args, *, event_sink):
        assert args is not caller_args
        assert args.runtime_receipt is True
        catalog = json.loads((catalog_root / "catalog.json").read_text())
        assert len(catalog["streams"]) == len(catalog["runs"]) == 2
        assert not args.output.exists()
        for entry in catalog["streams"]:
            snapshot = json.loads((catalog_root / entry["snapshot_path"]).read_text())
            assert snapshot["run"]["status"] == "running"
            assert snapshot["run"]["runtime_receipt"] is None
            assert entry["max_event_bytes"] == 8 * 1024 * 1024
        return fake_measurement(args, event_sink=event_sink)

    monkeypatch.setattr(measurement, "run", run)
    catalog = live.record_model(args, catalog_output=catalog_root)
    assert args.runtime_receipt is False
    for entry, stream in zip(catalog["runs"], catalog["streams"]):
        canonical = json.loads((catalog_root / entry["path"]).read_text())
        snapshot = json.loads((catalog_root / stream["snapshot_path"]).read_text())
        events = [
            json.loads(line)
            for line in (catalog_root / stream["journal_path"]).read_text().splitlines()
        ]
        assert canonical == snapshot["run"]
        assert canonical["id"] == model_run_id(args.output, canonical["mode"])
        assert canonical["status"] == "complete"
        assert canonical["runtime_receipt"]["validation"]["status"] == "pending"
        assert canonical["clean_requests"][0]["output"]["actions"]["action.x"][
            "values"
        ] == [[[1.0], [2.0]]]
        assert snapshot["state"]["status"] == "ended"
        assert events[-1]["type"] == "run.ended"
        assert any(
            row.get("projection", {}).get("collection") == "clean_requests"
            for row in events
            if row.get("projection")
        )
        assert all(
            row["payload"].get("mode") in (None, canonical["mode"]) for row in events
        )


def test_failure_preserves_real_partial_inputs_and_reports_failed_terminal(
    tmp_path, monkeypatch
):
    from coexecution.libero import measurement
    from coexecution.replay import live
    from coexecution.replay.producer import emit_event

    def fail(args, *, event_sink):
        args.output.mkdir()
        emit_event(
            event_sink,
            "observation.loaded",
            t_s=4.0,
            sources={"observation_id": "actual-input"},
            payload={"observation_id": "actual-input", "sha256": "b" * 64},
        )
        raise RuntimeError("fixture model failure")

    monkeypatch.setattr(measurement, "run", fail)
    output = tmp_path / "catalog"
    with pytest.raises(RuntimeError, match="fixture model failure"):
        live.record_model(arguments(tmp_path), catalog_output=output)
    catalog = json.loads((output / "catalog.json").read_text())
    for entry in catalog["streams"]:
        snapshot = json.loads((output / entry["snapshot_path"]).read_text())
        assert snapshot["state"]["status"] == "failed"
        assert snapshot["run"]["observations"][0]["observation_id"] == "actual-input"
        assert snapshot["run"]["status"] == "failed"
        assert snapshot["run"]["runtime_receipt"] is None


def test_record_model_rejects_multiple_modes_and_existing_measurements_before_run(
    tmp_path, monkeypatch
):
    from coexecution.libero import measurement
    from coexecution.replay import live

    monkeypatch.setattr(
        measurement,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Must fail before model execution"),
    )
    args = arguments(tmp_path)
    args.modes.append("baseline")
    with pytest.raises(ValueError, match="one mode"):
        live.record_model(args, catalog_output=tmp_path / "catalog")
    args.modes = ["kv_reuse"]
    args.output.mkdir()
    with pytest.raises(ValueError, match="new measurement"):
        live.record_model(args, catalog_output=tmp_path / "catalog")


def test_cli_help_exposes_measurement_flags_without_loading_neural_modules():
    code = "from coexecution.replay.__main__ import main; import sys;\ntry: main(['record-model', '--help'])\nexcept SystemExit as e: assert e.code == 0\nassert not {'torch', 'numpy', 'av'} & sys.modules.keys()"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "--catalog-output" in result.stdout and "kv_reuse" in result.stdout
    assert "--quantization-shape-tactics" in result.stdout


def test_actual_asset_publication_and_append_preserve_prior_catalog(
    tmp_path, monkeypatch
):
    from coexecution.libero import measurement
    from coexecution.replay import live
    from coexecution.replay.producer import emit_event

    args = arguments(tmp_path, precisions=("bf16",))
    args.replay_assets = True

    def with_assets(args, *, event_sink):
        def sink(kind, **fields):
            accepted = event_sink(kind, **fields)
            if kind == "observation.loaded":
                (args.output / "actual.png").write_bytes(b"owned-fixture-pixels")
                asset = {
                    **fields["payload"],
                    "state": {"x": [[1.0]]},
                    "assets": [
                        {
                            "camera": "image",
                            "frame_index": 0,
                            "path": "actual.png",
                            "mime": "image/png",
                        }
                    ],
                }
                (args.output / "observations-replay.json").write_text(
                    json.dumps([asset])
                )
                emit_event(
                    event_sink,
                    "observation.assets_ready",
                    t_s=1.1,
                    sources=fields["sources"],
                    payload={"observation": asset},
                )
            return accepted

        return fake_measurement(args, event_sink=sink)

    monkeypatch.setattr(measurement, "run", with_assets)
    output = tmp_path / "catalog"
    first = live.record_model(args, catalog_output=output)
    manifest = json.loads((output / "media.json").read_text())
    assert len(manifest["assets"]) == 1
    assert (
        output / manifest["assets"][0]["path"]
    ).read_bytes() == b"owned-fixture-pixels"
    events = [
        json.loads(line)
        for line in (output / first["streams"][0]["journal_path"])
        .read_text()
        .splitlines()
    ]
    actual = next(row for row in events if row["type"] == "observation.assets_ready")
    assert (
        actual["projection"]["items"][0]["assets"][0]["media_id"]
        == manifest["assets"][0]["id"]
    )
    assert "path" not in actual["payload"]["observation"]["assets"][0]
    args.output = tmp_path / "model-next"
    args.replay_assets = False
    monkeypatch.setattr(measurement, "run", fake_measurement)
    second = live.record_model(args, catalog_output=output)
    assert len(second["runs"]) == len(second["streams"]) == 2
    assert json.loads((output / "media.json").read_text()) == manifest


def test_attestation_is_explicit_and_cannot_change_measured_identity(
    tmp_path, monkeypatch
):
    from coexecution.libero import measurement
    from coexecution.replay import live

    monkeypatch.setattr(measurement, "run", fake_measurement)
    args = arguments(tmp_path, precisions=("bf16",))
    output = tmp_path / "catalog"

    def wrong_identity(run):
        assert run["status"] == "complete"
        return {**run["runtime_receipt"], "identity_sha256": "different"}

    with pytest.raises(ValueError, match="exact measured identity"):
        live.record_model(args, catalog_output=output, attest=wrong_identity)
    catalog = json.loads((output / "catalog.json").read_text())
    snapshot = json.loads((output / catalog["streams"][0]["snapshot_path"]).read_text())
    assert snapshot["state"]["status"] == "failed"
    assert snapshot["run"]["runtime_receipt"]["validation"]["status"] == "pending"


@pytest.mark.parametrize("mutation", ["body", "proof_link", "missing_evidence", None])
def test_attestation_changes_only_validation_and_requires_linked_gate(
    tmp_path, monkeypatch, mutation
):
    from coexecution.libero import measurement
    from coexecution.replay import live

    monkeypatch.setattr(measurement, "run", fake_measurement)

    def attest(run):
        receipt = {
            **run["runtime_receipt"],
            "validation": {
                "status": "gpu_verified",
                "runtime_identity_sha256": run["runtime_receipt"]["identity_sha256"],
                "evidence_sha256": "c" * 64,
            },
        }
        if mutation == "body":
            receipt["configuration"] = {"unmeasured": True}
        elif mutation == "proof_link":
            receipt["validation"]["runtime_identity_sha256"] = "unmatched"
        elif mutation == "missing_evidence":
            receipt["validation"]["evidence_sha256"] = None
        return receipt

    args, output = arguments(tmp_path, precisions=("bf16",)), tmp_path / "catalog"
    if mutation is not None:
        with pytest.raises(ValueError, match="attestation"):
            live.record_model(args, catalog_output=output, attest=attest)
    else:
        catalog = live.record_model(args, catalog_output=output, attest=attest)
        run = json.loads((output / catalog["runs"][0]["path"]).read_text())
        assert run["runtime_receipt"]["validation"]["status"] == "gpu_verified"


def test_failed_instrumented_invocation_remains_visible_without_gpu_timeline(
    tmp_path, monkeypatch
):
    from coexecution.libero import measurement
    from coexecution.replay import live
    from coexecution.replay.producer import emit_event

    def fail(args, *, event_sink):
        args.output.mkdir()
        sources = {"observation_id": "actual-input", "request_id": "profile-r0-o0"}
        emit_event(
            event_sink,
            "observation.loaded",
            t_s=1.0,
            sources={"observation_id": "actual-input"},
            payload={"observation_id": "actual-input", "sha256": "b" * 64},
        )
        emit_event(
            event_sink,
            "model.request.started",
            t_s=2.0,
            sources=sources,
            payload={"mode": "bf16-kv_reuse", "instrumented": True, "seed": 17},
        )
        emit_event(
            event_sink,
            "model.stage.failed",
            t_s=2.2,
            sources=sources,
            payload={
                "mode": "bf16-kv_reuse",
                "stage": "vlm",
                "error": "actual failure",
            },
        )
        raise RuntimeError("actual failure")

    monkeypatch.setattr(measurement, "run", fail)
    output = tmp_path / "catalog"
    with pytest.raises(RuntimeError, match="actual failure"):
        live.record_model(
            arguments(tmp_path, precisions=("bf16",)), catalog_output=output
        )
    catalog = json.loads((output / "catalog.json").read_text())
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    request = run["profile_requests"][0]
    assert request["live_status"] == "failed"
    assert request["host_started"]["value"] == 2.0
    assert request["host_failed"]["value"] == 2.2
    assert request["timeline"] == [] and request["host_wall_ms"] is None
    assert request["output"]["status"] == "unavailable"


@pytest.mark.parametrize(
    ("boundary", "scope"),
    [("model-only", "prepared_model"), ("comparison", "preparation_model")],
)
def test_live_profile_projects_observed_host_stages_before_completion(
    tmp_path, boundary, scope
):
    from contextlib import nullcontext

    from coexecution.replay.http import ReplayFiles
    from coexecution.replay.live import _Recording
    from coexecution.replay.producer import ModelRequestEvents, emit_event

    args = arguments(tmp_path, precisions=("bf16",))
    args.measurement_boundary = boundary
    recording = _Recording(args, tmp_path / "catalog", None)
    mode = "bf16-kv_reuse"
    files = ReplayFiles(tmp_path / "catalog")
    run_id = recording.runs[mode]["id"]

    def request():
        recording.writers[mode].flush()
        rows = files.snapshot(run_id)["run"]["profile_requests"]
        assert len(rows) == 1
        return rows[0]

    try:
        emit_event(
            recording.emit,
            "model.request.started",
            t_s=10.0,
            sources={"request_id": "profile-r0-o0", "observation_id": "obs-0"},
            payload={"mode": mode, "instrumented": True, "seed": 17},
        )
        initial = request()
        assert initial["scope"] == scope
        assert initial["host_progress_ms"] == 0
        assert initial["stage_spans"] == []
        assert initial["host_wall_ms"] is None
        assert initial["output"]["status"] == "unavailable"

        events = ModelRequestEvents(
            recording.emit,
            request_id="profile-r0-o0",
            observation_id="obs-0",
            mode=mode,
            clock=iter([10.25, 10.5, 10.75]).__next__,
        )
        with events.stage("vlm", nullcontext()):
            running = request()
            span = running["stage_spans"][0]
            assert span["start_ms"] == 250
            assert span["end_ms"] is None and span["status"] == "running"
            assert span["domain"] == "host_submit"
            assert span["clock_id"] == "host-monotonic"
            assert running["host_progress_ms"] == 250
        submitted = request()
        assert submitted["stage_spans"][0]["end_ms"] == 500
        assert submitted["stage_spans"][0]["status"] == "submitted"
        events.completed()
        completed = request()
        assert completed["id"] == initial["id"]
        assert completed["host_progress_ms"] == 750
        assert completed["live_status"] == "cuda_completion_observed"
        assert completed["host_wall_ms"] is None and completed["timeline"] == []
        assert completed["output"]["status"] == "unavailable"
    finally:
        recording.close()


def test_live_profile_stage_failure_updates_existing_request(tmp_path):
    from contextlib import nullcontext

    from coexecution.replay.http import ReplayFiles
    from coexecution.replay.live import _Recording
    from coexecution.replay.producer import ModelRequestEvents, emit_event

    recording = _Recording(
        arguments(tmp_path, precisions=("bf16",)), tmp_path / "catalog", None
    )
    mode = "bf16-kv_reuse"
    try:
        emit_event(
            recording.emit,
            "model.request.started",
            t_s=1.0,
            sources={"request_id": "profile-r0-o0", "observation_id": "obs-0"},
            payload={"mode": mode, "instrumented": True},
        )
        events = ModelRequestEvents(
            recording.emit,
            request_id="profile-r0-o0",
            observation_id="obs-0",
            mode=mode,
            clock=iter([1.25, 1.5]).__next__,
        )
        with (
            pytest.raises(RuntimeError, match="stage failure"),
            events.stage("action", nullcontext()),
        ):
            raise RuntimeError("stage failure")
        recording.writers[mode].flush()
        run = ReplayFiles(tmp_path / "catalog").snapshot(recording.runs[mode]["id"])[
            "run"
        ]
        assert len(run["profile_requests"]) == 1
        request = run["profile_requests"][0]
        assert request["live_status"] == "failed"
        assert request["host_progress_ms"] == 500
        assert request["stage_spans"][0]["end_ms"] == 500
        assert request["stage_spans"][0]["status"] == "failed"
        assert request["device_span_ms"] is None
    finally:
        recording.close()


def test_recording_preserves_original_failure_without_python311_add_note(
    tmp_path, monkeypatch
):
    from coexecution.libero import measurement
    from coexecution.replay import live

    class OriginalFailure(RuntimeError):
        add_note = None

    original = OriginalFailure("original model failure")

    def fail(*_args, **_kwargs):
        raise original

    def fail_cleanup(*_args, **_kwargs):
        raise ValueError("partial artifact failure")

    monkeypatch.setattr(measurement, "run", fail)
    monkeypatch.setattr(live._Recording, "finish", fail_cleanup)
    with pytest.raises(OriginalFailure) as captured:
        live.record_model(
            arguments(tmp_path, precisions=("bf16",)),
            catalog_output=tmp_path / "catalog",
        )
    assert captured.value is original
    assert "partial artifact failure" in original.__notes__[0]


def test_new_recording_catalog_defaults_to_model_profile(tmp_path):
    from coexecution.replay.live import _Recording

    args = arguments(tmp_path, precisions=("bf16",))
    recording = _Recording(args, tmp_path / "catalog", None)
    try:
        assert recording.catalog["default_view"] == "model"
        assert recording.catalog["default_record_type"] == "profile"
        recording.catalog.update(default_view="deployment", default_record_type="clean")
        recording.persist()
    finally:
        recording.close()
    args.output = tmp_path / "next-model"
    following = _Recording(args, tmp_path / "catalog", None)
    try:
        assert following.catalog["default_view"] == "deployment"
        assert following.catalog["default_record_type"] == "clean"
    finally:
        following.close()


@pytest.mark.parametrize("started", [None, 1.0])
def test_live_completion_never_fills_missing_stage_timestamps(tmp_path, started):
    from coexecution.replay.http import ReplayFiles
    from coexecution.replay.live import _Recording
    from coexecution.replay.producer import emit_event

    recording = _Recording(
        arguments(tmp_path, precisions=("bf16",)), tmp_path / "catalog", None
    )
    mode = "bf16-kv_reuse"
    sources = {"request_id": "profile-r0-o0", "observation_id": "obs-0"}
    payload = {"mode": mode, "instrumented": True}
    try:
        emit_event(
            recording.emit,
            "model.request.started",
            t_s=started,
            sources=sources,
            payload=payload,
        )
        emit_event(
            recording.emit,
            "model.stage.started",
            t_s=1.25,
            sources=sources,
            payload={**payload, "stage": "vlm", "timing_boundary": "host_submission"},
        )
        emit_event(
            recording.emit,
            "model.request.completed",
            t_s=1.5,
            sources=sources,
            payload={**payload, "gpu_completion_observed": True},
        )
        recording.writers[mode].flush()
        request = ReplayFiles(tmp_path / "catalog").snapshot(
            recording.runs[mode]["id"]
        )["run"]["profile_requests"][0]
        assert request["host_wall_ms"] is None
        if started is None:
            assert request["host_progress_ms"] is None
            assert request["stage_spans"] == []
        else:
            assert request["host_progress_ms"] == 500
            assert request["stage_spans"][0]["end_ms"] is None
            assert request["stage_spans"][0]["status"] == "incomplete"
    finally:
        recording.close()


def test_final_canonical_profile_replaces_live_host_projection(tmp_path, monkeypatch):
    from copy import deepcopy

    from coexecution.replay import live
    from coexecution.replay.http import ReplayFiles
    from coexecution.replay.producer import emit_event

    recording = live._Recording(
        arguments(tmp_path, precisions=("bf16",)), tmp_path / "catalog", None
    )
    mode = "bf16-kv_reuse"
    try:
        emit_event(
            recording.emit,
            "model.request.started",
            t_s=10.0,
            sources={"request_id": "profile-r0-o0", "observation_id": "obs-0"},
            payload={"mode": mode, "instrumented": True},
        )
        canonical = deepcopy(recording.runs[mode])
        request = canonical["profile_requests"][0]
        request.pop("stage_spans")
        request.pop("host_progress_ms")
        request.update(
            host_wall_ms=40.0,
            timeline_status="recorded",
            timeline=[{"stage": "vlm", "start_ms": 2.0, "duration_ms": 15.0}],
        )
        canonical["status"] = "complete"
        monkeypatch.setattr(live, "model_runs", lambda *_args, **_kwargs: [canonical])
        recording.finish()
        recording.writers[mode].flush()
        actual = ReplayFiles(tmp_path / "catalog").snapshot(canonical["id"])["run"]
        assert actual == canonical
        assert "host_progress_ms" not in actual["profile_requests"][0]
    finally:
        recording.close()
