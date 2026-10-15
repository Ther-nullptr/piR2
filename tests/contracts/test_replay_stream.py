"""Append-only replay transport contracts; no model, CUDA, or array imports."""

import json
import threading

import pytest

from coexecution.replay.stream import (
    BoundedEventWriter,
    ReplayJournal,
    StreamGap,
    StreamTail,
    apply_event,
    validate_event,
)


def run():
    return {
        "schema": "pir2.replay.run",
        "version": 1,
        "id": "run-a",
        "kind": "model",
        "precision": "w8a8",
        "mode": "w8a8-clean",
        "observations": [],
        "clean_requests": [],
        "profile_requests": [],
        "kernels": [],
    }


CLOCKS = {"host-monotonic": {"unit": "s", "domain": "host_monotonic"}}


def occurrence(value=12.0):
    return {
        "value": value,
        "unit": "s",
        "domain": "host_monotonic",
        "clock_id": "host-monotonic",
    }


def append_observation(journal, name="obs-1", when=12.0):
    return journal.append(
        "observation.captured",
        occurrence=occurrence(when),
        sources={"observation_id": name},
        projection={
            "op": "append",
            "collection": "observations",
            "items": [{"observation_id": name, "sha256": None}],
        },
    )


def test_journal_projects_original_normalized_rows_and_keeps_occurrence_clock(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS, checkpoint_every=2)
    first = append_observation(journal)
    second = journal.append(
        "model.stage.started",
        occurrence=occurrence(11.5),
        sources={"request_id": "run-a/request-1"},
        payload={"stage": "vlm", "end": None},
    )
    assert first["sequence"] == 1 and second["sequence"] == 2
    assert second["occurrence"]["value"] < first["occurrence"]["value"]
    snapshot = journal.snapshot()
    assert snapshot["run"]["observations"] == [
        {"observation_id": "obs-1", "sha256": None}
    ]
    assert snapshot["last_sequence"] == 2
    assert snapshot["stream_available"] is True
    assert json.loads(journal.snapshot_path.read_text()) == snapshot
    assert len(journal.journal_path.read_text().splitlines()) == 2
    journal.close()


def test_immutable_append_collision_and_replace_identity_are_rejected_before_commit(
    tmp_path,
):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    append_observation(journal)
    with pytest.raises(ValueError, match="identity"):
        append_observation(journal)
    for key, value in (
        ("id", "another-run"),
        ("kind", "deployment"),
        ("precision", "bf16"),
        ("schema", "other"),
    ):
        with pytest.raises(ValueError):
            journal.append(
                "run.updated",
                occurrence=occurrence(),
                projection={"op": "replace", "run": run() | {key: value}},
            )
    assert journal.snapshot()["last_sequence"] == 1
    journal.close()


def test_event_rejects_wrong_run_clock_nonfinite_and_unknown_projection(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    event = append_observation(journal)
    for changes in (
        {"run_id": "other"},
        {"occurrence": occurrence() | {"unit": "ms"}},
        {"occurrence": occurrence() | {"domain": "cuda_event"}},
        {"occurrence": occurrence(float("nan"))},
    ):
        with pytest.raises(ValueError):
            validate_event(event | changes, "run-a", journal.snapshot()["clocks"])
    with pytest.raises(ValueError, match="collection"):
        journal.append(
            "run.updated",
            occurrence=occurrence(),
            projection={"op": "append", "collection": "__proto__", "items": []},
        )
    journal.close()


def test_reducer_duplicate_sequence_does_not_duplicate_normalized_evidence(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    initial = journal.snapshot()
    event = append_observation(journal)
    changed = apply_event(initial, event)
    duplicate = apply_event(changed, event)
    assert duplicate == changed
    with pytest.raises(StreamGap):
        apply_event(initial, event | {"sequence": 3})
    journal.close()


def test_registration_preserves_explicit_eight_mib_limit_and_rejects_unbounded_limits(
    tmp_path,
):
    journal = ReplayJournal(tmp_path, run(), CLOCKS, max_event_bytes=8 * 1024 * 1024)
    assert journal.registration(tmp_path)["max_event_bytes"] == 8 * 1024 * 1024
    journal.close()
    with pytest.raises(ValueError, match="limit"):
        ReplayJournal(
            tmp_path / "invalid", run(), CLOCKS, max_event_bytes=8 * 1024 * 1024 + 1
        )
    with pytest.raises(ValueError, match="limit"):
        StreamTail(lambda: None, "run-a", CLOCKS, max_event_bytes=True)


def test_tail_bounds_each_poll_by_bytes_without_skipping_committed_records(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS, max_event_bytes=1024)
    rows = [
        journal.append(
            "metadata", occurrence=occurrence(), payload={"value": "x" * 300}
        )
        for _ in range(3)
    ]
    tail = StreamTail(
        lambda: journal.journal_path.open("rb"),
        "run-a",
        journal.snapshot()["clocks"],
        max_event_bytes=1024,
    )
    seen = []
    for _ in range(3):
        batch = tail.poll(max_bytes=1024)
        assert len(batch) == 1
        seen.extend(batch)
    assert seen == rows
    journal.close()


def test_tail_ignores_partial_line_and_resumes_exactly(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    first = append_observation(journal)
    tail = StreamTail(
        lambda: journal.journal_path.open("rb"),
        "run-a",
        journal.snapshot()["clocks"],
        after=0,
    )
    assert tail.poll() == [first]
    second = first | {"sequence": 2, "type": "model.stage.started", "projection": None}
    encoded = json.dumps(second).encode()
    with journal.journal_path.open("ab") as stream:
        stream.write(encoded[:20])
    assert tail.poll() == []
    with journal.journal_path.open("ab") as stream:
        stream.write(encoded[20:] + b"\n")
    assert tail.poll() == [second]
    assert tail.poll() == []
    journal.close()


def test_tail_duplicate_conflict_gap_future_cursor_and_replacement(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    event = append_observation(journal)
    tail = StreamTail(
        lambda: journal.journal_path.open("rb"),
        "run-a",
        journal.snapshot()["clocks"],
        after=0,
    )
    assert tail.poll() == [event]
    with journal.journal_path.open("ab") as stream:
        stream.write(json.dumps(event).encode() + b"\n")
    assert tail.poll() == []
    with journal.journal_path.open("ab") as stream:
        stream.write(
            json.dumps(event | {"payload": {"changed": True}}).encode() + b"\n"
        )
    with pytest.raises(StreamGap, match="duplicate"):
        tail.poll()
    journal.close()
    other = ReplayJournal(tmp_path / "other", run(), CLOCKS)
    append_observation(other)
    future = StreamTail(
        lambda: other.journal_path.open("rb"),
        "run-a",
        other.snapshot()["clocks"],
        after=9,
    )
    with pytest.raises(StreamGap, match="ahead"):
        future.poll()
    reader = StreamTail(
        lambda: other.journal_path.open("rb"),
        "run-a",
        other.snapshot()["clocks"],
        after=0,
    )
    reader.poll()
    other.journal_path.write_bytes(b"")
    with pytest.raises(StreamGap, match="truncated"):
        reader.poll()
    other.close()


def test_queue_full_loss_is_persisted_even_if_close_is_next_operation(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    entered, release = threading.Event(), threading.Event()
    original = journal.append

    def gated(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(2)
        return original(*args, **kwargs)

    journal.append = gated
    writer = BoundedEventWriter(journal, max_pending=1)
    try:
        assert writer.emit(
            "model.stage.started", occurrence=occurrence(), payload={"stage": "vlm"}
        )
        assert entered.wait(2)
        assert writer.emit("model.stage.submitted", occurrence=occurrence(13)) is False
        assert writer.emit("model.stage.submitted", occurrence=occurrence(14)) is False
        release.set()
        writer.close(status="ended", occurrence=occurrence(15))
        snapshot = journal.snapshot()
        assert snapshot["state"]["dropped_events"] == 2
        assert snapshot["state"]["status"] == "ended"
        rows = [
            json.loads(line) for line in journal.journal_path.read_text().splitlines()
        ]
        assert (
            sum(
                row["payload"]["count"]
                for row in rows
                if row["type"] == "stream.dropped"
            )
            == 2
        )
        assert rows[-1]["type"] == "run.ended"
        assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    finally:
        release.set()
        writer.close()


def test_close_without_terminal_status_does_not_invent_run_completion(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    writer = BoundedEventWriter(journal)
    writer.emit(
        "model.stage.started",
        occurrence=occurrence(),
        payload={"stage": "action", "duration": None},
    )
    writer.close()
    assert journal.snapshot()["state"]["status"] == "running"


def test_flush_waits_for_inflight_and_loss_then_allows_final_projection(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    entered, release = threading.Event(), threading.Event()
    loss_entered, loss_release = threading.Event(), threading.Event()
    flushed = threading.Event()
    original = journal.append

    def gated(kind, **kwargs):
        if kind == "model.stage.started":
            entered.set()
            assert release.wait(2)
        if kind == "stream.dropped":
            loss_entered.set()
            assert loss_release.wait(2)
        return original(kind, **kwargs)

    journal.append = gated
    writer = BoundedEventWriter(journal, max_pending=1)
    errors = []

    def flush():
        try:
            writer.flush(timeout=2)
            flushed.set()
        except Exception as error:  # noqa: BLE001 - return test thread failure
            errors.append(error)

    waiter = threading.Thread(target=flush)
    try:
        assert writer.emit("model.stage.started", occurrence=occurrence())
        assert entered.wait(2)
        assert not writer.emit("model.stage.submitted", occurrence=occurrence())
        with pytest.raises(TimeoutError):
            writer.flush(timeout=0)
        release.set()
        assert loss_entered.wait(2)
        with pytest.raises(TimeoutError):
            writer.flush(timeout=0)
        waiter.start()
        loss_release.set()
        assert flushed.wait(2)
        assert not errors
        assert journal.snapshot()["state"]["dropped_events"] == 1
        assert writer.emit(
            "run.updated",
            occurrence=occurrence(),
            projection={"op": "replace", "run": run() | {"status": "complete"}},
        )
        writer.flush(timeout=2)
        assert journal.snapshot()["run"]["status"] == "complete"
        assert journal.snapshot()["state"]["status"] == "running"
    finally:
        release.set()
        loss_release.set()
        if waiter.ident is not None:
            waiter.join(2)
        writer.close()
    writer.flush(timeout=0)


def test_flush_surfaces_writer_failure_and_validates_timeout(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    writer = BoundedEventWriter(journal)
    for timeout in (-1, float("nan"), True):
        with pytest.raises(ValueError, match="timeout"):
            writer.flush(timeout=timeout)
    writer.emit("model.stage.started", occurrence=occurrence() | {"unit": "ms"})
    with pytest.raises(RuntimeError, match="recording failed"):
        writer.flush(timeout=2)
    with pytest.raises(RuntimeError, match="recording failed"):
        writer.close()


def test_writer_failure_is_visible_and_does_not_claim_model_failure(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    writer = BoundedEventWriter(journal)
    writer.emit("model.stage.started", occurrence=occurrence() | {"unit": "ms"})
    with pytest.raises(RuntimeError, match="recording failed"):
        writer.close()
    snapshot = journal.snapshot()
    assert snapshot["state"]["status"] == "failed"
    row = json.loads(journal.journal_path.read_text().splitlines()[-1])
    assert row["type"] == "stream.failed"
    assert row["payload"]["model_failure"] is False


def test_writer_byte_limit_is_accounted_on_cold_close(tmp_path):
    journal = ReplayJournal(tmp_path, run(), CLOCKS)
    writer = BoundedEventWriter(journal, max_pending_bytes=128)
    assert not writer.emit(
        "model.stage.started", occurrence=occurrence(), payload={"data": "x" * 300}
    )
    writer.close()
    snapshot = journal.snapshot()
    assert snapshot["state"]["dropped_events"] == 1
    assert snapshot["state"]["status"] == "running"
