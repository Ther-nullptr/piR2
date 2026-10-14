"""Preview exposes an explicitly supplied report, never its artifact directory."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "queue_preview", Path(__file__).with_name("serve.py")
)
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)


@pytest.mark.parametrize("connected", [False, True])
def test_explicit_report_routes_and_file_whitelist(tmp_path, connected):
    report = tmp_path / "index.html"
    report.write_text("<h1>local evidence</h1>")
    (tmp_path / "private.json").write_text("private")
    server = preview.make_server(0, report if connected else None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/experiment-status.json") as response:
            assert json.load(response) == {"available": connected}
        for path in [
            "/simulator/",
            "/simulator/model.js",
            "/simulator/measured-model.js",
        ]:
            with urllib.request.urlopen(base + path) as response:
                assert response.status == 200
        for route in [
            "/simulator/",
            "/simulator/replay-view.js",
            "/simulator/replay.css",
        ]:
            request = urllib.request.Request(
                base + route,
                headers={"If-Modified-Since": "Wed, 21 Oct 2099 07:28:00 GMT"},
            )
            with urllib.request.urlopen(request) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/") as response:
            assert response.url.endswith("/simulator/")
        if connected:
            with urllib.request.urlopen(
                base + "/experiment-report.html?embedded=1"
            ) as response:
                assert response.read().decode() == report.read_text()
        blocked = [
            "/private.json",
            "/simulator/%2e%2e/AGENTS.md",
            "/.local/",
            "/experiment-report.html/private.json",
        ]
        if not connected:
            blocked.append("/experiment-report.html")
        for path in blocked:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def replay_fixture(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "media").mkdir()
    (tmp_path / "runs/a.json").write_text('{"id":"a","kind":"model"}')
    (tmp_path / "media/a.webm").write_bytes(b"0123456789")
    (tmp_path / "media/a.png").write_bytes(b"PNGbytes")
    (tmp_path / "media.json").write_text(
        json.dumps(
            {
                "assets": [
                    {"id": "video", "path": "media/a.webm", "mime": "video/webm"},
                    {"id": "image", "path": "media/a.png", "mime": "image/png"},
                ]
            }
        )
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema": "pir2.replay.catalog",
                "version": 1,
                "runs": [{"id": "a", "path": "runs/a.json"}],
                "media_manifest": "media.json",
            }
        )
    )
    return catalog


def test_replay_explicit_routes_ranges_and_head(tmp_path):
    catalog = replay_fixture(tmp_path)
    server = preview.make_server(0, replay=catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/") as response:
            assert response.url.endswith("/simulator/replay.html")
        with urllib.request.urlopen(base + "/api/replay/catalog") as response:
            assert json.load(response)["runs"][0]["id"] == "a"
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/api/replay/run/a") as response:
            assert json.load(response)["id"] == "a"
        for header, body, content_range in [
            ("bytes=2-4", b"234", "bytes 2-4/10"),
            ("bytes=8-", b"89", "bytes 8-9/10"),
            ("bytes=-3", b"789", "bytes 7-9/10"),
        ]:
            req = urllib.request.Request(
                base + "/api/replay/media/video", headers={"Range": header}
            )
            with urllib.request.urlopen(req) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == content_range
                assert int(response.headers["Content-Length"]) == len(body)
                assert response.read() == body
        with urllib.request.urlopen(
            urllib.request.Request(base + "/api/replay/media/image", method="HEAD")
        ) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.headers["Content-Length"] == "8"
            assert response.read() == b""
        for route, code, headers in [
            ("/api/replay/media/video", 416, {"Range": "bytes=20-30"}),
            ("/api/replay/media/video", 416, {"Range": "bytes=0-2,5-8"}),
            ("/api/replay/media/%2e%2e%2fa.webm", 404, {}),
            ("/api/replay/run/a.json", 404, {}),
            ("/api/replay/media/", 404, {}),
            ("/runs/a.json", 404, {}),
            ("/api/replay/media/no", 404, {}),
        ]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(
                    urllib.request.Request(base + route, headers=headers)
                )
            assert error.value.code == code
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("escape", ["../secret.json", "runs/link.json", "/etc/passwd"])
def test_replay_rejects_manifest_path_escape(tmp_path, escape):
    catalog = replay_fixture(tmp_path)
    (tmp_path / "runs/link.json").symlink_to("/etc/passwd")
    data = json.loads(catalog.read_text())
    data["runs"][0]["path"] = escape
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="path|root"):
        preview.make_server(0, replay=catalog)


def test_replay_rechecks_symlink_and_blocks_unlisted_media(tmp_path):
    catalog = replay_fixture(tmp_path)
    server = preview.make_server(0, replay=catalog)
    (tmp_path / "media/a.webm").unlink()
    (tmp_path / "media/a.webm").symlink_to("/etc/passwd")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/api/replay/media/video"
            )
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def stream_fixture(tmp_path):
    from coexecution.replay.stream import ReplayJournal

    catalog = replay_fixture(tmp_path)
    run = {
        "schema": "pir2.replay.run",
        "version": 1,
        "id": "a",
        "kind": "model",
        "precision": "w8a8",
        "mode": "w8a8-clean",
        "observations": [],
        "clean_requests": [],
        "profile_requests": [],
        "kernels": [],
    }
    (tmp_path / "runs/a.json").write_text(json.dumps(run))
    clocks = {"host": {"unit": "s", "domain": "host_monotonic"}}
    journal = ReplayJournal(tmp_path / "streams", run, clocks)
    data = json.loads(catalog.read_text())
    data["streams"] = [journal.registration(tmp_path)]
    catalog.write_text(json.dumps(data))
    return catalog, journal


def test_refresh_discovers_new_run_stream_and_media_without_server_restart(tmp_path):
    from coexecution.replay.stream import ReplayJournal

    catalog, original = stream_fixture(tmp_path)
    server = preview.make_server(0, replay=catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    added = None
    try:
        data = json.loads(catalog.read_text())
        new_run = original.snapshot()["run"] | {"id": "new-run"}
        (tmp_path / "runs/new.json").write_text(json.dumps(new_run))
        added = ReplayJournal(
            tmp_path / "new-stream",
            new_run,
            {"host": {"unit": "s", "domain": "host_monotonic"}},
            max_event_bytes=8 * 1024 * 1024,
        )
        data["runs"].append({"id": "new-run", "path": "runs/new.json"})
        data["streams"].append(added.registration(tmp_path))
        replacement = tmp_path / "catalog.next"
        replacement.write_text(json.dumps(data))
        replacement.replace(catalog)
        with urllib.request.urlopen(base + "/api/replay/snapshot/new-run") as response:
            assert json.load(response)["stream_available"] is True
        with urllib.request.urlopen(base + "/api/replay/catalog") as response:
            assert len(json.load(response)["runs"]) == 2
        media = json.loads((tmp_path / "media.json").read_text())
        (tmp_path / "media/new.png").write_bytes(b"new media")
        media["assets"].append(
            {"id": "new-frame", "path": "media/new.png", "mime": "image/png"}
        )
        (tmp_path / "media.json").write_text(json.dumps(media))
        with urllib.request.urlopen(base + "/api/replay/media/new-frame") as response:
            assert response.read() == b"new media"
    finally:
        original.close()
        if added is not None:
            added.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_registered_large_event_reconciles_and_streams_using_same_byte_limit(tmp_path):
    catalog, journal = stream_fixture(tmp_path)
    journal.close()
    from coexecution.replay.stream import ReplayJournal

    journal = ReplayJournal(
        tmp_path / "large",
        journal.snapshot()["run"],
        {"host": {"unit": "s", "domain": "host_monotonic"}},
        max_event_bytes=8 * 1024 * 1024,
    )
    data = json.loads(catalog.read_text())
    data["streams"] = [journal.registration(tmp_path)]
    catalog.write_text(json.dumps(data))
    when = {"value": 1, "unit": "s", "clock_id": "host", "domain": "host_monotonic"}
    journal.append(
        "run.updated",
        occurrence=when,
        projection={
            "op": "replace",
            "run": journal.snapshot()["run"]
            | {"fixture_metadata": "x" * (2 * 1024 * 1024)},
        },
    )
    server = preview.make_server(0, replay=catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/replay/snapshot/a") as response:
            result = json.load(response)
            assert result["last_sequence"] == 1
            assert len(result["run"]["fixture_metadata"]) == 2 * 1024 * 1024
        journal.append("run.ended", occurrence=when)
        with urllib.request.urlopen(
            base + "/api/replay/events/a?after=0", timeout=3
        ) as response:
            body = response.read()
            assert b"id: 1\n" in body and b"id: 2\n" in body
            assert b"event: end\n" in body
    finally:
        journal.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_invalid_refresh_keeps_previous_indexes_and_never_opens_escape(tmp_path):
    catalog = replay_fixture(tmp_path)
    files = preview.ReplayFiles(catalog)
    previous = files.runs
    data = json.loads(catalog.read_text())
    data["runs"].append({"id": "escape", "path": "../secret.json"})
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="root"):
        files.refresh()
    assert files.runs is previous
    assert "escape" not in files.runs
    catalog.unlink()
    catalog.symlink_to("/etc/passwd")
    with pytest.raises(OSError):
        files.refresh()


@pytest.mark.parametrize("limit", [True, 0, 8 * 1024 * 1024 + 1])
def test_stream_registration_rejects_invalid_byte_limit(tmp_path, limit):
    catalog, journal = stream_fixture(tmp_path)
    journal.close()
    data = json.loads(catalog.read_text())
    data["streams"][0]["max_event_bytes"] = limit
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="limit"):
        preview.ReplayFiles(catalog)


def test_malformed_manifest_refresh_preserves_previous_index(tmp_path):
    catalog = replay_fixture(tmp_path)
    files = preview.ReplayFiles(catalog)
    before = files.media
    for invalid in ([], {"assets": [None]}, {"assets": {"id": "not-a-list"}}):
        (tmp_path / "media.json").write_text(json.dumps(invalid))
        with pytest.raises(TypeError, match="manifest|rows|object"):
            files.refresh()
        assert files.media is before


def test_stream_snapshot_reconciles_checkpoint_and_sse_resumes_then_ends(tmp_path):
    catalog, journal = stream_fixture(tmp_path)
    occurrence = {
        "value": 1.0,
        "unit": "s",
        "clock_id": "host",
        "domain": "host_monotonic",
    }
    journal.append(
        "observation.loaded",
        occurrence=occurrence,
        sources={"observation_id": "obs"},
        projection={
            "op": "append",
            "collection": "observations",
            "items": [{"observation_id": "obs"}],
        },
    )
    server = preview.make_server(0, replay=catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/replay/snapshot/a") as response:
            snapshot = json.load(response)
            assert snapshot["last_sequence"] == 1
            assert snapshot["run"]["observations"] == [{"observation_id": "obs"}]
        journal.append("run.ended", occurrence=occurrence | {"value": 2.0})
        request = urllib.request.Request(
            base + "/api/replay/events/a?after=0", headers={"Last-Event-ID": "1"}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            body = response.read().decode()
            assert (
                response.headers["Content-Type"] == "text/event-stream; charset=utf-8"
            )
            assert "id: 1\n" not in body and "id: 2\n" in body
            assert "event: end\n" in body
            assert body.index("id: 2\n") < body.index("event: end\n")
        with urllib.request.urlopen(
            base + "/api/replay/events/a?after=99", timeout=2
        ) as response:
            body = response.read().decode()
            assert "event: resnapshot\n" in body and "cursor_ahead" in body
        for route, expected in (
            ("/api/replay/events/no", 404),
            ("/api/replay/snapshot/..%2Fa", 404),
            ("/api/replay/events/a?after=-1", 400),
        ):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + route)
            assert error.value.code == expected
    finally:
        journal.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_legacy_snapshot_is_historical_without_starting_a_stream(tmp_path):
    catalog, journal = stream_fixture(tmp_path)
    journal.close()
    data = json.loads(catalog.read_text())
    data.pop("streams")
    catalog.write_text(json.dumps(data))
    server = preview.make_server(0, replay=catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/replay/snapshot/a") as response:
            snapshot = json.load(response)
            assert snapshot["stream_available"] is False
            assert snapshot["state"]["status"] == "historical"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/api/replay/events/a")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_open_sse_delivers_newly_committed_events_and_preserves_occurrence(tmp_path):
    from coexecution.replay.stream import BoundedEventWriter

    catalog, journal = stream_fixture(tmp_path)
    writer = BoundedEventWriter(journal)
    server = preview.make_server(0, replay=catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(
            base + "/api/replay/events/a?after=0", timeout=3
        ) as response:
            occurrence = {
                "value": 7.25,
                "unit": "s",
                "clock_id": "host",
                "domain": "host_monotonic",
            }
            assert writer.emit(
                "model.stage.started",
                occurrence=occurrence,
                payload={"stage": "vlm", "duration": None},
            )
            writer.close(status="ended", occurrence=occurrence | {"value": 7.5})
            lines = response.read().decode().splitlines()
            data = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
            assert data[0]["occurrence"] == occurrence
            assert data[0]["payload"]["duration"] is None
            assert [row.get("sequence") for row in data[:-1]] == [1, 2]
            assert data[-1] == {"run_id": "a", "last_sequence": 2, "status": "ended"}
    finally:
        writer.close()
        server.shutdown()
        server.server_close()
        thread.join()
