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
