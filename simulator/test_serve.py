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
