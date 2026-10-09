"""Loopback-only preview of the teaching tool and stable tracked documentation."""

import argparse
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = [ROOT / "simulator"]
ALLOWED_FILES = {
    ROOT / "coexecution/README.md",
    ROOT / "docs/coexecution.md",
    ROOT / "docs/timing-protocols.md",
}


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        route = urlsplit(self.path).path
        report = self.server.report_path
        if route == "/":
            self.send_response(302)
            self.send_header("Location", "/simulator/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if route == "/experiment-status.json":
            data = json.dumps(
                {"available": report is not None and report.is_file()}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return io.BytesIO(data)
        if route == "/experiment-report.html":
            if report is None or not report.is_file():
                self.send_error(404, "No experiment report configured")
                return None
            data = report.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return io.BytesIO(data)
        requested = (ROOT / unquote(urlsplit(self.path).path).lstrip("/")).resolve()
        allowed = requested in ALLOWED_FILES or any(
            requested == parent or parent in requested.parents for parent in ALLOWED
        )
        if not allowed:
            self.send_error(404)
            return None
        return super().send_head()


def make_server(port, report=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), PreviewHandler)
    server.report_path = Path(report).resolve() if report is not None else None
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--report", type=Path, help="Explicit local self-contained measured report HTML"
    )
    args = parser.parse_args()
    if args.report is not None and not args.report.is_file():
        parser.error("--report must name an existing HTML file")
    server = make_server(args.port, args.report)
    print(f"Preview: http://127.0.0.1:{args.port}/simulator/", flush=True)
    server.serve_forever()
