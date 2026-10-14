"""Loopback-only preview of the simulator, explicit replay evidence and reports."""

import argparse
import io
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coexecution.replay.http import ReplayFiles, byte_range

ALLOWED = [ROOT / "simulator"]
ALLOWED_FILES = {
    ROOT / "coexecution/README.md",
    ROOT / "docs/coexecution.md",
    ROOT / "docs/timing-protocols.md",
}


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        if getattr(self, "no_store_static", False):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "response_bytes", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining:
            data = source.read(min(65536, remaining))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)

    def replay_head(self, route):
        replay = self.server.replay_files
        if replay is None:
            self.send_error(404, "No replay catalog configured")
            return None
        if route == "/api/replay/catalog":
            data = json.dumps(replay.catalog).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return io.BytesIO(data)
        parts = route.split("/")
        if len(parts) != 5 or parts[3] not in {"run", "media"}:
            self.send_error(404)
            return None
        media = parts[3] == "media"
        entry = (replay.media if media else replay.runs).get(parts[4])
        if entry is None:
            self.send_error(404)
            return None
        try:
            stream = replay.open(entry[0])
        except OSError:
            self.send_error(404, "Listed replay file unavailable")
            return None
        size = stream.seek(0, 2)
        start, end = 0, size - 1
        partial = media and "Range" in self.headers
        if partial:
            try:
                start, end = byte_range(self.headers["Range"], size)
            except ValueError:
                stream.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
        stream.seek(start)
        self.response_bytes = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", entry[1])
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(self.response_bytes))
        self.send_header(
            "Cache-Control", "private, max-age=3600" if media else "no-store"
        )
        if media:
            self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        return stream

    def list_directory(self, path):
        self.send_error(404)

    def send_head(self):
        self.response_bytes = None
        self.no_store_static = False
        route = urlsplit(self.path).path
        report = self.server.report_path
        if route.startswith("/api/replay/"):
            return self.replay_head(route)
        if route == "/":
            self.send_response(302)
            self.send_header(
                "Location",
                "/simulator/replay.html" if self.server.replay_files else "/simulator/",
            )
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
        self.no_store_static = requested.is_dir() or requested.suffix in {
            ".html",
            ".js",
            ".css",
        }
        if self.no_store_static and "If-Modified-Since" in self.headers:
            del self.headers["If-Modified-Since"]
        return super().send_head()


def make_server(port, report=None, replay=None):
    replay_files = ReplayFiles(replay) if replay is not None else None
    server = ThreadingHTTPServer(("127.0.0.1", port), PreviewHandler)
    server.replay_files = replay_files
    server.report_path = Path(report).resolve() if report is not None else None
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--report", type=Path, help="Explicit local self-contained measured report HTML"
    )
    parser.add_argument(
        "--replay", type=Path, help="Explicit replay catalog JSON or directory"
    )
    args = parser.parse_args()
    if args.report is not None and not args.report.is_file():
        parser.error("--report must name an existing HTML file")
    try:
        server = make_server(args.port, args.report, args.replay)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Preview: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()
