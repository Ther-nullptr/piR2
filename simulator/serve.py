"""Loopback-only preview of the teaching tool and stable tracked documentation."""

import argparse
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
        requested = (ROOT / unquote(urlsplit(self.path).path).lstrip("/")).resolve()
        allowed = requested in ALLOWED_FILES or any(
            requested == parent or parent in requested.parents for parent in ALLOWED
        )
        if not allowed:
            self.send_error(404)
            return None
        return super().send_head()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PreviewHandler)
    print(f"Preview: http://127.0.0.1:{args.port}/simulator/", flush=True)
    server.serve_forever()
