"""Read-only HTTP catalog allowlist. Standard library only; no inference imports."""

import json
import os
import re
import stat
from pathlib import Path

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
MIMES = {"video/webm", "image/png"}


class ReplayFiles:
    """Only explicitly listed files beneath the selected catalog root are public."""

    def __init__(self, catalog):
        path = Path(catalog)
        if path.is_dir():
            path = path / "catalog.json"
        path = path.resolve(strict=True)
        self.root = path.parent
        self.catalog = json.loads(path.read_text())
        if (
            self.catalog.get("schema") != "pir2.replay.catalog"
            or self.catalog.get("version") != 1
        ):
            raise ValueError("Unsupported replay catalog schema/version")
        self.runs = self._index(self.catalog.get("runs", []))
        manifest = self._relative(self.catalog["media_manifest"])
        with self.open(manifest) as stream:
            self.media = self._index(json.load(stream).get("assets", []), media=True)

    def _relative(self, value):
        path = Path(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(p in {"..", "."} for p in path.parts)
        ):
            raise ValueError("Replay path must stay beneath catalog root")
        current = self.root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Replay path must not use symlinks")
        return path

    def _index(self, rows, media=False):
        out = {}
        for row in rows:
            identifier = row.get("id", "")
            if (
                not isinstance(identifier, str)
                or not IDENTIFIER.fullmatch(identifier)
                or identifier in out
            ):
                raise ValueError("Invalid or duplicate replay ID")
            mime = row.get("mime") if media else "application/json; charset=utf-8"
            if media and mime not in MIMES:
                raise ValueError("Unsupported replay media type")
            out[identifier] = (self._relative(row["path"]), mime)
        return out

    def open(self, path):
        # openat + O_NOFOLLOW on every component also closes a symlink-swap race.
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = child
            descriptor = os.open(
                path.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise OSError("Replay asset is not a regular file")
            return os.fdopen(descriptor, "rb")
        finally:
            os.close(directory)


def byte_range(header, size):
    """Single RFC byte range; malformed/multipart/unsatisfiable requests fail."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or size == 0:
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - length), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, end
