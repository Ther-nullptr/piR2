"""Read-only HTTP catalog allowlist. Standard library only; no inference imports."""

import json
import os
import re
import stat
import threading
from pathlib import Path

from .stream import (
    MAX_EVENT_BYTES,
    StreamTail,
    apply_event,
    event_byte_limit,
    initial_snapshot,
    validate_snapshot,
)

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
MIMES = {"video/webm", "video/mp4", "image/png"}


class ReplayFiles:
    """Only explicitly listed files beneath the selected catalog root are public."""

    def __init__(self, catalog):
        path = Path(catalog)
        if path.is_dir():
            path = path / "catalog.json"
        path = path.resolve(strict=True)
        self.root = path.parent
        self._catalog_path = Path(path.name)
        self._lock = threading.RLock()
        self._indexes = ({}, {}, {}, {})
        self._fingerprints = None
        self.refresh()

    @property
    def catalog(self):
        return self._indexes[0]

    @property
    def runs(self):
        return self._indexes[1]

    @property
    def media(self):
        return self._indexes[2]

    @property
    def streams(self):
        return self._indexes[3]

    @staticmethod
    def _fingerprint(source):
        metadata = os.fstat(source.fileno())
        return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns

    def refresh(self):
        """Atomically reload changed allowlists, never paths supplied by HTTP."""
        with self._lock:
            with self.open(self._catalog_path) as source:
                catalog_stamp = self._fingerprint(source)
                catalog = (
                    self.catalog
                    if self._fingerprints and catalog_stamp == self._fingerprints[0]
                    else json.load(source)
                )
            if (
                not isinstance(catalog, dict)
                or catalog.get("schema") != "pir2.replay.catalog"
                or catalog.get("version") != 1
            ):
                raise ValueError("Unsupported replay catalog schema/version")
            manifest = self._relative(catalog["media_manifest"])
            with self.open(manifest) as source:
                stamps = catalog_stamp, self._fingerprint(source)
                if stamps == self._fingerprints:
                    return False
                media_manifest = json.load(source)
                if not isinstance(media_manifest, dict):
                    raise TypeError("Replay media manifest must be an object")
                media_rows = media_manifest.get("assets", [])
            indexes = self._build_indexes(catalog, media_rows)
            self._indexes = indexes
            self._fingerprints = stamps
            return True

    def _build_indexes(self, catalog, media_rows):
        runs = self._index(catalog.get("runs", []))
        media = self._index(media_rows, media=True)
        streams = {}
        for row in self._rows(catalog.get("streams", [])):
            identifier = row.get("run_id")
            if (
                not isinstance(identifier, str)
                or identifier not in runs
                or identifier in streams
            ):
                raise ValueError("Stream ID must name one unique registered replay run")
            streams[identifier] = (
                self._relative(row["snapshot_path"]),
                self._relative(row["journal_path"]),
                event_byte_limit(row.get("max_event_bytes", MAX_EVENT_BYTES)),
            )
        return catalog, runs, media, streams

    def snapshot(self, identifier):
        self.refresh()
        return self._snapshot(identifier, self._indexes)

    def _snapshot(self, identifier, indexes):
        _, runs, _, streams = indexes
        if identifier not in runs:
            raise KeyError("Unknown replay run")
        if identifier not in streams:
            with self.open(runs[identifier][0]) as source:
                return validate_snapshot(
                    initial_snapshot(json.load(source), {}, available=False), identifier
                )
        snapshot_path, journal_path, max_event_bytes = streams[identifier]
        with self.open(snapshot_path) as source:
            snapshot = validate_snapshot(json.load(source), identifier)
        if not snapshot["stream_available"]:
            raise ValueError("Registered live stream must declare availability")
        # A committed JSONL suffix may be newer than its atomic checkpoint. The
        # read-only server reconciles it in memory without modifying artifacts.
        tail = StreamTail(
            lambda: self.open(journal_path),
            identifier,
            snapshot["clocks"],
            after=snapshot["last_sequence"],
            max_event_bytes=max_event_bytes,
        )
        while True:
            events = tail.poll()
            if not events:
                break
            for event in events:
                snapshot = apply_event(snapshot, event)
        return snapshot

    def stream_tail(self, identifier, *, after=0, clocks=None):
        self.refresh()
        indexes = self._indexes
        if clocks is None:
            clocks = self._snapshot(identifier, indexes)["clocks"]
        return self._stream_tail(identifier, indexes, after=after, clocks=clocks)

    def stream_session(self, identifier, *, after=0):
        """Bind initial snapshot and cursor to the same allowlist generation."""
        self.refresh()
        indexes = self._indexes
        snapshot = self._snapshot(identifier, indexes)
        tail = self._stream_tail(
            identifier, indexes, after=after, clocks=snapshot["clocks"]
        )
        return snapshot, tail

    def _stream_tail(self, identifier, indexes, *, after, clocks):
        streams = indexes[3]
        if identifier not in streams:
            raise KeyError("Replay run has no registered live stream")
        _, journal_path, max_event_bytes = streams[identifier]
        return StreamTail(
            lambda: self.open(journal_path),
            identifier,
            clocks,
            after=after,
            max_event_bytes=max_event_bytes,
        )

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
        for row in self._rows(rows):
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

    @staticmethod
    def _rows(rows):
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise TypeError("Replay manifest rows must be a list of objects")
        return rows

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
