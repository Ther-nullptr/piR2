"""Append-only replay evidence, bounded recording, and read-only stream cursors.

Only standard-library metadata and explicitly normalized replay v1 records are
handled here. No timing, kernel event, observation, or model output is inferred.
"""

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path

from .schema import PRECISIONS

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
EVENT_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
DOMAINS = frozenset(("host_monotonic", "cuda_event", "media", "simulation", "utc"))
UNITS = frozenset(("s", "ms", "ns"))
MAX_SEQUENCE = 2**53 - 1
MAX_EVENT_BYTES = 1024 * 1024
MAX_REGISTERED_EVENT_BYTES = 8 * 1024 * 1024
MAX_POLL_BYTES = 16 * 1024 * 1024
RECORDER_CLOCK = "recorder-monotonic"
COLLECTIONS = {
    "observations": "observation_id",
    "clean_requests": "id",
    "profile_requests": "id",
    "retained_outputs": "id",
    "episodes": "id",
    "kernels": None,
}


class StreamGap(ValueError):
    """The consumer must fetch an authoritative snapshot before proceeding."""


def _json(value):
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Replay stream requires finite JSON metadata") from error


def _loads(value):
    def invalid(_value):
        raise ValueError("Replay stream contains nonfinite JSON")

    return json.loads(value, parse_constant=invalid)


def _sequence(value, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= MAX_SEQUENCE:
        raise ValueError("Invalid replay sequence")
    return value


def event_byte_limit(value):
    """One explicit producer/reader limit, capped at eight MiB per record."""
    if type(value) is not int or not 256 <= value <= MAX_REGISTERED_EVENT_BYTES:
        raise ValueError("Event byte limit must be between 256 bytes and 8 MiB")
    return value


def _run(value, run_id):
    if not isinstance(value, dict) or (
        value.get("schema") != "pir2.replay.run"
        or type(value.get("version")) is not int
        or value["version"] != 1
        or value.get("id") != run_id
        or value.get("kind") not in ("model", "deployment")
        or value.get("precision") not in PRECISIONS
    ):
        raise ValueError(
            "Normalized replay run identity/schema/kind/precision is invalid"
        )
    _json(value)
    return value


def validate_clocks(clocks):
    if not isinstance(clocks, dict):
        raise TypeError("Clock registry must be an object")
    for clock_id, definition in clocks.items():
        if (
            not isinstance(clock_id, str)
            or not IDENTIFIER.fullmatch(clock_id)
            or not isinstance(definition, dict)
            or definition.get("unit") not in UNITS
            or definition.get("domain") not in DOMAINS
        ):
            raise ValueError("Invalid explicit clock definition")
    _json(clocks)


def validate_event(event, run_id, clocks):
    if (
        not isinstance(event, dict)
        or event.get("schema") != "pir2.replay.event"
        or type(event.get("version")) is not int
        or event["version"] != 1
    ):
        raise ValueError("Unsupported replay event schema/version")
    if event.get("run_id") != run_id:
        raise ValueError("Replay event belongs to a different run")
    _sequence(event.get("sequence"))
    if not isinstance(event.get("type"), str) or not EVENT_TYPE.fullmatch(
        event["type"]
    ):
        raise ValueError("Invalid replay event type")
    occurrence = event.get("occurrence")
    if not isinstance(occurrence, dict) or set(occurrence) != {
        "value",
        "unit",
        "clock_id",
        "domain",
    }:
        raise ValueError("Event occurrence needs explicit value/unit/clock_id/domain")
    definition = clocks.get(occurrence["clock_id"])
    if definition is None or any(
        occurrence[key] != definition[key] for key in ("unit", "domain")
    ):
        raise ValueError("Event clock does not match its declared clock registry")
    value = occurrence["value"]
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value)
    ):
        raise ValueError("Occurrence time must be finite or explicitly unknown")
    sources = event.get("sources")
    if not isinstance(sources, dict) or any(
        not isinstance(key, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]*_id", key)
        or value is not None
        and (not isinstance(value, str) or not value or len(value) > 1024)
        for key, value in sources.items()
    ):
        raise ValueError("Source IDs must be named opaque strings or null")
    if not isinstance(event.get("payload"), dict):
        raise TypeError("Event payload must be a JSON object")
    _json(event)
    return event


def validate_snapshot(snapshot, run_id=None):
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema") != "pir2.replay.snapshot"
        or type(snapshot.get("version")) is not int
        or snapshot["version"] != 1
    ):
        raise ValueError("Unsupported replay snapshot schema/version")
    identifier = snapshot.get("run_id")
    if (
        not isinstance(identifier, str)
        or not IDENTIFIER.fullmatch(identifier)
        or run_id is not None
        and identifier != run_id
    ):
        raise ValueError("Snapshot run ID is invalid")
    _sequence(snapshot.get("last_sequence"), zero=True)
    _run(snapshot.get("run"), identifier)
    validate_clocks(snapshot.get("clocks"))
    state = snapshot.get("state")
    if not isinstance(state, dict) or state.get("status") not in (
        "running",
        "ended",
        "failed",
        "historical",
    ):
        raise ValueError("Invalid stream state")
    for key in ("dropped_events", "recorded_events"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError("Invalid stream accounting")
    if type(snapshot.get("stream_available")) is not bool:
        raise ValueError("Snapshot must declare stream availability")
    _json(snapshot)
    return snapshot


def initial_snapshot(run, clocks, *, available=True):
    identifier = run.get("id")
    snapshot = {
        "schema": "pir2.replay.snapshot",
        "version": 1,
        "run_id": identifier,
        "last_sequence": 0,
        "run": copy.deepcopy(run),
        "stream_available": available,
        "state": {
            "status": "running" if available else "historical",
            "dropped_events": 0,
            "recorded_events": 0,
            "error": None,
        },
        "clocks": copy.deepcopy(clocks),
    }
    return validate_snapshot(snapshot)


def _item_identity(collection, item):
    key = COLLECTIONS[collection]
    value = item if key is None else item.get(key) if isinstance(item, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError("Normalized append item requires an immutable identity")
    return value


def apply_event(snapshot, event):
    validate_event(event, snapshot["run_id"], snapshot["clocks"])
    sequence = event["sequence"]
    if sequence <= snapshot["last_sequence"]:
        return copy.deepcopy(snapshot)
    if sequence != snapshot["last_sequence"] + 1:
        raise StreamGap("sequence_gap")
    if snapshot["state"]["status"] in ("ended", "failed"):
        raise ValueError("Cannot append after terminal stream state")
    result = copy.deepcopy(snapshot)
    projection = event.get("projection")
    if projection is not None:
        if not isinstance(projection, dict):
            raise ValueError("Projection must be a JSON object")
        if projection.get("op") == "replace":
            replacement = _run(projection.get("run"), snapshot["run_id"])
            if any(
                replacement[key] != snapshot["run"][key]
                for key in ("kind", "precision")
            ):
                raise ValueError(
                    "Replacement cannot change run identity/kind/precision"
                )
            result["run"] = copy.deepcopy(replacement)
        elif projection.get("op") == "append":
            collection = projection.get("collection")
            if collection not in COLLECTIONS or not isinstance(
                projection.get("items"), list
            ):
                raise ValueError("Unknown normalized append collection")
            target = result["run"].setdefault(collection, [])
            if not isinstance(target, list):
                raise ValueError("Append target must remain a normalized list")
            identities = {_item_identity(collection, item) for item in target}
            for item in projection["items"]:
                identity = _item_identity(collection, item)
                if identity in identities:
                    raise ValueError("Normalized append identity collision")
                identities.add(identity)
                target.append(copy.deepcopy(item))
        else:
            raise ValueError("Unknown normalized projection operation")
    result["last_sequence"] = sequence
    state = result["state"]
    state["recorded_events"] += 1
    if event["type"] == "stream.dropped":
        count = event["payload"].get("count")
        if type(count) is not int or count < 1:
            raise ValueError("Dropped-event accounting requires a positive count")
        state["dropped_events"] += count
    if event["type"] in ("run.ended", "run.failed", "stream.failed"):
        state["status"] = "ended" if event["type"] == "run.ended" else "failed"
        if state["status"] == "failed":
            state["error"] = event["payload"].get("error", "Run reported failure")
    return result


class ReplayJournal:
    """Single-writer committed JSONL plus atomic normalized checkpoints."""

    def __init__(
        self,
        directory,
        run,
        clocks,
        *,
        checkpoint_every=32,
        max_event_bytes=MAX_EVENT_BYTES,
    ):
        if type(checkpoint_every) is not int or checkpoint_every < 1:
            raise ValueError("Journal limits must be positive bounded integers")
        event_byte_limit(max_event_bytes)
        clocks = copy.deepcopy(clocks)
        recorder = {"unit": "s", "domain": "host_monotonic"}
        if RECORDER_CLOCK in clocks and clocks[RECORDER_CLOCK] != recorder:
            raise ValueError("Recorder clock is reserved for recording-loss metadata")
        clocks[RECORDER_CLOCK] = recorder
        self._snapshot = initial_snapshot(run, clocks)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = directory / f"{run['id']}.snapshot.json"
        self.journal_path = directory / f"{run['id']}.events.jsonl"
        if self.snapshot_path.exists() or self.journal_path.exists():
            raise ValueError("Stream journal requires a new run destination")
        self.checkpoint_every, self.max_event_bytes = checkpoint_every, max_event_bytes
        self._lock = threading.RLock()
        self._owner = None
        self._closed = False
        self._stream = self.journal_path.open("xb")
        self.checkpoint()

    def registration(self, catalog_root):
        root = Path(catalog_root).resolve()
        return {
            "run_id": self._snapshot["run_id"],
            "snapshot_path": str(self.snapshot_path.resolve().relative_to(root)),
            "journal_path": str(self.journal_path.resolve().relative_to(root)),
            "max_event_bytes": self.max_event_bytes,
        }

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def checkpoint(self):
        with self._lock:
            data = _json(self._snapshot).encode() + b"\n"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.snapshot_path.parent,
                    prefix=".replay-snapshot-",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                    stream.flush()
                os.replace(temporary, self.snapshot_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def append(
        self, event_type, *, occurrence, sources=None, payload=None, projection=None
    ):
        with self._lock:
            if self._closed:
                raise RuntimeError("Replay journal is closed")
            owner = threading.get_ident()
            if self._owner is None:
                self._owner = owner
            if owner != self._owner:
                raise RuntimeError("Replay journal requires its single writer")
            event = {
                "schema": "pir2.replay.event",
                "version": 1,
                "run_id": self._snapshot["run_id"],
                "sequence": self._snapshot["last_sequence"] + 1,
                "type": event_type,
                "occurrence": occurrence,
                "sources": {} if sources is None else sources,
                "payload": {} if payload is None else payload,
                "projection": projection,
            }
            updated = apply_event(self._snapshot, event)
            data = _json(event).encode() + b"\n"
            if len(data) > self.max_event_bytes:
                raise ValueError("Replay event exceeds the declared byte bound")
            self._stream.write(data)
            self._stream.flush()  # A newline is the committed-record boundary.
            self._snapshot = updated
            if event["sequence"] % self.checkpoint_every == 0 or event_type in (
                "stream.dropped",
                "run.ended",
                "run.failed",
                "stream.failed",
            ):
                self.checkpoint()
            return _loads(data)

    def close(self):
        with self._lock:
            if not self._closed:
                self.checkpoint()
                self._stream.close()
                self._closed = True


class BoundedEventWriter:
    """Nonblocking producer admission; disk/projection work runs on one writer.

    Capacity includes the in-flight record. Loss is a bounded side counter and
    is committed even if close immediately follows a full-queue rejection.
    """

    def __init__(self, journal, *, max_pending=256, max_pending_bytes=4 * 1024 * 1024):
        if any(
            type(value) is not int or value < 1
            for value in (max_pending, max_pending_bytes)
        ):
            raise ValueError("Producer limits must be positive integers")
        self.journal, self.max_pending, self.max_pending_bytes = (
            journal,
            max_pending,
            max_pending_bytes,
        )
        self._condition = threading.Condition()
        self._queue = deque()
        self._used_bytes = 0
        self._inflight = False
        self._closing = self._closed = False
        self._loss_count = self.dropped_events = self.accepted_events = 0
        self._loss_types, self._loss_reasons = Counter(), Counter()
        self._first_loss_s = None
        self._terminal = None
        self.error = None
        self.failure_record_error = None
        self._thread = threading.Thread(
            target=self._run, name="replay-journal-writer", daemon=True
        )
        self._thread.start()

    def _drop(self, event_type, reason):
        self._loss_count += 1
        self.dropped_events += 1
        key = (
            event_type
            if event_type in self._loss_types or len(self._loss_types) < 32
            else "other"
        )
        self._loss_types[key] += 1
        self._loss_reasons[reason] += 1
        if self._first_loss_s is None:
            self._first_loss_s = time.monotonic()
        self._condition.notify_all()

    def emit(
        self, event_type, *, occurrence, sources=None, payload=None, projection=None
    ):
        if not isinstance(event_type, str) or not EVENT_TYPE.fullmatch(event_type):
            raise ValueError("Invalid replay event type")
        with self._condition:
            if self._closing or self.error is not None:
                return False
            if len(self._queue) + int(self._inflight) >= self.max_pending:
                self._drop(event_type, "queue_full")
                return False
            data = _json(
                {
                    "occurrence": occurrence,
                    "sources": sources,
                    "payload": payload,
                    "projection": projection,
                }
            ).encode()
            if (
                len(data) > self.journal.max_event_bytes - 512
                or self._used_bytes + len(data) > self.max_pending_bytes
            ):
                self._drop(event_type, "byte_limit")
                return False
            self._queue.append((event_type, data))
            self._used_bytes += len(data)
            self.accepted_events += 1
            self._condition.notify_all()
            return True

    def flush(self, timeout=None):
        """Wait for accepted events and loss records outside inference timing.

        Producers must stop submitting before using this barrier to reserve
        capacity for a final normalized projection. This does not end the run.
        """
        if timeout is not None and (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("Flush timeout must be finite and nonnegative")
        with self._condition:
            drained = self._condition.wait_for(
                lambda: (
                    self.error is not None
                    or (
                        not self._queue
                        and not self._inflight
                        and not self._loss_count
                        and (not self._closing or self._closed)
                    )
                ),
                timeout=timeout,
            )
            if self.error is not None:
                raise RuntimeError("Replay event recording failed") from self.error
            if not drained:
                raise TimeoutError("Replay event flush timed out")

    def close(self, *, status=None, occurrence=None, payload=None):
        if (
            status not in (None, "ended", "failed")
            or status is not None
            and occurrence is None
        ):
            raise ValueError("Explicit terminal status requires its actual occurrence")
        with self._condition:
            if not self._closing:
                self._closing = True
                if status is not None:
                    self._terminal = (
                        "run.ended" if status == "ended" else "run.failed",
                        {
                            "occurrence": copy.deepcopy(occurrence),
                            "payload": copy.deepcopy(payload),
                        },
                    )
                self._condition.notify_all()
        self._thread.join()
        if self.error is not None:
            raise RuntimeError("Replay event recording failed") from self.error

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._queue or self._loss_count or self._closing
                    )
                    if self._loss_count:
                        task = (
                            "stream.dropped",
                            {
                                "occurrence": {
                                    "value": self._first_loss_s,
                                    "unit": "s",
                                    "clock_id": RECORDER_CLOCK,
                                    "domain": "host_monotonic",
                                },
                                "payload": {
                                    "count": self._loss_count,
                                    "event_types": dict(self._loss_types),
                                    "reasons": dict(self._loss_reasons),
                                },
                            },
                        )
                        self._loss_count, self._first_loss_s = 0, None
                        self._loss_types.clear()
                        self._loss_reasons.clear()
                        size = 0
                    elif self._queue:
                        kind, data = self._queue.popleft()
                        task, size = (kind, _loads(data)), len(data)
                    else:
                        break
                    self._inflight = True
                self.journal.append(task[0], **task[1])
                with self._condition:
                    self._used_bytes -= size
                    self._inflight = False
                    self._condition.notify_all()
                task = None
            if self._terminal is not None:
                self.journal.append(self._terminal[0], **self._terminal[1])
        except BaseException as error:  # noqa: BLE001 - propagate writer failures at close
            with self._condition:
                self.error = error
                self._condition.notify_all()
            try:
                self.journal.append(
                    "stream.failed",
                    occurrence={
                        "value": time.monotonic(),
                        "unit": "s",
                        "clock_id": RECORDER_CLOCK,
                        "domain": "host_monotonic",
                    },
                    payload={
                        "error": repr(error),
                        "scope": "event_recording",
                        "model_failure": False,
                    },
                )
            except Exception as recording_error:  # noqa: BLE001 - retain original failure
                # Storage failure can also prevent recording the failure itself;
                # close still raises the original error to the explicit producer.
                self.failure_record_error = repr(recording_error)
        finally:
            try:
                self.journal.close()
            except Exception as error:  # noqa: BLE001 - surface final checkpoint failure
                if self.error is None:
                    self.error = error
            finally:
                with self._condition:
                    self._queue.clear()
                    self._used_bytes = 0
                    self._inflight = False
                    self._closed = True
                    self._condition.notify_all()


class StreamTail:
    """Read complete committed lines, retaining bounded duplicate fingerprints."""

    def __init__(
        self, opener, run_id, clocks, *, after=0, max_event_bytes=MAX_EVENT_BYTES
    ):
        self.opener, self.run_id, self.clocks = opener, run_id, clocks
        self.after = _sequence(after, zero=True)
        self.max_event_bytes = event_byte_limit(max_event_bytes)
        self.offset = self.last_sequence = 0
        self.identity = None
        self._seen = OrderedDict()

    def poll(self, *, limit=256, max_bytes=MAX_POLL_BYTES):
        if (
            type(limit) is not int
            or limit < 1
            or type(max_bytes) is not int
            or not self.max_event_bytes <= max_bytes <= MAX_POLL_BYTES
        ):
            raise ValueError("Tail polling requires bounded record and byte limits")
        output = []
        output_bytes = 0
        with self.opener() as stream:
            metadata = os.fstat(stream.fileno())
            identity = (metadata.st_dev, metadata.st_ino)
            if self.identity is not None and identity != self.identity:
                raise StreamGap("journal_replaced")
            if metadata.st_size < self.offset:
                raise StreamGap("journal_truncated")
            self.identity = identity
            stream.seek(self.offset)
            while len(output) < limit:
                beginning = stream.tell()
                line = stream.readline(self.max_event_bytes + 1)
                if not line:
                    if self.after > self.last_sequence:
                        raise StreamGap("cursor_ahead")
                    break
                if len(line) > self.max_event_bytes:
                    raise StreamGap("event_size_limit")
                if not line.endswith(b"\n"):
                    stream.seek(beginning)
                    break
                if output and output_bytes + len(line) > max_bytes:
                    break
                try:
                    event = validate_event(_loads(line), self.run_id, self.clocks)
                except (ValueError, TypeError, KeyError) as error:
                    raise StreamGap("invalid_event") from error
                sequence = event["sequence"]
                fingerprint = hashlib.sha256(
                    json.dumps(
                        event, sort_keys=True, allow_nan=False, separators=(",", ":")
                    ).encode()
                ).digest()
                if sequence <= self.last_sequence:
                    if self._seen.get(sequence) != fingerprint:
                        raise StreamGap("conflicting_or_unverifiable_duplicate")
                elif sequence != self.last_sequence + 1:
                    raise StreamGap("sequence_gap")
                else:
                    self.last_sequence = sequence
                    self._seen[sequence] = fingerprint
                    while len(self._seen) > 256:
                        self._seen.popitem(last=False)
                    if sequence > self.after:
                        output.append(event)
                        output_bytes += len(line)
                        self.after = sequence
                self.offset = stream.tell()
        return output


def sse(event_type, data, *, sequence=None):
    lines = [f"event: {event_type}"]
    if sequence is not None:
        lines.append(f"id: {_sequence(sequence)}")
    lines.append("data: " + _json(data))
    return ("\n".join(lines) + "\n\n").encode()
