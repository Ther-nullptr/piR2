"""Bounded, owner-qualified in-process feature leases; no CUDA IPC or model cache.

The pool never expires or overwrites a live lease. Clients own a session and
release handles only after their latest/selected/pending references disappear.
An explicit session close reclaims replies lost in transit after consumers stop.
CUDA is imported only by the caller of the GPU snapshot/borrow helpers.
"""

import math
import re
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

SCHEMA = "libero.cuda_features"
VERSION = 1
_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
_SESSION_KEYS = {"owner", "epoch", "session"}
_HANDLE_KEYS = _SESSION_KEYS | {
    "schema",
    "version",
    "encoding",
    "lease",
    "device",
    "fields",
}


def validate_handle(handle):
    """Validate all host metadata before an exchange lookup or a CUDA API call."""
    if (
        not isinstance(handle, dict)
        or set(handle) != _HANDLE_KEYS
        or handle["schema"] != SCHEMA
        or type(handle["version"]) is not int
        or handle["version"] != VERSION
        or handle["encoding"] != "cuda_local"
        or type(handle["epoch"]) is not int
        or handle["epoch"] < 0
        or any(
            not isinstance(handle[key], str)
            or not re.fullmatch(r"[0-9a-f]{32}", handle[key])
            for key in ("owner", "session", "lease")
        )
        or not isinstance(handle["device"], str)
        or not re.fullmatch(r"cuda:[0-9]+", handle["device"])
        or not isinstance(handle["fields"], dict)
        or not handle["fields"]
    ):
        raise ValueError("Invalid resident feature handle")
    for key, field in handle["fields"].items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(field, dict)
            or set(field) != {"dtype", "shape", "nbytes"}
            or not isinstance(field["dtype"], str)
            or field["dtype"] not in _BYTES
            or not isinstance(field["shape"], list)
            or any(type(size) is not int or size < 0 for size in field["shape"])
            or type(field["nbytes"]) is not int
            or field["nbytes"] != math.prod(field["shape"]) * _BYTES[field["dtype"]]
        ):
            raise ValueError("Invalid resident feature dtype/shape")
    return handle


def resident_evidence(handle):
    validate_handle(handle)
    return {
        "encoding": "cuda_local",
        "schema_version": VERSION,
        "payload_nbytes": 0,
        "resident_nbytes": sum(field["nbytes"] for field in handle["fields"].values()),
        "field_dtypes": {
            key: field["dtype"] for key, field in handle["fields"].items()
        },
        "source": "vlm_cache_owned_cuda_snapshot",
    }


@dataclass
class _Entry:
    handle: dict
    features: dict
    meta: dict
    ready: object
    readers: int = 0
    leased: bool = True


@dataclass(frozen=True)
class BorrowedFeature:
    features: dict
    ready: object


class FeatureLeasePool:
    def __init__(self, device, *, max_entries=16):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Resident max_entries must be a positive integer")
        if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
            raise ValueError("Resident features require an explicit CUDA device")
        self.device, self.max_entries = device, max_entries
        self.owner, self.epoch = uuid.uuid4().hex, 0
        self._session = None
        self._entries = {}
        self._lock = threading.RLock()
        self._closed = False
        self._published = self._released = self._peak = 0

    def open_session(self, session=None):
        with self._lock:
            if session is not None and (
                not isinstance(session, dict)
                or set(session) != _SESSION_KEYS
                or session["owner"] != self.owner
                or type(session["epoch"]) is not int
                or session["epoch"] != self.epoch
                or not isinstance(session["session"], str)
                or not re.fullmatch(r"[0-9a-f]{32}", session["session"])
            ):
                raise ValueError("Invalid client resident feature session")
            if not self._closed and session is not None and session == self._session:
                return dict(self._session)
            if self._closed or self._session is not None:
                raise RuntimeError(
                    "Resident feature session unavailable or already owned"
                )
            self._session = (
                dict(session)
                if session is not None
                else {
                    "owner": self.owner,
                    "epoch": self.epoch,
                    "session": uuid.uuid4().hex,
                }
            )
            return dict(self._session)

    def _check_session(self, session):
        if (
            self._closed
            or not isinstance(session, dict)
            or set(session) != _SESSION_KEYS
            or type(session["epoch"]) is not int
            or self._session is None
            or session != self._session
        ):
            raise ValueError("Stale or foreign resident feature session")

    def _check_handle(self, handle, *, missing=False):
        validate_handle(handle)
        self._check_session({key: handle[key] for key in _SESSION_KEYS})
        if handle["device"] != self.device:
            raise ValueError("Resident feature device mismatch")
        entry = self._entries.get(handle["lease"])
        if entry is None:
            if missing:
                return None
            raise ValueError("Stale resident feature lease")
        if entry.handle != handle:
            raise ValueError("Resident feature handle differs from its owned tensors")
        return entry

    def publish(self, session, features, fields, meta, *, ready):
        with self._lock:
            self._check_session(session)
            if len(self._entries) >= self.max_entries:
                raise RuntimeError(
                    "Resident feature lease capacity exhausted; no lease was evicted"
                )
            handle = {
                "schema": SCHEMA,
                "version": VERSION,
                "encoding": "cuda_local",
                **session,
                "lease": uuid.uuid4().hex,
                "device": self.device,
                "fields": deepcopy(fields),
            }
            validate_handle(handle)
            if set(features) != set(fields):
                raise ValueError("Resident feature fields do not match owned tensors")
            self._entries[handle["lease"]] = _Entry(
                handle, dict(features), deepcopy(meta), ready
            )
            self._published += 1
            self._peak = max(self._peak, len(self._entries))
            return deepcopy(handle)

    @contextmanager
    def borrow(self, handle, meta):
        with self._lock:
            entry = self._check_handle(handle)
            if not entry.leased:
                raise ValueError("Released resident feature lease")
            if entry.meta != meta:
                raise ValueError("Resident feature metadata differs from publication")
            entry.readers += 1
        try:
            yield BorrowedFeature(dict(entry.features), entry.ready)
        finally:
            with self._lock:
                entry.readers -= 1
                self._retire(entry)

    def _retire(self, entry):
        if not entry.leased and not entry.readers:
            self._entries.pop(entry.handle["lease"], None)

    def release(self, session, handles):
        with self._lock:
            self._check_session(session)
            if not isinstance(handles, list):
                raise ValueError("Resident releases must be a list")  # noqa: TRY004 -- invalid RPC wire
            entries = [self._check_handle(handle, missing=True) for handle in handles]
            for entry in entries:
                if entry is not None:
                    self._released += int(entry.leased)
                    entry.leased = False
                    self._retire(entry)

    def close_session(self, session):
        with self._lock:
            self._check_session(session)
            for entry in tuple(self._entries.values()):
                entry.leased = False
                self._retire(entry)
            self._session = None
            self.epoch += 1

    def reset(self):
        with self._lock:
            if self._session is not None:
                self.close_session(self._session)
            else:
                self.epoch += 1

    def close(self):
        with self._lock:
            self.reset()
            self._closed = True

    def evidence(self):
        with self._lock:
            return {
                "owner": self.owner,
                "epoch": self.epoch,
                "live_leases": len(self._entries),
                "max_leases": self.max_entries,
                "peak_leases": self._peak,
                "published_leases": self._published,
                "released_leases": self._released,
                "active_readers": sum(
                    entry.readers for entry in self._entries.values()
                ),
            }

    def publish_cuda(
        self, session, features, meta, *, torch_module, stream, started_s=None
    ):
        """Snapshot on producer stream and complete the real VLM before publication."""
        # Check capacity before allocating. The producer is serial; consumers can
        # only free entries while this snapshot is prepared.
        with self._lock:
            self._check_session(session)
            if len(self._entries) >= self.max_entries:
                raise RuntimeError("Resident feature lease capacity exhausted")
        if not features or any(
            not torch_module.is_tensor(value)
            or not value.is_cuda
            or str(value.device) != self.device
            or value.layout != torch_module.strided
            for value in features.values()
        ):
            raise ValueError(
                "Resident snapshot requires CUDA tensors on the producer device"
            )
        owned = {key: value.detach().clone() for key, value in features.items()}
        fields = {
            key: {
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape),
                "nbytes": value.numel() * value.element_size(),
            }
            for key, value in owned.items()
        }
        ready = torch_module.cuda.Event()
        ready.record(stream)
        ready.synchronize()
        # Include the complete immutable wire metadata in the authenticated entry.
        published_meta = deepcopy(meta)
        if started_s is not None:
            published_meta["vlm_seconds"] = time.monotonic() - started_s
        published_meta["feature_wire"] = {
            "encoding": "cuda_local",
            "schema_version": VERSION,
            "payload_nbytes": 0,
            "resident_nbytes": sum(field["nbytes"] for field in fields.values()),
            "field_dtypes": {key: field["dtype"] for key, field in fields.items()},
            "source": "vlm_cache_owned_cuda_snapshot",
        }
        handle = self.publish(session, owned, fields, published_meta, ready=ready)
        return handle, published_meta

    @contextmanager
    def borrow_cuda(self, handle, meta, *, device, stream):
        if str(device) != self.device:
            raise ValueError("Resident consumer device mismatch")
        with self.borrow(handle, meta) as record:
            stream.wait_event(record.ready)
            for value in record.features.values():
                value.record_stream(stream)
            yield record.features


class _LeasedReply(dict):
    """Reference counting spans mailbox, executor arguments and fallback cache."""


class ClientFeatureLeases:
    def __init__(self, session):
        self.session = deepcopy(session)
        self._pending = {}
        self._lock = threading.Lock()
        self.closed = False

    @classmethod
    def new(cls, identity):
        # Know our unguessable token before the open RPC, including a lost reply.
        return cls(
            {
                "owner": identity["owner"],
                "epoch": identity["epoch"],
                "session": uuid.uuid4().hex,
            }
        )

    def open(self, call):
        received = call(
            "feature_control", {"operation": "open", "session": self.session}
        )
        if received != self.session:
            raise ValueError("Resident server did not open the requested owned session")

    def vision(self, call, payload):
        request = self.request()
        result = self.attach(call("vision", {**payload, **request}))
        self.acknowledge(request)
        return result

    def _release(self, handle):
        with self._lock:
            if not self.closed:
                self._pending[handle["lease"]] = handle

    def attach(self, result):
        handle = deepcopy(validate_handle(result["vl_embeds"]))
        if {key: handle[key] for key in _SESSION_KEYS} != self.session or self.closed:
            raise ValueError("Resident reply belongs to another client session")
        reply = _LeasedReply(result)
        weakref.finalize(reply, self._release, handle)
        return reply

    def request(self):
        with self._lock:
            if self.closed:
                raise RuntimeError("Resident client session is closed")
            return {
                "feature_session": deepcopy(self.session),
                "released_features": deepcopy(list(self._pending.values())),
            }

    def acknowledge(self, request):
        with self._lock:
            for handle in request["released_features"]:
                self._pending.pop(handle["lease"], None)

    def close(self, call):
        """Caller first joins all action consumers, then uses a thread-owned RPC."""
        with self._lock:
            if self.closed:
                return
        call("feature_control", {"operation": "close", "session": self.session})
        with self._lock:
            self.closed = True
            self._pending.clear()


@contextmanager
def feature_rpc_client(port):
    """Own a feature-control connection with bounded socket teardown on failure."""
    import zmq
    from gr00t.policy.server_client import PolicyClient

    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=120000)
    try:
        # Configure before sending: the pinned client replaces its REQ socket
        # after a timeout. Context defaults cover every replacement, and the
        # already-created socket needs the same setting explicitly.
        client.context.setsockopt(zmq.LINGER, 0)
        client.socket.setsockopt(zmq.LINGER, 0)
        yield client
    finally:
        # This fresh context and all its sockets belong only to this thread.
        # Destroy also closes any replaced socket retained by a traceback.
        client.context.destroy(linger=0)


def close_remote_features(leases, port):
    """A fresh thread-owned connection also drains leases after an RPC timeout."""
    if leases is None or leases.closed:
        return
    with feature_rpc_client(port) as client:
        leases.close(client.call_endpoint)


@contextmanager
def remote_feature_session(client, port, transport):
    """Short lived warmup/calibration scope; consumers finish before its exit."""
    leases = None
    try:
        if transport == "cuda_local":
            identity = client.call_endpoint("identity", requires_input=False)
            leases = ClientFeatureLeases.new(identity["feature_exchange"])
            leases.open(client.call_endpoint)
        yield leases
    finally:
        close_remote_features(leases, port)
