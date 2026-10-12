"""Host ownership tests; fake resources are metadata, never model inference."""

import gc
from copy import deepcopy

import pytest


def pool(limit=2):
    from coexecution.libero.resident import FeatureLeasePool

    return FeatureLeasePool("cuda:0", max_entries=limit)


def publish(exchange, session, value=None):
    return exchange.publish(
        session,
        {"features": object() if value is None else value},
        {"features": {"dtype": "bfloat16", "shape": [1, 2], "nbytes": 4}},
        {"capture_s": 1.0, "source_tick": 1},
        ready=object(),
    )


def test_pool_is_bounded_and_release_does_not_overwrite_an_active_reader():
    exchange = pool(1)
    session = exchange.open_session()
    value = object()
    handle = publish(exchange, session, value)
    with exchange.borrow(handle, {"capture_s": 1.0, "source_tick": 1}) as record:
        exchange.release(session, [handle])
        assert record.features["features"] is value
        with pytest.raises(RuntimeError, match="capacity"):
            publish(exchange, session)
    assert exchange.evidence()["live_leases"] == 0
    publish(exchange, session)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "foreign"),
        ("epoch", 9),
        ("session", "foreign"),
        ("lease", "foreign"),
        ("device", "cuda:1"),
        ("version", True),
        ("fields", {"features": {"dtype": "float32", "shape": [1, 2], "nbytes": 8}}),
    ],
)
def test_modified_or_foreign_handles_fail_closed(field, value):
    exchange = pool()
    session = exchange.open_session()
    handle = publish(exchange, session)
    bad = deepcopy(handle)
    bad[field] = value
    with (
        pytest.raises(ValueError),
        exchange.borrow(bad, {"capture_s": 1.0, "source_tick": 1}),
    ):
        pytest.fail("Invalid handle reached consumer")
    assert exchange.evidence()["live_leases"] == 1


def test_release_validates_whole_batch_without_deleting_an_innocent_lease():
    exchange = pool()
    session = exchange.open_session()
    handle = publish(exchange, session)
    bad = {**handle, "owner": "foreign"}
    with pytest.raises(ValueError):
        exchange.release(session, [handle, bad])
    assert exchange.evidence()["live_leases"] == 1
    exchange.release(session, [handle, handle])
    exchange.release(session, [handle])  # Safe retransmission after a lost reply.
    assert exchange.evidence()["live_leases"] == 0


def test_reset_retires_reader_without_revoking_its_owned_references():
    exchange = pool(1)
    session = exchange.open_session()
    handle = publish(exchange, session)
    with exchange.borrow(handle, {"capture_s": 1.0, "source_tick": 1}) as record:
        exchange.reset()
        assert record.features
        with pytest.raises(ValueError):
            exchange.release(session, [handle])
        replacement = exchange.open_session()
        with pytest.raises(RuntimeError, match="capacity"):
            publish(exchange, replacement)
    publish(exchange, replacement)


def test_session_close_cleans_unacknowledged_reply_and_cannot_adopt_foreign_session():
    exchange = pool()
    session = exchange.open_session()
    publish(exchange, session)  # Producer completed but network response was lost.
    with pytest.raises(RuntimeError, match="session"):
        exchange.open_session()
    with pytest.raises(ValueError):
        exchange.close_session({**session, "session": "foreign"})
    assert exchange.evidence()["live_leases"] == 1
    exchange.close_session(session)
    assert exchange.evidence()["live_leases"] == 0


def test_metadata_is_bound_to_lease_and_pool_owns_mapping():
    exchange = pool()
    session = exchange.open_session()
    handle = publish(exchange, session)
    with (
        pytest.raises(ValueError, match="metadata"),
        exchange.borrow(handle, {"capture_s": 0.9, "source_tick": 1}),
    ):
        pass
    with exchange.borrow(handle, {"capture_s": 1.0, "source_tick": 1}) as record:
        record.features.clear()
    with exchange.borrow(handle, {"capture_s": 1.0, "source_tick": 1}) as record:
        assert record.features


def test_client_lease_survives_latest_replacement_and_pending_action_reference():
    from coexecution.libero.resident import ClientFeatureLeases

    exchange = pool()
    session = exchange.open_session()
    leases = ClientFeatureLeases(session)
    handle = publish(exchange, session)
    latest = leases.attach({"vl_embeds": handle, "meta": {"capture_s": 1.0}})
    pending = latest
    del latest
    gc.collect()
    assert leases.request()["released_features"] == []
    del pending
    gc.collect()
    request = leases.request()
    assert request["released_features"] == [handle]
    assert leases.request() == request  # Failed RPC does not lose release IDs.
    leases.acknowledge(request)
    assert leases.request()["released_features"] == []


def test_backend_resident_install_binds_exact_meta_and_retains_after_release(tmp_path):
    from tests.contracts.test_feature_transport import backend

    exchange = pool()
    session = exchange.open_session()
    handle = publish(exchange, session)
    instance = backend(tmp_path)
    instance.feature_transport = "cuda_local"
    instance.feature_exchange = exchange
    calls = []
    instance.stream.wait_event = lambda ready: calls.append(ready)
    value = exchange._entries[handle["lease"]].features["features"]
    # Metadata resources emulate only stream/lifetime calls, never NN execution.
    from types import SimpleNamespace

    tensor = SimpleNamespace(record_stream=lambda stream: calls.append(stream))
    exchange._entries[handle["lease"]].features["features"] = tensor
    instance.install(handle, {"capture_s": 1.0, "source_tick": 1})
    exchange.release(session, [handle])
    assert instance.core._vl_cache["features"] is tensor
    assert len(calls) == 2
    assert value is not tensor


def test_paired_s2_reset_does_not_reseed_action_rng(tmp_path):
    from tests.contracts.test_feature_transport import backend

    instance = backend(tmp_path)
    instance.role, instance.paired = "vlm", True
    seeds = []
    instance.torch.manual_seed = seeds.append
    instance.core.reset_streaming_buffer = lambda: None
    instance.reset({"seed": 7})
    assert seeds == []
    instance.role = "action"
    instance.reset({"seed": 7})
    assert seeds == [7]


@pytest.mark.parametrize("releases", [False, 0, "", {}])
def test_vision_rejects_malformed_falsy_release_batches_before_neural_work(
    tmp_path, releases
):
    from tests.contracts.test_feature_transport import backend

    instance = backend(tmp_path)
    instance.feature_transport, instance.role = "cuda_local", "vlm"
    instance.feature_exchange = pool()
    session = instance.feature_exchange.open_session()
    with pytest.raises(ValueError, match="releases"):
        instance.vision({}, 1.0, 1, feature_session=session, released_features=releases)


def test_client_generated_session_is_idempotent_after_lost_open_reply():
    from coexecution.libero.resident import ClientFeatureLeases

    exchange = pool()
    leases = ClientFeatureLeases.new(exchange.evidence())
    session = exchange.open_session(leases.session)
    assert exchange.open_session(leases.session) == session
    publish(exchange, session)
    exchange.close_session(leases.session)
    assert exchange.evidence()["live_leases"] == 0


def test_failed_vision_keeps_releases_until_ack_and_close_revokes_unknown_reply():
    from coexecution.libero.resident import ClientFeatureLeases

    exchange = pool()
    session = exchange.open_session()
    leases = ClientFeatureLeases(session)
    reply = leases.attach({"vl_embeds": publish(exchange, session), "meta": {}})
    del reply
    gc.collect()

    def failed_call(name, payload):
        exchange.release(payload["feature_session"], payload["released_features"])
        publish(exchange, session)
        raise TimeoutError("lost reply")

    with pytest.raises(TimeoutError):
        leases.vision(failed_call, {})
    assert leases.request()["released_features"]
    leases.close(lambda name, payload: exchange.close_session(payload["session"]))
    assert exchange.evidence()["live_leases"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        leases.request()


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 2),
        ("physical_device_uuid", "other"),
        ("server_instance", "other"),
        ("cuda_stream", 11),
        ("policy_instance", "action"),
    ],
)
def test_evaluator_rejects_foreign_or_shared_paired_worker_identity(field, value):
    from coexecution.libero.features import validate_worker_pair

    action = {
        "role": "action",
        "pid": 1,
        "paired_same_process": True,
        "server_layout": "paired",
        "server_instance": "one",
        "policy_instance": "action",
        "cuda_stream": 11,
        "physical_device_uuid": "gpu",
        "feature_exchange": {"owner": "pool"},
    }
    vision = {**action, "role": "vlm", "policy_instance": "vision", "cuda_stream": 12}
    validate_worker_pair(action, vision, "paired", "cuda_local")
    vision[field] = value
    with pytest.raises(ValueError):
        validate_worker_pair(action, vision, "paired", "cuda_local")


def test_graph_execution_evidence_exposes_cold_capture_without_relabeling_time(
    tmp_path,
):
    from types import SimpleNamespace

    from tests.contracts.test_feature_transport import backend

    instance = backend(tmp_path)
    instance.optimization = SimpleNamespace(
        graphs=SimpleNamespace(capture_count=1, replay_count=4, failure_count=0),
        static_vlm=SimpleNamespace(counts={"graph_captures": 2, "graph_replays": 3}),
    )
    before = instance.graph_counters()
    instance.optimization.graphs.capture_count += 1
    evidence = instance.graph_execution(before)
    assert evidence["capture_delta"] == {"dit": 1, "vlm": 0}
    assert evidence["cold_capture"]


@pytest.mark.parametrize("lost_reply", [False, True])
def test_real_vision_worker_closes_owned_session_including_lost_reply(
    monkeypatch, lost_reply
):
    import sys
    import threading
    from types import SimpleNamespace

    from libero_queue_trace import QueueTrace
    from libero_wallclock import Observation, VisionWorker

    exchange = pool(3)
    published, calls = threading.Event(), []
    connection_owners = []

    class Trace(QueueTrace):
        def emit(self, kind, **fields):
            result = super().emit(kind, **fields)
            if kind == "feature_published":
                published.set()
            return result

    class Client:
        def __init__(self, **kwargs):
            owner = threading.get_ident()
            connection_owners.append(owner)

            def close():
                assert threading.get_ident() == owner

            self.socket, self.context = (
                SimpleNamespace(close=close, setsockopt=lambda *args: None),
                SimpleNamespace(
                    term=close,
                    setsockopt=lambda *args: None,
                    destroy=lambda linger: close(),
                ),
            )

        def call_endpoint(self, endpoint, payload=None, **kwargs):
            if endpoint == "identity":
                return {"feature_exchange": exchange.evidence()}
            if endpoint == "feature_control":
                if payload["operation"] == "open":
                    return exchange.open_session(payload["session"])
                return exchange.close_session(payload["session"])
            assert endpoint == "vision"
            calls.append(payload)
            session = payload["feature_session"]
            exchange.release(session, payload["released_features"])
            meta = {
                "capture_s": payload["capture_s"],
                "source_tick": payload["source_tick"],
                "vlm_forward_calls": 1,
                "vlm_seconds": 0.001,
            }
            handle = exchange.publish(
                session,
                {"features": object()},
                {"features": {"dtype": "bfloat16", "shape": [1], "nbytes": 2}},
                meta,
                ready=object(),
            )
            if lost_reply:
                published.set()
                raise TimeoutError("producer completed; reply lost")
            return {"vl_embeds": handle, "meta": meta}

    monkeypatch.setitem(
        sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyClient=Client)
    )
    monkeypatch.setitem(sys.modules, "zmq", SimpleNamespace(LINGER=17))
    worker = VisionWorker(1, events=Trace(), feature_transport="cuda_local")
    worker.offer(Observation({}, 0, 1.0))
    assert published.wait(2)
    if lost_reply:
        with pytest.raises(RuntimeError, match="VLM worker failed"):
            worker.close()
    else:
        selected = worker.ready(Observation({}, 0, 1.0))
        for tick in (1, 2):
            published.clear()
            worker.offer(Observation({}, tick, tick + 1.0))
            assert published.wait(2)
        assert exchange.evidence()["live_leases"] == 3
        del selected
        published.clear()
        worker.offer(Observation({}, 3, 4.0))
        assert published.wait(2)
        assert calls[-1]["released_features"]
        worker.close()
    assert exchange.evidence()["live_leases"] == 0
    assert len(connection_owners) == 2
    assert connection_owners[0] != connection_owners[1]


def test_cleanup_rpc_to_unavailable_server_has_bounded_native_zmq_shutdown():
    import ctypes.util
    import subprocess
    import sys
    from pathlib import Path

    if ctypes.util.find_library("zmq") is None:
        pytest.skip("Native libzmq is needed for the host transport regression")
    command = [
        sys.executable,
        "-c",
        (
            "from tests.contracts.test_resident_features import _native_zmq_cleanup_probe; "
            "_native_zmq_cleanup_probe()"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"Cleanup hung after a native ZMQ receive timeout: {error.stdout!r}"
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cleanup_timeout_propagated_and_context_closed" in result.stdout


def _native_zmq_cleanup_probe():
    """Exercise actual libzmq with the pinned client's timeout/recreate lifecycle.

    CI intentionally has no model or pyzmq dependencies. The small binding uses
    real queued REQ sends, receive timeouts, linger and context termination; the
    subprocess parent supplies the hard deadline even for a broken shutdown.
    """
    import ctypes as ct
    import ctypes.util
    import errno
    import socket
    import sys
    from types import SimpleNamespace

    from coexecution.libero.resident import ClientFeatureLeases, close_remote_features

    library = ct.CDLL(ctypes.util.find_library("zmq"))
    signatures = {
        "zmq_ctx_new": (ct.c_void_p, []),
        "zmq_ctx_term": (ct.c_int, [ct.c_void_p]),
        "zmq_socket": (ct.c_void_p, [ct.c_void_p, ct.c_int]),
        "zmq_close": (ct.c_int, [ct.c_void_p]),
        "zmq_connect": (ct.c_int, [ct.c_void_p, ct.c_char_p]),
        "zmq_setsockopt": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t]),
        "zmq_getsockopt": (
            ct.c_int,
            [ct.c_void_p, ct.c_int, ct.c_void_p, ct.POINTER(ct.c_size_t)],
        ),
        "zmq_send": (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]),
        "zmq_recv": (ct.c_int, [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]),
        "zmq_errno": (ct.c_int, []),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(library, name)
        function.restype, function.argtypes = restype, argtypes
    linger, receive_timeout, send_timeout, request_socket = 17, 27, 28, 3
    send_lingers, created_lingers, contexts = [], [], []

    class Socket:
        def __init__(self, handle):
            self.handle, self.closed = handle, False

        def setsockopt(self, option, value):
            number = ct.c_int(value)
            assert (
                library.zmq_setsockopt(
                    self.handle, option, ct.byref(number), ct.sizeof(number)
                )
                == 0
            )

        def getsockopt(self, option):
            number, size = ct.c_int(), ct.c_size_t(ct.sizeof(ct.c_int))
            assert (
                library.zmq_getsockopt(
                    self.handle, option, ct.byref(number), ct.byref(size)
                )
                == 0
            )
            return number.value

        def close(self, linger=None):
            if not self.closed:
                if linger is not None:
                    self.setsockopt(17, linger)
                assert library.zmq_close(self.handle) == 0
                self.closed = True

    class Context:
        def __init__(self):
            self.handle = library.zmq_ctx_new()
            assert self.handle
            self.defaults, self.sockets, self.closed = {}, [], False
            contexts.append(self)

        def setsockopt(self, option, value):
            self.defaults[option] = value

        def socket(self):
            result = Socket(library.zmq_socket(self.handle, request_socket))
            for option, value in self.defaults.items():
                result.setsockopt(option, value)
            self.sockets.append(result)
            created_lingers.append(result.getsockopt(linger))
            return result

        def term(self):
            print("context_term_entered_after_timeout", flush=True)
            assert library.zmq_ctx_term(self.handle) == 0
            self.closed = True

        def destroy(self, linger=None):
            for current in self.sockets:
                current.close(linger=linger)
            self.term()

    class Client:
        def __init__(self, host, port, timeout_ms):
            assert timeout_ms == 120000
            self.context, self.host, self.port = Context(), host, port
            self._init_socket()

        def _init_socket(self):
            previous = getattr(self, "socket", None)
            self.socket = self.context.socket()
            self.socket.setsockopt(receive_timeout, 10)
            self.socket.setsockopt(send_timeout, 10)
            assert (
                library.zmq_connect(
                    self.socket.handle, f"tcp://{self.host}:{self.port}".encode()
                )
                == 0
            )
            # The pinned client drops its previous socket on replacement. Close
            # with its current linger value, as the pyzmq destructor does.
            if previous is not None:
                previous.close()

        def call_endpoint(self, endpoint, payload):
            assert endpoint == "feature_control" and payload["operation"] == "close"
            for _ in range(2):
                send_lingers.append(self.socket.getsockopt(linger))
                message, reply = (
                    ct.create_string_buffer(b"close"),
                    ct.create_string_buffer(32),
                )
                assert library.zmq_send(self.socket.handle, message, 5, 0) == 5
                assert library.zmq_recv(self.socket.handle, reply, len(reply), 0) == -1
                assert library.zmq_errno() == errno.EAGAIN
                print("native_receive_timeout", flush=True)
                self._init_socket()
            raise TimeoutError("cleanup RPC timed out")

    sys.modules["gr00t.policy.server_client"] = SimpleNamespace(PolicyClient=Client)
    sys.modules["zmq"] = SimpleNamespace(LINGER=linger)
    leases = ClientFeatureLeases({"owner": "owner", "epoch": 0, "session": "owned"})
    # Reserve a non-listening TCP port, so no unrelated server can accept this
    # test's REQ message. libzmq queues sends before a peer connects by default.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        try:
            close_remote_features(leases, reserved.getsockname()[1])
        except TimeoutError:
            pass
        else:
            raise AssertionError("Cleanup timeout must reach the caller")
    assert not leases.closed
    assert send_lingers == [0, 0]
    assert created_lingers == [-1, 0, 0]
    assert all(
        context.closed and all(current.closed for current in context.sockets)
        for context in contexts
    )
    print("cleanup_timeout_propagated_and_context_closed", flush=True)
