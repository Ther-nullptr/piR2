"""Check paired device mapping without loading or validating a model on CPU."""

import json
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from libero_inference_backend import endpoint_layout


def test_same_gpu_is_two_endpoints_on_one_device():
    layout = endpoint_layout("cuda:0", "cuda:0", 5570, 5572, "action")
    assert [(x["role"], x["device"], x["port"]) for x in layout] == [
        ("action", "cuda:0", 5570),
        ("vlm", "cuda:0", 5572),
    ]


def test_dual_gpu_preserves_explicit_mapping():
    layout = endpoint_layout("cuda:0", "cuda:1", 5570, 5572, "action")
    assert layout[1]["device"] == "cuda:1"


@pytest.mark.parametrize(
    "device,slow,port,slow_port,role",
    [
        ("cpu", None, 5570, None, "action"),
        ("cuda:0", "cpu", 5570, 5572, "action"),
        ("cuda:0", "cuda:0", 5570, 5570, "action"),
        ("cuda:0", "cuda:1", 5570, None, "action"),
        ("cuda:0", None, 5570, 5572, "action"),
        ("cuda:0", "cuda:1", 5570, 5572, "vlm"),
    ],
)
def test_invalid_layout_is_rejected(device, slow, port, slow_port, role):
    with pytest.raises(ValueError):
        endpoint_layout(device, slow, port, slow_port, role)


def test_paired_startup_failure_reaches_parent():
    import threading

    from serve_libero_protocol import supervise_workers

    release = threading.Event()

    def worker(value):
        if value == "fail":
            raise OSError("port in use")
        release.wait(2)

    try:
        with pytest.raises(RuntimeError, match="port in use"):
            supervise_workers([("peer",), ("fail",)], worker)
    finally:
        release.set()


@pytest.mark.parametrize(
    "flags",
    [["--operator-fusion"], ["--inference-precision", "w8a8"], ["--dit-cuda-graph"]],
)
@pytest.mark.parametrize("vlm_device", ["cuda:0", "cuda:1"])
def test_optimized_paired_service_is_rejected_before_model_loading(
    monkeypatch, flags, vlm_device
):
    import serve_libero_protocol as service

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_libero_protocol.py",
            "--checkpoint",
            "unused-checkpoint",
            "--output",
            "unused-output",
            "--variant",
            "flow",
            "--vlm-device",
            vlm_device,
            "--vlm-port",
            "5572",
            *flags,
        ],
    )
    with pytest.raises(ValueError, match="single worker"):
        service.main()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [None, "bind", "run"])
def test_service_optimization_owns_worker_stream_and_restores_on_exit(
    tmp_path, monkeypatch, enabled, failure
):
    import serve_libero_protocol as service

    from coexecution.groot_optimization import OptimizationConfig

    active = []
    endpoints = {}
    stream = object()
    core = object()

    @contextmanager
    def context(value):
        active.append(value)
        try:
            yield
        finally:
            assert active.pop() == value

    def install(scope):
        assert scope.policy is core
        assert active == ["cuda:1", stream]
        active.append("optimized")
        scope.stack.callback(active.pop)

    monkeypatch.setattr(service.GrootOptimizations, "_install", install)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device=context,
                stream=context,
                nvtx=SimpleNamespace(range=lambda _: nullcontext()),
            )
        ),
    )

    class Server:
        def __init__(self, policy, *, host, port):
            assert policy is core
            assert (host, port) == ("127.0.0.1", 5570)
            assert active == ["cuda:1", stream] + (["optimized"] if enabled else [])
            if failure == "bind":
                raise OSError("bind failure")

        def register_endpoint(self, name, handler, **kwargs):
            endpoints[name] = handler

        def run(self):
            observed = endpoints["identity"]()
            assert ("inference_optimization" in observed) == enabled
            if enabled:
                saved = json.loads((tmp_path / "identity.json").read_text())
                assert saved == observed
                assert saved["inference_optimization"]["config"]["fusion"]
            assert endpoints["plan"](request_tick=12) == {"request_tick": 12}
            if failure == "run":
                raise OSError("run failure")

    monkeypatch.setitem(
        sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyServer=Server)
    )
    backend = SimpleNamespace(core=core, stream=stream, output=tmp_path)
    for name in ["reset", "bootstrap", "vision", "install", "plan"]:
        setattr(backend, name, lambda **kwargs: kwargs)
    expected = pytest.raises(OSError, match=failure) if failure else nullcontext()
    with expected:
        service.serve(
            backend,
            {"device": "cuda:1", "role": "action", "port": 5570},
            {"variant": "flow"},
            OptimizationConfig(fusion=True) if enabled else None,
        )
    assert active == []
