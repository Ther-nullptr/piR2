"""Check paired device mapping without loading or validating a model on CPU."""

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
