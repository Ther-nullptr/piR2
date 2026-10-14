"""Opt-in bit/ownership/stream gates for the actual CUDA feature wire codec."""

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PIR2_GPU_TESTS") != "1", reason="explicit CUDA validation only"
)


def test_all_bfloat16_bits_and_other_dtypes_survive_owned_cuda_roundtrip():
    import torch

    from coexecution.libero.features import decode_features, encode_features

    device = torch.device("cuda:0")
    producer, consumer = (
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
    )
    with torch.cuda.stream(producer):
        bits = torch.from_numpy(np.arange(65536, dtype=np.uint16).reshape(256, 256)).to(
            device
        )
        features = {
            "backbone_features": bits.view(torch.bfloat16),
            "mask": torch.tensor([[True, False]], device=device),
            "integer": torch.tensor(7, dtype=torch.int64, device=device),
            "float16": torch.tensor([0.125, -0.0], dtype=torch.float16, device=device),
            "float32": torch.tensor(
                [0.1234567, -0.0], dtype=torch.float32, device=device
            ),
            "float64": torch.tensor(
                [0.123456789012345, -0.0], dtype=torch.float64, device=device
            ),
        }
        payload = encode_features(features, "numpy_bf16", torch_module=torch)
        features["backbone_features"].zero_()
    with torch.cuda.stream(consumer):
        decoded = decode_features(
            payload, torch_module=torch, device=device, legacy_dtype=torch.bfloat16
        )
        decoded_again = decode_features(
            payload, torch_module=torch, device=device, legacy_dtype=torch.bfloat16
        )
    consumer.synchronize()
    assert np.array_equal(
        decoded["backbone_features"].view(torch.uint16).cpu().numpy(),
        np.arange(65536, dtype=np.uint16).reshape(256, 256),
    )
    for name, value in decoded.items():
        assert value.device == device
        assert value.dtype == features[name].dtype
        assert value.shape == features[name].shape
        assert value.data_ptr() != decoded_again[name].data_ptr()
        assert value.data_ptr() != features[name].data_ptr()
        storage = value.view(torch.uint16) if value.dtype == torch.bfloat16 else value
        assert (
            storage.cpu().numpy().tobytes() == payload["fields"][name]["data"].tobytes()
        )
    payload["fields"]["backbone_features"]["data"][:] = 0
    assert decoded["backbone_features"].view(torch.uint16)[255, 255].item() == 65535


def test_backend_install_finishes_on_owning_stream_with_fresh_tensors(tmp_path):
    import torch

    from coexecution.libero.backend import Backend
    from coexecution.libero.features import encode_features

    backend = Backend.__new__(Backend)
    backend.torch, backend.BatchFeature = torch, dict
    backend.stream = torch.cuda.Stream(device="cuda:0")
    backend.cache_meta, backend.cache_install_timing = {}, {}
    backend.core = SimpleNamespace(
        model=SimpleNamespace(device="cuda:0", dtype=torch.bfloat16),
        _vl_cache_lock=threading.Lock(),
    )
    observed = []

    def prime(features, capture):
        assert backend.stream.query()
        observed.append(features)
        return len(observed), capture

    backend.core._prime_vlm_cache = prime
    with torch.cuda.stream(backend.stream):
        features = {
            "backbone_features": torch.randn(
                2, 8, device="cuda:0", dtype=torch.bfloat16
            ),
            "mask": torch.ones(2, 8, device="cuda:0", dtype=torch.bool),
        }
        payload = encode_features(features, "numpy_bf16", torch_module=torch)
        backend.install(payload, {"capture_s": 1.0})
        backend.install(payload, {"capture_s": 2.0})
    for key, value in features.items():
        assert torch.equal(observed[0][key], value)
        assert observed[0][key].data_ptr() != observed[1][key].data_ptr()


@pytest.mark.parametrize("encoding", ["numpy_fp32", "numpy_bf16"])
def test_pinned_rpc_serializer_preserves_feature_dtype_shape_and_metadata(encoding):
    from gr00t.policy.server_client import MsgSerializer

    from coexecution.libero.features import encode_features, validate_features

    payload = encode_features(
        {
            "words": np.arange(65536, dtype=np.uint16),
            "mask": np.array(True),
            "indices": np.arange(6, dtype=np.int64).reshape(2, 3),
        },
        encoding,
    )
    if encoding == "numpy_bf16":
        payload["fields"]["words"]["dtype"] = "bfloat16"
    meta = {
        "capture_s": 12.5,
        "source_tick": 4,
        "requested_tick": 3,
        "frame_hashes": {"image": "abc"},
    }
    wire = MsgSerializer.to_bytes({"vl_embeds": payload, "feature_meta": meta})
    received = MsgSerializer.from_bytes(wire)
    actual_encoding, actual = validate_features(received["vl_embeds"])
    _, expected = validate_features(payload)
    assert actual_encoding == encoding
    assert received["feature_meta"] == meta
    for name, (dtype, array) in expected.items():
        assert actual[name][0] == dtype
        assert actual[name][1].shape == array.shape
        assert actual[name][1].dtype == array.dtype
        assert actual[name][1].tobytes() == array.tobytes()
