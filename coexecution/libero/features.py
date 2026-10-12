"""Versioned VLM feature transport, independent of weight/activation precision.

``numpy_fp32`` retains the upstream NumPy dictionary and floating install cast.
``numpy_bf16`` transports BF16 as explicitly tagged uint16 words and preserves
every other tensor dtype. Validation is host-only; torch is supplied by callers.
"""

import math
from numbers import Real

import numpy as np

FEATURE_TRANSPORTS = ("numpy_fp32", "numpy_bf16")
FEATURE_RPC_MODES = ("separate", "combined")
TIMING_VERSION = 2
SCHEMA = "libero.features"
SCHEMA_VERSION = 1
_NUMPY_DTYPES = {
    name: np.dtype(name)
    for name in (
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
}


def validate_feature_options(transport="numpy_fp32", rpc="separate"):
    if transport not in FEATURE_TRANSPORTS or rpc not in FEATURE_RPC_MODES:
        raise ValueError("Unknown feature transport or feature RPC mode")


def add_feature_arguments(parser, *, rpc=False):
    parser.add_argument(
        "--feature-transport",
        choices=FEATURE_TRANSPORTS,
        default="numpy_fp32",
        help="VLM wire representation; numpy_bf16 preserves exact BF16 bits",
    )
    if rpc:
        parser.add_argument(
            "--feature-rpc",
            choices=FEATURE_RPC_MODES,
            default="separate",
            help="Deployment: install a new feature within the action plan RPC",
        )


def validate_feature_identity(identity, requested):
    """An evaluator must use the representation declared for this condition."""
    actual = identity.get("feature_transport", "numpy_fp32")
    if actual != requested:
        raise ValueError(
            f"Feature transport mismatch: requested {requested}, worker {actual}"
        )


def feature_use_time(timing):
    """Prefer explicit server planning; keep legacy RPC-entry semantics labeled."""
    if timing.get("server_plan_started_s") is not None:
        return "server_plan_started_s", timing["server_plan_started_s"]
    return "plan_rpc_started_s", timing.get("plan_rpc_started_s")


def _array(value):
    if not isinstance(value, (np.ndarray, np.generic)):
        raise ValueError("Feature data must be a NumPy array or scalar")  # noqa: TRY004 -- invalid wire
    array = np.asarray(value)
    if array.dtype.name not in _NUMPY_DTYPES or not array.dtype.isnative:
        raise ValueError(f"Unsupported feature dtype: {array.dtype}")
    return array


def validate_features(payload):
    """Validate the complete wire before any transfer, returning tagged arrays."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Feature payload must be a nonempty dictionary")
    if "schema" not in payload:
        fields = {
            key: (array.dtype.name, array)
            for key, value in payload.items()
            for array in (_array(value),)
        }
        encoding = "numpy_fp32"
    else:
        if (
            set(payload) != {"schema", "version", "encoding", "fields"}
            or payload["schema"] != SCHEMA
            or type(payload["version"]) is not int
            or payload["version"] != SCHEMA_VERSION
            or payload["encoding"] != "numpy_bf16"
            or not isinstance(payload["fields"], dict)
            or not payload["fields"]
        ):
            raise ValueError("Unknown or malformed feature wire schema")
        encoding, fields = payload["encoding"], {}
        for key, field in payload["fields"].items():
            if not isinstance(field, dict) or set(field) != {"dtype", "shape", "data"}:
                raise ValueError("Malformed feature field")
            dtype, shape = field["dtype"], field["shape"]
            if not isinstance(dtype, str) or dtype not in {*_NUMPY_DTYPES, "bfloat16"}:
                raise ValueError("Unknown feature dtype tag")
            if not isinstance(shape, (list, tuple)) or any(
                type(size) is not int or size < 0 for size in shape
            ):
                raise ValueError("Malformed feature shape")
            array = _array(field["data"])
            storage = (
                np.dtype("uint16") if dtype == "bfloat16" else _NUMPY_DTYPES[dtype]
            )
            if array.dtype != storage or tuple(shape) != array.shape:
                raise ValueError("Feature dtype/shape does not match declared storage")
            fields[key] = (dtype, array)
    if any(not isinstance(key, str) or not key for key in fields):
        raise ValueError("Feature names must be nonempty strings")
    return encoding, fields


def encode_features(features, encoding="numpy_fp32", *, torch_module=None):
    """Snapshot into owned host arrays; BF16 uses a view, never a float cast.

    CUDA callers own the producing stream and must establish readiness before
    encoding. The blocking CPU copies complete before this function returns.
    """
    validate_feature_options(encoding)
    fields = {}
    for key, value in features.items():
        if torch_module is not None and torch_module.is_tensor(value):
            tensor = value.detach()
            if encoding == "numpy_bf16" and tensor.dtype == torch_module.bfloat16:
                dtype = "bfloat16"
                tensor = tensor.contiguous().view(torch_module.uint16)
            else:
                if encoding == "numpy_fp32" and tensor.dtype in (
                    torch_module.bfloat16,
                    torch_module.float16,
                ):
                    tensor = tensor.to(dtype=torch_module.float32)
                dtype = str(tensor.dtype).removeprefix("torch.")
            array = tensor.to("cpu").contiguous().numpy().copy()
        else:
            array = _array(value).copy(order="C")
            dtype = array.dtype.name
        fields[key] = {"dtype": dtype, "shape": list(array.shape), "data": array}
    payload = (
        {key: field["data"] for key, field in fields.items()}
        if encoding == "numpy_fp32"
        else {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "encoding": encoding,
            "fields": fields,
        }
    )
    validate_features(payload)
    return payload


def decode_features(payload, *, torch_module, device, legacy_dtype):
    """Decode fresh tensors on the caller's device/stream after full validation."""
    encoding, fields = validate_features(payload)
    decoded = {}
    for key, (dtype, array) in fields.items():
        # copy=True also owns storage when a caller requests the host device.
        tensor = torch_module.from_numpy(array.copy(order="C"))
        if dtype == "bfloat16":
            tensor = tensor.view(torch_module.bfloat16)
        target_dtype = (
            legacy_dtype
            if encoding == "numpy_fp32" and tensor.dtype.is_floating_point
            else tensor.dtype
        )
        decoded[key] = tensor.to(device=device, dtype=target_dtype, copy=True)
    return decoded


def feature_evidence(payload):
    encoding, fields = validate_features(payload)
    return {
        "encoding": encoding,
        "schema_version": SCHEMA_VERSION if encoding == "numpy_bf16" else None,
        "payload_nbytes": sum(array.nbytes for _, array in fields.values()),
        "field_dtypes": {key: dtype for key, (dtype, _) in fields.items()},
    }


def validate_feature_meta(meta, *, state_capture_s=None, request_tick=None):
    if not isinstance(meta, dict):
        raise ValueError("Feature metadata must be a dictionary")  # noqa: TRY004 -- invalid wire
    capture = meta.get("capture_s")
    if (
        isinstance(capture, bool)
        or not isinstance(capture, Real)
        or not math.isfinite(capture)
    ):
        raise ValueError("Feature capture must be finite")
    if state_capture_s is not None and (
        not math.isfinite(state_capture_s) or capture > state_capture_s + 1e-6
    ):
        raise ValueError("Future image cannot condition an older state")
    tick = meta.get("source_tick")
    if tick is not None and (
        type(tick) is not int or (request_tick is not None and tick > request_tick)
    ):
        raise ValueError("Future or invalid source tick")
