"""Public-safe runtime identities captured explicitly outside inference timers.

Importing this module loads no neural framework and performs no device work.
Receipts describe the installed configuration and observed loaded binaries;
only a separate source-matched numerical gate may attest GPU validation.
"""

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

SCHEMA = "pir2.runtime.receipt"
VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".c", ".cc", ".inl"}
PATH_CONFIG = {
    "coverage",
    "tactics",
    "group_tactics",
    "shape_tactics",
    "operator_profile",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE = re.compile(r"^[A-Za-z0-9_ .():+-]{1,160}$")
_LIBRARY = re.compile(
    r"^(pir2_[a-z0-9_]+)_([0-9a-f]{64})_ext(?:_v[0-9]+)?\.(?:so|pyd|dll)$"
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path):
    path = Path(path)
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_ino,
    ):
        raise RuntimeError("Runtime input changed while hashing")
    return value.hexdigest()


def public_value(value):
    """Allow structured operator facts, excluding paths, credentials and free text."""
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return (
            value
            if _SAFE.fullmatch(value) and not value.startswith(("hf_", "sk-"))
            else None
        )
    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: public_value(item)
            for key, item in value.items()
            if isinstance(key, str)
            and _SAFE.fullmatch(key)
            and not any(
                word in key.lower()
                for word in (
                    "password",
                    "secret",
                    "auth",
                    "token",
                    "path",
                    "directory",
                    "filename",
                )
            )
        }
    return None


def source_manifest(source_root=None):
    """Hash the actual owned runtime tree, including untracked integration files."""
    root = ROOT if source_root is None else Path(source_root).resolve()
    if root != ROOT:
        raise ValueError(
            "Declared runtime root differs from the imported receipt module"
        )
    files = {}
    for path in sorted((root / "coexecution").rglob("*")):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and not {"vendor", "third_party", "__pycache__"}.intersection(
                relative.parts
            )
        ):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Runtime source escapes the declared root")
            files[relative.as_posix()] = file_digest(path)
    for name in ("sources.lock.json", "coexecution/kernels/SOURCE_MANIFEST.json"):
        if (root / name).is_file():
            files[name] = file_digest(root / name)
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("coexecution."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Loaded coexecution module came from another runtime root")
        relative = path.relative_to(root).as_posix()
        if relative in files:
            loaded[name] = relative
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    head = commit.stdout.strip()
    return {
        "manifest_sha256": digest(files),
        "files": files,
        "loaded_modules": loaded,
        "git_commit": head if re.fullmatch(r"[0-9a-f]{40,64}", head) else None,
        "identity_basis": "actual_source_content_not_commit_or_date_alone",
    }


def loaded_binaries(torch_module):
    """Observe loaded paths, never search a requested extension/cache directory."""
    if torch_module is None:
        return {
            "status": "unobserved",
            "libraries": [],
            "scope": "process_loaded_inventory_not_dispatch_proof",
        }
    rows = []
    for filename in sorted(torch_module.ops.loaded_libraries):
        path = Path(filename)
        match = _LIBRARY.fullmatch(path.name)
        if match is None:
            continue
        namespace, embedded = match.groups()
        namespace_ops = getattr(torch_module.ops, namespace, None)
        method = getattr(namespace_ops, "source_identity", None)
        registered = method() if callable(method) else None
        if registered is not None and not _SHA.fullmatch(str(registered)):
            raise ValueError("Loaded operator returned an invalid source identity")
        if registered is not None and registered != embedded:
            raise RuntimeError("Loaded binary and registered operator source disagree")
        rows.append(
            {
                "filename": path.name,
                "sha256": file_digest(path),
                "namespace": namespace,
                "embedded_source_sha256": embedded,
                "registered_source_sha256": registered,
                "identity_verified": registered == embedded,
            }
        )
    return {
        "status": "observed",
        "libraries": rows,
        "scope": "process_loaded_inventory_not_dispatch_proof",
    }


def loaded_model_sources():
    """Identify the loaded GR00T package without exposing its local directory."""
    package = sys.modules.get("gr00t")
    filename = getattr(package, "__file__", None)
    if filename is None:
        return {"status": "unobserved", "files": {}, "manifest_sha256": None}
    root = Path(filename).resolve().parent
    files = {}
    for name, module in tuple(sys.modules.items()):
        if name != "gr00t" and not name.startswith("gr00t."):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Loaded model modules came from different source roots")
        files["gr00t/" + path.relative_to(root).as_posix()] = file_digest(path)
    return {"status": "observed", "files": files, "manifest_sha256": digest(files)}


def capture_runtime_receipt(
    session,
    *,
    variant,
    measurement_boundary,
    torch_module=None,
    source_root=None,
    validation=None,
):
    """Capture installed facts after warmup/execution, outside clean model timers.

    ``variant`` describes actual model semantics (Flow4 or piR2), not a frontend
    display choice. Callers retain the matching receipt with each mode/run.
    Loading or selecting a profile alone never changes validation to verified.
    """
    from coexecution.optimization.config import OptimizationConfig

    if variant not in ("flow", "pir2"):
        raise ValueError("Unknown model variant")
    if measurement_boundary not in (
        "prepared_model",
        "policy",
        "preparation_transfer",
        "comparison",
        "model-only",
        "deployment",
    ):
        raise ValueError("Unknown inference measurement boundary")
    evidence = session.evidence()
    model = getattr(getattr(session, "policy", None), "model", None)
    model_config = getattr(model, "config", None)
    streaming = getattr(model_config, "streaming", None)
    if streaming is not None and bool(streaming) != (variant == "pir2"):
        raise ValueError("Receipt variant differs from the loaded model")
    effective = evidence["config"]
    config = {}
    for field in fields(OptimizationConfig):
        value = effective.get(field.name)
        if field.name in PATH_CONFIG:
            checksum = value.get("sha256") if isinstance(value, dict) else None
            if value is not None and (
                not isinstance(checksum, str) or not _SHA.fullmatch(checksum)
            ):
                raise ValueError("File-backed configuration requires an actual SHA256")
            config[field.name] = None if value is None else {"sha256": checksum}
        else:
            config[field.name] = public_value(value)
    coverage = evidence.get("coverage", {})
    linears = coverage.get("linears", [])
    profile = evidence.get("operator_selection")
    sources = source_manifest(source_root)
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "variant": variant,
        "measurement_boundary": measurement_boundary,
        "model_contract": {
            "streaming": streaming,
            "action_horizon": getattr(model_config, "action_horizon", None),
            "num_inference_timesteps": getattr(
                getattr(model, "action_head", None), "num_inference_timesteps", None
            ),
        },
        "source": sources,
        "model_source": loaded_model_sources(),
        "configuration": config,
        "configuration_sha256": digest(config),
        "operator_profile": public_value(profile),
        "operator_profile_sha256": None
        if config.get("operator_profile") is None
        else config["operator_profile"]["sha256"],
        "coverage": {
            "linear_count": len(linears),
            "linears_sha256": digest(public_value(linears)),
            "category_linear_count": len(coverage.get("category_linears", [])),
            "operator_adapters": public_value(coverage.get("operator_adapters", {})),
        },
        "cache_clear_policy": public_value(evidence.get("cache_clear_policy")),
        "enabled_adapters": {
            name: public_value(evidence.get(name))
            for name in (
                "compiled_host_guards",
                "input_preparation",
                "tokenizer_cache",
                "image_array_preparation",
                "dit_graph",
                "action_graph",
                "static_vlm",
            )
            if evidence.get(name) is not None
        },
        "backend": public_value(
            {
                key: evidence.get("backend", {}).get(key)
                for key in (
                    "backend_id",
                    "backend_revision",
                    "device_arch",
                    "quant_format",
                    "runtime",
                )
            }
        ),
        "binaries": loaded_binaries(torch_module),
        "jit_binary_identity": "not_collected",
        "freshness": "identity_recorded_no_automatic_current_label",
    }
    if torch_module is not None and model is not None:
        device = model.device
        props = torch_module.cuda.get_device_properties(device)
        gpu_uuid = getattr(props, "uuid", None)
        receipt["device"] = {
            "logical_device": str(device),
            "name": public_value(props.name),
            "compute_capability": [props.major, props.minor],
            "uuid_sha256": None
            if gpu_uuid is None
            else hashlib.sha256(str(gpu_uuid).encode()).hexdigest(),
            "torch": str(torch_module.__version__),
            "cuda": str(torch_module.version.cuda),
            "clock_sample": None,
        }
    # Replay counters and graph IDs describe observations, not a different
    # installed implementation. Keep their snapshots outside the stable identity.
    identity = {
        key: value
        for key, value in receipt.items()
        if key not in ("source", "enabled_adapters")
    }
    identity["source_manifest_sha256"] = sources["manifest_sha256"]
    identity["compiled_host_guards"] = receipt["enabled_adapters"].get(
        "compiled_host_guards"
    )
    receipt["identity_sha256"] = digest(identity)
    receipt["validation"] = {
        "status": "pending",
        "evidence_sha256": None,
        "runtime_identity_sha256": receipt["identity_sha256"],
        "task_quality": "not_established",
    }
    if validation is not None:
        status = validation.get("status")
        if status not in ("pending", "metadata_only", "gpu_verified"):
            raise ValueError("Unknown runtime validation status")
        if status == "gpu_verified" and (
            validation.get("runtime_identity_sha256") != receipt["identity_sha256"]
            or not _SHA.fullmatch(str(validation.get("evidence_sha256", "")))
        ):
            raise ValueError(
                "GPU validation must reference this exact runtime identity"
            )
        receipt["validation"].update(
            status=status,
            evidence_sha256=validation.get("evidence_sha256")
            if status == "gpu_verified"
            else None,
        )
    return receipt
