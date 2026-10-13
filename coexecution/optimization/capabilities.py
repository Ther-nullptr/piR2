"""Backend metadata and opt-in checks; importing this module is GPU-free.

Integer loading preserves the pinned backend's CUDA contract without adding a
device whitelist. SM89 is measured coverage, not a claim that other devices
work. Thor floating adapters retain their existing SM110 requirement. Loader
failures propagate; there is no automatic precision or backend fallback.
"""

import hashlib
import importlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

BACKENDS = {
    "w8a8": {
        "backend_id": "robotics_integer",
        "quant_format": "w8a8-symmetric-row-v1",
        "architectures": None,
        "validated_architectures": ("sm_89",),
        "module": "robotics_kernels.ampere_ada.integer",
        "loader": "load_integer_extension",
        "operators": ("gemm", "gemm_biasless"),
    },
    "w4a4": {
        "backend_id": "robotics_integer",
        "quant_format": "w4a4-symmetric-row-v1",
        "architectures": None,
        "validated_architectures": ("sm_89",),
        "module": "robotics_kernels.ampere_ada.integer",
        "loader": "load_integer_extension",
        "operators": ("gemm", "gemm_biasless"),
    },
    "fp8": {
        "backend_id": "robotics_cutlass_fp8",
        "quant_format": "fp8-e4m3-per-tensor-v1",
        "architectures": ((11, 0),),
        "validated_architectures": ("sm_110",),
        "module": "robotics_kernels.blackwell.fp8_linear",
        "loader": "require_fp8_backend",
        "operators": (
            "pack_linear_weight",
            "quantize_bf16",
            "linear_forward_packed_tactic",
        ),
    },
    "fp4": {
        "backend_id": "robotics_cutlass_fp4",
        "quant_format": "fp4-e2m1-block16-ue4m3-v1",
        "architectures": ((11, 0),),
        "validated_architectures": ("sm_110",),
        "module": "robotics_kernels.blackwell.fp4_linear",
        "loader": "require_fp4_backend",
        "operators": (
            "pack_linear_weight_gemm_bf16",
            "pack_activation_gemm_bf16",
            "linear_forward_packed_gemm_bf16_tactic",
        ),
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(module):
    """Record source evidence independently of the binary loaded into this process."""
    pin = json.loads(
        (Path(__file__).resolve().parents[2] / "sources.lock.json").read_text()
    )["sources"]["robotics-kernels"]
    result = {
        "expected_revision": pin["revision"],
        "url": pin["url"],
        "revision": "unknown",
        "dirty": "unknown",
        "module_tracked": "unknown",
        "module_sha256": "unknown",
    }
    filename = getattr(module, "__file__", None)
    if filename is None:
        return result
    path = Path(filename).resolve()
    try:
        result["module_sha256"] = _sha256(path)
        command = ["git", "-C", str(path.parent)]
        repository = Path(
            subprocess.check_output(
                [*command, "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        ).resolve()
        # Do not mistake a surrounding application checkout for the backend source.
        if (
            not path.relative_to(repository)
            .as_posix()
            .startswith("src/robotics_kernels/")
        ):
            return result
        try:
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repository),
                    "ls-files",
                    "--error-unmatch",
                    "--",
                    path.relative_to(repository).as_posix(),
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except subprocess.CalledProcessError:
            result["module_tracked"] = False
            return result
        result["module_tracked"] = True
        result["revision"] = subprocess.check_output(
            [*command, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        result["dirty"] = bool(
            subprocess.check_output(
                [*command, "status", "--porcelain", "--untracked-files=all"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def _loaded_libraries(torch):
    return {str(Path(path).resolve()) for path in torch.ops.loaded_libraries}


def _binary_identity(before, after, namespace, had_operators):
    """Hash only actual loaded paths, never an environment variable's requested .so."""
    candidates = []
    for filename in sorted(after):
        path = Path(filename)
        if namespace not in path.name:
            continue
        try:
            digest = _sha256(path)
        except OSError:
            digest = "unknown"
        candidates.append(
            {
                "filename": path.name,
                "sha256": digest,
                "newly_loaded": filename not in before,
            }
        )
    observed = [row for row in candidates if row["newly_loaded"]]
    identified = not had_operators and len(observed) == 1
    return {
        "status": "observed_backend_load" if identified else "unknown",
        "selected": observed[0] if identified else None,
        "loaded_candidates": candidates,
        "note": (
            "Library appeared while the backend loader registered missing operators"
            if identified
            else "Preloaded operators or ambiguous libraries cannot be attributed from torch.ops"
        ),
    }


def _missing_operators(torch, spec):
    namespace = getattr(torch.ops, spec["backend_id"], None)
    return [
        name
        for name in spec["operators"]
        if not callable(getattr(namespace, name, None))
    ]


def preflight_backend(config, device):
    """Validate the selected runtime before model patches; never execute a model."""
    import torch

    if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
        raise ValueError("Optimization preflight requires a CUDA device")
    precision = config.precision
    if precision not in ("bf16", *BACKENDS):
        raise ValueError(f"Unknown optimization precision: {precision}")
    capability = tuple(torch.cuda.get_device_capability(device))
    required_arches = (
        None if precision == "bf16" else BACKENDS[precision]["architectures"]
    )
    if required_arches is not None and capability not in required_arches:
        raise ValueError(
            f"{precision} has no supported backend for SM{capability[0]}{capability[1]}"
        )
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = "unknown"
    report = {
        "backend_id": "torch",
        "backend_revision": str(torch.__version__),
        "device_arch": f"sm_{capability[0]}{capability[1]}",
        "quant_format": "bf16",
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "triton": triton_version,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
    }
    if precision != "bf16":
        spec = BACKENDS[precision]
        if precision == "fp4" and getattr(config, "fp4_swap_ab", False):
            spec = dict(
                spec,
                operators=(
                    *spec["operators"],
                    "linear_forward_packed_gemm_bf16_swap_ab",
                ),
            )
        before = _loaded_libraries(torch)
        missing_before = _missing_operators(torch, spec)
        backend = importlib.import_module(spec["module"])
        loader = getattr(backend, spec["loader"])
        loader() if precision in ("w8a8", "w4a4") else loader(device)
        missing = _missing_operators(torch, spec)
        if missing:
            raise RuntimeError(
                f"{spec['backend_id']} is missing required GR00T operators: {', '.join(missing)}"
            )
        source = _source_identity(backend)
        report.update(
            backend_id=spec["backend_id"],
            backend_revision=source["revision"]
            if source["dirty"] is False
            else "unknown",
            quant_format=spec["quant_format"],
            source=source,
            validated_architectures=list(spec["validated_architectures"]),
            runtime_arch_validated=report["device_arch"]
            in spec["validated_architectures"],
            architecture_policy=(
                "Pinned integer loader; no additional device whitelist"
                if precision in ("w8a8", "w4a4")
                else "Existing Thor adapter requires SM110"
            ),
            binary=_binary_identity(
                before,
                _loaded_libraries(torch),
                spec["backend_id"],
                len(missing_before) != len(spec["operators"]),
            ),
            required_operators=list(spec["operators"]),
        )
    if config.shape_tactics is not None:
        from coexecution.optimization.tactics import read_shape_tactics

        read_shape_tactics(config.shape_tactics, backend_evidence=report)
    return report
