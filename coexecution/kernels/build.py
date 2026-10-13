"""Resolve and explicitly build repository-owned CUDA translation units.

Metadata access imports neither Torch nor Triton. CUDA is compiled only when
``build_extension`` or the module's command-line build is explicitly invoked.
There are no external source-root or prebuilt-library overrides.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_FAILED_BUILDS = set()
BACKENDS = {
    "integer": {
        "namespace": "pir2_integer",
        "sources": ("ampere_ada/csrc/integer_gemm.cu",),
        "cutlass": True,
        "cuda_flags": ("--fmad=false",),
        "operators": ("gemm", "gemm_biasless"),
    },
    "integer_dispatch": {
        "namespace": "pir2_integer_dispatch",
        "sources": ("ampere_ada/csrc/integer_dispatch.cu",),
        "cutlass": False,
        "cuda_flags": ("--fmad=false", "--ftz=false"),
        "operators": ("linear", "prepare"),
    },
    "bf16_fusion": {
        "namespace": "pir2_bf16_fusion",
        "sources": ("common/csrc/fusion_torch.cu",),
        "headers": ("common/csrc/vision_rope.cu", "common/csrc/adaln_modulation.cu"),
        "cutlass": False,
        "cuda_flags": ("--fmad=false", "--ftz=false"),
        "operators": ("vision_rope", "adaln_modulation"),
    },
    "fp8": {
        "namespace": "pir2_cutlass_fp8",
        "sources": ("blackwell/csrc/fp8_cutlass_linear.cu",),
        "headers": ("blackwell/csrc/sampling_bf16.cuh",),
        "cutlass": True,
        "cuda_flags": ("--use_fast_math",),
        "arch": "11.0a",
        "operators": (
            "supports_device",
            "set_decode_policy",
            "set_swap_ab_enabled",
            "pack_linear_weight",
            "quantize_bf16",
            "linear_forward_packed_tactic",
        ),
    },
    "fp4": {
        "namespace": "pir2_cutlass_fp4",
        "sources": ("blackwell/csrc/fp4_cutlass_linear.cu",),
        "headers": ("blackwell/csrc/sampling_bf16.cuh",),
        "cutlass": True,
        "cuda_flags": ("--use_fast_math",),
        "arch": "11.0a",
        "operators": (
            "supports_device",
            "set_decode_policy",
            "pack_linear_weight_gemm_bf16",
            "pack_activation_gemm_bf16",
            "linear_forward_packed_gemm_bf16_tactic",
            "linear_forward_packed_gemm_bf16_swap_ab",
        ),
    },
}


def _owned_path(relative):
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(
            f"Kernel build input escapes the owned source tree: {relative}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Owned kernel build input is missing: {relative}")
    return path


def build_inputs(backend="integer"):
    """Return build metadata without importing Torch or probing the GPU."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown owned kernel backend: {backend}")
    spec = BACKENDS[backend]
    sources = [_owned_path(name) for name in spec["sources"]]
    sources.append(_owned_path("common/source_identity.cpp"))
    headers = [_owned_path(name) for name in spec.get("headers", ())]
    includes = []
    cutlass_revision = None
    if spec["cutlass"]:
        from coexecution.kernels.sources import require_submodule

        cutlass = require_submodule()
        directory = Path(cutlass["directory"])
        includes = [
            directory / "include",
            directory / "tools/util/include",
        ]
        cutlass_revision = cutlass["revision"]
    return {
        "backend": backend,
        "namespace": spec["namespace"],
        "sources": [str(path) for path in sources],
        "headers": [str(path) for path in headers],
        "include_paths": [str(path) for path in includes],
        "cutlass_revision": cutlass_revision,
        "cuda_flags": list(spec["cuda_flags"]),
        "architecture": spec.get("arch"),
        "operators": list(spec["operators"]),
    }


def _registered_extension(ops, inputs, identity):
    if (
        not hasattr(ops, "source_identity")
        or ops.source_identity() != identity["source_sha256"]
        or any(not hasattr(ops, name) for name in inputs["operators"])
    ):
        raise RuntimeError(
            f"Registered {inputs['namespace']} source identity differs; restart the process"
        )
    return ops


def build_extension(backend, *, verbose=False):
    """Explicitly build/load the selected source; failures never select a fallback."""
    inputs = build_inputs(backend)
    from coexecution.kernels.sources import verified_source_identity

    identity = verified_source_identity(backend=backend)
    import torch

    ops = getattr(torch.ops, inputs["namespace"])
    if hasattr(ops, "source_identity") or any(
        hasattr(ops, name) for name in inputs["operators"]
    ):
        return _registered_extension(ops, inputs, identity)
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Build and warm owned kernels before CUDA Graph capture")
    name = inputs["namespace"] + "_" + identity["source_sha256"] + "_ext"
    if name in _FAILED_BUILDS:
        raise RuntimeError(
            "Owned extension build previously failed; restart after fixing the build"
        )
    from torch.utils.cpp_extension import load

    definitions = [
        f"-DPIR2_KERNEL_NAMESPACE={inputs['namespace']}",
        f'-DPIR2_BUILD_SOURCE_ID=\\"{identity["source_sha256"]}\\"',
    ]
    flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        *inputs["cuda_flags"],
        # Includes the actual dependency revision, or its header fingerprint
        # for modified/non-Git trees; the recommended pin is not a version gate.
        f"-DPIR2_SOURCE_{identity['source_sha256']}=1",
        *definitions,
    ]
    if inputs["architecture"] is not None:
        arch = inputs["architecture"].replace(".", "")
        flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ.setdefault("MAX_JOBS", "2")
    try:
        load(
            name=name,
            sources=inputs["sources"],
            extra_include_paths=inputs["include_paths"],
            extra_cflags=["-O3", "-std=c++17", *definitions],
            extra_cuda_cflags=flags,
            is_python_module=False,
            verbose=verbose,
        )
        return _registered_extension(ops, inputs, identity)
    except Exception:
        _FAILED_BUILDS.add(name)
        raise
    finally:
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--describe", action="store_true", help="Print metadata only")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(build_inputs(args.backend), indent=2))
    else:
        print(build_extension(args.backend, verbose=args.verbose))


if __name__ == "__main__":
    main()
