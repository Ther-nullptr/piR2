"""Optional exact output epilogues for the pinned packed integer GEMM.

The reusable run template is SHA256-verified and adapted only in Torch's local
build cache. The residual writer preserves the original BF16 GEMM boundary,
then adds a BF16 residual in FP32 and rounds to BF16 again. The grouped writer
stores that original GEMM result directly into contiguous sibling segments.
Load and warm the extension before CUDA Graph capture. Default paths are unchanged.
"""

import hashlib
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

from coexecution.kernels.ampere_ada._candidate_support import (
    is_ada_device,
    owned_build_identity,
    projection_for_rows,
    require_pre_capture_build,
)

_BACKEND_SHA256 = "78689a96089653d3cf2f9fabb1fa64eeb5bb7ee835b7d3316297c7ef884a431f"
_CUTLASS_REVISION = "982748aa7356fa838c2ea4994ddcb0b2a4b4cefa"
_DISPATCH_MARKER = b"template <int Bits, bool WithBias>\nat::Tensor dispatch_tactic"
_CACHED_HEADER = "pir2_integer_epilogue_backend.cuh"
_FAILED_BUILDS = set()


def _pinned_run_header(inputs):
    """Adapt the pinned run template's output tree, excluding its dispatch."""
    if inputs.get("cutlass_revision") != _CUTLASS_REVISION:
        raise RuntimeError("CUTLASS revision does not match the pinned snapshot")
    try:
        source = Path(inputs["sources"][0]).read_bytes()
    except (OSError, KeyError, IndexError) as error:
        raise RuntimeError("Pinned integer GEMM source is unavailable") from error
    if hashlib.sha256(source).hexdigest() != _BACKEND_SHA256:
        raise RuntimeError("Integer GEMM source does not match the pinned SHA256")
    prefix, marker, _ = source.partition(_DISPATCH_MARKER)
    if not marker:
        raise RuntimeError("Pinned integer GEMM source has no run-template boundary")
    # VisitorCompute converts its result to ElementOutput before returning its
    # fragment. This identity node therefore rounds Sum to BF16 before Add
    # converts both children to FP32. AuxStore performs the final BF16 round.
    replacements = (
        (
            b"          int WarpN, int Stages>",
            b"          int WarpN, int Stages, bool GroupWrite = false>",
        ),
        (
            b"const at::Tensor &sb, const c10::optional<at::Tensor> &bias) {",
            (
                b"const at::Tensor &sb, const c10::optional<at::Tensor> &bias,\n"
                b"               const at::Tensor &residual,\n"
                b"               pir2_epilogue::GroupSizes group_sizes = {}) {"
            ),
        ),
        (
            (
                b"  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<\n"
                b"      Map, Output, cutlass::FloatRoundStyle::round_to_nearest,\n"
                b"      Stride<int64_t, _1, int64_t>>;"
            ),
            (
                b"  using Round = cutlass::epilogue::threadblock::VisitorCompute<\n"
                b"      pir2_epilogue::Identity, Output, float,\n"
                b"      cutlass::FloatRoundStyle::round_to_nearest>;\n"
                b"  using Rounded = cutlass::epilogue::threadblock::Sm80EVT<Round, Sum>;\n"
                b"  using Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<\n"
                b"      Map, Output, Stride<int64_t, _1, int64_t>>;\n"
                b"  using ResidualSum =\n"
                b"      cutlass::epilogue::threadblock::Sm80EVT<Add, Rounded, Residual>;\n"
                b"  using Child = std::conditional_t<GroupWrite, Sum, ResidualSum>;\n"
                b"  using AffineStore = cutlass::epilogue::threadblock::VisitorAuxStore<\n"
                b"      Map, Output, cutlass::FloatRoundStyle::round_to_nearest,\n"
                b"      Stride<int64_t, _1, int64_t>>;\n"
                b"  using Store = std::conditional_t<\n"
                b"      GroupWrite, pir2_epilogue::GroupStore<\n"
                b"          Map, Output, cutlass::FloatRoundStyle::round_to_nearest,\n"
                b"          Stride<int64_t, _1, int64_t>>, AffineStore>;"
            ),
        ),
        (
            b"Sm80EVT<Store, Sum>;",
            b"Sm80EVT<Store, Child>;",
        ),
        (
            b"  auto output = at::empty({m, n}, a.options().dtype(at::kBFloat16));",
            (
                b"  auto output = [&]() {\n"
                b"    if constexpr (GroupWrite)\n"
                b"      return at::empty({m * n}, a.options().dtype(at::kBFloat16));\n"
                b"    else\n"
                b"      return at::empty({m, n}, a.options().dtype(at::kBFloat16));\n"
                b"  }();"
            ),
        ),
        (
            (
                b"  typename Tree::Arguments callbacks{\n"
                b"      sum, {reinterpret_cast<Output *>(output.data_ptr()), {n, _1{}, m * n}}};"
            ),
            (
                b"  typename Tree::Arguments callbacks = [&]() {\n"
                b"    if constexpr (GroupWrite) {\n"
                b"      return typename Tree::Arguments{\n"
                b"          sum, {{reinterpret_cast<Output *>(output.data_ptr()),\n"
                b"                 {n, _1{}, m * n}}, group_sizes}};\n"
                b"    } else {\n"
                b"      typename Rounded::Arguments rounded{sum, {}};\n"
                b"      typename Residual::Arguments residual_args{\n"
                b"          reinterpret_cast<Output *>(residual.data_ptr()),\n"
                b"          Output(0), {n, _1{}, m * n}};\n"
                b"      typename ResidualSum::Arguments residual_sum{\n"
                b"          rounded, residual_args, {}};\n"
                b"      return typename Tree::Arguments{\n"
                b"          residual_sum,\n"
                b"          {reinterpret_cast<Output *>(output.data_ptr()),\n"
                b"           {n, _1{}, m * n}}};\n"
                b"    }\n"
                b"  }();"
            ),
        ),
    )
    for original, replacement in replacements:
        if prefix.count(original) != 1:
            raise RuntimeError(
                "Pinned integer GEMM source has no unique epilogue boundary"
            )
        prefix = prefix.replace(original, replacement, 1)
    return b"#pragma once\n" + prefix + b"} // namespace\n"


def _cache_run_header(directory, content):
    """Publish a whole generated header atomically inside the build cache."""
    header = directory / _CACHED_HEADER
    if header.exists() and header.read_bytes() == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, header)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _registered_extension(identity):
    extension = torch.ops.pir2_integer_epilogue
    if (
        not hasattr(extension, "gemm")
        or not hasattr(extension, "source_identity")
        or extension.source_identity() != identity
    ):
        raise RuntimeError(
            "Registered epilogue extension source identity differs; restart the process"
        )
    return extension


@lru_cache(maxsize=1)
def load_epilogue_extension():
    """Load the separate Ada output writer from verified, fixed build inputs."""
    from coexecution.kernels.ampere_ada.build import build_inputs

    inputs = build_inputs()
    header = _pinned_run_header(inputs)
    wrapper = Path(__file__).parent / "csrc/integer_epilogue.cu"
    identity = hashlib.sha256(
        header + b"\0" + wrapper.read_bytes() + b"\0" + owned_build_identity()
    ).hexdigest()
    name = f"pir2_integer_epilogue_{identity}_ext"
    if name in _FAILED_BUILDS:
        raise RuntimeError(
            "Epilogue extension build previously failed for this source identity; "
            "restart the process after fixing the build"
        )
    if hasattr(torch.ops.pir2_integer_epilogue, "gemm"):
        return _registered_extension(identity)
    require_pre_capture_build()
    from torch.utils.cpp_extension import _get_build_directory, load

    directory = Path(_get_build_directory(name, verbose=False))
    _cache_run_header(directory, header)
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "2"
    source_define = f'-DPIR2_EPILOGUE_SOURCE_ID=\\"{identity}\\"'
    try:
        load(
            name=name,
            sources=[str(wrapper)],
            build_directory=str(directory),
            extra_include_paths=[str(directory), *inputs["include_paths"]],
            extra_cflags=["-O3", "-std=c++17", source_define],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--fmad=false",
                "--ftz=false",
                "-gencode=arch=compute_89,code=sm_89",
                source_define,
            ],
            is_python_module=False,
            verbose=False,
        )
        return _registered_extension(identity)
    except Exception:
        # JIT records a version before compilation finishes. Do not let a retry
        # of that failed identity reuse a previously compiled binary.
        _FAILED_BUILDS.add(name)
        raise
    finally:
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs


def eligible_shape(rows, q):
    """Report the bounded physical shapes/tactics with a native residual writer."""
    q = projection_for_rows(q, rows)
    weight = q.packed_weight
    if q.bits not in (4, 8) or weight.ndim != 2:
        return False
    n, stored_k = weight.shape
    k = stored_k * (2 if q.bits == 4 else 1)
    return (
        type(q.tactic) is int
        and q.tactic in (0, 2, 7)
        and 0 < rows <= 64
        and 128 <= k <= 8192
        and k % 128 == 0
        and 0 < n <= 8192
        and n % 8 == 0
    )


def _validate_pack(packed, q):
    if (
        packed.bits != q.bits
        or packed.logical_k != q.in_features
        or packed.format_version != "symmetric-row-v1"
    ):
        raise ValueError("Incompatible activation pack")
    a, sa = packed.data, packed.scales
    b, sb = q.packed_weight, q.weight_scale
    storage = torch.uint8 if q.bits == 4 else torch.int8
    tensors = (a, sa, b, sb) if q.bias is None else (a, sa, b, sb, q.bias)
    if not (
        q.bits in (4, 8)
        and type(q.in_features) is int
        and type(q.out_features) is int
        and q.in_features > 0
        and q.out_features > 0
        and type(q.tactic) is int
        and q.tactic in range(8)
        and a.ndim == b.ndim == 2
        and a.shape[1] == b.shape[1]
        and a.dtype == b.dtype == storage
        and sa.dtype == sb.dtype == torch.float32
        and sa.ndim == sb.ndim == 1
        and sa.shape[0] == a.shape[0]
        and sb.shape[0] == b.shape[0]
        and b.shape[0] == (q.out_features + 7) // 8 * 8
        and a.shape[1] * (2 if q.bits == 4 else 1) == (q.in_features + 127) // 128 * 128
        and all(type(dimension) is int and dimension >= 0 for dimension in packed.shape)
        and math.prod(packed.shape) == a.shape[0]
        and all(
            tensor.is_cuda and tensor.device == a.device and tensor.is_contiguous()
            for tensor in tensors
        )
        and (
            q.bias is None
            or (
                q.bias.dtype == torch.bfloat16
                and q.bias.ndim == 1
                and q.bias.shape[0] == b.shape[0]
            )
        )
    ):
        raise ValueError("Invalid packed buffers for integer epilogue")
    if a.shape[0] and any(tensor.data_ptr() % 16 for tensor in (a, b, sb)):
        raise ValueError("Invalid aligned packed buffers for integer epilogue")
    if a.shape[0] and q.bias is not None and q.bias.data_ptr() % 16:
        raise ValueError("Invalid aligned packed buffers for integer epilogue")


def _validate_residual(packed, q, residual):
    rows, physical_n = packed.data.shape[0], q.packed_weight.shape[0]
    shapes = (
        (rows, physical_n),
        (rows, q.out_features),
        (*packed.shape, q.out_features),
    )
    if not (
        hasattr(residual, "is_cuda")
        and residual.is_cuda
        and residual.device == packed.data.device
        and residual.dtype == torch.bfloat16
        and tuple(residual.shape) in shapes
    ):
        raise ValueError(
            "Invalid BF16 residual: expected same-device CUDA [M,Npad] or logical rows"
        )


def _native_residual_layout(packed, q, residual):
    return (
        residual.ndim == 2
        and tuple(residual.shape) == (packed.data.shape[0], q.packed_weight.shape[0])
        and residual.is_contiguous()
        and residual.data_ptr() % 16 == 0
    )


def eligible(packed, q, residual):
    """Return whether this validated call uses the fused native residual writer."""
    try:
        _validate_pack(packed, q)
        _validate_residual(packed, q, residual)
        return (
            is_ada_device(packed.data.device)
            and eligible_shape(packed.data.shape[0], q)
            and _native_residual_layout(packed, q, residual)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _reference_with_residual(packed, q, residual):
    """Execute the existing packed operator and its separate BF16 addition."""
    with torch.cuda.device(packed.data.device):
        stream = torch.cuda.current_stream(packed.data.device)
        arguments = (
            packed.data,
            packed.scales,
            q.packed_weight,
            q.weight_scale,
            q.bias,
            q.bits,
        )
        tensors = arguments[:4] if q.bias is None else (*arguments[:4], q.bias)
        for tensor in (*tensors, residual):
            tensor.record_stream(stream)
        reference = q._ops.gemm_biasless if q.biasless_epilogue else q._ops.gemm
        output = reference(*arguments, q.tactic)
        rows, physical_n = output.shape
        if residual.ndim == 2 and tuple(residual.shape) == (rows, physical_n):
            matrix = residual
        else:
            matrix = residual.reshape(rows, q.out_features)
            matrix = torch.nn.functional.pad(matrix, (0, physical_n - q.out_features))
        return output + matrix


def residual_gemm(packed, q, residual):
    """Return padded, contiguous BF16 [M,Npad] after exact residual addition.

    Native calls require W4/W8, q.tactic 0/2/7, 1<=M<=64, padded K in
    [128,8192] with K%128=0, and padded N in [8,8192] with N%8=0. The residual
    must be a contiguous, 16-byte-aligned CUDA BF16 [M,Npad] on the packed input
    device. Other valid shapes/tactics/layouts use q's existing GEMM plus BF16
    addition. A logical residual [M,Nlogical] or [*packed.shape,Nlogical] is
    reshaped and zero-padded in fallback; noncontiguous residuals also use that
    path. Caller owns logical crop/reshape. Zero rows skip extension loading.
    """
    q = projection_for_rows(q, packed.data.shape[0])
    _validate_pack(packed, q)
    _validate_residual(packed, q, residual)
    rows = packed.data.shape[0]
    if rows == 0:
        return torch.empty(
            (0, q.packed_weight.shape[0]),
            device=packed.data.device,
            dtype=torch.bfloat16,
        )
    if not (
        is_ada_device(packed.data.device)
        and eligible_shape(rows, q)
        and _native_residual_layout(packed, q, residual)
    ):
        return _reference_with_residual(packed, q, residual)
    return load_epilogue_extension().gemm(
        packed.data,
        packed.scales,
        q.packed_weight,
        q.weight_scale,
        q.bias,
        q.bits,
        q.tactic,
        q.biasless_epilogue,
        residual,
    )


def _validate_group(group):
    sizes, offsets = tuple(group.sizes), tuple(group.offsets)
    if not (
        len(sizes) >= 2
        and len(sizes) == len(offsets)
        and all(type(size) is int and size > 0 for size in sizes)
        and all(type(offset) is int and offset >= 0 for offset in offsets)
        and all(
            offset + size <= group.linear.out_features
            for offset, size in zip(offsets, sizes)
        )
        and all(
            offsets[index] >= offsets[index - 1] + sizes[index - 1]
            for index in range(1, len(sizes))
        )
    ):
        raise ValueError("Invalid projection group sizes or physical offsets")
    return sizes, offsets


def _native_group_layout(rows, group, sizes, offsets):
    q = projection_for_rows(group.linear, rows)
    n, stored_k = q.packed_weight.shape
    k = stored_k * (2 if q.bits == 4 else 1)
    return (
        is_ada_device(q.packed_weight.device)
        and q.bits in (4, 8)
        and type(q.tactic) is int
        and q.tactic in (0, 2, 7)
        and 0 < rows <= 4096
        and 128 <= k <= 8192
        and k % 128 == 0
        and 0 < n <= 8192
        and len(sizes) in (2, 3)
        and all(size % 8 == 0 for size in sizes)
        and sum(sizes) == q.out_features == n
        and offsets == tuple(sum(sizes[:index]) for index in range(len(sizes)))
        and group.contiguous_outputs is True
    )


def group_eligible(packed, group):
    """Report whether validated siblings use direct segmented BF16 writeback."""
    try:
        _validate_pack(packed, group.linear)
        sizes, offsets = _validate_group(group)
        return _native_group_layout(packed.data.shape[0], group, sizes, offsets)
    except (AttributeError, TypeError, ValueError):
        return False


def _reference_group_outputs(packed, group, sizes, offsets):
    q = group.linear
    with torch.cuda.device(packed.data.device):
        stream = torch.cuda.current_stream(packed.data.device)
        arguments = (
            packed.data,
            packed.scales,
            q.packed_weight,
            q.weight_scale,
            q.bias,
            q.bits,
            q.tactic,
        )
        tensors = arguments[:4] if q.bias is None else (*arguments[:4], q.bias)
        for tensor in tensors:
            tensor.record_stream(stream)
        operation = q._ops.gemm_biasless if q.biasless_epilogue else q._ops.gemm
        matrix = operation(*arguments)
        return tuple(
            matrix.narrow(1, offset, size).contiguous().view(*packed.shape, size)
            for offset, size in zip(offsets, sizes)
        )


def group_outputs(packed, group):
    """Return contiguous BF16 siblings shaped [*packed.shape, logical width].

    Native W4/W8 calls use tactics 0/2/7, 1<=M<=4096, padded K in [128,8192],
    and 2/3 positive widths divisible by eight, with cumulative logical offsets
    and their sum equal to the physical packed N<=8192. One fresh allocation
    holds [all Q rows][all K rows][all V rows]; returned slices are views, so no
    post-GEMM copy is launched. The original scale/bias tree rounds to BF16 once.

    Other valid groups, including logical padding tails and condition groups,
    use the existing CUDA GEMM followed by physical-offset slices and contiguous
    copies. Both paths preserve logical widths and leading dimensions. Empty
    rows skip extension loading. Warm native builds before CUDA Graph capture.
    """
    _validate_pack(packed, group.linear)
    sizes, offsets = _validate_group(group)
    rows = packed.data.shape[0]
    if rows == 0:
        return tuple(
            torch.empty(
                (*packed.shape, size),
                device=packed.data.device,
                dtype=torch.bfloat16,
            )
            for size in sizes
        )
    if not _native_group_layout(rows, group, sizes, offsets):
        return _reference_group_outputs(packed, group, sizes, offsets)
    q = projection_for_rows(group.linear, rows)
    flat = load_epilogue_extension().group_gemm(
        packed.data,
        packed.scales,
        q.packed_weight,
        q.weight_scale,
        q.bias,
        q.bits,
        q.tactic,
        q.biasless_epilogue,
        list(sizes),
    )
    return tuple(
        flat.narrow(0, rows * offset, rows * size).view(*packed.shape, size)
        for offset, size in zip(offsets, sizes)
    )
