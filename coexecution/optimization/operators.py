"""GPU-free inventory, explicit profiles and comparable-measurement selection.

Inventory describes implementations and their contracts, never a speed ranking.
Profiles are opt-in partial overrides: missing families retain runtime defaults.
Selection consumes retained measurements; it runs no benchmark and does not
turn a kernel/chain result into a deployment or LIBERO task-quality claim.
"""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType

PRECISIONS = ("bf16", "w8a8", "w4a4", "fp8", "fp4")
INTEGER_PRECISIONS = ("w8a8", "w4a4")


@dataclass(frozen=True)
class OperatorImplementation:
    family: str
    implementation: str
    source: str
    language: str
    precisions: tuple[str, ...]
    architectures: tuple[str, ...] | None
    shape_contract: str
    numerical_contract: str


def _implementation(
    family,
    implementation,
    source,
    language,
    shape,
    numerical,
    precisions=PRECISIONS,
    architectures=None,
):
    return OperatorImplementation(
        family,
        implementation,
        source,
        language,
        precisions,
        architectures,
        shape,
        numerical,
    )


_REFERENCE = "Original same-precision computation; not equivalence to BF16 policy"
_BF16 = "Preserves reference BF16 operation/rounding boundaries"
_INTEGER = "Preserves integer dot, FP32 scale/bias order and BF16 output boundary"
_LAYERNORM = "BF16 contiguous inference rows; width 128..8192 divisible by 4"
_GEMM = "Packed M/K/N, K padded to 128, N padded to 8; runtime validates layout"
_INVENTORY = (
    *(
        _implementation(
            family,
            "reference",
            "coexecution.groot_fusion",
            "pytorch",
            "Original upstream tensor contract",
            _REFERENCE,
        )
        for family in ("vision_rope", "text_rope", "rmsnorm", "adaln")
    ),
    _implementation(
        "vision_rope",
        "triton",
        "coexecution.fused_ops.vision_rope",
        "triton",
        "Q/K [tokens,heads,even_dim], trig [tokens,dim]; supports packed QKV strides",
        _BF16,
    ),
    _implementation(
        "vision_rope",
        "native",
        "coexecution/kernels/common/csrc/fusion_torch.cu",
        "cuda",
        "Q/K [tokens,16,64], tokens 1..4096; token stride 1024 or 3072; contiguous trig",
        _BF16,
    ),
    _implementation(
        "text_rope",
        "triton",
        "coexecution.fused_ops.text_rope",
        "triton",
        "Q/K [batch,heads,tokens,even_dim], BF16 trig [batch,tokens,dim], unsqueeze_dim=1",
        _BF16,
    ),
    _implementation(
        "rmsnorm",
        "triton",
        "coexecution.fused_ops.rmsnorm",
        "triton+pytorch",
        "Contiguous BF16 input/weight with matching final width",
        "FP32 square and native PyTorch mean; BF16 normalize then BF16 weight multiply",
    ),
    _implementation(
        "rmsnorm",
        "native",
        "coexecution.kernels.ampere_ada.rms_statistics",
        "cuda",
        _LAYERNORM
        + "; pinned PyTorch 2.7.1+cu126 Reduce.cuh identity; fallback otherwise",
        "Same native FP32 reduction order and original RMS tail rounding",
        architectures=("sm_89",),
    ),
    _implementation(
        "adaln",
        "triton",
        "coexecution.groot_fusion.modulation",
        "triton",
        "BF16 [B,T,D] normalized input and feature-contiguous [B,D] scale/shift",
        _BF16,
    ),
    _implementation(
        "adaln",
        "native",
        "coexecution/kernels/common/csrc/fusion_torch.cu",
        "cuda",
        "Contiguous BF16 [1,41,1536] and [1,1536] scale/shift; fallback otherwise",
        _BF16,
    ),
    _implementation(
        "integer_gemm",
        "reference",
        "coexecution.kernels.ampere_ada.integer",
        "cuda",
        _GEMM,
        _INTEGER,
        INTEGER_PRECISIONS,
    ),
    _implementation(
        "integer_gemm",
        "int4_gemv",
        "coexecution.kernels.ampere_ada.int4_gemv",
        "cuda",
        "M=1, Kpad 128..8192 divisible by 128; positive Npad divisible by 8",
        _INTEGER,
        ("w4a4",),
        ("sm_89",),
    ),
    _implementation(
        "integer_gemm",
        "small_m",
        "coexecution.kernels.ampere_ada.int4_small_m",
        "cuda",
        "M 1..64, Kpad 128..8192 divisible by 128; Npad 8..8192 divisible by 8; tactic 0..5",
        _INTEGER,
        ("w4a4",),
        ("sm_89",),
    ),
    _implementation(
        "integer_gemm",
        "split_k",
        "coexecution.kernels.ampere_ada.split_k",
        "cuda",
        "M 1..64, Kpad 4096..8192 divisible by 128; Npad 8..8192 divisible by 8; slices 2/4/8",
        _INTEGER,
        INTEGER_PRECISIONS,
        ("sm_89",),
    ),
    _implementation(
        "activation_pack",
        "triton",
        "coexecution.optimization.backends.integer.linear.prepare",
        "triton",
        "BF16 rowwise dynamic quantization, padded to 128, INT8 or packed INT4",
        "Symmetric row scale/rounding and zero padding; fused LUT paths retain BF16 boundaries",
        INTEGER_PRECISIONS,
    ),
    _implementation(
        "activation_pack",
        "native",
        "coexecution/kernels/ampere_ada/csrc/integer_dispatch.cu",
        "cuda",
        "BF16 regularly strided rows with feature stride 1; K 1..65536, padded to 128",
        "FP32 rowmax/scale, ties-even symmetric quantization; INT4 low nibble first",
        INTEGER_PRECISIONS,
    ),
    _implementation(
        "layernorm_pack",
        "reference",
        "coexecution.kernels.ampere_ada.norm",
        "pytorch+triton",
        "Native LayerNorm followed by separate modulation/packing",
        _REFERENCE,
        INTEGER_PRECISIONS,
    ),
    _implementation(
        "layernorm_pack",
        "native",
        "coexecution.kernels.ampere_ada.norm",
        "cuda",
        _LAYERNORM + "; PyTorch 2.7.1 vectorized Welford contract; fallback otherwise",
        "Native Welford order, BF16 normalization/modulation, identical quantized packing",
        INTEGER_PRECISIONS,
        ("sm_89",),
    ),
    *(
        _implementation(
            family,
            "reference",
            "coexecution.optimization.groot.integer",
            "pytorch",
            "Separate BF16 residual addition or contiguous projection copies",
            _REFERENCE,
            INTEGER_PRECISIONS,
        )
        for family in ("integer_residual", "group_writeback")
    ),
    _implementation(
        "integer_residual",
        "native",
        "coexecution.kernels.ampere_ada.epilogue.residual_gemm",
        "cuda",
        "M 1..64, Kpad 128..8192, Npad 8..8192; tactics 0/2/7; aligned contiguous residual",
        "Round GEMM to BF16 before adding residual and rounding again",
        INTEGER_PRECISIONS,
        ("sm_89",),
    ),
    _implementation(
        "group_writeback",
        "native",
        "coexecution.kernels.ampere_ada.epilogue.group_outputs",
        "cuda",
        "M 1..4096, Kpad 128..8192, Npad 8..8192; tactics 0/2/7; 2/3 widths multiple 8, no tails",
        "Same GEMM values, contiguous sibling views sharing one new allocation",
        INTEGER_PRECISIONS,
        ("sm_89",),
    ),
    _implementation(
        "floating_gemm",
        "native",
        "coexecution.optimization.backends.thor.linear",
        "cuda+cutlass",
        "Thor SM110; FP8 K/N aligned 16, FP4 aligned 32; independent scale formats",
        "FP8 E4M3 or FP4 E2M1/block16 UE4M3 scales, FP32 accumulation, BF16 output",
        ("fp8", "fp4"),
        ("sm_110",),
    ),
    _implementation(
        "floating_gemm",
        "swap_ab",
        "coexecution/kernels/blackwell/csrc/fp4_cutlass_linear.cu",
        "cuda+cutlass",
        "Thor FP4 with bias: exact (M,K,N)=(41,256,1536)/(41,1536,1536); native otherwise",
        "Preserves NVFP4 format; tested-input equivalence, not universal BF16 bitwise equality",
        ("fp4",),
        ("sm_110",),
    ),
    *(
        _implementation(
            family,
            "reference",
            "coexecution.optimization.groot.producers",
            "pytorch+triton",
            "Original separate producer calls and packed inputs",
            _REFERENCE,
            INTEGER_PRECISIONS,
        )
        for family in ("context_pack", "rmsnorm_pack", "final_modulation_pack")
    ),
    _implementation(
        "context_pack",
        "shared",
        "coexecution.optimization.groot.producers",
        "pytorch",
        "Complete selected DiT cross-attention group; one read-only input per forward",
        "Same packed bytes/scales shared within that invocation; never across forwards",
        INTEGER_PRECISIONS,
    ),
    _implementation(
        "rmsnorm_pack",
        "native",
        "coexecution.kernels.ampere_ada.producers",
        "cuda+triton",
        _LAYERNORM + "; pinned native RMS statistics and guarded fallback",
        "Native FP32 RMS reduction, BF16 normalize/weight boundaries, same packed bytes/scales",
        INTEGER_PRECISIONS,
        ("sm_89",),
    ),
    _implementation(
        "final_modulation_pack",
        "triton",
        "coexecution.kernels.ampere_ada.producers",
        "triton",
        "Final DiT normalized BF16 output plus scale/shift feeding an integer projection",
        "BF16 modulation boundaries precede rowwise quantization/packing",
        INTEGER_PRECISIONS,
    ),
)
_BY_KEY = {(row.family, row.implementation): row for row in _INVENTORY}


def inventory(family=None):
    """Return metadata only. ``architectures=None`` is not a validation claim."""
    rows = tuple(row for row in _INVENTORY if family is None or row.family == family)
    if not rows:
        raise ValueError(f"Unknown operator family: {family}")
    return rows


def _target(architecture, precision):
    if not isinstance(architecture, str) or not re.fullmatch(
        r"sm_[0-9]{2,3}", architecture
    ):
        raise ValueError("architecture must be an explicit CUDA SM identifier")
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision in ("fp8", "fp4") and architecture != "sm_110":
        raise ValueError("Floating precision requires Thor architecture sm_110")


def _choice(family, implementation, architecture, precision):
    try:
        row = _BY_KEY[family, implementation]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown operator choice: {family}={implementation}") from exc
    if precision not in row.precisions:
        raise ValueError(
            f"{family}={implementation} does not support precision {precision}"
        )
    if row.architectures is not None and architecture not in row.architectures:
        raise ValueError(
            f"{family}={implementation} does not support architecture {architecture}"
        )


def _parameters(family, implementation, values):
    allowed = {}
    if family == "integer_gemm" and implementation == "small_m":
        allowed = {"tactic": range(6)}
    elif family == "integer_gemm" and implementation == "split_k":
        allowed = {"slices": (2, 4, 8), "tile": range(3)}
    if not isinstance(values, Mapping):
        raise TypeError(f"Parameters for {family} must be an object")
    for name, value in values.items():
        if name not in allowed or type(value) is not int or value not in allowed[name]:
            raise ValueError(
                f"Unsupported {family}={implementation} parameter {name}={value}"
            )
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class OperatorProfile:
    architecture: str
    precision: str
    implementations: Mapping[str, str]
    parameters: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        _target(self.architecture, self.precision)
        if not isinstance(self.implementations, Mapping) or not isinstance(
            self.parameters, Mapping
        ):
            raise TypeError("Operator implementations/parameters must be objects")
        for family, implementation in self.implementations.items():
            _choice(family, implementation, self.architecture, self.precision)
        if self.parameters.keys() - self.implementations.keys():
            raise ValueError("Parameters require an explicit family implementation")
        parameters = {
            family: _parameters(family, self.implementations[family], values)
            for family, values in self.parameters.items()
        }
        object.__setattr__(
            self, "implementations", MappingProxyType(dict(self.implementations))
        )
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    def get(self, family, default=None):
        return self.implementations.get(family, default)

    def options(self, family):
        return dict(self.parameters.get(family, {}))

    def to_dict(self):
        return {
            "schema_version": 1,
            "architecture": self.architecture,
            "precision": self.precision,
            "implementations": dict(self.implementations),
            "parameters": {
                family: dict(values) for family, values in self.parameters.items()
            },
        }


def _document(path, required, optional=()):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    keys = value.keys() - {"schema_version"}
    if set(required) - keys or keys - set(required) - set(optional):
        raise ValueError("Missing or unknown JSON fields")
    return {key: val for key, val in value.items() if key != "schema_version"}


def read_operator_profile(path):
    return OperatorProfile(
        **_document(
            path, ("architecture", "precision", "implementations"), ("parameters",)
        )
    )


load_operator_profile = read_operator_profile


def validate_profile(profile, *, architecture, precision):
    """Match actual runtime identity before installing any model adapters."""
    _target(architecture, precision)
    for name, actual in (("architecture", architecture), ("precision", precision)):
        if getattr(profile, name) != actual:
            raise ValueError(f"Operator profile {name} does not match runtime {actual}")
    return profile


def _identity(name, value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in ("unknown", "unavailable")
    ):
        raise ValueError(f"{name} must identify retained evidence")


@dataclass(frozen=True)
class MeasurementContext:
    """All comparison invariants; cohort identifies one fair measured round.

    Workload identifies checkpoint, inputs, shapes/layouts and policy settings;
    environment identifies the physical GPU, clocks and software/build stack.
    Shape is an exact signature under that workload (e.g. M/K/N for GEMM).
    Numerical gate identifies both the reference and accepted error bounds.
    These identities must come from artifacts, not guessed platform labels.
    """

    architecture: str
    precision: str
    shape: tuple[int, ...]
    cohort: str
    workload: str
    environment: str
    boundary: str
    timing: str
    statistic: str
    numerical_gate: str

    def __post_init__(self):
        _target(self.architecture, self.precision)
        if (
            not isinstance(self.shape, (tuple, list))
            or not self.shape
            or any(type(size) is not int or size <= 0 for size in self.shape)
        ):
            raise ValueError(
                "shape must contain an exact nonempty positive-integer signature"
            )
        object.__setattr__(self, "shape", tuple(self.shape))
        for name in ("cohort", "workload", "environment", "numerical_gate"):
            _identity(name, getattr(self, name))
        if self.boundary not in ("operator", "operator_chain", "full_policy"):
            raise ValueError("boundary must be operator, operator_chain or full_policy")
        if self.timing not in ("wall", "cuda_event", "gpu_work"):
            raise ValueError("timing must be wall, cuda_event or gpu_work")
        if self.statistic not in ("p50", "p95", "mean"):
            raise ValueError("statistic must be p50, p95 or mean")


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate_id: str
    family: str
    implementation: str
    context: MeasurementContext
    latency_ms: float
    samples: int
    numerical_passed: bool
    source_revision: str
    artifact: str
    parameters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("candidate_id", "source_revision", "artifact"):
            _identity(name, getattr(self, name))
        if not isinstance(self.context, MeasurementContext):
            raise TypeError("context must be MeasurementContext")
        _choice(
            self.family,
            self.implementation,
            self.context.architecture,
            self.context.precision,
        )
        object.__setattr__(
            self,
            "parameters",
            dict(_parameters(self.family, self.implementation, self.parameters)),
        )
        if (
            type(self.latency_ms) not in (int, float)
            or not math.isfinite(self.latency_ms)
            or self.latency_ms <= 0
        ):
            raise ValueError("latency_ms must be a positive finite measured value")
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError("samples must be a positive integer")
        if type(self.numerical_passed) is not bool:
            raise ValueError(
                "numerical_passed must explicitly record the numerical gate"
            )


@dataclass(frozen=True)
class MeasuredSelection:
    winner: CandidateMeasurement
    baseline: CandidateMeasurement
    rejected: tuple[str, ...]
    scope: str

    @property
    def relative_gain(self):
        return 1.0 - self.winner.latency_ms / self.baseline.latency_ms


def read_measurements(path):
    """Read normalized retained evidence; does not infer missing run metadata."""
    rows = _document(path, ("candidates",))["candidates"]
    if not isinstance(rows, list):
        raise TypeError("candidates must be a JSON list")
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("context"), dict):
            raise TypeError("Each candidate requires an explicit context object")
        try:
            result.append(
                CandidateMeasurement(
                    **{**row, "context": MeasurementContext(**row["context"])}
                )
            )
        except TypeError as exc:
            raise ValueError("Missing or unknown candidate/context fields") from exc
    return tuple(result)


def select_measured(
    candidates, *, baseline, deployment=False, minimum_relative_gain=0.0
):
    """Select only within one comparable family/context, retaining ties/baseline.

    Failed numerical candidates are excluded explicitly. Mismatched contexts
    raise instead of silently combining incompatible runs. ``deployment=True``
    requires complete policy wall latency, not profiler totals or a kernel win.
    The result applies to this exact context only and does not change defaults.
    """
    if (
        type(minimum_relative_gain) not in (int, float)
        or not math.isfinite(minimum_relative_gain)
        or not 0 <= minimum_relative_gain < 1
    ):
        raise ValueError("minimum_relative_gain must be finite and in [0, 1)")
    rows = tuple(candidates)
    if not rows or any(not isinstance(row, CandidateMeasurement) for row in rows):
        raise ValueError("Selection requires CandidateMeasurement records")
    by_id = {row.candidate_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate candidate_id in comparison")
    if baseline not in by_id:
        raise ValueError("Selection requires an explicit measured baseline")
    reference = by_id[baseline]
    if not reference.numerical_passed:
        raise ValueError("The baseline failed its numerical gate")
    for row in rows:
        if row.family != reference.family:
            raise ValueError("Cannot compare different operator family records")
        for descriptor in fields(MeasurementContext):
            name = descriptor.name
            if getattr(row.context, name) != getattr(reference.context, name):
                raise ValueError(f"Candidate {row.candidate_id} has mismatched {name}")
    context = reference.context
    if deployment and (context.boundary != "full_policy" or context.timing != "wall"):
        raise ValueError(
            "Deployment selection requires full_policy boundary and wall timing"
        )
    passed = [row for row in rows if row.numerical_passed]
    winner = min(
        passed,
        key=lambda row: (
            row.latency_ms,
            row.candidate_id != baseline,
            row.candidate_id,
        ),
    )
    if 1.0 - winner.latency_ms / reference.latency_ms < minimum_relative_gain:
        winner = reference
    return MeasuredSelection(
        winner,
        reference,
        tuple(row.candidate_id for row in rows if not row.numerical_passed),
        "deployment_latency" if deployment else context.boundary,
    )
