"""Opt-in GR00T configuration and CLI flags; no GPU imports or kernel loading."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OptimizationConfig:
    precision: str = "bf16"
    fusion: bool = False
    dit_graph: bool = False
    action_graph: bool = False
    action_kv_reuse: bool = False
    scope: str = "transformer"
    category_id: int | None = None
    coverage: Path | None = None
    tactics: Path | None = None
    group_tactics: Path | None = None
    shape_tactics: Path | None = None
    group_conditioning: bool = False
    vision_channels_last: bool = False
    norm_modulation_quant: bool = False
    residual_norm_quant: bool = False
    fp8_fast_quant: bool = False
    fp4_swap_ab: bool = False
    fp_shared_inputs: bool = False
    fp_grouped: bool = False
    fp_swiglu: bool = False
    component: str = "all"
    input_preparation: bool = False
    tokenizer_cache: bool = False
    static_vlm: bool = False
    vlm_graph: bool = False
    image_array_preparation: bool = False
    compiled_host_guards: bool = False
    component_cache_clear: bool = False
    fusion_mode: str = "all"
    operator_profile: Path | None = None

    def __post_init__(self):
        if type(self.action_kv_reuse) is not bool:
            raise ValueError("Action K/V reuse must be a bool")
        if self.action_kv_reuse and not self.action_graph:
            raise ValueError("Action K/V reuse requires --action-cuda-graph")
        if type(self.component_cache_clear) is not bool:
            raise ValueError("component_cache_clear must be boolean")
        if self.component_cache_clear and self.precision not in ("w8a8", "w4a4"):
            raise ValueError("Component cache clearing requires W8A8 or W4A4")
        if self.action_graph and self.precision not in ("bf16", "w8a8", "w4a4"):
            raise ValueError("Action graph requires BF16, W8A8 or W4A4")
        if self.vlm_graph and not self.static_vlm:
            raise ValueError("VLM graph requires static_vlm")
        if self.fusion_mode not in (
            "vision_rope",
            "text_rope",
            "rope",
            "rmsnorm",
            "rope_rms",
            "all",
        ):
            raise ValueError("Unknown operator fusion mode")
        if self.fusion_mode != "all" and not self.fusion:
            raise ValueError("Selective mode requires operator fusion")
        if self.fusion_mode != "all" and (
            self.group_conditioning or self.norm_modulation_quant
        ):
            raise ValueError("DiT producer options require all operator fusion")
        if self.operator_profile is not None and not (
            self.fusion or self.precision != "bf16"
        ):
            raise ValueError("Operator profile requires fusion or low-bit precision")
        if self.precision not in ("bf16", "w8a8", "w4a4", "fp8", "fp4"):
            raise ValueError("Precision must be bf16, w8a8, w4a4, fp8 or fp4")
        if self.fp8_fast_quant and self.precision != "fp8":
            raise ValueError("Fast FP8 quantization requires precision=fp8")
        if self.fp4_swap_ab and self.precision != "fp4":
            raise ValueError("FP4 Swap-AB requires precision=fp4")
        if self.fp_shared_inputs and self.precision not in ("fp8", "fp4"):
            raise ValueError("Shared floating inputs require precision=fp8 or fp4")
        if (self.fp_grouped or self.fp_swiglu) and self.precision not in ("fp8", "fp4"):
            raise ValueError("Floating fusion requires precision=fp8 or fp4")
        if self.scope not in ("transformer", "all"):
            raise ValueError("Quantization scope must be transformer or all")
        if self.component not in ("all", "vlm", "dit"):
            raise ValueError("Quantization component must be all, vlm or dit")
        if self.component == "vlm" and (
            self.category_id is not None
            or self.group_conditioning
            or self.norm_modulation_quant
        ):
            raise ValueError(
                "Category, condition and norm producers require the DiT component"
            )
        if self.category_id is not None and self.category_id < 0:
            raise ValueError("Category ID must be nonnegative")
        if self.category_id is not None and self.scope != "all":
            raise ValueError("Category projections require scope=all")
        if self.precision in ("fp8", "fp4") and (
            self.category_id is not None
            or self.tactics is not None
            or self.group_tactics is not None
            or self.shape_tactics is not None
            or self.group_conditioning
        ):
            raise ValueError(
                "Floating precision does not support integer tactic, category or condition grouping flags"
            )
        if self.group_conditioning and (
            not self.fusion or self.scope != "all" or self.precision == "bf16"
        ):
            raise ValueError(
                "Condition grouping requires fusion, scope=all and integer precision"
            )
        if self.norm_modulation_quant and (not self.fusion or self.precision == "bf16"):
            raise ValueError(
                "Norm quantization fusion requires fusion and low-bit precision"
            )
        if self.residual_norm_quant and not self.norm_modulation_quant:
            raise ValueError("Residual fusion requires norm_modulation_quant")

    @property
    def enabled(self):
        return (
            self.precision != "bf16"
            or self.fusion
            or self.dit_graph
            or self.action_graph
            or self.vision_channels_last
            or self.input_preparation
            or self.tokenizer_cache
            or self.static_vlm
            or self.vlm_graph
            or self.image_array_preparation
            or self.compiled_host_guards
        )

    def validate_variant(self, variant):
        if self.action_graph and variant != "flow":
            raise ValueError("Action graph requires non-streaming Flow4")
        if variant == "pir2" and self.group_conditioning:
            raise ValueError("Streaming piR2 condition grouping is not validated")


def add_optimization_arguments(parser):
    parser.add_argument(
        "--inference-precision",
        choices=["bf16", "w8a8", "w4a4", "fp8", "fp4"],
        default="bf16",
    )
    parser.add_argument("--operator-fusion", action="store_true")
    parser.add_argument(
        "--operator-fusion-mode",
        choices=["vision_rope", "text_rope", "rope", "rmsnorm", "rope_rms", "all"],
        default="all",
    )
    parser.add_argument("--operator-profile", type=Path)
    parser.add_argument("--dit-cuda-graph", action="store_true")
    parser.add_argument("--action-cuda-graph", action="store_true")
    parser.add_argument("--action-kv-reuse", action="store_true")
    parser.add_argument(
        "--quantization-scope", choices=["transformer", "all"], default="transformer"
    )
    parser.add_argument(
        "--quantization-component", choices=["all", "vlm", "dit"], default="all"
    )
    parser.add_argument("--quantization-category-id", type=int)
    parser.add_argument("--quantization-coverage", type=Path)
    parser.add_argument("--quantization-tactics", type=Path)
    parser.add_argument("--quantization-group-tactics", type=Path)
    parser.add_argument("--quantization-shape-tactics", type=Path)
    parser.add_argument("--group-conditioning", action="store_true")
    parser.add_argument("--vision-channels-last", action="store_true")
    parser.add_argument("--input-preparation", action="store_true")
    parser.add_argument("--tokenizer-cache", action="store_true")
    parser.add_argument("--static-vlm", action="store_true")
    parser.add_argument("--vlm-cuda-graph", action="store_true")
    parser.add_argument("--image-array-preparation", action="store_true")
    parser.add_argument("--compiled-host-guards", action="store_true")
    parser.add_argument("--component-cache-clear", action="store_true")
    parser.add_argument("--norm-modulation-quant", action="store_true")
    parser.add_argument("--residual-norm-quant", action="store_true")
    parser.add_argument("--fp8-fast-quant", action="store_true")
    parser.add_argument("--fp4-swap-ab", action="store_true")
    parser.add_argument("--fp-shared-inputs", action="store_true")
    parser.add_argument("--fp-grouped", action="store_true")
    parser.add_argument("--fp-swiglu", action="store_true")


def optimization_config(args):
    return OptimizationConfig(
        precision=args.inference_precision,
        fusion=args.operator_fusion,
        fusion_mode=args.operator_fusion_mode,
        operator_profile=args.operator_profile,
        dit_graph=args.dit_cuda_graph,
        action_graph=args.action_cuda_graph,
        action_kv_reuse=args.action_kv_reuse,
        scope=args.quantization_scope,
        component=args.quantization_component,
        category_id=args.quantization_category_id,
        coverage=args.quantization_coverage,
        tactics=args.quantization_tactics,
        group_tactics=args.quantization_group_tactics,
        shape_tactics=args.quantization_shape_tactics,
        group_conditioning=args.group_conditioning,
        vision_channels_last=args.vision_channels_last,
        input_preparation=args.input_preparation,
        tokenizer_cache=args.tokenizer_cache,
        static_vlm=args.static_vlm,
        vlm_graph=args.vlm_cuda_graph,
        image_array_preparation=args.image_array_preparation,
        compiled_host_guards=args.compiled_host_guards,
        component_cache_clear=getattr(args, "component_cache_clear", False),
        norm_modulation_quant=args.norm_modulation_quant,
        residual_norm_quant=args.residual_norm_quant,
        fp8_fast_quant=args.fp8_fast_quant,
        fp4_swap_ab=args.fp4_swap_ab,
        fp_shared_inputs=args.fp_shared_inputs,
        fp_grouped=args.fp_grouped,
        fp_swiglu=args.fp_swiglu,
    )


def selected_component(name, component):
    """S2 is the backbone; S1 includes every action-head projection, including VL."""
    if component not in ("all", "vlm", "dit"):
        raise ValueError("Quantization component must be all, vlm or dit")
    return component == "all" or name.startswith(
        "backbone." if component == "vlm" else "action_head."
    )
