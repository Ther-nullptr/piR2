"""Opt-in, reversible GR00T fusion and integer inference configuration.

Importing this module does not import Torch or compile kernels. GPU dependencies
are loaded only when an enabled configuration enters its scope.
"""

import hashlib
import json
import math
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class OptimizationConfig:
    precision: str = "bf16"
    fusion: bool = False
    dit_graph: bool = False
    scope: str = "transformer"
    category_id: int | None = None
    coverage: Path | None = None
    tactics: Path | None = None
    group_tactics: Path | None = None
    group_conditioning: bool = False
    vision_channels_last: bool = False

    def __post_init__(self):
        if self.precision not in ("bf16", "w8a8", "w4a4"):
            raise ValueError("Precision must be bf16, w8a8 or w4a4")
        if self.scope not in ("transformer", "all"):
            raise ValueError("Quantization scope must be transformer or all")
        if self.category_id is not None and self.category_id < 0:
            raise ValueError("Category ID must be nonnegative")
        if self.category_id is not None and self.scope != "all":
            raise ValueError("Category projections require scope=all")
        if self.group_conditioning and (
            not self.fusion or self.scope != "all" or self.precision == "bf16"
        ):
            raise ValueError(
                "Condition grouping requires fusion, scope=all and integer precision"
            )

    @property
    def enabled(self):
        return (
            self.precision != "bf16"
            or self.fusion
            or self.dit_graph
            or self.vision_channels_last
        )

    def validate_variant(self, variant):
        if variant == "pir2" and self.group_conditioning:
            raise ValueError("Streaming piR2 condition grouping is not validated")


def add_optimization_arguments(parser):
    parser.add_argument(
        "--inference-precision", choices=["bf16", "w8a8", "w4a4"], default="bf16"
    )
    parser.add_argument("--operator-fusion", action="store_true")
    parser.add_argument("--dit-cuda-graph", action="store_true")
    parser.add_argument(
        "--quantization-scope", choices=["transformer", "all"], default="transformer"
    )
    parser.add_argument("--quantization-category-id", type=int)
    parser.add_argument("--quantization-coverage", type=Path)
    parser.add_argument("--quantization-tactics", type=Path)
    parser.add_argument("--quantization-group-tactics", type=Path)
    parser.add_argument("--group-conditioning", action="store_true")
    parser.add_argument("--vision-channels-last", action="store_true")


def optimization_config(args):
    return OptimizationConfig(
        precision=args.inference_precision,
        fusion=args.operator_fusion,
        dit_graph=args.dit_cuda_graph,
        scope=args.quantization_scope,
        category_id=args.quantization_category_id,
        coverage=args.quantization_coverage,
        tactics=args.quantization_tactics,
        group_tactics=args.quantization_group_tactics,
        group_conditioning=args.group_conditioning,
        vision_channels_last=args.vision_channels_last,
    )


def read_json(path, default):
    return default if path is None else json.loads(Path(path).read_text())


def checked_tactic(value):
    if type(value) is not int or value not in range(8):
        raise ValueError("Integer tactic must be an integer in [0, 7]")
    return value


def apply_tactics(controller, extra, config, coverage):
    singles = read_json(config.tactics, {})
    groups = read_json(config.group_tactics, [])
    for bits in (8, 4):
        models = {**controller.quant[bits], **extra.quant[bits]}
        legacy = [row for row in groups if row["bits"] == bits and "shape" in row]
        for name, tactic in singles.get(str(bits), {}).items():
            if name in models:
                models[name].tactic = checked_tactic(tactic)
        for names, group in controller.grouped[bits]:
            q = group.linear
            matching = [
                row
                for row in groups
                if row["bits"] == bits and row.get("members") == list(names)
            ]
            if not matching and legacy and names[0] in coverage:
                shape = coverage[names[0]]["shape"]
                signature = [
                    math.prod(shape[:-1]),
                    q.in_features,
                    q.out_features,
                    q.bias is not None,
                ]
                matching = [row for row in legacy if row["shape"] == signature]
            choices = {checked_tactic(row["tactic"]) for row in matching}
            if len(choices) > 1:
                raise ValueError("Conflicting grouped tactics")
            q.tactic = choices.pop() if choices else 0


class GrootOptimizations:
    """One serial, frozen BF16 CUDA policy; close before replacing its weights.

    This scope neither selects a checkpoint nor changes sampling, observations
    or the policy's VLM cache. No optimization is enabled by default.
    Vision Conv3D layout conversion requires the explicit vision_channels_last flag.
    """

    def __init__(self, policy, config=None):
        self.policy = policy
        self.config = OptimizationConfig() if config is None else config
        self.stack = ExitStack()
        self.coverage = {}

    def __enter__(self):
        if not self.config.enabled:
            return self
        try:
            self._install()
        except BaseException:
            self.close()
            raise
        return self

    def _install(self):
        import torch

        model = self.policy.model
        if model.training or next(model.parameters()).dtype != torch.bfloat16:
            raise ValueError("Optimizations require an eval-mode BF16 model")
        if next(model.parameters()).device.type != "cuda":
            raise ValueError("Optimizations require CUDA")
        self.config.validate_variant(
            "pir2" if getattr(model.config, "streaming", False) else "flow"
        )
        if self.config.vision_channels_last:
            proj = model.backbone.model.visual.patch_embed.proj
            self.stack.callback(setattr, proj.weight, "data", proj.weight.data)
            proj.weight.data = proj.weight.data.contiguous(
                memory_format=torch.channels_last_3d
            )
            hook = proj.register_forward_pre_hook(
                lambda _module, inputs: (
                    inputs[0].contiguous(memory_format=torch.channels_last_3d),
                )
            )
            self.stack.callback(hook.remove)
        if self.config.fusion:
            from coexecution.groot_fusion import NonGemmAdapters
            from coexecution.groot_pointwise import PointwiseFusion

            fusion = NonGemmAdapters(self.policy, "native")
            self.stack.callback(fusion.close)
            fusion.set("all")
            pointwise = PointwiseFusion(self.policy)
            self.stack.callback(pointwise.close)
            pointwise.set("dit")
        if self.config.precision != "bf16":
            from coexecution.quantization import TransformerINT, inventory
            from coexecution.quantization_connections import AdditionalConnections

            expanded = self.config.scope == "all"
            sites = inventory(model, expanded=expanded)[0]
            recorded = read_json(self.config.coverage, {}).get("linears", {})
            executed = recorded if self.config.coverage is not None else sites
            categories = {
                name: (module, self.config.category_id)
                for name, module in model.named_modules()
                if type(module).__name__ == "CategorySpecificLinear"
                and self.config.category_id is not None
            }
            if not sites.keys() & executed.keys() and not categories:
                raise ValueError("Quantization selected no matching projections")
            controller = TransformerINT(
                model,
                executed,
                native=True,
                expanded=expanded,
                condition_group=self.config.group_conditioning,
            )
            self.stack.callback(controller.close)
            extra = AdditionalConnections(self.policy, controller, categories)
            self.stack.callback(extra.close)
            apply_tactics(controller, extra, self.config, recorded)
            controller.set(self.config.precision)
            extra.set(self.config.precision, expanded=expanded, fuse=self.config.fusion)
            self.coverage = {
                "linears": sorted(controller.sites),
                "category_linears": sorted(categories),
                "scope": self.config.scope,
                "category_id": self.config.category_id,
                "projection_groups": controller.groups,
                "tactic_policy": "Explicit entries reused; unmatched entries use 0",
            }
        if self.config.dit_graph:
            from coexecution.quantization_connections import DitGraphs

            graphs = DitGraphs(model.action_head.model)
            self.stack.callback(graphs.close)
            graphs.set(self.config.precision)

    def evidence(self):
        config = asdict(self.config)
        for key in ("coverage", "tactics", "group_tactics"):
            path = config[key]
            config[key] = (
                None
                if path is None
                else {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            )
        return {
            "config": config,
            "coverage": self.coverage,
            "scope": "Experimental serial inference; replay checks do not establish task quality",
        }

    def close(self):
        self.stack.close()

    def __exit__(self, *exception):
        self.close()
