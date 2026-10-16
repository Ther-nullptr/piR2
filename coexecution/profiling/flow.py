"""Official Spatial Flow4 scheduling diagnostic; model execution is CUDA opt-in."""

import argparse
import math
import os
import re
import sys
from pathlib import Path

from coexecution.libero.preprocessing import add_image_arguments, image_config
from coexecution.optimization.config import OptimizationConfig

from .common import save
from .flow_protocol import MODES, FlowCase, validate_start_gate

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = ROOT / "configs/quantization/groot-libero-spatial-ada"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    add_image_arguments(result)
    for name in ("checkpoint", "dataset", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument(
        "--groot-root", type=Path, default=ROOT / "upstream/learning/Isaac-GR00T"
    )
    result.add_argument("--precision", choices=("bf16", "w8a8", "w4a4"), default="bf16")
    result.add_argument(
        "--group", choices=("optimized", "cooperative"), default="optimized"
    )
    result.add_argument("--action-device", default="cuda:0")
    result.add_argument("--vlm-device", default="cuda:0")
    result.add_argument("--modes", choices=MODES, nargs="+")
    result.add_argument("--offsets-ms", type=float, nargs="+", default=[0.0, 5.0])
    result.add_argument("--iterations", type=int, default=30)
    result.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Passes over every fixture on every execution stream, followed by one pass per case",
    )
    result.add_argument("--fixtures", dest="observations", type=int, default=3)
    result.add_argument("--episodes", type=int, nargs="+")
    result.add_argument("--start-step", type=int, default=10)
    result.add_argument("--step-stride", type=int, default=7)
    result.add_argument("--video-backend", default="torchcodec")
    result.add_argument("--seed", type=int, default=1000)
    result.add_argument("--torch-threads", type=int, default=2)
    result.add_argument("--vlm-chunk-blocks", type=int, choices=(1, 2), default=1)
    result.add_argument(
        "--profile-iterations",
        type=int,
        default=0,
        help="Separate instrumented pairs per case after clean measurements",
    )
    result.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket only separate profile pairs with cudaProfilerStart/Stop for Nsight",
    )
    result.add_argument(
        "--start-gate",
        type=Path,
        help="Fresh release-file path; publish start-ready.json after warmup, then wait before clean measurements",
    )
    result.add_argument("--gate-timeout-s", type=float, default=600.0)
    return result


def configuration(args, role):
    if role not in ("vlm", "dit"):
        raise ValueError("Unknown component role")
    quantized = args.precision != "bf16"
    w8 = args.precision == "w8a8"
    return OptimizationConfig(
        precision=args.precision,
        fusion=True,
        scope="all",
        # The full W8 profile admits both VLM and action producer families.
        # Each independent policy keeps all parameters; only its role executes.
        component="all",
        category_id=2 if quantized else None,
        tactics=PROFILE_ROOT / "site-tactics.json" if quantized else None,
        group_tactics=PROFILE_ROOT / "group-tactics.json" if quantized else None,
        shape_tactics=PROFILE_ROOT / "w8-shape-tactics.json" if w8 else None,
        operator_profile=PROFILE_ROOT / "w8-splitk-full.json" if w8 else None,
        group_conditioning=quantized,
        norm_modulation_quant=args.precision == "w4a4",
        residual_norm_quant=args.precision == "w4a4",
        action_graph=role == "dit",
        action_kv_reuse=role == "dit",
        input_preparation=True,
        tokenizer_cache=True,
        static_vlm=role == "vlm",
        vlm_graph=role == "vlm" and args.group == "optimized",
        image_array_preparation=True,
        compiled_host_guards=True,
    )


def cases(args):
    if (
        not args.offsets_ms
        or len(set(args.offsets_ms)) != len(args.offsets_ms)
        or any(not math.isfinite(value) or value < 0 for value in args.offsets_ms)
    ):
        raise ValueError("Unique finite nonnegative offsets are required")
    modes = args.modes or list(MODES if args.group == "cooperative" else MODES[:5])
    if len(set(modes)) != len(modes):
        raise ValueError("Unique modes are required")
    if args.group != "cooperative" and any(FlowCase(mode).fenced for mode in modes):
        raise ValueError(
            "Fenced/cooperative modes require the cooperative group without a whole VLM graph"
        )
    return [
        FlowCase(mode, offset)
        for mode in modes
        for offset in (args.offsets_ms if FlowCase(mode).concurrent else [0.0])
    ]


def validate_options(args):
    image_config(args)
    cases(args)
    validate_start_gate(args.start_gate, args.gate_timeout_s)
    if args.cuda_profiler_range and not args.profile_iterations:
        raise ValueError("CUDA profiler range requires separate profile iterations")
    for device in (args.action_device, args.vlm_device):
        if re.fullmatch(r"cuda:\d+", device) is None:
            raise ValueError("An explicit CUDA device index is required")
    if (
        not 1 <= args.observations <= 3
        or args.iterations < 1
        or args.warmup < 1
        or args.profile_iterations < 0
        or args.torch_threads < 1
        or args.start_step < 0
        or args.step_stride < 1
    ):
        raise ValueError("Invalid fixture, iteration, warmup, thread or step counts")
    if args.episodes is not None and (
        not args.episodes or any(value < 0 for value in args.episodes)
    ):
        raise ValueError("Nonnegative episode positions are required")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_options(args)
    for name in ("checkpoint", "dataset", "groot_root"):
        if not getattr(args, name).is_dir():
            raise ValueError(
                f"--{name.replace('_', '-')} must be an existing directory"
            )
    if not (args.groot_root / "gr00t").is_dir():
        raise ValueError("--groot-root must contain gr00t")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.groot_root.resolve()))
    try:
        from .flow_execution import run_group

        return run_group(args)
    except BaseException as error:
        save(
            args.output / "failure.json",
            {"error": repr(error), "pid": os.getpid(), "measurements_valid": False},
        )
        raise


if __name__ == "__main__":
    main()
