"""Same-precision replay equivalence gate before any timing is accepted."""

import json

import torch

from .common import save, tensor_digest


def compare_tensor(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise RuntimeError("Output shape or dtype changed under scheduling")
    if reference.is_floating_point():
        if not bool(torch.isfinite(reference).all()) or not bool(
            torch.isfinite(candidate).all()
        ):
            raise RuntimeError("Nonfinite model output")
        maximum = float((reference.float() - candidate.float()).abs().max())
    else:
        maximum = 0.0 if torch.equal(reference, candidate) else 1.0
    record = {
        "equal": bool(torch.equal(reference, candidate)),
        "max_abs_error": maximum,
        "shape": list(reference.shape),
        "reference_sha256": tensor_digest(reference),
        "candidate_sha256": tensor_digest(candidate),
    }
    # Same precision, same RNG and buffers: strict bitwise agreement is required.
    if not record["equal"]:
        raise RuntimeError(
            "Same-precision scheduling output mismatch: " + json.dumps(record)
        )
    return record


def validate(runner, adapter, args):
    checks = []
    try:
        with torch.cuda.nvtx.range("pir2/phase=validation"):
            for index in range(args.frames):
                _, reference = runner.pair("serial", index, index)
                for mode in (
                    "concurrent_equal",
                    "concurrent_dit_priority",
                    "cooperative_dit_priority",
                ):
                    row, candidate = runner.pair(mode, index, index)
                    check = {
                        "fixture": index,
                        "mode": mode,
                        "timing_excluded": True,
                        "dit": compare_tensor(reference["dit"], candidate["dit"]),
                        "vlm": {
                            key: compare_tensor(value, candidate["vlm"][key])
                            for key, value in reference["vlm"].items()
                        },
                        "cache_generations": [
                            row["s1_generation"],
                            row["s2_generation"],
                        ],
                    }
                    checks.append(check)
                    adapter.assert_clear()
        result = {
            "passed": True,
            "criterion": "bitwise equal tensors at fixed precision, restored BF16 bootstrap buffer and CUDA RNG",
            "checks": checks,
        }
    except BaseException as error:
        result = {"passed": False, "error": repr(error), "checks": checks}
        save(args.output / "equivalence.json", result)
        if "Same-precision scheduling output mismatch" in str(error):
            try:
                if args.capture == "cuda":
                    with torch.cuda.nvtx.range(
                        "pir2/phase=failed_equivalence_diagnostic"
                    ):
                        runner.pair("serial", index, index, capture=True)
                        runner.pair(mode, index, index, capture=True)
                else:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as failed_trace:
                        runner.pair("serial", index, index)
                        runner.pair(mode, index, index)
                    failed_trace.export_chrome_trace(
                        str(args.output / "failed-equivalence.chrome.json")
                    )
            except (RuntimeError, ValueError, OSError) as diagnostic_error:
                save(
                    args.output / "failed-equivalence-trace-error.json",
                    {"error": repr(diagnostic_error)},
                )
        raise
    save(args.output / "equivalence.json", result)
    return result
