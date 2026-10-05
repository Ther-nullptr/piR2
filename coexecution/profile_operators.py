"""Short diagnostic capture with module, ATen, and real tensor-layout evidence.

Dispatch instrumentation changes CPU launch overhead. Never use this entrypoint
as a latency benchmark; the ordinary coexecution.run path stays uninstrumented.
"""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from coexecution.run import Runner
from coexecution.workload import PiR2Workload


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class OperatorRecorder:
    def __init__(self):
        self.records = []
        self.signatures = {}
        self.lock = Lock()

    def identify(self, operator, args, kwargs):
        record = {
            "operator": str(operator),
            "args": describe(args),
            "kwargs": describe(kwargs),
        }
        signature = json.dumps(record, sort_keys=True)
        with self.lock:
            if signature not in self.signatures:
                self.signatures[signature] = len(self.records)
                self.records.append(record)
            return self.signatures[signature]


class OperatorRanges(TorchDispatchMode):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        index = self.recorder.identify(func, args, kwargs)
        with torch.cuda.nvtx.range(f"OP/{func}/{index}"):
            return func(*args, **kwargs)


@contextmanager
def module_ranges(workload):
    handles, metadata = [], {}
    roots = {
        "S1": workload.model.action_head,
        "S2": workload.model.backbone.model.model,
    }
    for role, root in roots.items():
        for name, module in root.named_modules():
            label = f"MODULE/{role}/{name or 'root'}"
            metadata[label] = {
                "class": type(module).__name__,
                "parameters": {
                    key: describe(value)
                    for key, value in module.named_parameters(recurse=False)
                },
            }

            def before(_module, _args, label=label):
                torch.cuda.nvtx.range_push(label)

            def after(_module, _args, _output):
                torch.cuda.nvtx.range_pop()

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
    try:
        yield metadata
    finally:
        for handle in handles:
            handle.remove()


class DiagnosticRunner(Runner):
    def __init__(self, workload, recorder):
        super().__init__(workload)
        self.recorder = recorder

    def fast_job(self, *args, **kwargs):
        with OperatorRanges(self.recorder):
            return super().fast_job(*args, **kwargs)

    def slow_job(self, *args, **kwargs):
        with OperatorRanges(self.recorder):
            return super().slow_job(*args, **kwargs)


def main(args):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    workload = PiR2Workload(args.checkpoint)
    reference_runner = Runner(workload)
    _, reference, _ = reference_runner.paired(
        "serial", args.iterations, args.warmup, keep_outputs=True
    )
    reference_runner.close()
    recorder = OperatorRecorder()
    runner = DiagnosticRunner(workload, recorder)
    results = {"rows": [], "correctness": {}, "workload": workload.metadata()}
    results.update(
        physical_visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_count=torch.cuda.device_count(),
        instrumented=True,
        note="Diagnostic kernel attribution only: added Python/NVTX overhead changes submission and overlap.",
    )
    with module_ranges(workload) as modules:
        torch.cuda.cudart().cudaProfilerStart()
        outputs = {}
        for mode in ["serial", "concurrent"]:
            rows, outputs[mode], _ = runner.paired(
                mode, args.iterations, args.warmup, keep_outputs=True
            )
            results["rows"].extend(rows)
        torch.cuda.cudart().cudaProfilerStop()
    for mode, actual in outputs.items():
        errors = []
        for expected, candidate in zip(reference, actual):
            torch.testing.assert_close(expected, candidate, rtol=0, atol=0)
            errors.append(float((expected.float() - candidate.float()).abs().max()))
        results["correctness"][mode] = {"compared": len(errors), "max_abs": max(errors)}
    runner.close()
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    (args.output / "operators.json").write_text(json.dumps(recorder.records, indent=2))
    (args.output / "modules.json").write_text(json.dumps(modules, indent=2))
    print(
        json.dumps(
            {
                "correctness": results["correctness"],
                "operator_signatures": len(recorder.records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/pir2-so100-smoke/checkpoint-10"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    main(parser.parse_args())
