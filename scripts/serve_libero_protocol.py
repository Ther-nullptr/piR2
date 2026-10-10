"""Serve one worker or paired S1/S2 workers with explicit CUDA placement."""

import argparse
import json
import logging
import os
import threading
from pathlib import Path

from libero_inference_backend import Backend, endpoint_layout

from coexecution.groot_optimization import (
    GrootOptimizations,
    add_optimization_arguments,
    optimization_config,
)

LOGGER = logging.getLogger(__name__)


def load_core(checkpoint, output, device):
    import torch
    from gr00t.policy import gr00t_policy as policy_module
    from gr00t.policy.decoupled_policy import DecoupledGr00tPolicy
    from libero_model_loading import task_checkpoint_backbone

    original = policy_module.AutoModel.from_pretrained

    def checked(*pos, **kwargs):
        model, info = original(*pos, **kwargs, output_loading_info=True)
        (output / "weight-loading.json").write_text(json.dumps(info, indent=2) + "\n")
        if any(
            info.get(k)
            for k in [
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            ]
        ):
            raise RuntimeError(str(info))
        return model

    policy_module.AutoModel.from_pretrained = checked
    try:
        with task_checkpoint_backbone(), torch.cuda.device(device):
            core = DecoupledGr00tPolicy("LIBERO_PANDA", str(checkpoint), device=device)
            # Startup only: parameters must be ready before an endpoint owns its stream.
            torch.cuda.synchronize(device)
    finally:
        policy_module.AutoModel.from_pretrained = original
    return core


def serve(backend, spec, identity, optimization=None):
    import torch
    from gr00t.policy.server_client import PolicyServer

    # Construct and run the ZMQ socket on its owning thread. A distinct stream per
    # endpoint permits same-context overlap on one GPU; fences are stream-local.
    with (
        torch.cuda.device(spec["device"]),
        torch.cuda.stream(backend.stream),
        GrootOptimizations(backend.core, optimization) as optimized,
    ):
        if optimized.config.enabled:
            identity["inference_optimization"] = optimized.evidence()
            (backend.output / "identity.json").write_text(
                json.dumps(identity, indent=2) + "\n"
            )
        server = PolicyServer(backend.core, host="127.0.0.1", port=spec["port"])
        server.register_endpoint("identity", lambda: identity, requires_input=False)
        server.register_endpoint(
            "profile_start",
            lambda: int(torch.cuda.cudart().cudaProfilerStart()),
            requires_input=False,
        )
        server.register_endpoint(
            "profile_stop",
            lambda: int(torch.cuda.cudart().cudaProfilerStop()),
            requires_input=False,
        )
        for name in ["reset", "bootstrap", "vision", "install", "plan"]:
            method = getattr(backend, name)

            def handler(_method=method, _name=name, **kwargs):
                with torch.cuda.nvtx.range(f"{spec['role']}:{_name}"):
                    return _method(**kwargs)

            server.register_endpoint(name, handler)
        print("TIMED_PROTOCOL_SERVER_READY", identity["variant"], spec, flush=True)
        server.run()


def supervise_workers(workers, worker_fn=serve):
    """Either endpoint exiting terminates paired service, including bind failures."""
    finished = threading.Event()
    errors = []

    def run(worker):
        try:
            worker_fn(*worker)
        except BaseException as error:
            LOGGER.exception("Paired inference endpoint exited with an error")
            errors.append(error)
        finally:
            finished.set()

    threads = [
        threading.Thread(target=run, args=(worker,), daemon=True) for worker in workers
    ]
    for thread in threads:
        thread.start()
    finished.wait()
    if errors:
        raise RuntimeError(f"Paired server failed: {errors[0]}") from errors[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--variant", choices=["flow", "pir2"], required=True)
    parser.add_argument("--role", choices=["action", "vlm"], default="action")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--vlm-device", help="Optional paired VLM worker in this process"
    )
    parser.add_argument("--vlm-port", type=int)
    parser.add_argument("--torch-threads", type=int, default=2)
    add_optimization_arguments(parser)
    args = parser.parse_args()
    optimization = optimization_config(args)
    optimization.validate_variant(args.variant)
    layout = endpoint_layout(
        args.device, args.vlm_device, args.port, args.vlm_port, args.role
    )
    if optimization.enabled and len(layout) > 1:
        raise ValueError(
            "Experimental optimizations require a single worker; "
            "paired S1/S2 integration is not validated"
        )
    import torch
    from libero_experiment_utils import checkpoint_identity

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    base_identity = checkpoint_identity(args.checkpoint)
    workers = []
    for spec in layout:
        output = args.output / spec["role"] if len(layout) > 1 else args.output
        output.mkdir(parents=True, exist_ok=True)
        core = load_core(args.checkpoint, output, spec["device"])
        if spec["role"] == "action":
            assert core.model.config.streaming == (args.variant == "pir2")
            assert (
                core.model.config.action_horizon
                == len(core.modality_configs["action"].delta_indices)
                == 40
            )
        backend = Backend(core, args.variant, output)
        identity = {
            **base_identity,
            "variant": args.variant,
            "role": spec["role"],
            "logical_device": spec["device"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pid": os.getpid(),
            "cuda_stream": backend.stream.cuda_stream,
            "paired_same_process": len(layout) > 1,
            "stream_local_synchronization": True,
            "feature_transport": "CPU numpy RPC, identical for single/dual layouts",
        }
        (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        workers.append((backend, spec, identity, optimization))
    if len(workers) == 1:
        serve(*workers[0])
    else:
        supervise_workers(workers)


if __name__ == "__main__":
    main()
