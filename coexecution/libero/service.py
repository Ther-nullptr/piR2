"""Serve one worker or paired S1/S2 workers with explicit CUDA placement."""

import argparse
import json
import logging
import os
import threading
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

from coexecution.libero.backend import Backend, endpoint_layout
from coexecution.libero.features import (
    SCHEMA,
    SCHEMA_VERSION,
    TIMING_VERSION,
    add_feature_arguments,
)
from coexecution.optimization.config import (
    add_optimization_arguments,
    optimization_config,
)
from coexecution.optimization.session import GrootOptimizations

LOGGER = logging.getLogger(__name__)


def validate_server_layout(
    layout, device, vlm_device, transport, variant, optimization
):
    if layout not in ("split", "paired"):
        raise ValueError("Unknown server layout")
    if (layout == "paired") != (vlm_device is not None):
        raise ValueError("Paired layout requires a second explicit worker device")
    if layout == "paired" and optimization.action_graph:
        raise ValueError("paired action graph is not validated; use per-DiT graphs")
    if transport == "cuda_local" and (
        layout != "paired" or device != vlm_device or variant != "flow"
    ):
        raise ValueError(
            "cuda_local requires paired Flow workers on the same CUDA device"
        )


@contextmanager
def worker_scope(backend, spec, optimization, torch):
    """Startup/teardown can submit CUDA too; exclude them from peer capture."""
    coordinator = getattr(backend, "coordinator", None)
    guard = coordinator.capture if coordinator is not None else nullcontext
    stack = ExitStack()
    try:
        with guard():
            stack.enter_context(torch.cuda.device(spec["device"]))
            stack.enter_context(torch.cuda.stream(backend.stream))
            optimized = stack.enter_context(
                GrootOptimizations(backend.core, optimization)
            )
        yield optimized
    finally:
        with guard():
            stack.close()


def policy_horizons(core, variant):
    predicted = core.model.config.action_horizon
    decoded = len(core.modality_configs["action"].delta_indices)
    if core.model.config.streaming != (variant == "pir2") or predicted != 40:
        raise ValueError("The declared variant requires the fixed H40 model")
    if not 5 < decoded <= predicted:
        raise ValueError(
            "Native decoded horizon must exceed the five-tick delay budget"
        )
    return {"predicted_horizon": predicted, "decoded_horizon": decoded}


def load_core(checkpoint, output, device):
    import torch
    from gr00t.policy import gr00t_policy as policy_module
    from gr00t.policy.decoupled_policy import DecoupledGr00tPolicy

    from coexecution.libero.loading import task_checkpoint_backbone

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
    with worker_scope(backend, spec, optimization, torch) as optimized:
        backend.optimization = optimized
        if optimized.config.enabled:
            identity["inference_optimization"] = optimized.evidence()
            (backend.output / "identity.json").write_text(
                json.dumps(identity, indent=2) + "\n"
            )
        server = PolicyServer(backend.core, host="127.0.0.1", port=spec["port"])
        coordinator = getattr(backend, "coordinator", None)
        submission = coordinator.submission if coordinator is not None else nullcontext
        # The pinned server also installs its upstream policy endpoints. Every
        # CUDA path must join the same guard, even when a diagnostic calls one.
        for name, endpoint in tuple(getattr(server, "_endpoints", {}).items()):

            def guarded(_method=endpoint.handler, **kwargs):
                with submission():
                    return _method(**kwargs)

            server.register_endpoint(
                name, guarded, requires_input=endpoint.requires_input
            )

        def current_identity():
            observed = dict(identity)
            if getattr(backend, "feature_exchange", None) is not None:
                observed["feature_exchange"] = backend.feature_exchange.evidence()
            return observed

        server.register_endpoint("identity", current_identity, requires_input=False)

        def profile(start):
            with submission():
                api = torch.cuda.cudart()
                return int(api.cudaProfilerStart() if start else api.cudaProfilerStop())

        server.register_endpoint(
            "profile_start",
            lambda: profile(True),
            requires_input=False,
        )
        server.register_endpoint(
            "profile_stop",
            lambda: profile(False),
            requires_input=False,
        )
        names = ["reset", "bootstrap", "vision", "install", "plan"]
        if hasattr(backend, "feature_control"):
            names.append("feature_control")
        for name in names:
            method = getattr(backend, name)

            def handler(_method=method, _name=name, **kwargs):
                with submission(), torch.cuda.nvtx.range(f"{spec['role']}:{_name}"):
                    return _method(**kwargs)

            server.register_endpoint(name, handler)
        print("TIMED_PROTOCOL_SERVER_READY", identity["variant"], spec, flush=True)
        try:
            server.run()
        finally:
            if hasattr(server, "socket"):
                server.socket.close()
                server.context.term()


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
    parser.add_argument("--server-layout", choices=("split", "paired"))
    parser.add_argument("--feature-max-leases", type=int, default=16)
    add_feature_arguments(parser)
    add_optimization_arguments(parser)
    args = parser.parse_args()
    optimization = optimization_config(args)
    optimization.validate_variant(args.variant)
    layout = endpoint_layout(
        args.device, args.vlm_device, args.port, args.vlm_port, args.role
    )
    server_layout = args.server_layout or ("paired" if len(layout) > 1 else "split")
    validate_server_layout(
        server_layout,
        args.device,
        args.vlm_device,
        args.feature_transport,
        args.variant,
        optimization,
    )
    if args.feature_max_leases < 1:
        raise ValueError("Feature max leases must be positive")
    import torch

    from scripts.libero_experiment_utils import checkpoint_identity

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    from coexecution.libero.resident import FeatureLeasePool
    from coexecution.optimization.execution.submission import CudaSubmissionCoordinator

    coordinator = (
        CudaSubmissionCoordinator(
            quiesce=lambda: [
                torch.cuda.synchronize(device)
                for device in sorted({spec["device"] for spec in layout})
            ]
        )
        if len(layout) > 1
        else None
    )
    exchange = (
        FeatureLeasePool(args.device, max_entries=args.feature_max_leases)
        if args.feature_transport == "cuda_local"
        else None
    )
    base_identity = checkpoint_identity(args.checkpoint)
    server_instance = uuid.uuid4().hex
    workers = []
    for spec in layout:
        output = args.output / spec["role"] if len(layout) > 1 else args.output
        output.mkdir(parents=True, exist_ok=True)
        core = load_core(args.checkpoint, output, spec["device"])
        horizons = policy_horizons(core, args.variant)
        backend = Backend(
            core,
            args.variant,
            output,
            feature_transport=args.feature_transport,
            feature_exchange=exchange,
            role=spec["role"],
            paired=len(layout) > 1,
        )
        backend.coordinator = coordinator
        identity = {
            **base_identity,
            **horizons,
            "variant": args.variant,
            "role": spec["role"],
            "logical_device": spec["device"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pid": os.getpid(),
            "cuda_stream": backend.stream.cuda_stream,
            "paired_same_process": len(layout) > 1,
            "server_layout": server_layout,
            "server_instance": server_instance,
            "policy_instance": uuid.uuid4().hex,
            "physical_device_uuid": str(
                torch.cuda.get_device_properties(spec["device"]).uuid
            ),
            "capture_coordination": "process_reader_writer_with_quiescent_cold_capture"
            if coordinator
            else "single_endpoint",
            "vision_reset_rng": "deterministic_no_reseed"
            if len(layout) > 1
            else "legacy_global_seed",
            "stream_local_synchronization": True,
            "feature_transport": args.feature_transport,
            "feature_wire": {
                "schema": SCHEMA
                if args.feature_transport == "numpy_bf16"
                else "legacy_numpy_dict",
                "version": SCHEMA_VERSION
                if args.feature_transport == "numpy_bf16"
                else None,
                "source": "vlm_cache_raw_tensor"
                if args.feature_transport == "numpy_bf16"
                else "upstream_return_features",
                "payload_bytes_evidence": "response meta.feature_wire.payload_nbytes; array storage only",
                "installation": "owned tensors on endpoint CUDA stream, complete before plan",
            },
            "feature_rpc_modes": ["separate", "combined"],
            "timing_version": TIMING_VERSION,
        }
        if exchange is not None:
            from coexecution.libero.resident import SCHEMA as RESIDENT_SCHEMA
            from coexecution.libero.resident import VERSION

            identity["feature_exchange"] = exchange.evidence()
            identity["feature_wire"] = {
                "schema": RESIDENT_SCHEMA,
                "version": VERSION,
                "source": "vlm_cache_owned_cuda_snapshot",
                "payload_bytes_evidence": "metadata only; tensor bytes in resident_nbytes",
                "installation": "producer event wait and record_stream; independent consumer tensor references",
            }
        (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        workers.append((backend, spec, identity, optimization))
    try:
        if len(workers) == 1:
            serve(*workers[0])
        else:
            supervise_workers(workers)
    finally:
        if exchange is not None:
            with coordinator.capture():
                exchange.close()


if __name__ == "__main__":
    main()
