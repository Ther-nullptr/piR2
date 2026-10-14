"""Resident real LIBERO fixtures and independently owned Flow4 policy instances."""

import threading
import time
from collections import Counter
from contextlib import ExitStack

from coexecution.kernels.ampere_ada._candidate_support import _ProjectionTactic

from .common import metadata, save, tensor_digest
from .flow_protocol import validate_flow_head


class ProjectionEvidence:
    """Observe selected integer dispatch/packing during cold warmup only.

    Packing is included because the segmented group-output implementation can
    bypass ``dispatch``. Counts are adapter visits, never GEMM/kernel counts.
    The wrappers remain installed and inactive during steady execution so frozen
    graph guards see the same callables throughout the experiment.
    """

    def __init__(self, session, role):
        self.role, self.active = role, True
        self.counts = Counter()
        self.controller = getattr(session._clear_cache, "__self__", None)
        self.expected = {
            name
            for field in ("linears", "category_linears")
            for name in session.coverage.get(field, [])
            if name.startswith("backbone." if role == "vlm" else "action_head.")
        }
        self.inactive = {}
        if (
            role == "vlm"
            and "backbone.model.lm_head" in self.expected
            and getattr(getattr(session, "config", None), "static_vlm", False)
        ):
            from coexecution.optimization.groot.vlm import PolicyStaticS2

            adapter = getattr(session, "static_vlm", None)
            if type(adapter) is PolicyStaticS2 and adapter._entered:
                self.inactive["backbone.model.lm_head"] = {
                    "name": "backbone.model.lm_head",
                    "reason": "The installed static feature path calls the inner Qwen3VLModel and returns captured features; it does not invoke the outer language-logit projection.",
                    "evidence": "PolicyStaticS2._forward -> StaticS2.forward_inputs -> self.inner(...); PolicyStaticS2._check_model binds outer.model to core.inner and rejects outer/lm_head hooks.",
                    "source": "coexecution/static_s2.py:StaticS2.forward_inputs",
                    "adapter": "coexecution.optimization.groot.vlm.PolicyStaticS2",
                    "adapter_entered": True,
                }
        self.expected_active = self.expected - self.inactive.keys()
        self.names = {}
        self.originals = []
        if self.controller is None:
            return
        controller = self.controller
        for bits, entries in controller.quant.items():
            for name, linear in entries.items():
                self.names[id(linear)] = (bits, (name,))
            for names, group in controller.grouped[bits]:
                self.names[id(group.linear)] = (bits, tuple(names))
        modules = dict(session.policy.model.named_modules())
        for name in session.coverage.get("category_linears", []):
            for value in getattr(modules[name].forward, "__defaults__", ()) or ():
                if hasattr(value, "packed_weight") and hasattr(value, "bits"):
                    self.names[id(value)] = (value.bits, (name,))
        targets = [
            (controller, method, method)
            for method in ("dispatch", "_pack_dispatch_input")
        ]
        candidate = getattr(controller, "candidates", None)
        if candidate is not None:
            # Native residual writeback reaches the candidate packer directly,
            # bypassing both controller hooks above. Observe that actual path.
            targets.append((candidate, "pack_input", "candidate.pack_input"))
        for owner, method, label in targets:
            original = getattr(owner, method)
            self.originals.append((owner, method, original))

            def observed(q, *args, _original=original, _method=label, **kwargs):
                if self.active:
                    projection = q
                    while isinstance(projection, _ProjectionTactic):
                        projection = projection.projection
                    bits, names = self.names.get(
                        id(projection),
                        (getattr(projection, "bits", None), ("unmapped",)),
                    )
                    for name in names:
                        self.counts[(name, bits, _method)] += 1
                return _original(q, *args, **kwargs)

            setattr(owner, method, observed)

    def evidence(self):
        observed = {name for name, _, _ in self.counts}
        unexpected_inactive = observed & self.inactive.keys()
        return {
            "component": self.role,
            "scope": "Observed integer dispatch, group input-pack or candidate input-pack adapter visits (including fused residual projections) during feature preparation/warmup/capture; counts are not timed kernel counts.",
            "passes": self.expected_active <= observed
            and "unmapped" not in observed
            and not unexpected_inactive,
            "expected_selected_projections": sorted(self.expected),
            "selected_projections": sorted(self.expected),
            "expected_active_projections": sorted(self.expected_active),
            "inactive_selected_projections": list(self.inactive.values()),
            "unexpected_inactive_execution": sorted(unexpected_inactive),
            "missing": sorted(self.expected_active - observed),
            "observed": [
                {"name": name, "bits": bits, "adapter": adapter, "visits": count}
                for (name, bits, adapter), count in sorted(self.counts.items())
            ],
        }

    def assert_clear(self):
        if self.controller is not None:
            for records in self.controller.grouped.values():
                for _, group in records:
                    if (
                        group._input is not None
                        or group._output is not None
                        or group._remaining
                    ):
                        raise RuntimeError(
                            "A component projection cache survived completion"
                        )

    def close(self):
        for owner, method, original in reversed(self.originals):
            setattr(owner, method, original)


class FlowWorkload:
    """Two full resident policies; no mutable model/graph/cache object is shared."""

    def __init__(self, args):
        self.args = args
        self.stack = ExitStack()
        self.context = threading.local()
        self.policies, self.sessions, self.coverage = {}, {}, {}
        self.hooks = []

    def __enter__(self):
        try:
            self._load()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        self.stack.close()
        for evidence in self.coverage.values():
            evidence.close()
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def _load(self):
        import torch

        from coexecution.libero.inference import (
            PreparedInferenceEngine,
            prepare_observations,
        )
        from coexecution.libero.measurement import _observations
        from coexecution.libero.service import load_core
        from coexecution.optimization.session import GrootOptimizations
        from coexecution.replay.assets import export_observations

        from .flow import configuration

        self.torch = torch
        self.devices = {"vlm": self.args.vlm_device, "dit": self.args.action_device}
        for role, device in self.devices.items():
            directory = self.args.output / role
            directory.mkdir()
            policy = load_core(self.args.checkpoint, directory, device)
            policy.model.eval()
            validate_flow_head(policy.model.action_head)
            self.policies[role] = policy
        left = {id(value) for value in self.policies["vlm"].model.parameters()}
        if left & {id(value) for value in self.policies["dit"].model.parameters()}:
            raise RuntimeError("Independent component policies share parameter objects")
        self.layer_modules = [
            (name, module)
            for name, module in self.policies["vlm"].model.backbone.named_modules()
            if type(module).__name__
            in ("Qwen3VLVisionBlock", "Qwen3VLTextDecoderLayer")
        ]
        if self.args.group == "cooperative":
            if not self.layer_modules:
                raise RuntimeError("Completed VLM layer boundaries were not found")
            # Install once before static metadata binds. All cooperative-group
            # controls pay the same no-op hook overhead; only fenced cases wait.
            for name, module in self.layer_modules:
                self.hooks.append(
                    module.register_forward_hook(
                        lambda _m, _a, _o, name=name: self._boundary(name)
                    )
                )
        for role, policy in self.policies.items():
            with torch.cuda.device(self.devices[role]):
                session = self.stack.enter_context(
                    GrootOptimizations(policy, configuration(self.args, role))
                )
                self.sessions[role] = session
                self.coverage[role] = ProjectionEvidence(session, role)
        observations, self.fixtures = _observations(self.policies["vlm"], self.args)
        export_observations(self.args.output, observations, self.fixtures)
        identities = [item["observation_id"] for item in self.fixtures]
        self.prepared, self.engines = {}, {}
        for role, policy in self.policies.items():
            with torch.cuda.device(self.devices[role]):
                self.prepared[role] = prepare_observations(
                    policy, observations, identities, limit=3
                )
                self.engines[role] = PreparedInferenceEngine(policy.model)
                torch.cuda.synchronize(self.devices[role])
        self.streams = {}
        for key, role, priority in (
            ("vlm", "vlm", 0),
            ("dit_equal", "dit", 0),
            ("dit_priority", "dit", -1),
        ):
            with torch.cuda.device(self.devices[role]):
                low, high = torch.cuda.current_stream().priority_range()
                if not high <= priority <= low:
                    raise RuntimeError(
                        f"Requested stream priority {priority} unavailable: {(low, high)}"
                    )
                self.streams[key] = torch.cuda.Stream(
                    device=self.devices[role], priority=priority
                )
        self.stream_evidence = {
            name: {
                "device": str(stream.device),
                "id": stream.cuda_stream,
                "priority": stream.priority,
            }
            for name, stream in self.streams.items()
        }
        self.reference_features, self.feature_records = [], []
        rng_before = torch.cuda.get_rng_state(self.devices["vlm"])
        for index, inputs in enumerate(self.prepared["vlm"]):
            stream = self.streams["vlm"]
            with torch.cuda.device(self.devices["vlm"]), torch.cuda.stream(stream):
                result = self.engines["vlm"].vlm(inputs.backbone)
                source = {key: value.detach().clone() for key, value in result.items()}
            stream.synchronize()
            with torch.cuda.device(self.devices["dit"]):
                resident = {
                    key: value.to(self.devices["dit"], copy=True)
                    for key, value in source.items()
                }
                torch.cuda.synchronize(self.devices["dit"])
            self.reference_features.append(resident)
            self.feature_records.append(
                {
                    "generation": index,
                    "fixture": index,
                    "observation_id": identities[index],
                    "ready_ns": time.perf_counter_ns(),
                    "tensors": metadata(resident),
                    "sha256": {
                        key: tensor_digest(value) for key, value in resident.items()
                    },
                    "source_sha256": {
                        key: tensor_digest(value) for key, value in source.items()
                    },
                }
            )
        if not torch.equal(rng_before, torch.cuda.get_rng_state(self.devices["vlm"])):
            raise RuntimeError("VLM feature preparation consumed CUDA RNG")
        self.rng_states = [
            torch.Generator(device=self.devices["dit"])
            .manual_seed(self.args.seed + index)
            .get_state()
            for index in range(len(self.fixtures))
        ]
        self.vlm_rng = torch.cuda.get_rng_state(self.devices["vlm"])
        save(self.args.output / "fixtures.json", self.fixtures)
        save(self.args.output / "features.json", self.feature_records)

    def _boundary(self, name):
        context = getattr(self.context, "current", None)
        if context is None or not context[0].fenced:
            return
        _case, gate, counter = context
        counter[0] += 1
        if counter[0] % self.args.vlm_chunk_blocks == 0 or counter[0] == len(
            self.layer_modules
        ):
            with self.torch.cuda.nvtx.range(f"Flow4/S2/completed_boundary/{name}"):
                self.streams["vlm"].synchronize()
                gate.boundary(name)

    def restore(self, index):
        # Caller invokes only after both workers and their own completion events
        # finished. Never reset all GPUs or the peer device's generator.
        self.torch.cuda.set_rng_state(self.rng_states[index], self.devices["dit"])

    def check_rng(self, index, has_dit, expected_after):
        torch = self.torch
        actual = torch.cuda.get_rng_state(self.devices["dit"])
        expected = expected_after[index] if has_dit else self.rng_states[index]
        if not torch.equal(actual, expected):
            raise RuntimeError(
                "CUDA RNG consumption differs from the matched serial request"
            )
        if self.devices["vlm"] != self.devices["dit"] and not torch.equal(
            self.vlm_rng, torch.cuda.get_rng_state(self.devices["vlm"])
        ):
            raise RuntimeError("VLM consumed CUDA RNG on its own device")

    def assert_features_unchanged(self):
        for values, record in zip(self.reference_features, self.feature_records):
            if {key: tensor_digest(value) for key, value in values.items()} != record[
                "sha256"
            ]:
                raise RuntimeError(
                    "An old feature generation was modified during the campaign"
                )
