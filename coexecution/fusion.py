"""Scoped model adapters; the eager baseline is restored when the scope exits."""

from collections import Counter
from contextlib import nullcontext
from types import MethodType

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from coexecution import fused_ops
from coexecution.patching import ScopedReplacements

VARIANTS = {
    "baseline": set(),
    "vision_rope": {"vision_rope"},
    "rope": {"vision_rope", "text_rope"},
    "rms": {"rmsnorm"},
    "rope_rms": {"vision_rope", "text_rope", "rmsnorm"},
}


class FusionPatch(ScopedReplacements):
    def __init__(self, workload, variant, validate=False, profile=False):
        super().__init__()
        self.workload = workload
        self.variant = variant
        self.enabled = VARIANTS[variant]
        self.validate = validate
        self.profile = profile
        self.calls = Counter()
        self.validated = Counter()
        self.samples = {}

    def invoke(self, name, reference, candidate, args, kwargs):
        use_fused = name in self.enabled
        self.calls[name + ("/fused" if use_fused else "/reference")] += 1
        scope = (
            torch.cuda.nvtx.range(f"CHAIN/{name}") if self.profile else nullcontext()
        )
        with scope:
            result = (candidate if use_fused else reference)(*args, **kwargs)
        if self.validate and use_fused:
            expected = reference(*args, **kwargs)
            pairs = (
                zip(expected, result)
                if isinstance(result, tuple)
                else [(expected, result)]
            )
            for first, second in pairs:
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                assert first.stride() == second.stride()
                assert torch.isfinite(second).all()
            self.validated[name] += 1
            signature = name + ":" + str(tuple(args[0].shape))
            if signature not in self.samples:
                # Validation-only capture; never clone input tensors in timing runs.
                captured = tuple(
                    value.detach().clone() if torch.is_tensor(value) else value
                    for value in args
                )
                self.samples[signature] = (reference, candidate, captured, dict(kwargs))
        return result

    def _install(self):
        for attr, name, candidate in [
            ("apply_rotary_pos_emb_vision", "vision_rope", fused_ops.vision_rope),
            ("apply_rotary_pos_emb", "text_rope", fused_ops.text_rope),
        ]:
            if name not in self.enabled and not self.profile:
                continue
            original = getattr(qwen, attr)

            def wrapper(
                *args, _name=name, _reference=original, _candidate=candidate, **kwargs
            ):
                return self.invoke(_name, _reference, _candidate, args, kwargs)

            self.replace(qwen, attr, wrapper)
        if "rmsnorm" in self.enabled or self.profile:
            for module in self.workload.model.backbone.modules():
                if isinstance(module, qwen.Qwen3VLTextRMSNorm):
                    original = module.forward

                    def forward(instance, x, _reference=original):
                        def candidate(values):
                            return fused_ops.rmsnorm(
                                values, instance.weight, instance.variance_epsilon
                            )

                        return self.invoke("rmsnorm", _reference, candidate, (x,), {})

                    self.replace(module, "forward", MethodType(forward, module))
        return self

    def evidence(self):
        return {
            "variant": self.variant,
            "enabled": sorted(self.enabled),
            "calls": dict(self.calls),
            "validated_calls": dict(self.validated),
            "fallbacks": 0,
        }
