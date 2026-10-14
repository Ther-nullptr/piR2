"""Explicit BF16 activation calibration and instance-owned INT4 balancing.

No class-global monkeypatches, mutable process-global precision map or shared
calibration cache. Original model parameters are never changed.
"""

import torch

from coexecution.kernels.ampere_ada.integer import IntegerProjectionGroup
from coexecution.kernels.common.fused import prepare_gelu
from coexecution.optimization.backends.integer.linear import PreparedLinear
from coexecution.optimization.config import selected_component
from coexecution.optimization.groot.projections import inventory


class ActivationCalibration:
    """Collect per-input-channel absmax on a caller-run BF16 CUDA baseline."""

    def __init__(self, policy, *, expanded=True, component="all", category_id=None):
        model = policy.model
        if (
            model.training
            or next(model.parameters()).dtype != torch.bfloat16
            or next(model.parameters()).device.type != "cuda"
        ):
            raise ValueError("Calibration requires an eval-mode BF16 CUDA baseline")
        self.modules = {
            name: value[0]
            for name, value in inventory(model, expanded)[0].items()
            if selected_component(name, component)
        }
        if category_id is not None:
            if not expanded:
                raise ValueError("Category calibration requires expanded scope")
            self.modules.update(
                {
                    name: module
                    for name, module in model.named_modules()
                    if type(module).__name__ == "CategorySpecificLinear"
                    and selected_component(name, component)
                }
            )
        self.maxima = {}
        self.calls = {}
        self.handles = []

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Calibration already active")
        try:
            for name, module in self.modules.items():

                def hook(_module, args, _output, name=name):
                    x = args[0].detach()
                    maximum = x.reshape(-1, x.shape[-1]).float().abs().amax(0)
                    self.maxima[name] = (
                        maximum
                        if name not in self.maxima
                        else torch.maximum(self.maxima[name], maximum)
                    )
                    self.calls[name] = self.calls.get(name, 0) + 1

                self.handles.append(module.register_forward_hook(hook))
        except BaseException:
            self.__exit__()
            raise
        return self

    def statistics(self):
        if set(self.maxima) != set(self.modules):
            raise ValueError("Some candidate projections were not calibrated")
        if any(not bool(torch.isfinite(x).all()) for x in self.maxima.values()):
            raise ValueError("Non-finite calibration statistics")
        return {n: x.detach().cpu().clone() for n, x in self.maxima.items()}

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_balance(controller, extra, path, *, alpha=0.5, route="register"):
    stats = torch.load(path, map_location="cpu", weights_only=True)
    weights = {n: m.weight for n, (m, _, _) in controller.sites.items()}
    weights.update({n: m.weight for n, m in extra.linears.items()})
    if not isinstance(stats, dict) or not set(weights) <= set(stats):
        raise ValueError("Calibration must cover all selected integer projections")
    for name, weight in weights.items():
        a = stats[name]
        if (
            not isinstance(a, torch.Tensor)
            or a.shape != (weight.shape[1],)
            or not bool(torch.isfinite(a).all())
            or bool((a < 0).any())
        ):
            raise ValueError(f"Invalid per-channel calibration: {name}")
    if route == "register":
        from coexecution.kernels.build import build_extension

        build_extension("int4_prepare")
    scales = {}

    def scale_for(names):
        if alpha == 0:
            return torch.ones(
                weights[names[0]].shape[1],
                device=weights[names[0]].device,
                dtype=torch.float32,
            )
        w = (
            torch.stack([weights[n].float().abs().amax(0) for n in names])
            .amax(0)
            .clamp_min(1e-4)
        )
        a = (
            torch.stack([stats[n].to(w.device, dtype=torch.float32) for n in names])
            .amax(0)
            .clamp_min(1e-4)
        )
        scale = (a.pow(alpha) / w.pow(1 - alpha)).clamp(1e-3, 1e3)
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("Non-finite channel balance")
        return scale

    for names in controller.groups:
        scale = scale_for(names)
        scales.update({n: scale for n in names})
    for name in weights:
        if name not in scales:
            scales[name] = scale_for([name])

    def attach(q, inverse):
        q.register_buffer("inverse_balance", inverse)
        q.prepare_route = route
        q.forward = lambda x, q=q: controller.dispatch(q, x)
        q.forward_gelu = lambda x, up=None, approximate="tanh", q=q: (
            controller.dispatch(q, x, prepare_gelu(x.device, x.dtype, approximate), up)
        )

    for name, weight in weights.items():
        original = (
            controller.sites[name][0]
            if name in controller.sites
            else extra.linears[name]
        )
        replacement = torch.nn.Linear(
            weight.shape[1], weight.shape[0], device="meta", dtype=torch.bfloat16
        )
        replacement.weight = torch.nn.Parameter(
            (weight.float() * scales[name][None, :]).to(torch.bfloat16),
            requires_grad=False,
        )
        replacement.bias = original.bias
        q = PreparedLinear.from_linear(
            replacement, bits=4, pack_reuse=True, biasless_epilogue=True
        )
        attach(q, scales[name].reciprocal().contiguous())
        (controller.quant if name in controller.sites else extra.quant)[4][name] = q
    grouped = []
    for names, old in controller.grouped[4]:
        group = IntegerProjectionGroup(
            [controller.quant[4][n] for n in names],
            contiguous_outputs=old.contiguous_outputs,
        )
        attach(group.linear, scales[names[0]].reciprocal().contiguous())
        grouped.append((names, group))
    controller.grouped[4] = grouped
    return {
        "alpha": alpha,
        "prepare_route": route,
        "selected_candidates": len(weights),
        "weight_rounding": "FP32 balance multiply -> BF16 -> signed INT4 pack",
        "activation_rounding": "BF16 nonlinear boundary -> FP32 balance -> RN row quantize",
        "scales": {
            n: {"min": float(s.min()), "max": float(s.max())} for n, s in scales.items()
        },
    }
