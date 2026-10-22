"""SmoothQuant channel balancing for the existing Thor floating GEMMs.

The absmax statistic and alpha formula follow mit-han-lab/smoothquant at
c61476d728e42ae0d8a35e7e78494edcac3237b5 (MIT; see the retained license).
This is an FP8/FP4 adaptation, not the upstream INT8 model implementation.
For PyTorch weights [N,K], use (X / s) @ (W * s).T + bias. The input division
is fused into packing, including AdaLN/residual/SwiGLU producers. Original
parameters are never edited. Calibration hooks are for eager inference only.
"""

import hashlib
import math
from contextlib import ExitStack
from pathlib import Path

import torch

UPSTREAM_REVISION = "c61476d728e42ae0d8a35e7e78494edcac3237b5"
PROFILE_SCHEMA = "pir2.thor.smoothquant.v1"


def weight_digest(weight):
    raw = weight.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def channel_scales(activation_absmax, weights, alpha):
    """Upstream balancing formula, with FP32 statistics and shared-consumer max."""
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("SmoothQuant alpha must be finite and in [0,1]")
    if not weights:
        raise ValueError("SmoothQuant needs at least one weight")
    device = weights[0].device
    act = activation_absmax.detach().to(device=device, dtype=torch.float32)
    if act.ndim != 1 or not torch.isfinite(act).all() or (act < 0).any():
        raise ValueError("Invalid activation channel statistics")
    if any(w.ndim != 2 or w.shape[1] != act.numel() for w in weights):
        raise ValueError("SmoothQuant channel count differs from weight inputs")
    weight_max = torch.stack([w.detach().float().abs().amax(0) for w in weights])
    weight_max = weight_max.amax(0).clamp(min=1e-5)
    scales = (act.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)
    if not torch.isfinite(scales).all():
        raise ValueError("Nonfinite SmoothQuant scales")
    return scales.contiguous()


def projection_groups(sites, groups):
    """Partition selected sites; shared QKV/KV/gate-up always share one scale."""
    result, used = [], set()
    for group in groups:
        members = [name for name in group if name in sites]
        if members:
            result.append(members)
            used.update(members)
    result.extend([name] for name in sites if name not in used)
    return result


class ActivationCalibration:
    """Per-input-channel maxima and bounded deterministic sample rows.

    A private generator never advances the model's RNG. Calibration is eager;
    CUDA Graph replay does not execute Python hooks and is explicitly rejected.
    Samples are only for parameter selection, never for held-out reporting.
    """

    def __init__(self, sites, *, rows_per_call=4, max_rows=128, seed=1000):
        if rows_per_call < 1 or max_rows < 1:
            raise ValueError("Calibration sample bounds must be positive")
        self.sites = sites
        self.rows_per_call, self.max_rows = rows_per_call, max_rows
        self.seed = seed
        self.maxima, self.samples, self.calls = {}, {}, {}
        self.stack = ExitStack()
        self.generator = None

    def __enter__(self):
        device = next(iter(self.sites.values()))[0].weight.device
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        try:
            for name, (module, _, _) in self.sites.items():

                def hook(_module, inputs, _output, name=name):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "SmoothQuant calibration requires eager inference"
                        )
                    value = inputs[0].detach().reshape(-1, _module.in_features)
                    maximum = value.float().abs().amax(0)
                    if not torch.isfinite(maximum).all():
                        raise ValueError(f"Nonfinite calibration activation: {name}")
                    self.maxima[name] = (
                        maximum
                        if name not in self.maxima
                        else torch.maximum(self.maxima[name], maximum)
                    )
                    self.calls[name] = self.calls.get(name, 0) + 1
                    indices = torch.randperm(
                        value.shape[0], device=value.device, generator=self.generator
                    )[: self.rows_per_call]
                    sample = value[indices].clone()
                    if name in self.samples:
                        sample = torch.cat((self.samples[name], sample))
                    if sample.shape[0] > self.max_rows:
                        indices = torch.randperm(
                            sample.shape[0],
                            device=value.device,
                            generator=self.generator,
                        )[: self.max_rows]
                        sample = sample[indices].contiguous()
                    self.samples[name] = sample

                self.stack.callback(module.register_forward_hook(hook).remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exception):
        self.stack.close()


def load_profile(path, sites, groups, precision):
    """Validate calibration provenance and bind immutable, shared GPU scales."""
    profile = torch.load(path, map_location="cpu", weights_only=True)
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("precision") != precision:
        raise ValueError("SmoothQuant profile schema or precision mismatch")
    if profile.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("SmoothQuant profile uses an unrecognized source revision")
    records = profile.get("linears", {})
    if not records or records.keys() - sites.keys():
        raise ValueError(
            "SmoothQuant profile has no sites or unsupported projection names"
        )
    scales = {}
    for names in projection_groups(sites, groups):
        present = [name for name in names if name in records]
        if not present:
            continue
        if len(present) != len(names):
            raise ValueError("SmoothQuant shared-input group has partial scales")
        first = records[names[0]]["scale"]
        for name in names:
            record, module = records[name], sites[name][0]
            scale = record["scale"]
            if (
                not isinstance(scale, torch.Tensor)
                or scale.dtype != torch.float32
                or scale.shape != (module.in_features,)
                or scale.requires_grad
                or not torch.isfinite(scale).all()
                or not (scale > 0).all()
            ):
                raise ValueError(f"Invalid SmoothQuant channel scale: {name}")
            if not torch.equal(scale, first):
                raise ValueError(
                    "SmoothQuant shared-input consumers have different scales"
                )
            if record.get("weight_sha256") != weight_digest(module.weight):
                raise ValueError(f"SmoothQuant profile weight mismatch: {name}")
        bound = first.to(device=sites[names[0]][0].weight.device).contiguous()
        for name in names:
            scales[name] = bound
    evidence = {
        "schema": PROFILE_SCHEMA,
        "upstream_revision": UPSTREAM_REVISION,
        "profile_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "smoothed_linears": sorted(scales),
        "unsmoothed_linears": sorted(sites.keys() - scales.keys()),
        "calibration": profile.get("calibration", {}),
        "method": "FP32 channel balancing; BF16 rounding; divide fused into packing",
        "original_weights_modified": False,
        "task_quality_acceptance": "not_assessed",
    }
    return scales, evidence
