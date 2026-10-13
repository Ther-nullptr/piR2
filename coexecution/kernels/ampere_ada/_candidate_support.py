"""Shared guards for explicitly selected Ada kernel candidates."""

from functools import cache


@cache
def is_ada_device(device):
    """Candidate cubins target SM89; other devices retain reference kernels."""
    import torch

    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 9)


def require_pre_capture_build():
    """JIT work must finish before a CUDA Graph starts capturing launches."""
    import torch

    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Load and warm integer candidates before CUDA Graph capture")


def owned_build_identity():
    """Include owned headers in JIT identities, which Torch otherwise omits."""
    from coexecution.kernels.sources import verified_source_identity

    identity = verified_source_identity(backend="integer")
    return identity["source_sha256"].encode()


class _ProjectionTactic:
    def __init__(self, projection, tactic):
        self.projection = projection
        self.tactic = tactic

    def __getattr__(self, name):
        return getattr(self.projection, name)


def projection_for_rows(projection, rows):
    """Honor the existing shape tactic without mutating resident projections."""
    tactic = getattr(projection, "shape_tactics", {}).get(rows, projection.tactic)
    if tactic == projection.tactic:
        return projection
    return _ProjectionTactic(projection, tactic)
