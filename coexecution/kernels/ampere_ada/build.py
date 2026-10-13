"""Integer source metadata shared by the canonical and candidate kernels."""

from coexecution.kernels.build import build_inputs as _build_inputs


def build_inputs():
    return _build_inputs("integer")
