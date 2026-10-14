"""Explicit integer candidate dispatch over the canonical packed representation.

The default returns control to the existing controller. Candidate selection never
autotunes, runs timing work, or persists an input. Bind this helper before graph
capture and warm the explicitly selected CUDA extension on representative input.
"""

import math

from coexecution.kernels.ampere_ada._candidate_support import (
    is_ada_device,
    projection_for_rows,
)


class IntegerCandidateDispatch:
    def __init__(
        self,
        controller,
        *,
        implementation="reference",
        small_m_tactic=0,
        split_k_slices=2,
        split_k_tile=0,
    ):
        if implementation not in ("reference", "int4_gemv", "small_m", "split_k"):
            raise ValueError("Unknown integer GEMM candidate")
        if type(small_m_tactic) is not int or small_m_tactic not in range(6):
            raise ValueError("Small-M tactic must be an integer from 0 to 5")
        if type(split_k_slices) is not int or split_k_slices not in (2, 4, 8):
            raise ValueError("Split-K slices must be 2, 4 or 8")
        if type(split_k_tile) is not int or split_k_tile not in (0, 1, 2):
            raise ValueError("Split-K tile must be 0, 1 or 2")
        self.controller = controller
        self.implementation = implementation
        self.small_m_tactic = small_m_tactic
        self.split_k_slices = split_k_slices
        self.split_k_tile = split_k_tile

    @staticmethod
    def _leading(q, x):
        if getattr(x, "_robotics_packed_input", False):
            return tuple(x.for_bits(q.bits).shape)
        return tuple(x.shape[:-1])

    def pack_input(self, q, x, lut=None, up=None):
        if getattr(x, "_robotics_packed_input", False):
            return x.for_bits(q.bits)
        if self.controller.native:
            import torch

            from coexecution.kernels.ampere_ada.integer import PackedActivation

            data, scales = torch.ops.pir2_integer_dispatch.prepare(x, lut, up, q.bits)
            return PackedActivation(
                data, scales, tuple(x.shape[:-1]), x.shape[-1], q.bits
            )
        from coexecution.optimization.backends.integer.linear import prepare

        return prepare(x, q.bits, lut, up)

    def _kernel(self, q, rows):
        """Resolve metadata before any activation packing or JIT build."""
        if self.implementation == "reference" or not is_ada_device(
            q.packed_weight.device
        ):
            return None
        if self.implementation == "int4_gemv":
            from coexecution.kernels.ampere_ada import int4_gemv

            return int4_gemv if int4_gemv.eligible_shape(rows, q) else None
        if self.implementation == "small_m":
            from coexecution.kernels.ampere_ada import int4_small_m

            return int4_small_m if int4_small_m.eligible_shape(rows, q) else None
        from coexecution.kernels.ampere_ada import split_k

        return split_k if split_k.eligible_shape(rows, q) else None

    def dispatch(self, q, x, lut=None, up=None):
        """Return a logical candidate result, or None for controller fallback."""
        if self.implementation == "reference":
            return None
        rows = math.prod(self._leading(q, x))
        q = projection_for_rows(q, rows)
        kernel = self._kernel(q, rows)
        if kernel is None:
            return None
        packed = self.pack_input(q, x, lut, up)
        if self.implementation == "int4_gemv":
            output = kernel.int4_gemv(packed, q)
        elif self.implementation == "small_m":
            output = kernel.small_m_gemm(packed, q, self.small_m_tactic)
        else:
            output = kernel.split_k_gemm(
                packed, q, self.split_k_slices, self.split_k_tile
            )
        return output[:, : q.out_features].reshape(*packed.shape, q.out_features)

    def supports(self, q, x):
        """Whether a selected GEMM candidate would replace this reference call."""
        if self.implementation == "reference":
            return False
        rows = math.prod(self._leading(q, x))
        return self._kernel(projection_for_rows(q, rows), rows) is not None

    def residual(self, q, x, residual, lut=None, up=None):
        """Fuse only eligible projections; preserve the selected GEMM elsewhere."""
        import torch

        from coexecution.kernels.ampere_ada import epilogue, split_k

        leading = self._leading(q, x)
        rows = math.prod(leading)
        q = projection_for_rows(q, rows)

        def fallback():
            return self.controller.dispatch(q, x, lut, up) + residual

        if not (
            is_ada_device(q.packed_weight.device)
            and 0 < rows <= 64
            and q.out_features == q.packed_weight.shape[0]
            and isinstance(residual, torch.Tensor)
            and residual.device == q.packed_weight.device
            and residual.dtype == torch.bfloat16
            and tuple(residual.shape) == (*leading, q.out_features)
            and residual.is_contiguous()
            and residual.data_ptr() % 16 == 0
        ):
            return fallback()
        selected = self._kernel(q, rows)
        # A residual writer must not silently replace a selected GEMV/small-M
        # implementation; only split-K has its own equivalent residual writer.
        if selected is not None and self.implementation != "split_k":
            return fallback()
        use_split = selected is split_k
        if not use_split and not epilogue.eligible_shape(rows, q):
            return fallback()
        packed = self.pack_input(q, x, lut, up)
        matrix = residual.view(rows, q.out_features)
        if use_split:
            output = split_k.split_k_gemm(
                packed,
                q,
                self.split_k_slices,
                self.split_k_tile,
                residual=matrix,
            )
        else:
            output = epilogue.residual_gemm(packed, q, matrix)
        return output.view(*leading, q.out_features)
