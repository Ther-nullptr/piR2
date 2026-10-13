"""Instance-local contiguous sibling writeback for existing integer groups."""


def _clear_group(group):
    group._input = group._output = None
    group._remaining.clear()


class GroupOutputAdapters:
    """Replace only contiguous-output group projects; reuse resident weights.

    The controller's sibling forwards resolve group.project dynamically. This
    adapter uses the same _input/_output/_remaining fields as the reference
    group, so controller precision/mode changes clear any partial cycle. Every
    fresh cycle prepares one input and allocates fresh output segments; returned
    tensors remain valid after subsequent cycles. Defaults stay on the existing
    group implementation until set(4) or set(8) is called.
    """

    def __init__(self, controller):
        self.controller = controller
        self.saved = []

    def set(self, bits=None):
        if bits is not None and (type(bits) is not int or bits not in (4, 8)):
            raise ValueError("Group output precision bits must be 4, 8 or None")
        self.close()
        if bits is None:
            return
        try:
            for _, group in self.controller.grouped[bits]:
                if group.contiguous_outputs is not True:
                    continue
                _clear_group(group)
                self.saved.append((group, "project" in group.__dict__, group.project))
                group.project = lambda x, index, group=group: self._project(
                    group, x, index
                )
        except BaseException:
            self.close()
            raise

    def _project(self, group, x, index):
        try:
            if type(index) is not int or not 0 <= index < len(group.sizes):
                raise IndexError("Projection group index is outside the sibling range")
            if group._input is not x or index not in group._remaining:
                _clear_group(group)
                outputs = self._outputs(group, x)
                group._input = x
                group._output = outputs
                group._remaining.update(range(len(group.sizes)))
            output = group._output[index]
            group._remaining.remove(index)
            if not group._remaining:
                _clear_group(group)
            return output
        except Exception:
            _clear_group(group)
            raise

    def _outputs(self, group, x):
        from coexecution.kernels.ampere_ada.epilogue import group_outputs

        candidate = getattr(self.controller, "candidates", None)
        if candidate is not None and candidate.supports(group.linear, x):
            # Segmented writeback owns only reference GEMM epilogues. Preserve
            # an explicitly selected GEMV/small-M/split-K, then split its output.
            output = self.controller.dispatch(group.linear, x)
            return tuple(
                output.narrow(-1, offset, size).contiguous()
                for offset, size in zip(group.offsets, group.sizes)
            )
        packed = self.controller._pack_dispatch_input(group.linear, x, None, None)
        return group_outputs(packed, group)

    def close(self):
        for group, had_project, original in self.saved:
            _clear_group(group)
            if had_project:
                group.project = original
            elif "project" in group.__dict__:
                del group.project
        self.saved.clear()
