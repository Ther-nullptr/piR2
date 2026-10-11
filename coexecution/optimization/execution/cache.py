"""Serial Python call boundaries for projection caches; no GPU dependencies.

The policy still requires one worker and its private CUDA stream. The Python lock
does not synchronize GPU work or make submissions from different streams safe.
"""

import threading


class SerialForwardBoundary:
    """Clear per-call projection state and reject overlapping roots or resets.

    Install after Graph wrappers, so capture and replay both enter this boundary.
    Close before Graph/controller restoration. A Graph may restore an eager forward
    after a capture failure; adopt that target while retaining this outer boundary.
    Neither this wrapper nor its clear callback modifies the rolling action buffer.
    """

    def __init__(self, modules, clear, *, reset_modules=()):
        self.clear = clear
        self.lock = threading.Lock()
        self.saved = []
        self.closed = False
        seen = set()
        targets = [(module, "forward") for module in modules]
        targets.extend((module, "reset_streaming_buffer") for module in reset_modules)
        try:
            for module, name in targets:
                if (id(module), name) in seen:
                    continue
                seen.add((id(module), name))
                original = getattr(module, name)
                self.saved.append((module, name, original, name in module.__dict__))
                self._install(module, name, original)
        except BaseException:
            self.close()
            raise

    def _install(self, module, name, original):
        target = original

        def forward(*args, **kwargs):
            nonlocal target
            if not self.lock.acquire(blocking=False):
                raise RuntimeError("Optimized policy requires serial submissions")
            if self.closed:
                self.lock.release()
                raise RuntimeError("Optimized serial boundary is closed")
            try:
                self.clear()
                return target(*args, **kwargs)
            finally:
                try:
                    # Graph failure restores its eager entry inside the call.
                    # Keep the serial boundary around that restored entry.
                    current = getattr(module, name)
                    if current is not forward:
                        target = current
                        setattr(module, name, forward)
                    self.clear()
                finally:
                    self.lock.release()

        setattr(module, name, forward)

    def close(self):
        if self.closed:
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(
                "Cannot close an optimized boundary during an active call"
            )
        try:
            self.closed = True
            for module, name, original, had in reversed(self.saved):
                if had:
                    setattr(module, name, original)
                elif name in module.__dict__:
                    delattr(module, name)
            self.saved.clear()
            self.clear()
        finally:
            self.lock.release()


def clear_floating_projection_cache(groups):
    """Drop owner-scoped floating inputs after a root completes or fails."""
    for group in groups:
        group.active, group.cache = False, None
