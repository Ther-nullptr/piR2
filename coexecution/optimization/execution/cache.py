"""Serial Python call boundaries for projection caches; no GPU dependencies.

The policy still requires one worker and its private CUDA stream. The Python lock
does not synchronize GPU work or make submissions from different streams safe.
"""

import threading


class ProjectionCacheClearer:
    """Cold-bound group ownership; preserves the ordinary cache release protocol."""

    def __init__(self, groups):
        self.groups = tuple(groups)

    def __call__(self):
        for group in self.groups:
            group._input = group._output = None
            group._remaining.clear()


def projection_cache_clearers(grouped, precision):
    """Bind one active integer format and one root without hot discovery.

    Every group, including inactive formats, must have an unambiguous root and
    distinct mutable storage. Precision changes require rebinding after a full
    clear; callers retain their original full callback for reset and teardown.
    """
    if precision not in ("w8a8", "w4a4"):
        raise ValueError("Component cache clearing requires an integer precision")
    bits = {"w8a8": 8, "w4a4": 4}[precision]
    prefixes = {
        "vlm": "backbone.",
        "s1_feature": "action_head.vl_self_attention.",
        "dit": "action_head.model.",
    }
    selected = {name: [] for name in prefixes}
    seen = set()
    if bits not in grouped:
        raise ValueError("Selected integer precision has no projection inventory")
    for group_bits, records in grouped.items():
        for names, group in records:
            owners = {
                next(
                    (
                        name
                        for name, prefix in prefixes.items()
                        if site.startswith(prefix)
                    ),
                    None,
                )
                for site in names
            }
            if len(owners) != 1 or None in owners:
                raise ValueError("Projection group must belong to exactly one root")
            if id(group) in seen:
                raise ValueError("Mutable projection group is shared across records")
            seen.add(id(group))
            if group_bits == bits:
                selected[owners.pop()].append(group)
    return {name: ProjectionCacheClearer(groups) for name, groups in selected.items()}


class SerialForwardBoundary:
    """Clear per-call projection state and reject overlapping roots or resets.

    Install after Graph wrappers, so capture and replay both enter this boundary.
    Close before Graph/controller restoration. A Graph may restore an eager forward
    after a capture failure; adopt that target while retaining this outer boundary.
    Neither this wrapper nor its clear callback modifies the rolling action buffer.
    Optional forward callbacks narrow cache ownership while retaining one shared
    lock. Resets and close always use the original full clear callback.
    """

    def __init__(self, modules, clear, *, reset_modules=(), forward_clears=()):
        modules = tuple(modules)
        known = {id(module) for module in modules}
        per_root = {}
        for module, callback in forward_clears:
            if id(module) not in known or id(module) in per_root:
                raise ValueError(
                    "Each forward clear must name one distinct guarded root"
                )
            if not callable(callback):
                raise TypeError("Forward cache clear must be callable")
            per_root[id(module)] = callback
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
                callback = (
                    per_root.get(id(module), clear) if name == "forward" else clear
                )
                self._install(module, name, original, callback)
        except BaseException:
            self.close()
            raise

    def _install(self, module, name, original, clear):
        target = original

        def forward(*args, **kwargs):
            nonlocal target
            if not self.lock.acquire(blocking=False):
                raise RuntimeError("Optimized policy requires serial submissions")
            if self.closed:
                self.lock.release()
                raise RuntimeError("Optimized serial boundary is closed")
            try:
                clear()
                return target(*args, **kwargs)
            finally:
                try:
                    # Graph failure restores its eager entry inside the call.
                    # Keep the serial boundary around that restored entry.
                    current = getattr(module, name)
                    if current is not forward:
                        target = current
                        setattr(module, name, forward)
                    clear()
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
