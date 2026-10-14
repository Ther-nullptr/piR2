"""Body-local cross-attention projections for a captured, unchanged Flow4.

This scope is used only inside ActionGraph's full TensorGraphCall function. Each
warmup/capture enters an empty scope: the first K/V computation is captured and
therefore executes again on every replay from freshly copied graph inputs.
No tensor identity/address is used as a freshness guarantee. Import is GPU-free.
"""

from collections import Counter
from contextlib import contextmanager


class _Projection:
    def __init__(self, scope):
        self.scope = scope
        self.calls = 0
        self.output = None

    def call(self, original, value):
        if self.scope is None or not self.scope.active:
            return original(value)
        self.calls += 1
        if self.calls > 4:
            raise RuntimeError("K/V reuse requires exactly four projection calls")
        if self.calls == 1:
            # Keep the returned storage alive independently of per-DiT group
            # cache resets. Existing grouped projections still execute intact.
            self.output = original(value)
        return self.output


class FlowKVReuse:
    """Reversible projection wrappers; all tensors die at full Flow4 body exit.

    The caller admits the pinned AlternateVLDiT and its frozen processors/methods.
    Q, self-attention, masks and timestep-dependent work remain untouched. Both the
    standard SDPA processor and PreparedAttention call these same K/V projections.
    """

    def __init__(self, dit):
        self.dit = dit
        self.targets = self._targets()
        self.saved = ()
        self.active = self.entered = False
        self.counts = Counter()
        self.last_body_proof = []

    def _targets(self):
        if not self.dit.config.interleave_self_attention:
            raise ValueError("K/V reuse requires interleaved cross/self attention")
        targets, seen = [], set()
        for index, block in enumerate(self.dit.transformer_blocks):
            attn = block.attn1
            if attn.is_cross_attention != (index % 2 == 0) or attn.norm_cross:
                raise ValueError("K/V reuse requires unchanged cross-attention context")
            for role in ("to_q", "to_k", "to_v"):
                projection = getattr(attn, role)
                if id(projection) in seen:
                    raise ValueError("K/V reuse requires independent layer projections")
                seen.add(id(projection))
                if index % 2 == 0 and role != "to_q":
                    targets.append((index, role, projection))
        if not targets:
            raise ValueError("K/V reuse requires cross-attention projections")
        return tuple(targets)

    def __enter__(self):
        if self.entered:
            raise RuntimeError("K/V reuse scope is already active")
        self.targets = self._targets()
        saved = []
        try:
            for index, role, module in self.targets:
                original, had = module.forward, "forward" in vars(module)
                projection = _Projection(self)

                # Keep the original callable in the wrapper's immutable defaults.
                # HostGuardPlan and captured_tensors traverse it, including hidden
                # integer group weights/tactics and reference fallback closures.
                def forward(value, original=original, projection=projection):
                    return projection.call(original, value)

                saved.append((index, role, module, original, had, forward, projection))
                module.forward = forward
        except BaseException:
            self.saved = tuple(saved)
            self.close()
            raise
        self.saved = tuple(saved)
        self.entered = True
        return self

    def freeze(self, guard):
        """Freeze wrapper dispatch plus original closures, excluding transient data."""
        guard.attribute(self.dit.config, "interleave_self_attention")
        guard.attribute(self, "body", method=True)
        for _index, _role, module, _original, _had, _wrapper, projection in self.saved:
            guard.attribute(module, "forward", method=True)
            guard.attribute(projection, "scope")
            guard.attribute(projection, "call", method=True)

    @property
    def retained_count(self):
        return sum(projection.output is not None for *_, projection in self.saved)

    @contextmanager
    def body(self):
        if not self.entered:
            raise RuntimeError("K/V reuse requires an installed scope")
        if self.active:
            raise RuntimeError("K/V reuse bodies cannot overlap")
        self.active = True
        for *_, projection in self.saved:
            projection.calls, projection.output = 0, None
        try:
            yield
            if any(projection.calls != 4 for *_, projection in self.saved):
                raise RuntimeError("K/V reuse requires exactly four projection calls")
            self.last_body_proof = [
                {
                    "block": index,
                    "projection": role,
                    "calls": 4,
                    "computes": 1,
                    "reuses": 3,
                }
                for index, role, *_ in self.saved
            ]
            self.counts["verified_body_count"] += 1
            self.counts["python_projection_compute_count"] += len(self.saved)
            self.counts["python_projection_reuse_count"] += 3 * len(self.saved)
        finally:
            self.active = False
            for *_, projection in self.saved:
                projection.output = None

    def close(self):
        if self.active:
            raise RuntimeError("Cannot close K/V reuse during an active body")
        for _index, _role, module, original, had, wrapper, projection in reversed(
            self.saved
        ):
            projection.output = None
            # A preserved owner may delegate to this wrapper after close. Retire
            # it permanently so reopening this scope cannot reactivate its cache.
            projection.scope = None
            # A frozen-execution rejection may be caused by a newer owner.
            # Restore only the forward installed by this scope.
            if getattr(module, "forward", None) is not wrapper:
                continue
            if had:
                module.forward = original
            elif "forward" in vars(module):
                del module.forward
        self.saved = ()
        self.entered = False

    def __exit__(self, *exception):
        self.close()

    def evidence(self, *, replays):
        projections = len(self.targets)
        return {
            "enabled": True,
            "scope": "one captured Flow4 body; no cross-observation cache",
            "unit": "K/V projection evaluations, not grouped GEMM/kernel launches",
            "cross_attention_layers": sorted({index for index, *_ in self.targets}),
            "projections_per_body": projections,
            **{
                name: self.counts[name]
                for name in (
                    "verified_body_count",
                    "python_projection_compute_count",
                    "python_projection_reuse_count",
                )
            },
            "last_body_proof": self.last_body_proof,
            "replay_count": replays,
            "expected_projection_count": replays * projections * 4,
            "executed_projection_count": replays * projections,
            "avoided_projection_count": replays * projections * 3,
            "count_source": "verified captured body multiplied by explicit replays",
            "python_counts": "successful warmup/capture bodies only; replay bypasses Python",
            "retained_tensors": self.retained_count,
        }
