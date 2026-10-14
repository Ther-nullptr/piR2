"""Flow4 projection lifetime and accounting contracts; no neural computation."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.execution.guards import HostGuardPlan


def adapter():
    name = "coexecution.optimization.groot.kv_reuse"
    assert importlib.util.find_spec(name) is not None, "Missing Flow4 K/V scope"
    return importlib.import_module(name)


class Projection:
    def __init__(self):
        self.calls = []

    def forward(self, value):
        output = object()
        self.calls.append((value, output))
        return output


def topology():
    blocks = [
        SimpleNamespace(
            attn1=SimpleNamespace(
                is_cross_attention=index % 2 == 0,
                norm_cross=False,
                to_q=Projection(),
                to_k=Projection(),
                to_v=Projection(),
            )
        )
        for index in range(4)
    ]
    return SimpleNamespace(
        config=SimpleNamespace(interleave_self_attention=True),
        transformer_blocks=blocks,
    )


def step(dit, context):
    return tuple(
        tuple(
            getattr(block.attn1, role).forward(context)
            for role in ("to_q", "to_k", "to_v")
        )
        for block in dit.transformer_blocks
    )


def test_reuse_requires_explicit_whole_action_graph_and_supports_all_admitted_precisions():
    assert hasattr(OptimizationConfig(), "action_kv_reuse")
    assert not OptimizationConfig().action_kv_reuse
    for options in ({}, {"dit_graph": True}):
        with pytest.raises(ValueError, match="action.*graph"):
            OptimizationConfig(action_kv_reuse=True, **options)
    for precision in ("bf16", "w8a8", "w4a4"):
        config = OptimizationConfig(
            precision=precision, action_graph=True, action_kv_reuse=True
        )
        assert not config.fusion
        config.validate_variant("flow")
        with pytest.raises(ValueError, match="Flow4"):
            config.validate_variant("pir2")
    with pytest.raises(ValueError, match="bool"):
        OptimizationConfig(action_graph=True, action_kv_reuse=1)


def test_only_cross_kv_reuses_four_steps_and_every_body_starts_fresh():
    dit = topology()
    scope = adapter().FlowKVReuse(dit)
    with scope:
        for _ in range(2):
            context = object()
            with scope.body():
                results = [step(dit, context) for _ in range(4)]
                for index in range(4):
                    assert len({id(row[index][0]) for row in results}) == 4
                    for column in (1, 2):
                        assert len({id(row[index][column]) for row in results}) == (
                            1 if index % 2 == 0 else 4
                        )
            assert not scope.active and not scope.retained_count
        proof = scope.evidence(replays=3)
        assert proof["verified_body_count"] == 2
        assert proof["python_projection_compute_count"] == 8
        assert proof["python_projection_reuse_count"] == 24
        assert proof["expected_projection_count"] == 48
        assert proof["executed_projection_count"] == 12
        assert proof["avoided_projection_count"] == 36
        assert all(
            row["calls"] == 4 and row["computes"] == 1 and row["reuses"] == 3
            for row in proof["last_body_proof"]
        )
    assert all(
        len(getattr(block.attn1, role).calls)
        == (2 if index % 2 == 0 and role != "to_q" else 8)
        for index, block in enumerate(dit.transformer_blocks)
        for role in ("to_q", "to_k", "to_v")
    )


def test_body_failure_incomplete_steps_and_reentry_clear_all_outputs():
    dit = topology()
    scope = adapter().FlowKVReuse(dit)
    originals = [block.attn1.to_k.forward for block in dit.transformer_blocks]
    with scope:
        with pytest.raises(RuntimeError, match="injected"), scope.body():
            step(dit, object())
            raise RuntimeError("injected")
        assert not scope.active and not scope.retained_count
        with pytest.raises(RuntimeError, match="four"), scope.body():
            step(dit, object())
        with scope.body():
            with pytest.raises(RuntimeError, match="overlap"), scope.body():
                pass
            with pytest.raises(RuntimeError, match="active"):
                scope.close()
            for _ in range(4):
                step(dit, object())
        assert scope.evidence(replays=0)["verified_body_count"] == 1
    assert [block.attn1.to_k.forward for block in dit.transformer_blocks] == originals
    with scope, scope.body():
        for _ in range(4):
            step(dit, object())
    assert not scope.retained_count


def test_outside_body_delegates_and_close_cannot_restore_over_new_owner():
    dit = topology()
    scope = adapter().FlowKVReuse(dit)
    with scope:
        first, second = step(dit, object()), step(dit, object())
        assert first[0][1] is not second[0][1]
    newer = lambda value: value
    dit.transformer_blocks[0].attn1.to_k.forward = newer
    scope.close()
    assert dit.transformer_blocks[0].attn1.to_k.forward is newer


@pytest.mark.parametrize("instance_forward", [False, True])
def test_active_scope_close_preserves_replacement_after_guard_rejection(
    instance_forward,
):
    dit = topology()
    projection = dit.transformer_blocks[0].attn1.to_k
    if instance_forward:
        projection.forward = lambda value: ("original", value)
    untouched = dit.transformer_blocks[0].attn1.to_v
    original_untouched = untouched.forward
    model = SimpleNamespace(training=False)
    model.named_modules = lambda: iter([("", model)])
    model.named_parameters = lambda: iter([])
    model.named_buffers = lambda: iter([])
    with adapter().FlowKVReuse(dit) as scope:
        guard = HostGuardPlan(model, methods=(), fields=(), global_bindings=())
        scope.freeze(guard)
        assert guard.unchanged()
        newer = lambda value: ("new owner", value)
        projection.forward = newer
        assert not guard.unchanged()
        assert scope.entered and not scope.active
    assert projection.forward is newer
    assert untouched.forward == original_untouched
    scope.close()
    assert projection.forward is newer


@pytest.mark.parametrize("prior_body", [False, True])
def test_retired_wrappers_never_reattach_when_preserved_owner_is_wrapped_again(
    prior_body,
):
    dit = topology()
    projection = dit.transformer_blocks[0].attn1.to_k
    scope = adapter().FlowKVReuse(dit)
    with scope:
        retired = scope.saved[0][-1]
        if prior_body:
            with scope.body():
                for _ in range(4):
                    step(dit, object())

        def newer(value, original=projection.forward):
            return original(value)

        projection.forward = newer
    assert projection.forward is newer
    assert retired.output is None
    bodies = []
    with scope:
        for _ in range(4):
            with scope.body():
                results = [step(dit, object()) for _ in range(4)]
                assert all(row[0][1] is results[0][0][1] for row in results)
                bodies.append(results[0][0][1])
            assert retired.output is None
            assert not scope.retained_count
    assert len({id(value) for value in bodies}) == 4
    assert len(projection.calls) == 4 + int(prior_body)
    assert projection.forward is newer
    assert retired.output is None


def test_hidden_original_closure_and_dispatch_remain_frozen():
    dit = topology()
    neural = ["frozen"]

    def original(value):
        return neural, value

    dit.transformer_blocks[0].attn1.to_k.forward = original
    model = SimpleNamespace(training=False)
    model.named_modules = lambda: iter([("", model)])
    model.named_parameters = lambda: iter([])
    model.named_buffers = lambda: iter([])
    with adapter().FlowKVReuse(dit) as scope:
        guard = HostGuardPlan(model, methods=(), fields=(), global_bindings=())
        scope.freeze(guard)
        assert guard.unchanged()
        with scope.body():
            for _ in range(4):
                step(dit, object())
        assert guard.unchanged()
        neural.append("changed")
        assert not guard.unchanged()
        neural.pop()
        assert guard.unchanged()
        dit.transformer_blocks[0].attn1.to_k.forward = lambda value: value
        assert not guard.unchanged()


def test_hidden_group_and_fallback_defaults_keep_packed_constants_frozen():
    dit = topology()
    packed = object()
    linear = SimpleNamespace(tactic=0, packed_weight=packed)
    group = SimpleNamespace(linear=linear)

    def fallback(value, group=group):
        return value, group

    def original(value, fallback=fallback):
        return fallback(value)

    dit.transformer_blocks[0].attn1.to_k.forward = original
    model = SimpleNamespace(training=False)
    model.named_modules = lambda: iter([("", model)])
    model.named_parameters = lambda: iter([])
    model.named_buffers = lambda: iter([])
    with adapter().FlowKVReuse(dit) as scope:
        guard = HostGuardPlan(
            model,
            methods=(),
            fields=("linear", "tactic", "packed_weight"),
            global_bindings=(),
        )
        scope.freeze(guard)
        assert guard.unchanged()
        linear.tactic = 1
        assert not guard.unchanged()
        linear.tactic = 0
        linear.packed_weight = object()
        assert not guard.unchanged()
        linear.packed_weight = packed
        assert guard.unchanged()
        dit.config.interleave_self_attention = False
        assert not guard.unchanged()


@pytest.mark.parametrize("mutation", ["interleave", "cross", "norm_cross", "shared"])
def test_unadmitted_topologies_fail_before_installation(mutation):
    dit = topology()
    if mutation == "interleave":
        dit.config.interleave_self_attention = False
    elif mutation == "cross":
        dit.transformer_blocks[1].attn1.is_cross_attention = True
    elif mutation == "norm_cross":
        dit.transformer_blocks[0].attn1.norm_cross = True
    else:
        dit.transformer_blocks[2].attn1.to_k = dit.transformer_blocks[0].attn1.to_k
    with pytest.raises(ValueError):
        adapter().FlowKVReuse(dit)
