"""Pure Python function rebinding; no Torch, Transformers or neural execution."""

import importlib
import importlib.util
from functools import wraps

import pytest

MODULE = "coexecution.optimization.execution.patching"


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, (
        "Pure patching module is missing"
    )
    return importlib.import_module(MODULE).clone_forward


def old_binding(value):
    return value + 1


def test_clone_rebinds_globals_without_modifying_the_original_function():
    def forward(value=3, *, multiplier=2):
        return old_binding(value) * multiplier

    forward.description = "original attribute"
    cloned = implementation()(forward, {"old_binding": lambda value: value + 10})
    assert forward() == 8
    assert cloned() == 26
    assert cloned(multiplier=3) == 39
    assert cloned.__defaults__ == forward.__defaults__
    assert cloned.__kwdefaults__ == forward.__kwdefaults__
    assert cloned.description == forward.description
    assert forward.__globals__["old_binding"] is old_binding


def test_clone_rebinds_wrapped_closure_and_preserves_wrapper_semantics():
    calls = []

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            calls.append("wrapper")
            return function(*args, **kwargs)

        return wrapper

    @decorator
    @decorator
    def forward(value):
        return old_binding(value)

    cloned = implementation()(forward, {"old_binding": lambda value: value + 20})
    assert cloned(3) == 23 and calls == ["wrapper", "wrapper"]
    calls.clear()
    assert forward(3) == 4 and calls == ["wrapper", "wrapper"]
    assert cloned.__wrapped__ is not forward.__wrapped__
    assert cloned.__wrapped__.__wrapped__ is not forward.__wrapped__.__wrapped__
    assert cloned.__name__ == forward.__name__


def test_clone_preserves_nonwrapped_closure_cells_and_annotations():
    sentinel = object()

    def forward(value: int) -> tuple:
        return sentinel, old_binding(value)

    cloned = implementation()(forward, {"old_binding": lambda value: value - 4})
    assert cloned(8) == (sentinel, 4)
    assert cloned.__annotations__ == forward.__annotations__
    assert cloned.__closure__[0] is forward.__closure__[0]


@pytest.mark.parametrize("invalid", [object(), 1, None])
def test_clone_requires_a_python_function(invalid):
    with pytest.raises(TypeError, match="Python function"):
        implementation()(invalid, {})


def test_effective_callable_stamp_is_stable_for_bound_methods_and_changes_on_replacement():
    stamp = getattr(importlib.import_module(MODULE), "effective_callable_stamp", None)
    assert callable(stamp), "Metadata producer callable stamp is missing"

    class Producer:
        def method(self, value):
            return value

    owner, other = Producer(), Producer()
    assert stamp(owner.method) == stamp(owner.method)
    assert stamp(owner.method) != stamp(other.method)
    before = stamp(owner.method)
    owner.method = lambda value: (value, "replacement")
    assert stamp(owner.method) != before


def test_metadata_producer_stamp_tracks_each_effective_producer():
    stamp = getattr(importlib.import_module(MODULE), "metadata_producer_stamp", None)
    assert callable(stamp), "Shared metadata producer stamp is missing"
    from types import SimpleNamespace

    inner = SimpleNamespace(get_rope_index=lambda values: values)
    visual = SimpleNamespace(
        fast_pos_embed_interpolate=lambda values: values,
        rot_pos_emb=lambda values: values,
        rotary_pos_emb=SimpleNamespace(forward=lambda values: values),
    )
    mask = lambda values: values
    before = stamp(inner, visual, mask)
    for owner, name in (
        (inner, "get_rope_index"),
        (visual, "fast_pos_embed_interpolate"),
        (visual, "rot_pos_emb"),
        (visual.rotary_pos_emb, "forward"),
    ):
        original = getattr(owner, name)
        setattr(owner, name, lambda values: (values, "new"))
        assert stamp(inner, visual, mask) != before
        setattr(owner, name, original)
    assert stamp(inner, visual, lambda values: values) != before
