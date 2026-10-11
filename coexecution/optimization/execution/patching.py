"""Instance-local Python rebinding without loading model or kernel dependencies."""

from types import FunctionType


def effective_callable_stamp(method):
    """Stable bound-method identity without retaining ephemeral method objects."""
    function = getattr(method, "__func__", method)
    return (
        id(function),
        id(getattr(method, "__self__", None)),
        getattr(function, "__code__", None),
    )


def metadata_producer_stamp(inner, visual, causal_mask):
    """One producer signature shared by input registration and vision caches."""
    return tuple(
        effective_callable_stamp(method)
        for method in (
            inner.get_rope_index,
            visual.fast_pos_embed_interpolate,
            visual.rot_pos_emb,
            visual.rotary_pos_emb.forward,
            causal_mask,
        )
    )


def clone_forward(function, replacements):
    """Clone globals and the wrapped-function closure, preserving decorators.

    Transformers' wrappers call a closed-over function rather than a global
    symbol. Rebinding only wrapper globals therefore leaves the model's actual
    helper untouched. Neither the original globals nor closure cells are edited.
    """
    if not isinstance(function, FunctionType):
        raise TypeError("clone_forward requires a Python function")
    wrapped = getattr(function, "__wrapped__", None)
    replacement_wrapped = (
        clone_forward(wrapped, replacements)
        if isinstance(wrapped, FunctionType)
        else wrapped
    )

    def cell(value):
        return (lambda: value).__closure__[0]

    closure = function.__closure__
    if wrapped is not None and closure:
        closure = tuple(
            cell(replacement_wrapped) if item.cell_contents is wrapped else item
            for item in closure
        )
    result = FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        function.__name__,
        function.__defaults__,
        closure,
    )
    result.__dict__.update(function.__dict__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__qualname__ = function.__qualname__
    result.__module__ = function.__module__
    result.__doc__ = function.__doc__
    if wrapped is not None:
        result.__wrapped__ = replacement_wrapped
    return result
