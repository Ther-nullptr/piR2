"""Build backbone structure while a complete task checkpoint supplies its weights."""

from contextlib import contextmanager


@contextmanager
def task_checkpoint_backbone():
    """Scope to serial, strict full-checkpoint loading; never use for a bare backbone."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    model_class = Qwen3VLForConditionalGeneration
    original = model_class.__dict__.get("from_pretrained")

    def from_config(cls, model_name, **kwargs):
        model_kwargs = {
            key: kwargs.pop(key)
            for key in ("torch_dtype", "dtype", "attn_implementation")
            if key in kwargs
        }
        config = Qwen3VLConfig.from_pretrained(model_name, **kwargs)
        return cls._from_config(config, **model_kwargs)

    model_class.from_pretrained = classmethod(from_config)
    try:
        yield
    finally:
        if original is None:
            del model_class.from_pretrained
        else:
            model_class.from_pretrained = original
