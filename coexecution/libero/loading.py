"""Build backbone structure while a complete task checkpoint supplies its weights."""

import json
from contextlib import contextmanager
from pathlib import Path


def checkpoint_decoded_horizon(checkpoint):
    """Read the LIBERO processor action length without constructing any model.

    Match the pinned policy loader: root processor metadata takes precedence;
    training checkpoints may instead save it under processor/. LIBERO_PANDA is
    the fixed service embodiment and resolves to the internal libero_sim key.
    Model config.action_horizon is the predicted horizon, not this decoded one.
    """
    checkpoint = Path(checkpoint)
    path = checkpoint / "processor_config.json"
    if not path.exists() and (checkpoint / "processor").is_dir():
        path = checkpoint / "processor" / "processor_config.json"
    try:
        metadata = json.loads(path.read_text())
        indices = metadata["processor_kwargs"]["modality_configs"]["libero_sim"][
            "action"
        ]["delta_indices"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Cannot read LIBERO decoded action horizon from {path}"
        ) from error
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int for index in indices)
    ):
        raise ValueError(f"Invalid LIBERO action delta_indices in {path}")
    return len(indices)


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
